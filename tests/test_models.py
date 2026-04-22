from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from amp_vae.config import CONDITION_COLUMNS
from amp_vae.data.tokenizer import build_vocab_bundle
from amp_vae.models.classifier import build_external_classifier
from amp_vae.models.cvae import build_cvae, infer_cvae_hparams_from_state_dict
from amp_vae.inference.generate import align_condition_to_model
from amp_vae.training.cvae import cvae_loss, cyclical_beta, get_kl_weight


class ModelShapeTest(unittest.TestCase):
    def setUp(self):
        self.vocab = build_vocab_bundle(["ACDEFG"], condition_cols=CONDITION_COLUMNS, max_len=10)

    def test_cvae_defaults_to_active_condition_schema(self):
        model = build_cvae(self.vocab)
        self.assertEqual(model.cond_dim, len(CONDITION_COLUMNS))

    def test_classifier_outputs_one_logit_per_active_label(self):
        model = build_external_classifier(self.vocab, num_labels=len(CONDITION_COLUMNS))
        x = torch.tensor([[self.vocab.sos_idx, self.vocab.char2idx["A"], self.vocab.eos_idx] + [self.vocab.pad_idx] * 7])
        lengths = torch.tensor([3])
        logits = model(x, lengths)
        self.assertEqual(tuple(logits.shape), (1, len(CONDITION_COLUMNS)))

    def test_infer_hparams_reads_cond_dim_from_checkpoint(self):
        model = build_cvae(self.vocab, cond_dim=len(CONDITION_COLUMNS))
        inferred = infer_cvae_hparams_from_state_dict(model.state_dict())
        self.assertEqual(inferred["cond_dim"], len(CONDITION_COLUMNS))
        self.assertEqual(inferred["latent_dim"], model.latent_dim)

    def test_align_condition_requires_exact_dimension_match(self):
        model = build_cvae(self.vocab, cond_dim=len(CONDITION_COLUMNS))
        aligned = align_condition_to_model([1, 0, 0, 0, 0, 0], model)
        self.assertEqual(aligned, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_cvae_stores_special_token_ids(self):
        model = build_cvae(self.vocab, cond_dim=len(CONDITION_COLUMNS))
        self.assertEqual(model.pad_idx, self.vocab.pad_idx)
        self.assertEqual(model.unk_idx, self.vocab.unk_idx)
        self.assertEqual(model.sos_idx, self.vocab.sos_idx)
        self.assertEqual(model.eos_idx, self.vocab.eos_idx)

    def test_word_dropout_preserves_sos_eos_pad(self):
        # Force rate=1.0 to verify mask exclusions: every replaceable token becomes UNK,
        # but PAD, SOS, EOS must stay intact.
        model = build_cvae(self.vocab, cond_dim=len(CONDITION_COLUMNS))
        model.train()
        torch.manual_seed(0)
        v = self.vocab
        x = torch.tensor(
            [[v.sos_idx, v.char2idx["A"], v.char2idx["C"], v.eos_idx, v.pad_idx, v.pad_idx]],
            dtype=torch.long,
        )
        z = torch.zeros(1, model.latent_dim)
        cond = torch.zeros(1, model.cond_dim)

        # Use a forward pre-hook on the embedding layer to capture the (possibly
        # UNK-replaced) token tensor the decoder actually looks up.
        seen = {}

        def pre_hook(_module, inputs):
            seen["tokens"] = inputs[0].detach().clone()
            return None

        handle = model.decoder.embedding.register_forward_pre_hook(pre_hook)
        try:
            model.decoder(
                x,
                z,
                cond,
                model.pad_idx,
                model.unk_idx,
                sos_idx=model.sos_idx,
                eos_idx=model.eos_idx,
                word_dropout_rate=1.0,
            )
        finally:
            handle.remove()

        mutated = seen["tokens"][0]
        self.assertEqual(int(mutated[0].item()), v.sos_idx, "SOS must be preserved")
        self.assertEqual(int(mutated[3].item()), v.eos_idx, "EOS must be preserved")
        self.assertEqual(int(mutated[4].item()), v.pad_idx, "PAD must be preserved")
        self.assertEqual(int(mutated[5].item()), v.pad_idx, "PAD must be preserved")
        # Interior AA tokens should all become UNK at rate=1.0.
        self.assertEqual(int(mutated[1].item()), v.unk_idx)
        self.assertEqual(int(mutated[2].item()), v.unk_idx)


class TrainingLossScheduleTest(unittest.TestCase):
    def test_cyclical_beta_fu_2019(self):
        period, R, beta_min = 10, 0.5, 1e-3
        # Epoch 0: floored at beta_min (tau = 0).
        self.assertAlmostEqual(cyclical_beta(0, period=period, R=R, beta_min=beta_min), beta_min)
        # Halfway through ramp (tau=0.25), should be around 0.5.
        self.assertAlmostEqual(cyclical_beta(2, period=period, R=R, beta_min=beta_min), 0.4, places=5)
        # End of ramp (tau>=R) -> 1.0.
        self.assertAlmostEqual(cyclical_beta(5, period=period, R=R, beta_min=beta_min), 1.0)
        # Hold region.
        self.assertAlmostEqual(cyclical_beta(9, period=period, R=R, beta_min=beta_min), 1.0)
        # New cycle resets to beta_min, not 0.
        self.assertAlmostEqual(cyclical_beta(10, period=period, R=R, beta_min=beta_min), beta_min)

    def test_get_kl_weight_after_annealing_is_one(self):
        # Past total_anneal_epochs = n_cycles * period, schedule clamps to 1.
        self.assertEqual(get_kl_weight(40, cyclical_period=10, n_cycles=4), 1.0)

    def test_cvae_loss_per_sequence_scale(self):
        # Doubling batch size with identical content should leave per-seq loss roughly
        # unchanged, confirming per-sequence convention (not per-token, not per-batch).
        torch.manual_seed(0)
        V, B, T, D = 24, 4, 6, 8
        logits = torch.randn(B, T, V)
        target = torch.randint(0, V, (B, T))
        mu = torch.randn(B, D)
        logvar = torch.zeros(B, D)
        criterion = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=0)
        _, recon_b, kl_b = cvae_loss(logits, target, mu, logvar, criterion, beta=1.0, free_bits=0.0)

        logits2 = logits.repeat(2, 1, 1)
        target2 = target.repeat(2, 1)
        mu2 = mu.repeat(2, 1)
        logvar2 = logvar.repeat(2, 1)
        _, recon_2b, kl_2b = cvae_loss(logits2, target2, mu2, logvar2, criterion, beta=1.0, free_bits=0.0)

        self.assertAlmostEqual(float(recon_b), float(recon_2b), places=4)
        self.assertAlmostEqual(float(kl_b), float(kl_2b), places=4)


if __name__ == "__main__":
    unittest.main()
