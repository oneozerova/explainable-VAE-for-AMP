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


if __name__ == "__main__":
    unittest.main()
