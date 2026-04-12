"""Generation helpers for the conditional VAE."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import torch

from ..config import CONDITION_COLUMNS, GENERATION_CONDITION_COLUMNS, VALID_AMINO_ACIDS
from ..data.tokenizer import VocabBundle


def condition_vector(**kwargs) -> list[int]:
    return [int(kwargs.get(name, 0)) for name in CONDITION_COLUMNS]


def generation_condition_vector(**kwargs) -> list[int]:
    """Build a condition vector for generation using the active 6-label schema."""
    return [int(kwargs.get(name, 0)) for name in GENERATION_CONDITION_COLUMNS]


def normalize_condition_name(name: str) -> str:
    name = str(name).lower().replace("+", "plus")
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


def is_valid_sequence(seq: str, min_len: int = 5, max_len: int = 50, valid_aa: Iterable[str] = VALID_AMINO_ACIDS) -> bool:
    if not seq:
        return False
    if not (min_len <= len(seq) <= max_len):
        return False
    return set(seq).issubset(set(valid_aa))


def decode_from_z_cond(
    model,
    z: torch.Tensor,
    cond_t: torch.Tensor,
    vocab: VocabBundle,
    max_gen_len: int | None = None,
    temperature: float = 0.8,
):
    model.eval()
    max_gen_len = int(max_gen_len or vocab.max_len)
    z_cond = torch.cat([z, cond_t], dim=1)
    h = torch.tanh(model.decoder.init_h(z_cond)).unsqueeze(0)
    c = torch.tanh(model.decoder.init_c(z_cond)).unsqueeze(0)

    token = torch.full((z.size(0), 1), vocab.sos_idx, dtype=torch.long, device=z.device)
    finished = torch.zeros(z.size(0), dtype=torch.bool, device=z.device)
    sequences = [[] for _ in range(z.size(0))]

    for _ in range(max_gen_len):
        emb = model.decoder.embedding(token)
        dec_inp = torch.cat([emb, z_cond.unsqueeze(1)], dim=-1)
        out, (h, c) = model.decoder.lstm(dec_inp, (h, c))
        logits = model.decoder.fc_out(out[:, -1, :]) / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        token = next_token

        for i, idx in enumerate(next_token.squeeze(1).tolist()):
            if finished[i]:
                continue
            if idx == vocab.eos_idx:
                finished[i] = True
            elif idx not in (vocab.pad_idx, vocab.sos_idx):
                sequences[i].append(vocab.idx2char.get(idx, ""))

        if bool(finished.all()):
            break

    return ["".join(seq) for seq in sequences]


@torch.no_grad()
def generate_batch(
    model,
    vocab: VocabBundle,
    condition: Iterable[int],
    n_samples: int = 128,
    temperature: float = 0.8,
    alpha: float = 1.0,
    device=None,
):
    device = device or next(model.parameters()).device
    cond_arr = np.asarray(list(condition), dtype=np.float32) * float(alpha)
    cond_t = torch.tensor(cond_arr, dtype=torch.float32, device=device).unsqueeze(0).repeat(n_samples, 1)
    z = torch.randn(n_samples, model.latent_dim, device=device)
    return decode_from_z_cond(model, z, cond_t, vocab=vocab, max_gen_len=vocab.max_len, temperature=temperature)
