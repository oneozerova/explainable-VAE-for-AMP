"""Scoring helpers for generated peptides."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from ..config import CONDITION_COLUMNS
from ..data.tokenizer import VocabBundle


def encode_sequences(sequences: Sequence[str], vocab: VocabBundle, device=None):
    device = device or "cpu"
    token_rows = []
    lengths = []
    for seq in sequences:
        tokens, seq_len = _encode_single(seq, vocab)
        token_rows.append(tokens)
        lengths.append(seq_len)
    return (
        torch.tensor(token_rows, dtype=torch.long, device=device),
        torch.tensor(lengths, dtype=torch.long, device=device),
    )


def _encode_single(seq: str, vocab: VocabBundle) -> tuple[list[int], int]:
    seq = str(seq).strip().upper()
    tokens = [vocab.sos_idx] + [vocab.char2idx.get(ch, vocab.unk_idx) for ch in seq] + [vocab.eos_idx]
    seq_len = min(len(tokens), vocab.max_len)
    tokens = tokens[: vocab.max_len]
    if len(tokens) < vocab.max_len:
        tokens = tokens + [vocab.pad_idx] * (vocab.max_len - len(tokens))
    return tokens, seq_len


@torch.no_grad()
def predict_probabilities(model, sequences: Sequence[str], vocab: VocabBundle, device=None, batch_size: int = 256) -> np.ndarray:
    device = device or next(model.parameters()).device
    outputs = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        if not batch:
            continue
        x, lengths = encode_sequences(batch, vocab, device=device)
        logits = model(x, lengths)
        outputs.append(torch.sigmoid(logits).cpu().numpy())
    if not outputs:
        return np.empty((0, len(CONDITION_COLUMNS)), dtype=np.float32)
    return np.concatenate(outputs, axis=0)


def load_thresholds(path: str | Path) -> dict[str, float]:
    import json

    with open(Path(path), "r", encoding="utf-8") as f:
        payload = json.load(f)
    thresholds = payload.get("best_thresholds", payload)
    return {str(k): float(v) for k, v in thresholds.items()}

