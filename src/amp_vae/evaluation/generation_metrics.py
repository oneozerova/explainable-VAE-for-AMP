"""Basic generation metrics."""

from __future__ import annotations

from typing import Iterable, Sequence


def validity_rate(sequences: Sequence[str], min_len: int = 5, max_len: int = 50, valid_aa: Iterable[str] = "ACDEFGHIKLMNPQRSTVWY") -> float:
    valid_aa = set(valid_aa)
    if not sequences:
        return 0.0
    valid = 0
    for seq in sequences:
        seq = str(seq).strip().upper()
        if seq and min_len <= len(seq) <= max_len and set(seq).issubset(valid_aa):
            valid += 1
    return valid / len(sequences)


def uniqueness_rate(sequences: Sequence[str]) -> float:
    if not sequences:
        return 0.0
    return len(set(sequences)) / len(sequences)


def novelty_rate(sequences: Sequence[str], reference_sequences: Sequence[str]) -> float:
    if not sequences:
        return 0.0
    reference = set(reference_sequences)
    novel = sum(1 for seq in sequences if seq not in reference)
    return novel / len(sequences)

