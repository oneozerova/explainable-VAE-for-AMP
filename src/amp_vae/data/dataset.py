"""Dataset wrappers used by training scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset

from .schema import SEQUENCE_COLUMN
from .tokenizer import VocabBundle, tokenize_sequence


class SequenceConditionDataset(Dataset):
    def __init__(self, sequences: Sequence[str], lengths: Sequence[int], conditions: Sequence[Sequence[float]], vocab: VocabBundle):
        self.vocab = vocab
        self.sequences = [str(seq) for seq in sequences]
        self.lengths = [int(length) for length in lengths]
        self.conditions = torch.tensor(conditions, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        tokens, seq_len = tokenize_sequence(self.sequences[idx], self.vocab)
        length = self.lengths[idx] if idx < len(self.lengths) else seq_len
        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            self.conditions[idx],
        )


class SequenceLabelDataset(Dataset):
    def __init__(self, frame, sequence_col: str, target_cols: Sequence[str], vocab: VocabBundle):
        self.frame = frame.reset_index(drop=True).copy()
        self.sequence_col = sequence_col
        self.target_cols = list(target_cols)
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        tokens, seq_len = tokenize_sequence(row[self.sequence_col], self.vocab)
        labels = row[self.target_cols].astype(float).tolist()
        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(seq_len, dtype=torch.long),
            torch.tensor(labels, dtype=torch.float32),
            str(row[self.sequence_col]),
        )

