"""External AMP classifier used for post-hoc screening."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ..config import CONDITION_COLUMNS
from ..data.tokenizer import VocabBundle


class ExternalAMPClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int | None, pad_idx: int, embed_dim: int = 128, hidden_dim: int = 256, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        num_labels = int(num_labels or len(CONDITION_COLUMNS))
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 if num_layers == 1 else dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim * 4)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_labels),
        )
        self.pad_idx = pad_idx

    def forward(self, x, lengths):
        emb = self.dropout(self.embedding(x))
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.encoder(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))

        mask = (x != self.pad_idx).unsqueeze(-1)
        out_masked = out.masked_fill(~mask, 0.0)
        denom = mask.sum(dim=1).clamp(min=1)
        mean_pool = out_masked.sum(dim=1) / denom

        neg_inf = torch.finfo(out.dtype).min
        max_pool = out.masked_fill(~mask, neg_inf).max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))

        feats = torch.cat([mean_pool, max_pool], dim=1)
        feats = self.norm(feats)
        return self.head(feats)


def build_external_classifier(vocab: VocabBundle, num_labels: int | None = None, embed_dim: int = 128, hidden_dim: int = 256, num_layers: int = 1, dropout: float = 0.3) -> ExternalAMPClassifier:
    return ExternalAMPClassifier(
        vocab_size=len(vocab.char2idx),
        num_labels=num_labels,
        pad_idx=vocab.pad_idx,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )


def load_external_classifier_checkpoint(checkpoint_path: str | Path, vocab: VocabBundle, num_labels: int | None = None, device=None, **kwargs) -> ExternalAMPClassifier:
    checkpoint_path = Path(checkpoint_path)
    model = build_external_classifier(vocab, num_labels=num_labels, **kwargs)
    if device is not None:
        model = model.to(device)

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model
