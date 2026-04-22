"""Integrated-gradient saliency for the external AMP classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from ..data.tokenizer import VocabBundle, tokenize_sequence
from ..models.classifier import ExternalAMPClassifier


@torch.no_grad()
def _tokenize_batch(sequences: Sequence[str], vocab: VocabBundle) -> tuple[torch.Tensor, torch.Tensor]:
    token_rows: list[list[int]] = []
    lengths: list[int] = []
    for seq in sequences:
        tokens, length = tokenize_sequence(seq, vocab, add_sos_eos=True)
        token_rows.append(tokens)
        lengths.append(length)
    tokens_tensor = torch.tensor(token_rows, dtype=torch.long)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    return tokens_tensor, lengths_tensor


def _classifier_forward_from_embeddings(
    model: ExternalAMPClassifier,
    embeddings: torch.Tensor,
    lengths: torch.Tensor,
    pad_mask: torch.Tensor,
) -> torch.Tensor:
    total_length = embeddings.size(1)
    packed = nn.utils.rnn.pack_padded_sequence(
        embeddings,
        lengths.cpu().clamp(min=1),
        batch_first=True,
        enforce_sorted=False,
    )
    packed_out, _ = model.encoder(packed)
    out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=total_length)

    mask = pad_mask.unsqueeze(-1)
    out_masked = out.masked_fill(~mask, 0.0)
    denom = mask.sum(dim=1).clamp(min=1)
    mean_pool = out_masked.sum(dim=1) / denom

    neg_inf = torch.finfo(out.dtype).min
    max_pool = out.masked_fill(~mask, neg_inf).max(dim=1).values
    max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))

    feats = torch.cat([mean_pool, max_pool], dim=1)
    feats = model.norm(feats)
    return model.head(feats)


def integrated_gradients_per_residue(
    model: ExternalAMPClassifier,
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    target_label_idx: int,
    n_steps: int = 32,
    sos_idx: int | None = None,
    eos_idx: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Integrated-gradient saliency per token.

    ``pad_mask`` is used to drive the classifier pooling; the returned
    saliency tensor has PAD positions zeroed. When ``sos_idx`` / ``eos_idx``
    are provided those token positions are also zeroed in the output so that
    downstream consumers don't see noise on the control tokens. (Previously
    only PAD was zeroed, which left visible saliency on ``<SOS>`` / ``<EOS>``.)
    """
    device = device or next(model.parameters()).device
    tokens = tokens.to(device)
    lengths = lengths.to(device)
    pad_mask = tokens != model.pad_idx

    with torch.no_grad():
        embeddings = model.embedding(tokens)
    baseline = torch.zeros_like(embeddings)
    diff = embeddings - baseline

    accumulated = torch.zeros_like(embeddings)
    for k in range(n_steps):
        alpha = (float(k) + 0.5) / float(n_steps)
        interp = (baseline + alpha * diff).detach().requires_grad_(True)
        logits = _classifier_forward_from_embeddings(model, interp, lengths, pad_mask)
        score = logits[:, target_label_idx].sum()
        grad = torch.autograd.grad(score, interp, retain_graph=False, create_graph=False)[0]
        accumulated = accumulated + grad

    avg_grad = accumulated / float(n_steps)
    ig = diff * avg_grad
    saliency = ig.norm(dim=-1)
    saliency = saliency.masked_fill(~pad_mask, 0.0)
    if sos_idx is not None:
        saliency = saliency.masked_fill(tokens == int(sos_idx), 0.0)
    if eos_idx is not None:
        saliency = saliency.masked_fill(tokens == int(eos_idx), 0.0)
    return saliency.detach()


def saliency_to_heatmap(
    sequences: Sequence[str],
    saliencies: list[np.ndarray],
    label_names: Sequence[str],
    out_path: str,
) -> None:
    if len(saliencies) != len(sequences) * len(label_names):
        raise ValueError("saliencies length must equal len(sequences) * len(label_names)")
    n_rows = len(saliencies)
    max_len = max(len(seq) for seq in sequences)
    fig_height = max(1.2 * n_rows, 2.0)
    fig_width = max(0.35 * max_len + 3.0, 6.0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_width, fig_height), squeeze=False)

    vmax_per_label: dict[str, float] = {}
    for seq_i, seq in enumerate(sequences):
        for li, label in enumerate(label_names):
            idx = seq_i * len(label_names) + li
            sal = np.asarray(saliencies[idx], dtype=float)
            sal = sal[: len(seq)]
            if sal.size:
                vmax_per_label[label] = max(vmax_per_label.get(label, 0.0), float(sal.max()))
    for label, v in vmax_per_label.items():
        if v <= 0:
            vmax_per_label[label] = 1.0

    idx = 0
    for seq_i, seq in enumerate(sequences):
        for label in label_names:
            ax = axes[idx, 0]
            sal = np.asarray(saliencies[idx], dtype=float)
            sal = sal[: len(seq)]
            if sal.size < len(seq):
                pad = np.zeros(len(seq) - sal.size)
                sal = np.concatenate([sal, pad])
            row = sal.reshape(1, -1)
            vmax = vmax_per_label[label]
            im = ax.imshow(row, aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax)
            ax.set_yticks([0])
            ax.set_yticklabels([f"{label}"], fontsize=8)
            ax.set_xticks(range(len(seq)))
            ax.set_xticklabels(list(seq), fontsize=8)
            for j, aa in enumerate(seq):
                ax.text(j, 0, aa, ha="center", va="center", fontsize=8, color="white")
            ax.set_title(f"seq[{seq_i}]={seq}  |  vmax[{label}]={vmax:.3f}", fontsize=8, loc="left")
            fig.colorbar(im, ax=ax, shrink=0.8)
            idx += 1

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
