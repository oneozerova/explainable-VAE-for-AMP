"""Concept vectors in cVAE latent space."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..config import CONDITION_COLUMNS
from ..data.dataset import SequenceConditionDataset
from ..data.schema import SEQUENCE_COLUMN
from ..data.tokenizer import VocabBundle
from ..inference.generate import align_condition_to_model, decode_from_z_cond


def build_condition_dataset(data_path: str, vocab: VocabBundle, indices: Sequence[int] | None = None):
    """Build a (dataset, condition_cols) pair, dropping rows with NaN condition labels.

    Previously used `.fillna(0)` which silently treated NaN conditions as negatives
    for every label and biased centroid estimates in `compute_concept_vectors` and
    `select_real_pairs`. We now drop rows that have NaN in any condition column and
    print the number dropped.

    `indices` — optional positional integer indices into the raw CSV
    (master_dataset.csv ordering). Applied BEFORE any dropna so callers can pass
    train-split indices from `amp_vae.utils.splits`.
    """
    frame = pd.read_csv(data_path)
    if indices is not None:
        frame = frame.iloc[list(indices)].copy()
    frame = frame.dropna(subset=[SEQUENCE_COLUMN]).copy()
    condition_cols = [col for col in CONDITION_COLUMNS if col in frame.columns]
    if not condition_cols:
        raise ValueError("No condition columns found in dataset.")
    before = len(frame)
    frame = frame.dropna(subset=condition_cols).copy()
    dropped = before - len(frame)
    if dropped > 0:
        print(f"[build_condition_dataset] dropped {dropped} rows with NaN in condition_cols={condition_cols}")
    seq_lengths = frame[SEQUENCE_COLUMN].astype(str).str.len()
    token_lengths = seq_lengths.add(2).clip(upper=vocab.max_len).astype(int)
    dataset = SequenceConditionDataset(
        sequences=frame[SEQUENCE_COLUMN].astype(str).tolist(),
        lengths=token_lengths.tolist(),
        conditions=frame[condition_cols].astype(float).values.tolist(),
        vocab=vocab,
    )
    return dataset, condition_cols


@torch.no_grad()
def collect_latent_embeddings(
    model,
    dataset,
    batch_size: int = 256,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = device or next(model.parameters()).device
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    mus: list[torch.Tensor] = []
    conds: list[torch.Tensor] = []
    for tokens, lengths, cond in loader:
        tokens = tokens.to(device)
        lengths = lengths.to(device)
        cond_dev = cond.to(device)
        mu, _ = model.encoder(tokens, lengths, cond_dev)
        mus.append(mu.detach().cpu().float())
        conds.append(cond.detach().cpu().float())
    return torch.cat(mus, dim=0), torch.cat(conds, dim=0)


def compute_concept_vectors(
    mu: torch.Tensor,
    conditions: torch.Tensor,
    label_names: Sequence[str],
    n_boot: int = 0,
    boot_seed: int = 0,
    ci: float = 0.95,
) -> dict[str, dict]:
    """Compute concept vectors as pos-minus-neg centroid differences.

    Raw centroid-difference is sensitive to class imbalance (rare-label centroid
    has higher variance). When `n_boot > 0` we resample each class with
    replacement `n_boot` times, recompute the centroid diff, and return the
    per-dim `ci` percentile interval alongside the point estimate.

    Returns a dict per label with keys:
        vector, magnitude, n_pos, n_neg,
        and (if n_boot > 0) ci_lo, ci_hi (both tensors of shape [latent_dim]),
        magnitude_ci_lo, magnitude_ci_hi (floats over the bootstrap magnitude).
    """
    result: dict[str, dict] = {}
    rng = np.random.default_rng(int(boot_seed))
    for k, name in enumerate(label_names):
        col = conditions[:, k]
        pos_mask = col >= 0.5
        neg_mask = ~pos_mask
        n_pos = int(pos_mask.sum().item())
        n_neg = int(neg_mask.sum().item())
        if n_pos == 0 or n_neg == 0:
            vec = torch.zeros(mu.size(1), dtype=torch.float32)
            info: dict = {
                "vector": vec.float(),
                "magnitude": float(torch.linalg.norm(vec).item()),
                "n_pos": n_pos,
                "n_neg": n_neg,
            }
        else:
            mu_pos = mu[pos_mask]
            mu_neg = mu[neg_mask]
            vec = mu_pos.mean(dim=0) - mu_neg.mean(dim=0)
            info = {
                "vector": vec.float(),
                "magnitude": float(torch.linalg.norm(vec).item()),
                "n_pos": n_pos,
                "n_neg": n_neg,
            }
            if int(n_boot) > 0:
                pos_np = mu_pos.cpu().numpy().astype(np.float64)
                neg_np = mu_neg.cpu().numpy().astype(np.float64)
                boot_vecs = np.empty((int(n_boot), mu.size(1)), dtype=np.float64)
                for b in range(int(n_boot)):
                    pi = rng.integers(0, n_pos, size=n_pos)
                    ni = rng.integers(0, n_neg, size=n_neg)
                    boot_vecs[b] = pos_np[pi].mean(axis=0) - neg_np[ni].mean(axis=0)
                lo_q = (1.0 - float(ci)) / 2.0
                hi_q = 1.0 - lo_q
                ci_lo = np.quantile(boot_vecs, lo_q, axis=0)
                ci_hi = np.quantile(boot_vecs, hi_q, axis=0)
                mags = np.linalg.norm(boot_vecs, axis=1)
                info["ci_lo"] = torch.from_numpy(ci_lo).float()
                info["ci_hi"] = torch.from_numpy(ci_hi).float()
                info["magnitude_ci_lo"] = float(np.quantile(mags, lo_q))
                info["magnitude_ci_hi"] = float(np.quantile(mags, hi_q))
        result[name] = info
    return result


def concept_vector_cosine(vectors: dict[str, torch.Tensor]) -> pd.DataFrame:
    names = list(vectors.keys())
    stacked = torch.stack([vectors[n].float() for n in names], dim=0)
    norms = torch.linalg.norm(stacked, dim=1, keepdim=True).clamp(min=1e-12)
    normed = stacked / norms
    sim = (normed @ normed.T).cpu().numpy()
    return pd.DataFrame(sim, index=names, columns=names)


@torch.no_grad()
def concept_vector_walk(
    model,
    vocab: VocabBundle,
    v: torch.Tensor,
    z0: torch.Tensor,
    alphas: Sequence[float],
    condition: Sequence[int],
    temperature: float = 0.8,
    device=None,
) -> list[tuple[float, str]]:
    device = device or next(model.parameters()).device
    aligned = align_condition_to_model(condition, model)
    cond_base = torch.tensor(aligned, dtype=torch.float32, device=device)
    v = v.to(device).view(-1).float()
    z0 = z0.to(device).view(-1).float()
    results: list[tuple[float, str]] = []
    for a in alphas:
        z = (z0 + float(a) * v).unsqueeze(0)
        cond_t = cond_base.unsqueeze(0)
        seqs = decode_from_z_cond(
            model,
            z,
            cond_t,
            vocab=vocab,
            max_gen_len=vocab.max_len,
            temperature=temperature,
        )
        results.append((float(a), seqs[0] if seqs else ""))
    return results
