"""Projection helpers for latent-space visualisations."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA


class ParametricTSNE(nn.Module):
    def __init__(self, in_dim: int = 32, hidden: int = 128, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_parametric_tsne(path: str | Path, device: torch.device | None = None) -> ParametricTSNE:
    device = device or torch.device("cpu")
    try:
        payload = torch.load(Path(path), map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(Path(path), map_location=device)
    state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    in_dim = int(state_dict["net.0.weight"].shape[1])
    hidden = int(state_dict["net.0.weight"].shape[0])
    out_dim = int(state_dict["net.4.weight"].shape[0])
    model = ParametricTSNE(in_dim=in_dim, hidden=hidden, out_dim=out_dim).to(device)
    model.load_state_dict(state_dict)
    model.train(False)
    return model


def fit_pca(mu: np.ndarray) -> PCA:
    pca = PCA(n_components=2)
    pca.fit(mu)
    return pca


def fit_umap(mu: np.ndarray, random_state: int = 42):
    import umap  # type: ignore

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=random_state,
    )
    reducer.fit(mu)
    return reducer


@torch.no_grad()
def project_parametric_tsne(
    model: ParametricTSNE,
    points: np.ndarray,
    device: torch.device | None = None,
    batch_size: int = 2048,
) -> np.ndarray:
    device = device or next(model.parameters()).device
    model.train(False)
    tensor = torch.as_tensor(points, dtype=torch.float32, device=device)
    outs: list[np.ndarray] = []
    for start in range(0, tensor.shape[0], batch_size):
        chunk = tensor[start : start + batch_size]
        outs.append(model(chunk).detach().cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, 2), dtype=np.float32)


def select_real_pairs(
    mu: np.ndarray,
    conditions: np.ndarray,
    cond_cols: Sequence[str],
    label_a: str,
    label_b: str,
    n_pairs: int,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray, str, str]]:
    """Return ``n_pairs`` (label_a, label_b) pairs sorted by descending L2 distance in μ-space.

    Subsamples each pool to at most 500 candidates before computing pairwise distances,
    then greedily picks the most distant non-overlapping pairs so the first pair spans
    the widest region of the latent cloud. Rows with any NaN are dropped first.
    """
    idx_a = cond_cols.index(label_a)
    idx_b = cond_cols.index(label_b)
    col_a = conditions[:, idx_a]
    col_b = conditions[:, idx_b]
    valid = ~np.isnan(conditions).any(axis=1)
    mask_a = valid & (col_a >= 0.5) & (col_b < 0.5)
    mask_b = valid & (col_b >= 0.5) & (col_a < 0.5)
    pool_a = np.where(mask_a)[0]
    pool_b = np.where(mask_b)[0]
    if pool_a.size == 0 or pool_b.size == 0:
        return []
    cap = 500
    sample_a = rng.choice(pool_a, size=min(cap, pool_a.size), replace=False)
    sample_b = rng.choice(pool_b, size=min(cap, pool_b.size), replace=False)
    # pairwise L2: shape (|sample_a|, |sample_b|)
    diff = mu[sample_a, np.newaxis, :] - mu[np.newaxis, sample_b, :]
    dists = np.linalg.norm(diff, axis=-1)
    flat_order = np.argsort(-dists.ravel())
    n_b = len(sample_b)
    pairs: list[tuple[np.ndarray, np.ndarray, str, str]] = []
    seen_a: set[int] = set()
    seen_b: set[int] = set()
    for fi in flat_order:
        ia, ib = divmod(int(fi), n_b)
        ai, bi = int(sample_a[ia]), int(sample_b[ib])
        if ai in seen_a or bi in seen_b:
            continue
        seen_a.add(ai)
        seen_b.add(bi)
        pairs.append((mu[ai].copy(), mu[bi].copy(), label_a, label_b))
        if len(pairs) >= n_pairs:
            break
    return pairs
