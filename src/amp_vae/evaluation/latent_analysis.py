"""Latent-dimension activity and traversal analysis for the cVAE."""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import pandas as pd
import torch

from ..data.tokenizer import VocabBundle
from ..inference.generate import align_condition_to_model, decode_from_z_cond


PROPERTY_KEYS: tuple[str, ...] = ("charge", "hydro", "moment", "length")


def active_units(mu: torch.Tensor, threshold: float = 0.01) -> pd.DataFrame:
    variances = mu.float().var(dim=0, unbiased=False).cpu().numpy()
    rows = []
    for dim, var in enumerate(variances):
        rows.append({
            "dim": int(dim),
            "variance": float(var),
            "is_active": bool(var > float(threshold)),
        })
    return pd.DataFrame(rows, columns=["dim", "variance", "is_active"])


@torch.no_grad()
def dim_traversal(
    model,
    vocab: VocabBundle,
    dim: int,
    z0: torch.Tensor,
    condition: Sequence[int],
    n_steps: int = 9,
    span_sigma: float = 2.0,
    temperature: float = 0.8,
    mu_std: float = 1.0,
    samples_per_step: int = 1,
    mode: Literal["offset", "absolute"] = "offset",
    device=None,
) -> list[tuple[float, list[str]]]:
    """Walk a single latent dim around a baseline and decode.

    Parameters
    ----------
    z0 : torch.Tensor
        Baseline latent vector (all other dims are held fixed at z0).
    dim : int
        Which latent dim to sweep.
    span_sigma, mu_std : float
        The sweep covers `a in [-span_sigma * mu_std, +span_sigma * mu_std]`
        where `mu_std` is the per-dim std of mu over the train set (caller
        provides). With `span_sigma=2.0` that reproduces "walk +/- 2 sigma".
    mode : {"offset", "absolute"}
        - ``"offset"`` (default): ``z[dim] = z0[dim] + a``. Matches the
          blogpost claim "freeze z_0, walk dim_i over [-2 sigma_i, +2 sigma_i]"
          relative to the baseline.
        - ``"absolute"``: ``z[dim] = a``. Overwrites the baseline's value
          for this dim entirely (ignores ``z0[dim]``). This was the behaviour
          prior to the offset-semantics fix and is kept for backwards
          reproducibility of older traversal artefacts.
    """
    device = device or next(model.parameters()).device
    aligned = align_condition_to_model(condition, model)
    cond_base = torch.tensor(aligned, dtype=torch.float32, device=device)
    z0 = z0.to(device).view(-1).float()

    span = float(span_sigma) * float(mu_std)
    alphas = torch.linspace(-span, span, int(n_steps))
    results: list[tuple[float, list[str]]] = []
    for a in alphas.tolist():
        z = z0.clone()
        if mode == "offset":
            z[int(dim)] = z0[int(dim)] + float(a)
        elif mode == "absolute":
            z[int(dim)] = float(a)
        else:
            raise ValueError(f"unknown traversal mode: {mode!r}")
        z_batch = z.unsqueeze(0).repeat(int(samples_per_step), 1)
        cond_batch = cond_base.unsqueeze(0).repeat(int(samples_per_step), 1)
        seqs = decode_from_z_cond(
            model,
            z_batch,
            cond_batch,
            vocab=vocab,
            max_gen_len=vocab.max_len,
            temperature=temperature,
        )
        results.append((float(a), seqs))
    return results


def summarize_dim_effects(
    mu: torch.Tensor,
    traversal_results: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    deltas: dict[int, dict[str, float]] = {}
    for dim, frame in traversal_results.items():
        row: dict[str, float] = {}
        for key in PROPERTY_KEYS:
            col = f"mean_{key}"
            if col in frame.columns and len(frame) > 0:
                values = frame[col].astype(float).to_numpy()
                finite = values[np.isfinite(values)]
                if finite.size == 0:
                    row[key] = 0.0
                else:
                    row[key] = float(np.max(finite) - np.min(finite))
            else:
                row[key] = 0.0
        deltas[int(dim)] = row

    if not deltas:
        cols = ["dim", *(f"delta_{k}" for k in PROPERTY_KEYS), "property_driven_most"]
        return pd.DataFrame(columns=cols)

    abs_matrix = np.array(
        [[abs(deltas[d][k]) for k in PROPERTY_KEYS] for d in deltas],
        dtype=np.float64,
    )
    maxes = abs_matrix.max(axis=0)
    maxes = np.where(maxes > 0, maxes, 1.0)
    normed = abs_matrix / maxes

    rows = []
    for i, dim in enumerate(deltas):
        driven_idx = int(np.argmax(normed[i]))
        row = {"dim": int(dim)}
        for k in PROPERTY_KEYS:
            row[f"delta_{k}"] = float(deltas[dim][k])
        row["property_driven_most"] = PROPERTY_KEYS[driven_idx]
        rows.append(row)
    return pd.DataFrame(rows)
