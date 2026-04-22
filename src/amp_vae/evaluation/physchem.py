"""Physicochemical descriptors for amino-acid sequences."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


KD_SCALE: dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "E": -3.5, "Q": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

CHARGE_MAP: dict[str, float] = {
    "K": 1.0, "R": 1.0, "D": -1.0, "E": -1.0, "H": 0.1,
}


def net_charge(seq: str) -> float:
    """Net charge at ~neutral pH. Returns NaN for empty input (matches the
    other descriptors; previously this returned 0.0 which was inconsistent)."""
    if not seq:
        return float("nan")
    return float(sum(CHARGE_MAP.get(aa, 0.0) for aa in seq))


def hydrophobicity(seq: str) -> float:
    if not seq:
        return float("nan")
    vals = [KD_SCALE.get(aa, 0.0) for aa in seq]
    return float(np.mean(vals))


def hydrophobic_moment_per_residue(seq: str, angle: float = 100.0) -> float:
    """Per-residue hydrophobic moment on the Kyte-Doolittle scale.

    NOTE: This is a per-residue variant that divides the raw Eisenberg 1982
    moment magnitude by the sequence length ``n``. It is NOT the raw Eisenberg
    1982 formula. The ``/n`` normalisation keeps values comparable across
    sequences of different lengths and keeps the numbers continuous with prior
    artefacts (docs/tables/*.csv, figures) that used this convention.

    If you need the raw-sum Eisenberg 1982 moment, compute it directly or drop
    the divide-by-n here.
    """
    if not seq:
        return float("nan")
    n = len(seq)
    idx = np.arange(n)
    theta = np.deg2rad(angle) * idx
    h = np.array([KD_SCALE.get(aa, 0.0) for aa in seq], dtype=np.float64)
    cos_sum = float(np.sum(h * np.cos(theta)))
    sin_sum = float(np.sum(h * np.sin(theta)))
    return float(np.sqrt(cos_sum * cos_sum + sin_sum * sin_sum) / n)


def hydrophobic_moment(seq: str, angle: float = 100.0) -> float:
    """Deprecated alias for :func:`hydrophobic_moment_per_residue`.

    Kept so existing callers (scripts, notebooks, downstream analyses) that
    imported ``hydrophobic_moment`` continue to return identical numbers to
    previously saved artefacts. Prefer the explicit name going forward.
    """
    return hydrophobic_moment_per_residue(seq, angle=angle)


def properties_frame(sequences: Iterable[str]) -> pd.DataFrame:
    rows = []
    for seq in sequences:
        s = str(seq) if seq is not None else ""
        if not s:
            rows.append({
                "sequence": s,
                "length": 0,
                "net_charge": float("nan"),
                "hydrophobicity": float("nan"),
                "hydrophobic_moment": float("nan"),
            })
            continue
        rows.append({
            "sequence": s,
            "length": len(s),
            "net_charge": net_charge(s),
            "hydrophobicity": hydrophobicity(s),
            "hydrophobic_moment": hydrophobic_moment(s),
        })
    return pd.DataFrame(rows, columns=["sequence", "length", "net_charge", "hydrophobicity", "hydrophobic_moment"])
