"""Centralised train/val/test splitting for cVAE + classifier.

Both trainers must import from here so that generated samples are never scored
by a classifier that was trained on them. Splits are persisted as positional
int64 index arrays into the master dataframe produced by ``build_dataset.py``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

SPLIT_DIR = Path("data/processed")
SPLIT_FILES = {
    "train": SPLIT_DIR / "split_train.npy",
    "val": SPLIT_DIR / "split_val.npy",
    "test": SPLIT_DIR / "split_test.npy",
}


def _try_iterstrat():
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit  # type: ignore

        return MultilabelStratifiedShuffleSplit
    except Exception:
        return None


def _fallback_stratified_split(
    df: pd.DataFrame,
    stratify_cols: Sequence[str],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic groupby-round-robin fallback when iterstrat is missing.

    Groups rows by joined label string, sorts groups deterministically, then
    distributes rows into train/val/test buckets via round-robin weighted by
    the requested fractions. This gives every label-combination approximately
    the same proportion across splits.
    """
    rng = np.random.default_rng(seed)
    # Joined label key per row
    label_key = df[list(stratify_cols)].astype(int).astype(str).agg("".join, axis=1).to_numpy()
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    # Fractions must sum to 1
    train_frac = 1.0 - val_frac - test_frac
    if train_frac <= 0:
        raise ValueError("val_frac + test_frac must be < 1.0")

    # Iterate keys in sorted order for determinism
    unique_keys = sorted(set(label_key.tolist()))
    for key in unique_keys:
        members = np.where(label_key == key)[0]
        rng.shuffle(members)
        n = len(members)
        n_val = int(round(n * val_frac))
        n_test = int(round(n * test_frac))
        # Guarantee at least 1 member in train when possible
        if n - n_val - n_test <= 0 and n >= 3:
            n_val = max(1, n // 10)
            n_test = max(1, n // 10)
        val_part = members[:n_val]
        test_part = members[n_val : n_val + n_test]
        train_part = members[n_val + n_test :]
        val_idx.extend(val_part.tolist())
        test_idx.extend(test_part.tolist())
        train_idx.extend(train_part.tolist())

    return (
        np.array(sorted(train_idx), dtype=np.int64),
        np.array(sorted(val_idx), dtype=np.int64),
        np.array(sorted(test_idx), dtype=np.int64),
    )


def _random_split(
    n: int,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(n * val_frac))
    n_test = int(round(n * test_frac))
    val = perm[:n_val]
    test = perm[n_val : n_val + n_test]
    train = perm[n_val + n_test :]
    return (
        np.sort(train).astype(np.int64),
        np.sort(val).astype(np.int64),
        np.sort(test).astype(np.int64),
    )


def load_or_create_splits(
    df: pd.DataFrame,
    seed: int = 42,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    stratify_cols: Sequence[str] | None = None,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(train_idx, val_idx, test_idx)`` as positional int64 arrays.

    Persists to ``SPLIT_FILES``. If all three files exist and ``force`` is
    False, loads them, verifies their combined length equals ``len(df)``, and
    returns them.
    """
    n = len(df)

    all_exist = all(p.exists() for p in SPLIT_FILES.values())
    if all_exist and not force:
        train = np.load(SPLIT_FILES["train"])
        val = np.load(SPLIT_FILES["val"])
        test = np.load(SPLIT_FILES["test"])
        total = len(train) + len(val) + len(test)
        if total != n:
            raise ValueError(
                f"Persisted splits cover {total} rows but df has {n}. "
                "Rebuild with force=True."
            )
        return train.astype(np.int64), val.astype(np.int64), test.astype(np.int64)

    # Build fresh splits
    if stratify_cols:
        MSSS = _try_iterstrat()
        if MSSS is not None:
            y = df[list(stratify_cols)].astype(int).to_numpy()
            # First split off test
            splitter1 = MSSS(n_splits=1, test_size=test_frac, random_state=seed)
            trainval_idx, test_idx = next(splitter1.split(np.zeros(n), y))
            # Then split train/val from the remainder
            y_tv = y[trainval_idx]
            # val fraction relative to remaining size
            rel_val = val_frac / (1.0 - test_frac)
            splitter2 = MSSS(n_splits=1, test_size=rel_val, random_state=seed)
            sub_train, sub_val = next(splitter2.split(np.zeros(len(trainval_idx)), y_tv))
            train_idx = trainval_idx[sub_train]
            val_idx = trainval_idx[sub_val]
            train = np.sort(train_idx).astype(np.int64)
            val = np.sort(val_idx).astype(np.int64)
            test = np.sort(test_idx).astype(np.int64)
        else:
            warnings.warn(
                "iterative-stratification (iterstrat) not installed; "
                "falling back to groupby-based deterministic stratification.",
                RuntimeWarning,
            )
            train, val, test = _fallback_stratified_split(
                df, stratify_cols, val_frac, test_frac, seed
            )
    else:
        train, val, test = _random_split(n, val_frac, test_frac, seed)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(SPLIT_FILES["train"], train)
    np.save(SPLIT_FILES["val"], val)
    np.save(SPLIT_FILES["test"], test)
    return train, val, test


def load_splits() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load persisted splits. Errors if any file is missing."""
    missing = [str(p) for p in SPLIT_FILES.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Split files not found: {missing}. Run build_dataset.py first."
        )
    train = np.load(SPLIT_FILES["train"]).astype(np.int64)
    val = np.load(SPLIT_FILES["val"]).astype(np.int64)
    test = np.load(SPLIT_FILES["test"]).astype(np.int64)
    return train, val, test
