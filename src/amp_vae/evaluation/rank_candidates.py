"""Heuristics for ranking generated candidates."""

from __future__ import annotations

import pandas as pd


def composite_candidate_score(frame: pd.DataFrame, requested_cols: list[str], off_target_cols: list[str] | None = None, hemolysis_col: str = "hemolytic_score") -> pd.Series:
    off_target_cols = off_target_cols or []
    requested_mean = frame[requested_cols].mean(axis=1) if requested_cols else pd.Series(0.0, index=frame.index)
    off_target_mean = frame[off_target_cols].mean(axis=1) if off_target_cols else pd.Series(0.0, index=frame.index)
    if hemolysis_col in frame.columns:
        safety_bonus = 1.0 - frame[hemolysis_col].astype(float)
    else:
        safety_bonus = pd.Series(1.0, index=frame.index)
    return 0.60 * requested_mean + 0.30 * safety_bonus - 0.10 * off_target_mean


def rank_candidates(frame: pd.DataFrame, requested_cols: list[str], off_target_cols: list[str] | None = None, hemolysis_col: str = "hemolytic_score", n_top: int = 50) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["composite_score"] = composite_candidate_score(ranked, requested_cols, off_target_cols=off_target_cols, hemolysis_col=hemolysis_col)
    return ranked.sort_values("composite_score", ascending=False).head(n_top).reset_index(drop=True)
