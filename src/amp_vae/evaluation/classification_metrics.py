"""Multi-label classification metrics and threshold tuning."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_multilabel_metrics(y_true, y_prob, threshold: float = 0.5, label_names: Sequence[str] | None = None):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    label_names = list(label_names or [f"label_{i}" for i in range(y_true.shape[1])])

    per_label = []
    for i, label in enumerate(label_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        ys = y_prob[:, i]

        if len(np.unique(yt)) < 2:
            auroc = np.nan
            auprc = np.nan
        else:
            auroc = roc_auc_score(yt, ys)
            auprc = average_precision_score(yt, ys)

        try:
            mcc = matthews_corrcoef(yt, yp)
        except Exception:
            mcc = np.nan

        per_label.append(
            {
                "label": label,
                "auroc": auroc,
                "auprc": auprc,
                "f1": f1_score(yt, yp, zero_division=0),
                "precision": precision_score(yt, yp, zero_division=0),
                "recall": recall_score(yt, yp, zero_division=0),
                "mcc": mcc,
                "balanced_acc": balanced_accuracy_score(yt, yp),
            }
        )

    summary = {
        "macro_auroc": float(np.nanmean([row["auroc"] for row in per_label])),
        "macro_auprc": float(np.nanmean([row["auprc"] for row in per_label])),
        "macro_f1": float(np.mean([row["f1"] for row in per_label])),
        "macro_precision": float(np.mean([row["precision"] for row in per_label])),
        "macro_recall": float(np.mean([row["recall"] for row in per_label])),
        "macro_mcc": float(np.nanmean([row["mcc"] for row in per_label])),
        "macro_balanced_acc": float(np.mean([row["balanced_acc"] for row in per_label])),
        "micro_f1": float(f1_score(y_true.reshape(-1), y_pred.reshape(-1), zero_division=0)),
    }

    return {"summary": summary, "per_label": per_label}


def tune_thresholds(y_true, y_prob, grid: Iterable[float] | None = None, label_names: Sequence[str] | None = None):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    grid = list(grid or np.arange(0.1, 0.91, 0.05))
    label_names = list(label_names or [f"label_{i}" for i in range(y_true.shape[1])])

    best = {}
    for i, label in enumerate(label_names):
        scores = []
        for thr in grid:
            pred = (y_prob[:, i] >= thr).astype(int)
            scores.append((f1_score(y_true[:, i], pred, zero_division=0), float(thr)))
        best[label] = max(scores)[1]
    return best

