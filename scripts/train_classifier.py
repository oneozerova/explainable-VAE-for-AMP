from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import torch

from amp_vae.config import CONDITION_COLUMNS, DEFAULT_VOCAB_PATH
from amp_vae.data.dataset import SequenceLabelDataset
from amp_vae.data.schema import SEQUENCE_COLUMN
from amp_vae.data.tokenizer import load_vocab_bundle
from amp_vae.evaluation.classification_metrics import compute_multilabel_metrics
from amp_vae.models.classifier import build_external_classifier, load_external_classifier_checkpoint
from amp_vae.training.classifier import (
    ClassifierTrainingConfig,
    fit_external_classifier,
    predict_classifier_loader,
)
from amp_vae.utils.paths import repo_path
from amp_vae.utils.seed import set_seed
from amp_vae.utils.splits import load_splits, load_or_create_splits


def parse_args():
    parser = argparse.ArgumentParser(description="Train the external AMP classifier.")
    parser.add_argument("--data", default=str(repo_path("data", "processed", "master_dataset.csv")))
    parser.add_argument("--vocab", default=str(repo_path(DEFAULT_VOCAB_PATH)))
    parser.add_argument("--output", default=str(repo_path("models", "best_external_amp_classifier.pt")))
    parser.add_argument(
        "--artifacts",
        default=str(repo_path("models", "external_amp_classifier_artifacts.json")),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-split", action="store_true", help="Rebuild the canonical split indices.")
    return parser.parse_args()


def _per_class_metrics(y_true: np.ndarray, y_prob: np.ndarray, label_names: list[str]) -> dict:
    metrics = compute_multilabel_metrics(y_true, y_prob, threshold=0.5, label_names=label_names)
    per_class = {}
    for row in metrics["per_label"]:
        name = row["label"]
        support = int(np.asarray(y_true)[:, label_names.index(name)].sum())
        per_class[name] = {
            "auroc": float(row["auroc"]) if row["auroc"] == row["auroc"] else None,
            "auprc": float(row["auprc"]) if row["auprc"] == row["auprc"] else None,
            "support": support,
        }
    summary = metrics["summary"]
    return {
        "per_class": per_class,
        "macro_auroc": summary["macro_auroc"],
        "macro_auprc": summary["macro_auprc"],
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    # Enforce deterministic cuDNN for classifier training.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[train_classifier] seed={args.seed}", flush=True)
    print(f"[train_classifier] loading dataset: {args.data}", flush=True)
    frame = pd.read_csv(args.data).dropna(subset=[SEQUENCE_COLUMN]).copy()
    vocab = load_vocab_bundle(args.vocab)

    # Classifier is aligned with cVAE schema. Require every condition column present.
    missing = [c for c in CONDITION_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Dataset missing required condition columns: {missing}")
    condition_cols = list(CONDITION_COLUMNS)

    # Canonical splits (shared with cVAE).
    if args.force_split:
        splits = load_or_create_splits(
            frame, seed=args.seed, stratify_cols=CONDITION_COLUMNS, force=True
        )
    else:
        try:
            splits = load_splits()
        except FileNotFoundError:
            splits = load_or_create_splits(
                frame, seed=args.seed, stratify_cols=CONDITION_COLUMNS, force=False
            )

    train_arr, val_arr, test_arr = splits
    train_idx = train_arr.tolist()
    val_idx = val_arr.tolist()
    test_idx = test_arr.tolist()

    train_df = frame.iloc[train_idx].reset_index(drop=True)
    val_df = frame.iloc[val_idx].reset_index(drop=True)
    test_df = frame.iloc[test_idx].reset_index(drop=True)

    train_ds = SequenceLabelDataset(train_df, SEQUENCE_COLUMN, condition_cols, vocab)
    val_ds = SequenceLabelDataset(val_df, SEQUENCE_COLUMN, condition_cols, vocab)
    test_ds = SequenceLabelDataset(test_df, SEQUENCE_COLUMN, condition_cols, vocab)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=128)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128)

    print(
        f"[train_classifier] rows={len(frame)} vocab_size={len(vocab.vocab_list)} max_len={vocab.max_len} "
        f"condition_cols={condition_cols}",
        flush=True,
    )
    print(
        f"[train_classifier] split train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}",
        flush=True,
    )
    print(f"[train_classifier] output checkpoint: {args.output}", flush=True)

    model = build_external_classifier(vocab, num_labels=len(condition_cols))
    y_train = train_df[condition_cols].values.astype(float)
    pos_counts = y_train.sum(axis=0)
    neg_counts = len(y_train) - pos_counts
    # Cap raised from 50 -> 100 to avoid under-weighting rare labels
    # (antiparasitic / antiviral / anticancer minority counts).
    pos_weight = (neg_counts / torch.clamp(torch.tensor(pos_counts), min=1.0).numpy()).clip(1.0, 100.0)
    print(f"[train_classifier] pos_weight={pos_weight.tolist()}", flush=True)

    config = ClassifierTrainingConfig(num_epochs=args.epochs, patience=args.patience, best_model_path=args.output)
    _, history, best_path, thresholds = fit_external_classifier(
        model, train_loader, val_loader, condition_cols, config, pos_weight=pos_weight
    )
    print(f"[train_classifier] saved checkpoint to {best_path}", flush=True)
    print(f"[train_classifier] thresholds={thresholds}", flush=True)
    print(f"[train_classifier] final_metrics={history[-1]}" if history else "[train_classifier] no history", flush=True)

    # Reload best checkpoint for held-out test evaluation.
    if config.device:
        device = torch.device(config.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    test_model = load_external_classifier_checkpoint(
        best_path, vocab, num_labels=len(condition_cols), device=device
    )
    test_criterion = torch.nn.BCEWithLogitsLoss()
    _, y_test, p_test, _ = predict_classifier_loader(test_model, test_loader, test_criterion, device)
    test_metrics = _per_class_metrics(y_test, p_test, condition_cols)
    print(f"[train_classifier] test_metrics={test_metrics}", flush=True)

    # Persist artifacts (v2 schema).
    artifacts_path = Path(args.artifacts)
    artifacts_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "target_labels": condition_cols,
        "target_cols": condition_cols,
        "best_thresholds": {name: float(thresholds[name]) for name in condition_cols},
        "test_metrics": test_metrics,
    }
    with open(artifacts_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[train_classifier] wrote artifacts -> {artifacts_path}", flush=True)


if __name__ == "__main__":
    main()
