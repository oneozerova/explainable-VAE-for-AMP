from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import torch

from amp_vae.config import CONDITION_COLUMNS, DEFAULT_VOCAB_PATH
from amp_vae.data.dataset import SequenceLabelDataset
from amp_vae.data.schema import SEQUENCE_COLUMN
from amp_vae.data.tokenizer import load_vocab_bundle
from amp_vae.models.classifier import build_external_classifier
from amp_vae.training.classifier import ClassifierTrainingConfig, fit_external_classifier
from amp_vae.utils.paths import repo_path
from amp_vae.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train the external AMP classifier.")
    parser.add_argument("--data", default=str(repo_path("data", "processed", "master_dataset.csv")))
    parser.add_argument("--vocab", default=str(repo_path(DEFAULT_VOCAB_PATH)))
    parser.add_argument("--output", default=str(repo_path("models", "best_external_amp_classifier.pt")))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    print(f"[train_classifier] seed={args.seed}", flush=True)
    print(f"[train_classifier] loading dataset: {args.data}", flush=True)
    frame = pd.read_csv(args.data).dropna(subset=[SEQUENCE_COLUMN]).copy()
    vocab = load_vocab_bundle(args.vocab)
    condition_cols = [col for col in CONDITION_COLUMNS if col in frame.columns]
    if not condition_cols:
        raise ValueError("No condition columns found in training dataset.")
    dataset = SequenceLabelDataset(frame, SEQUENCE_COLUMN, condition_cols, vocab)

    idx = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed)).tolist()
    n = len(idx)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    train_df = frame.iloc[train_idx].reset_index(drop=True)
    val_df = frame.iloc[val_idx].reset_index(drop=True)
    _test_df = frame.iloc[test_idx].reset_index(drop=True)
    train_ds = SequenceLabelDataset(train_df, SEQUENCE_COLUMN, condition_cols, vocab)
    val_ds = SequenceLabelDataset(val_df, SEQUENCE_COLUMN, condition_cols, vocab)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=128)

    print(
        f"[train_classifier] rows={len(frame)} vocab_size={len(vocab.vocab_list)} max_len={vocab.max_len} "
        f"condition_cols={condition_cols}",
        flush=True,
    )
    print(
        f"[train_classifier] split train={len(train_ds)} val={len(val_ds)} test={len(_test_df)}",
        flush=True,
    )
    print(f"[train_classifier] output checkpoint: {args.output}", flush=True)

    model = build_external_classifier(vocab, num_labels=len(condition_cols))
    y_train = train_df[condition_cols].values.astype(float)
    pos_counts = y_train.sum(axis=0)
    neg_counts = len(y_train) - pos_counts
    pos_weight = (neg_counts / torch.clamp(torch.tensor(pos_counts), min=1.0).numpy()).clip(1.0, 50.0)
    print(f"[train_classifier] pos_weight={pos_weight.tolist()}", flush=True)

    config = ClassifierTrainingConfig(num_epochs=args.epochs, patience=args.patience, best_model_path=args.output)
    _, history, best_path, thresholds = fit_external_classifier(model, train_loader, val_loader, condition_cols, config, pos_weight=pos_weight)
    print(f"[train_classifier] saved checkpoint to {best_path}", flush=True)
    print(f"[train_classifier] thresholds={thresholds}", flush=True)
    print(f"[train_classifier] final_metrics={history[-1]}" if history else "[train_classifier] no history", flush=True)


if __name__ == "__main__":
    main()
