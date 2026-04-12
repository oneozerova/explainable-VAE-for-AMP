from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from amp_vae.config import CONDITION_COLUMNS, DEFAULT_VOCAB_PATH
from amp_vae.data.dataset import SequenceConditionDataset
from amp_vae.data.schema import SEQUENCE_COLUMN
from amp_vae.data.tokenizer import load_vocab_bundle
from amp_vae.models.cvae import build_cvae
from amp_vae.training.cvae import CVAETrainingConfig, fit_cvae
from amp_vae.utils.paths import repo_path
from amp_vae.utils.seed import set_seed


def log(message: str) -> None:
    print(f"[train_cvae] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the cVAE model.")
    parser.add_argument("--data", default=str(repo_path("data", "processed", "master_dataset.csv")))
    parser.add_argument("--vocab", default=str(repo_path(DEFAULT_VOCAB_PATH)))
    parser.add_argument("--output", default=str(repo_path("models", "best_cvae.pt")))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    started_at = time.perf_counter()
    args = parse_args()
    set_seed(args.seed)
    log(f"seed={args.seed}")
    log(f"loading dataset: {args.data}")
    frame = pd.read_csv(args.data)
    log(f"loading vocab: {args.vocab}")
    vocab = load_vocab_bundle(args.vocab)
    frame = frame.dropna(subset=[SEQUENCE_COLUMN]).copy()
    seq_lengths = frame[SEQUENCE_COLUMN].astype(str).str.len()
    raw_lengths = pd.to_numeric(frame.get("Length"), errors="coerce") if "Length" in frame.columns else None
    invalid_length_count = 0 if raw_lengths is None else int(raw_lengths.isna().sum() + (raw_lengths <= 0).sum())
    # cVAE packs tokenized sequences, so derive lengths from the actual token stream.
    token_lengths = seq_lengths.add(2).clip(upper=vocab.max_len).astype(int)
    condition_cols = [col for col in CONDITION_COLUMNS if col in frame.columns]
    if not condition_cols:
        raise ValueError("No condition columns found in training dataset.")
    if list(vocab.condition_cols) != list(condition_cols):
        log(f"overriding vocab.condition_cols from {vocab.condition_cols} to {condition_cols}")
        vocab = type(vocab)(
            char2idx=vocab.char2idx,
            idx2char=vocab.idx2char,
            max_len=vocab.max_len,
            vocab_list=vocab.vocab_list,
            condition_cols=list(condition_cols),
        )
    log(
        f"rows={len(frame)} vocab_size={len(vocab.vocab_list)} max_len={vocab.max_len} "
        f"condition_cols={condition_cols}"
    )
    log(
        "sequence_length_stats="
        f"min={int(seq_lengths.min())} max={int(seq_lengths.max())} "
        f"mean={float(seq_lengths.mean()):.2f}"
    )
    dataset = SequenceConditionDataset(
        sequences=frame[SEQUENCE_COLUMN].astype(str).tolist(),
        lengths=token_lengths.tolist(),
        conditions=frame[condition_cols].fillna(0).astype(float).values.tolist(),
        vocab=vocab,
    )

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, _ = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(args.seed)
    )
    batch_size = 256
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    if raw_lengths is not None:
        log(f"length_source=sequence_tokens invalid_length_rows={invalid_length_count}")
    log(f"dataset_size={len(dataset)} split train={len(train_dataset)} val={len(val_dataset)} test={test_size}")
    log(f"dataloader batch_size={batch_size} train_batches={len(train_loader)} val_batches={len(val_loader)}")
    log(f"output checkpoint: {args.output}")

    model = build_cvae(vocab, cond_dim=len(condition_cols))
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"model cond_dim={model.cond_dim} latent_dim={model.latent_dim} parameters={n_params} trainable={n_trainable}")
    log("starting training")
    config = CVAETrainingConfig(num_epochs=args.epochs, patience=args.patience, best_model_path=args.output)
    _, history, best_path = fit_cvae(model, train_loader, val_loader, config)
    elapsed = time.perf_counter() - started_at
    log(f"saved checkpoint to {best_path}")
    if history["val_loss"]:
        best_val_loss = min(history["val_loss"])
        best_epoch = history["val_loss"].index(best_val_loss) + 1
        log(f"final_val_loss={history['val_loss'][-1]:.4f} best_val_loss={best_val_loss:.4f} best_epoch={best_epoch}")
    else:
        log("no history")
    log(f"elapsed_seconds={elapsed:.1f}")


if __name__ == "__main__":
    import torch

    main()
