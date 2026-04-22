from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import torch

from amp_vae.config import CONDITION_COLUMNS, DEFAULT_VOCAB_PATH
from amp_vae.data.dataset import SequenceConditionDataset
from amp_vae.data.schema import SEQUENCE_COLUMN
from amp_vae.data.tokenizer import load_vocab_bundle
from amp_vae.models.cvae import build_cvae
from amp_vae.training.cvae import (
    CVAETrainingConfig,
    cvae_loss,
    eval_cvae_epoch,
    fit_cvae,
    load_checkpoint,
)
from amp_vae.utils.paths import repo_path
from amp_vae.utils.seed import set_seed
from amp_vae.utils.splits import load_or_create_splits, load_splits


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
    parser.add_argument(
        "--force-split",
        action="store_true",
        help="Recompute and overwrite persisted train/val/test splits.",
    )
    parser.add_argument(
        "--test-metrics-output",
        default=str(repo_path("artifacts", "cvae_test_metrics.json")),
        help="Where to write held-out test-set ELBO/recon/KL metrics.",
    )
    return parser.parse_args()


def _build_subset_loader(dataset, indices: np.ndarray, batch_size: int, shuffle: bool, generator: torch.Generator | None):
    idx = np.asarray(indices, dtype=np.int64)
    subset = torch.utils.data.Subset(dataset, idx.tolist())
    return torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def main():
    started_at = time.perf_counter()
    args = parse_args()
    set_seed(args.seed)
    # Deterministic cuDNN for reproducible runs (P1 fix).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log(f"seed={args.seed}")

    log(f"loading dataset: {args.data}")
    frame = pd.read_csv(args.data)
    log(f"loading vocab: {args.vocab}")
    vocab = load_vocab_bundle(args.vocab)
    frame = frame.dropna(subset=[SEQUENCE_COLUMN]).reset_index(drop=True).copy()
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

    # Centralised split: shared with classifier so the two models see identical
    # train/val/test partitions (fixes P0 #5). load_or_create_splits persists indices
    # to data/processed/split_{train,val,test}.npy.
    if args.force_split:
        log("force-split: recomputing splits")
        train_idx, val_idx, test_idx = load_or_create_splits(
            frame,
            seed=args.seed,
            val_frac=0.1,
            test_frac=0.1,
            stratify_cols=condition_cols,
            force=True,
        )
    else:
        try:
            train_idx, val_idx, test_idx = load_splits()
        except (FileNotFoundError, Exception):
            log("persisted splits not found; creating new splits")
            train_idx, val_idx, test_idx = load_or_create_splits(
                frame,
                seed=args.seed,
                val_frac=0.1,
                test_frac=0.1,
                stratify_cols=condition_cols,
                force=False,
            )

    batch_size = 256
    # Seeded generator for DataLoader shuffle -> reproducible batch ordering.
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_loader = _build_subset_loader(dataset, train_idx, batch_size, shuffle=True, generator=loader_generator)
    val_loader = _build_subset_loader(dataset, val_idx, batch_size, shuffle=False, generator=None)
    test_loader = _build_subset_loader(dataset, test_idx, batch_size, shuffle=False, generator=None)

    if raw_lengths is not None:
        log(f"length_source=sequence_tokens invalid_length_rows={invalid_length_count}")
    log(
        f"dataset_size={len(dataset)} split train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
    )
    log(
        f"dataloader batch_size={batch_size} train_batches={len(train_loader)} "
        f"val_batches={len(val_loader)} test_batches={len(test_loader)}"
    )
    log(f"output checkpoint: {args.output}")

    model = build_cvae(vocab, cond_dim=len(condition_cols))
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"model cond_dim={model.cond_dim} latent_dim={model.latent_dim} parameters={n_params} trainable={n_trainable}")
    log("starting training")
    config = CVAETrainingConfig(num_epochs=args.epochs, patience=args.patience, best_model_path=args.output)
    trained_model, history, best_path = fit_cvae(model, train_loader, val_loader, config)
    elapsed = time.perf_counter() - started_at
    log(f"saved checkpoint to {best_path}")
    if history["val_loss"]:
        best_val_loss = min(history["val_loss"])
        best_epoch = history["val_loss"].index(best_val_loss) + 1
        log(f"final_val_loss={history['val_loss'][-1]:.4f} best_val_loss={best_val_loss:.4f} best_epoch={best_epoch}")
    else:
        log("no history")

    # Held-out test-set evaluation against the best checkpoint (fixes P0 #12).
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log(f"loading best checkpoint for test eval: {best_path}")
    load_checkpoint(best_path, trained_model, optimizer=None, map_location=device)
    trained_model = trained_model.to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=trained_model.pad_idx, reduction="sum")
    test_elbo, test_recon, test_kl = eval_cvae_epoch(trained_model, test_loader, criterion, device, config)
    test_metrics = {
        "test_elbo": float(test_elbo),
        "test_recon": float(test_recon),
        "test_kl": float(test_kl),
        "n_test": int(len(test_idx)),
        "seed": int(args.seed),
        "checkpoint": str(best_path),
    }
    test_out = Path(args.test_metrics_output)
    test_out.parent.mkdir(parents=True, exist_ok=True)
    with open(test_out, "w") as f:
        json.dump(test_metrics, f, indent=2)
    log(
        f"test_elbo={test_elbo:.4f} test_recon={test_recon:.4f} test_kl={test_kl:.4f} "
        f"n_test={len(test_idx)} written={test_out}"
    )

    log(f"elapsed_seconds={elapsed:.1f}")


if __name__ == "__main__":
    main()
