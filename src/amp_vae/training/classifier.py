"""Training loop for the external AMP classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..evaluation.classification_metrics import compute_multilabel_metrics, tune_thresholds


@dataclass
class ClassifierTrainingConfig:
    lr: float = 2e-3
    weight_decay: float = 1e-4
    num_epochs: int = 30
    patience: int = 7
    max_grad_norm: float = 5.0
    device: str | None = None
    best_model_path: str | Path = "models/best_external_amp_classifier.pt"
    verbose: bool = True


def train_classifier_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_n = 0
    for x, lengths, y, _ in loader:
        x = x.to(device)
        lengths = lengths.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x, lengths)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        total_n += x.size(0)
    return total_loss / max(total_n, 1)


@torch.no_grad()
def predict_classifier_loader(model, loader, criterion, device):
    model.eval()
    probs_all, labels_all, seqs_all = [], [], []
    total_loss, total_n = 0.0, 0
    for x, lengths, y, seqs in loader:
        x = x.to(device)
        lengths = lengths.to(device)
        y = y.to(device)

        logits = model(x, lengths)
        loss = criterion(logits, y)
        probs = torch.sigmoid(logits).cpu().numpy()

        probs_all.append(probs)
        labels_all.append(y.cpu().numpy())
        seqs_all.extend(seqs)

        total_loss += float(loss.item()) * x.size(0)
        total_n += x.size(0)

    probs_all = np.concatenate(probs_all, axis=0) if probs_all else np.empty((0, 0))
    labels_all = np.concatenate(labels_all, axis=0) if labels_all else np.empty((0, 0))
    return total_loss / max(total_n, 1), labels_all, probs_all, seqs_all


def fit_external_classifier(model, train_loader, val_loader, target_cols, config: ClassifierTrainingConfig, pos_weight=None):
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if pos_weight is not None:
        pos_weight_t = torch.tensor(pos_weight, dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    else:
        criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    best_score = -np.inf
    wait = 0
    best_path = Path(config.best_model_path)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    history = []

    if config.verbose:
        print(
            f"[classifier] device={device} epochs={config.num_epochs} patience={config.patience} "
            f"labels={list(target_cols)} train_batches={len(train_loader)} val_batches={len(val_loader)} "
            f"best_model={best_path}",
            flush=True,
        )

    for epoch in range(1, config.num_epochs + 1):
        train_loss = train_classifier_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, y_val, p_val, _ = predict_classifier_loader(model, val_loader, criterion, device)
        metrics = compute_multilabel_metrics(y_val, p_val, threshold=0.5, label_names=target_cols)
        macro = metrics["summary"]
        scheduler.step(macro["macro_auroc"])

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **macro})

        if config.verbose:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"[classifier] epoch={epoch}/{config.num_epochs} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} macro_auroc={macro['macro_auroc']:.4f} "
                f"macro_f1={macro['macro_f1']:.4f} micro_f1={macro['micro_f1']:.4f} lr={lr:.6f}",
                flush=True,
            )

        monitor = macro["macro_auroc"]
        if monitor > best_score:
            best_score = monitor
            wait = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "target_cols": list(target_cols),
                },
                best_path,
            )
            if config.verbose:
                print(
                    f"[classifier] saved new best checkpoint: {best_path} "
                    f"(macro_auroc={best_score:.4f})",
                    flush=True,
                )
        else:
            wait += 1
            if wait >= config.patience:
                if config.verbose:
                    print(
                        f"[classifier] early stopping after epoch {epoch}; best_macro_auroc={best_score:.4f}",
                        flush=True,
                    )
                break

    thresholds = tune_thresholds(y_val, p_val, label_names=target_cols)
    if config.verbose:
        print(f"[classifier] tuned thresholds: {thresholds}", flush=True)
    return model, history, best_path, thresholds
