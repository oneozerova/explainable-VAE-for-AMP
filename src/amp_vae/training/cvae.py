"""Training loop for the conditional VAE."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn


@dataclass
class CVAETrainingConfig:
    lr: float = 1e-3
    num_epochs: int = 150
    patience: int = 20
    free_bits: float = 0.25
    cyclical_period: int = 10
    n_cycles: int = 4
    lagging_epochs: int = 10
    lagging_inner_steps: int = 5
    max_grad_norm: float = 5.0
    word_dropout_rate: float = 0.3
    batch_log_interval: int = 5
    device: str | None = None
    best_model_path: str | Path = "models/best_cvae.pt"
    verbose: bool = True


def get_kl_weight(epoch: int, cyclical_period: int, n_cycles: int) -> float:
    total_anneal_epochs = n_cycles * cyclical_period
    if epoch >= total_anneal_epochs:
        return 1.0
    position_in_cycle = epoch % cyclical_period
    return position_in_cycle / max(1, cyclical_period - 1)


def cvae_loss(logits, target, mu, logvar, criterion, beta: float = 1.0, free_bits: float = 0.25):
    recon_loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = kl_per_dim.mean(dim=0)
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    kl_loss = kl_per_dim.sum()
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def train_cvae_epoch(model, loader, optimizer, criterion, device, epoch: int, config: CVAETrainingConfig, use_lagging: bool = False):
    model.train()
    total_loss = total_recon = total_kl = 0.0
    beta = get_kl_weight(epoch, config.cyclical_period, config.n_cycles)

    for batch_idx, (src, lengths, cond) in enumerate(loader, start=1):
        src, lengths, cond = src.to(device), lengths.to(device), cond.to(device)

        if use_lagging:
            for _ in range(config.lagging_inner_steps):
                optimizer.zero_grad()
                mu, logvar = model.encoder(src, lengths, cond)
                z = model.reparameterize(mu, logvar)
                logits_for_enc = model.decoder(
                    src[:, :-1],
                    z,
                    cond,
                    model.pad_idx,
                    model.unk_idx,
                    word_dropout_rate=config.word_dropout_rate,
                )
                target = src[:, 1:]
                loss_enc, _, _ = cvae_loss(logits_for_enc, target, mu, logvar, criterion, beta=beta, free_bits=config.free_bits)
                loss_enc.backward()
                for p in model.decoder.parameters():
                    if p.grad is not None:
                        p.grad.zero_()
                torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), max_norm=config.max_grad_norm)
                optimizer.step()

        optimizer.zero_grad()
        logits, target, mu, logvar = model(src, lengths, cond, word_dropout_rate=config.word_dropout_rate)
        loss, recon, kl = cvae_loss(logits, target, mu, logvar, criterion, beta=beta, free_bits=config.free_bits)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
        optimizer.step()

        total_loss += float(loss.item())
        total_recon += float(recon.item())
        total_kl += float(kl.item())

        if config.verbose:
            should_log_batch = (
                batch_idx == 1
                or batch_idx == len(loader)
                or (config.batch_log_interval > 0 and batch_idx % config.batch_log_interval == 0)
            )
            if should_log_batch:
                print(
                    f"[cVAE][train] epoch={epoch + 1} batch={batch_idx}/{len(loader)} "
                    f"loss={loss.item():.4f} recon={recon.item():.4f} kl={kl.item():.4f} beta={beta:.3f}",
                    flush=True,
                )

    n = max(len(loader), 1)
    return total_loss / n, total_recon / n, total_kl / n, beta


@torch.no_grad()
def eval_cvae_epoch(model, loader, criterion, device, config: CVAETrainingConfig):
    model.eval()
    total_loss = 0.0
    for src, lengths, cond in loader:
        src, lengths, cond = src.to(device), lengths.to(device), cond.to(device)
        logits, target, mu, logvar = model(src, lengths, cond, word_dropout_rate=0.0)
        loss, _, _ = cvae_loss(logits, target, mu, logvar, criterion, beta=1.0, free_bits=config.free_bits)
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


def fit_cvae(model, train_loader, val_loader, config: CVAETrainingConfig):
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=model.pad_idx)
    best_val = float("inf")
    wait = 0
    history = {"train_loss": [], "val_loss": [], "recon": [], "kl": [], "beta": []}
    best_path = Path(config.best_model_path)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    if config.verbose:
        print(
            f"[cVAE] device={device} epochs={config.num_epochs} patience={config.patience} "
            f"train_batches={len(train_loader)} val_batches={len(val_loader)} "
            f"batch_log_interval={config.batch_log_interval} best_model={best_path}",
            flush=True,
        )

    for epoch in range(config.num_epochs):
        use_lagging = epoch < config.lagging_epochs
        beta = get_kl_weight(epoch, config.cyclical_period, config.n_cycles)
        if config.verbose:
            print(
                f"[cVAE] epoch_start={epoch + 1}/{config.num_epochs} beta={beta:.3f} lagging={use_lagging}",
                flush=True,
            )
        train_loss, recon, kl, beta = train_cvae_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            config,
            use_lagging=use_lagging,
        )
        val_loss = eval_cvae_epoch(model, val_loader, criterion, device, config)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["recon"].append(recon)
        history["kl"].append(kl)
        history["beta"].append(beta)

        if config.verbose:
            print(
                f"[cVAE] epoch={epoch + 1}/{config.num_epochs} "
                f"train_loss={train_loss:.4f} recon={recon:.4f} kl={kl:.4f} "
                f"beta={beta:.3f} val_loss={val_loss:.4f} lagging={use_lagging}",
                flush=True,
            )

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save({"model_state_dict": model.state_dict(), "history": history}, best_path)
            if config.verbose:
                print(f"[cVAE] saved new best checkpoint: {best_path} (val_loss={best_val:.4f})", flush=True)
        else:
            wait += 1
            if wait >= config.patience:
                if config.verbose:
                    print(f"[cVAE] early stopping after epoch {epoch + 1}; best_val={best_val:.4f}", flush=True)
                break

    return model, history, best_path
