"""Training loop for the conditional VAE."""

from __future__ import annotations

import warnings
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
    # Fu 2019: fraction of each cycle spent ramping beta; hold at 1 for the remainder.
    cyclical_R: float = 0.5
    # Floor on beta to prevent full z-collapse at cycle reset (beta=0 is a trap).
    beta_min: float = 1e-3
    lagging_epochs: int = 10
    lagging_inner_steps: int = 5
    max_grad_norm: float = 5.0
    word_dropout_rate: float = 0.3
    batch_log_interval: int = 5
    device: str | None = None
    best_model_path: str | Path = "models/best_cvae.pt"
    verbose: bool = True


def cyclical_beta(
    epoch: int,
    period: int = 10,
    R: float = 0.5,
    beta_min: float = 1e-3,
) -> float:
    """Fu et al. 2019 cyclical KL annealing.

    Ramp linearly from ~0 to 1 over the first R fraction of each cycle, then hold at 1.
    Floor at `beta_min` so the decoder is never trained against a pure-AE objective,
    which is the standard z-collapse trap at cycle reset.
    """
    if period <= 0:
        return 1.0
    tau = (epoch % period) / period
    beta = min(1.0, tau / R) if R > 0 else 1.0
    return max(beta_min, float(beta))


# Backwards-compat shim for external callers.
def get_kl_weight(
    epoch: int,
    cyclical_period: int,
    n_cycles: int,
    R: float = 0.5,
    beta_min: float = 1e-3,
) -> float:
    total_anneal_epochs = n_cycles * cyclical_period
    if epoch >= total_anneal_epochs:
        return 1.0
    return cyclical_beta(epoch, period=cyclical_period, R=R, beta_min=beta_min)


def cvae_loss(logits, target, mu, logvar, criterion, beta: float = 1.0, free_bits: float = 0.25):
    """Per-sequence ELBO components.

    Convention (P1 fix): both recon and KL are per-sequence scalars.
      - recon: CE summed over tokens (PAD ignored via criterion.ignore_index), divided
        by batch size => "sum over tokens, averaged across the batch".
      - KL: summed over latent dims, averaged over batch.
    Decouples beta / free_bits from sequence length and latent_dim.

    `criterion` must be CrossEntropyLoss(reduction='sum', ignore_index=pad_idx).
    """
    batch_size = target.size(0)
    recon_sum = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
    recon_loss = recon_sum / max(batch_size, 1)

    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = kl_per_dim.mean(dim=0)  # mean over batch -> per-dim scalar
    kl_clamped = torch.clamp(kl_per_dim, min=free_bits)
    kl_loss = kl_clamped.sum()
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def train_cvae_epoch(model, loader, optimizer, criterion, device, epoch: int, config: CVAETrainingConfig, use_lagging: bool = False):
    model.train()
    total_loss = total_recon = total_kl = 0.0
    beta = get_kl_weight(epoch, config.cyclical_period, config.n_cycles, R=config.cyclical_R, beta_min=config.beta_min)

    for batch_idx, (src, lengths, cond) in enumerate(loader, start=1):
        src, lengths, cond = src.to(device), lengths.to(device), cond.to(device)

        if use_lagging:
            # He et al. 2019 "Lagging Inference Networks": update only the encoder
            # for several inner steps so q(z|x) catches up with p(x|z). Freeze decoder
            # via requires_grad=False so Adam does not advance moments for decoder
            # params with stale/zeroed grads (the previous zero-grad hack caused
            # decoder drift from non-zero m_hat / sqrt(v_hat)).
            for p in model.decoder.parameters():
                p.requires_grad_(False)
            try:
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
                        sos_idx=getattr(model, "sos_idx", None),
                        eos_idx=getattr(model, "eos_idx", None),
                        word_dropout_rate=config.word_dropout_rate,
                    )
                    target = src[:, 1:]
                    loss_enc, _, _ = cvae_loss(
                        logits_for_enc, target, mu, logvar, criterion, beta=beta, free_bits=config.free_bits
                    )
                    loss_enc.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.encoder.parameters(), max_norm=config.max_grad_norm
                    )
                    optimizer.step()
            finally:
                for p in model.decoder.parameters():
                    p.requires_grad_(True)

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
    """Return (val_elbo, val_recon, val_kl). free_bits=0 so best-ckpt tracks the
    true variational bound, not the clamped training surrogate."""
    model.eval()
    total_loss = total_recon = total_kl = 0.0
    for src, lengths, cond in loader:
        src, lengths, cond = src.to(device), lengths.to(device), cond.to(device)
        logits, target, mu, logvar = model(src, lengths, cond, word_dropout_rate=0.0)
        loss, recon, kl = cvae_loss(
            logits, target, mu, logvar, criterion, beta=1.0, free_bits=0.0
        )
        total_loss += float(loss.item())
        total_recon += float(recon.item())
        total_kl += float(kl.item())
    n = max(len(loader), 1)
    return total_loss / n, total_recon / n, total_kl / n


def _save_checkpoint(path: Path, model, optimizer, epoch: int, best_val_elbo: float, history: dict) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_val_elbo": float(best_val_elbo),
        "history": history,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model, optimizer=None, map_location=None):
    """Resumable loader.

    Backward-compat: accepts plain state_dict, a dict with `model_state_dict`, or the
    full training payload written by `_save_checkpoint`.
    """
    path = Path(path)
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=map_location)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"failed to restore optimizer state: {exc}")
        if ckpt.get("rng_state") is not None:
            try:
                torch.set_rng_state(ckpt["rng_state"])
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"failed to restore torch rng state: {exc}")
        if ckpt.get("cuda_rng_state") is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"failed to restore cuda rng state: {exc}")
        return ckpt
    # Plain state_dict fallback (legacy ckpts).
    model.load_state_dict(ckpt)
    return {"model_state_dict": ckpt}


def fit_cvae(model, train_loader, val_loader, config: CVAETrainingConfig):
    if config.device:
        device = torch.device(config.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    # reduction='sum' backs the per-sequence convention in cvae_loss.
    criterion = nn.CrossEntropyLoss(ignore_index=model.pad_idx, reduction="sum")
    best_val = float("inf")
    wait = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_elbo": [],
        "val_recon": [],
        "val_kl": [],
        "recon": [],
        "kl": [],
        "beta": [],
    }
    best_path = Path(config.best_model_path)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    if config.lagging_epochs > 0 and config.lagging_epochs % max(config.cyclical_period, 1) == 0:
        warnings.warn(
            f"lagging_epochs={config.lagging_epochs} ends exactly at a cyclical-beta reset "
            f"(period={config.cyclical_period}); first non-lagging epoch runs at beta_min "
            f"without inference lag correction. Consider offsetting by +/-1.",
            stacklevel=2,
        )

    if config.verbose:
        print(
            f"[cVAE] device={device} epochs={config.num_epochs} patience={config.patience} "
            f"train_batches={len(train_loader)} val_batches={len(val_loader)} "
            f"batch_log_interval={config.batch_log_interval} best_model={best_path}",
            flush=True,
        )

    for epoch in range(config.num_epochs):
        use_lagging = epoch < config.lagging_epochs
        beta = get_kl_weight(epoch, config.cyclical_period, config.n_cycles, R=config.cyclical_R, beta_min=config.beta_min)
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
        val_elbo, val_recon, val_kl = eval_cvae_epoch(model, val_loader, criterion, device, config)
        val_loss = val_elbo  # best-ckpt selection metric; no free-bits clamp.

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_elbo"].append(val_elbo)
        history["val_recon"].append(val_recon)
        history["val_kl"].append(val_kl)
        history["recon"].append(recon)
        history["kl"].append(kl)
        history["beta"].append(beta)

        if config.verbose:
            print(
                f"[cVAE] epoch={epoch + 1}/{config.num_epochs} "
                f"train_loss={train_loss:.4f} recon={recon:.4f} kl={kl:.4f} "
                f"beta={beta:.3f} val_elbo={val_elbo:.4f} "
                f"val_recon={val_recon:.4f} val_kl={val_kl:.4f} lagging={use_lagging}",
                flush=True,
            )

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            _save_checkpoint(best_path, model, optimizer, epoch, best_val, history)
            if config.verbose:
                print(f"[cVAE] saved new best checkpoint: {best_path} (val_elbo={best_val:.4f})", flush=True)
        else:
            wait += 1
            if wait >= config.patience:
                if config.verbose:
                    print(f"[cVAE] early stopping after epoch {epoch + 1}; best_val={best_val:.4f}", flush=True)
                break

    return model, history, best_path
