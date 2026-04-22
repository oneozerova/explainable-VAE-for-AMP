"""Retrain parametric t-SNE MLP on current cVAE mu distribution.

Supervised regression of sklearn t-SNE embedding on mu. The resulting MLP
mirrors the architecture used elsewhere (amp_vae.evaluation.projection.ParametricTSNE)
so the checkpoint drops in without any downstream change.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE

from amp_vae.config import DEFAULT_CVAE_CHECKPOINT, DEFAULT_VOCAB_PATH
from amp_vae.data.tokenizer import load_vocab_bundle
from amp_vae.evaluation.concept_vectors import (
    build_condition_dataset,
    collect_latent_embeddings,
)
from amp_vae.evaluation.projection import ParametricTSNE
from amp_vae.models.cvae import load_cvae_checkpoint
from amp_vae.utils.paths import repo_path
from amp_vae.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Retrain parametric t-SNE MLP on current cVAE mu distribution.")
    parser.add_argument("--data", default=str(repo_path("data", "processed", "master_dataset.csv")))
    parser.add_argument("--vocab", default=str(repo_path(DEFAULT_VOCAB_PATH)))
    parser.add_argument("--checkpoint", default=str(repo_path(DEFAULT_CVAE_CHECKPOINT)))
    parser.add_argument("--output", default=str(repo_path("models", "parametric_tsne.pt")))
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def r2_score(pred: np.ndarray, target: np.ndarray) -> float:
    sse = np.sum((target - pred) ** 2, axis=0)
    mean = target.mean(axis=0, keepdims=True)
    sst = np.sum((target - mean) ** 2, axis=0)
    r2_per_dim = 1.0 - (sse / np.clip(sst, 1e-12, None))
    return float(np.mean(r2_per_dim))


def main():
    args = parse_args()
    set_seed(args.seed, deterministic=False)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device={device}")

    vocab = load_vocab_bundle(args.vocab)
    model = load_cvae_checkpoint(args.checkpoint, vocab, device=device)

    # Fit t-SNE target on train mu only.
    from amp_vae.utils.splits import load_splits

    train_idx, _, _ = load_splits()
    dataset, condition_cols = build_condition_dataset(args.data, vocab, indices=train_idx)
    print(f"dataset_size={len(dataset)} condition_cols={condition_cols}")

    mu_t, _ = collect_latent_embeddings(model, dataset, batch_size=args.batch_size, device=device)
    mu = mu_t.cpu().numpy().astype(np.float32)
    print(f"mu shape={mu.shape} mean={mu.mean():.4f} std={mu.std():.4f}")

    t0 = time.time()
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        init="pca",
        random_state=args.seed,
        learning_rate="auto",
        max_iter=1000,
    )
    target_2d = tsne.fit_transform(mu).astype(np.float32)
    tsne_elapsed = time.time() - t0
    print(f"sklearn t-SNE fit in {tsne_elapsed:.2f}s target range=[{target_2d.min():.2f},{target_2d.max():.2f}]")

    in_dim = int(mu.shape[1])
    net = ParametricTSNE(in_dim=in_dim, hidden=128, out_dim=2).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    X = torch.as_tensor(mu, dtype=torch.float32, device=device)
    Y = torch.as_tensor(target_2d, dtype=torch.float32, device=device)
    n = X.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    best_loss = float("inf")
    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    bad = 0
    train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        net.train(True)
        perm = torch.randperm(n, generator=gen).to(device)
        total = 0.0
        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size]
            pred = net(X[idx])
            loss = mse(pred, Y[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item() * idx.shape[0]
        epoch_loss = total / n
        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if epoch % 20 == 0 or epoch == 1:
            print(f"epoch {epoch:4d}  loss={epoch_loss:.4f}  best={best_loss:.4f}  bad={bad}")
        if bad >= args.patience:
            print(f"early stop at epoch {epoch} (no improvement for {args.patience} epochs)")
            break
    train_elapsed = time.time() - train_start

    net.load_state_dict(best_state)
    net.train(False)
    with torch.no_grad():
        pred_all = net(X).cpu().numpy()
    final_mse = float(np.mean((pred_all - target_2d) ** 2))
    final_r2 = r2_score(pred_all, target_2d)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_path)

    reload_sd = torch.load(out_path, map_location="cpu", weights_only=False)
    print(f"checkpoint keys: {list(reload_sd.keys())}")
    print(f"saved to {out_path}")
    print(f"tsne_time={tsne_elapsed:.2f}s train_time={train_elapsed:.2f}s")
    print(f"FINAL_MSE={final_mse:.4f}  FINAL_R2={final_r2:.4f}")


if __name__ == "__main__":
    main()
