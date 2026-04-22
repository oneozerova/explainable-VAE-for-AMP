"""UMAP of cVAE mu — faceted one-vs-rest per condition label.

Fits UMAP on train mu, transforms all mu, then plots 6 one-vs-rest panels
(positive=coloured, rest=grey) plus one panel coloured by hydrophobic moment.

Usage::

    python3 scripts/plot_umap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import torch
import umap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from amp_vae.config import CONDITION_COLUMNS, DEFAULT_CVAE_CHECKPOINT, DEFAULT_VOCAB_PATH
from amp_vae.data.tokenizer import load_vocab_bundle
from amp_vae.evaluation.concept_vectors import build_condition_dataset, collect_latent_embeddings
from amp_vae.data.schema import SEQUENCE_COLUMN
from amp_vae.evaluation.physchem import hydrophobic_moment_per_residue
from amp_vae.models.cvae import load_cvae_checkpoint
from amp_vae.utils.paths import repo_path
from amp_vae.utils.splits import load_splits


LABEL_DISPLAY = {
    "is_anti_gram_positive": "Gram+",
    "is_anti_gram_negative": "Gram−",
    "is_antifungal": "Antifungal",
    "is_antiviral": "Antiviral",
    "is_antiparasitic": "Antiparasitic",
    "is_anticancer": "Anticancer",
}
COLORS = {
    "is_anti_gram_positive": "#4C72B0",
    "is_anti_gram_negative": "#55A868",
    "is_antifungal": "#C44E52",
    "is_antiviral": "#8172B2",
    "is_antiparasitic": "#CCB974",
    "is_anticancer": "#64B5CD",
}


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device={device}")

    vocab = load_vocab_bundle(str(repo_path(DEFAULT_VOCAB_PATH)))
    model = load_cvae_checkpoint(str(repo_path(DEFAULT_CVAE_CHECKPOINT)), vocab, device=device)

    train_idx, val_idx, test_idx = load_splits()
    all_idx = np.concatenate([train_idx, val_idx, test_idx])

    # collect all mu
    ds_all, cols = build_condition_dataset(
        str(repo_path("data", "processed", "master_dataset.csv")),
        vocab,
        indices=all_idx.tolist(),
    )
    mu_all, cond_all = collect_latent_embeddings(model, ds_all, device=device)
    mu = mu_all.cpu().numpy().astype(np.float32)
    cond = cond_all.cpu().numpy().astype(np.int32)
    print(f"mu shape={mu.shape}")

    # get sequences for moment computation
    frame = pd.read_csv(str(repo_path("data", "processed", "master_dataset.csv")))
    frame_sub = frame.iloc[all_idx.tolist()].reset_index(drop=True)
    # drop NaN seqs to match ds_all filtering
    frame_sub = frame_sub.dropna(subset=[SEQUENCE_COLUMN]).reset_index(drop=True)
    cond_cols_present = [c for c in CONDITION_COLUMNS if c in frame_sub.columns]
    frame_sub = frame_sub.dropna(subset=cond_cols_present).reset_index(drop=True)

    # align lengths
    n = min(len(mu), len(frame_sub))
    mu = mu[:n]
    cond = cond[:n]
    seqs = frame_sub[SEQUENCE_COLUMN].astype(str).tolist()[:n]

    # UMAP fit on train partition of mu
    n_train = len(train_idx)
    reducer = umap.UMAP(
        n_neighbors=50,
        min_dist=0.3,
        n_components=2,
        random_state=42,
        metric="euclidean",
        verbose=True,
    )
    print("Fitting UMAP on train mu...")
    reducer.fit(mu[:n_train])
    print("Transforming all mu...")
    emb = reducer.transform(mu)
    print(f"embedding shape={emb.shape} range=[{emb.min():.2f},{emb.max():.2f}]")

    # compute moment for colour
    moments = np.array([hydrophobic_moment_per_residue(s) for s in seqs], dtype=np.float32)
    moments = np.nan_to_num(moments, nan=0.0)

    # ── figure 1: one-vs-rest facets ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, col in zip(axes.flat, CONDITION_COLUMNS):
        if col not in cols:
            ax.set_visible(False)
            continue
        k = cols.index(col)
        pos_mask = cond[:, k] == 1
        ax.scatter(
            emb[~pos_mask, 0], emb[~pos_mask, 1],
            c="#cccccc", s=3, alpha=0.25, linewidths=0, rasterized=True,
        )
        ax.scatter(
            emb[pos_mask, 0], emb[pos_mask, 1],
            c=COLORS[col], s=6, alpha=0.7, linewidths=0, rasterized=True,
            label=f"{LABEL_DISPLAY[col]} (n={pos_mask.sum()})",
        )
        ax.set_title(LABEL_DISPLAY[col], fontsize=10)
        ax.set_xlabel("UMAP 1", fontsize=8)
        ax.set_ylabel("UMAP 2", fontsize=8)
        ax.tick_params(labelsize=7)
        leg = ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
        for h in leg.legend_handles:
            h.set_sizes([20])

    fig.suptitle("UMAP of cVAE μ — one-vs-rest per label", fontsize=12)
    plt.tight_layout()
    out1 = repo_path("docs", "figures", "umap_one_vs_rest.png")
    Path(out1).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out1), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out1}")

    # ── figure 2: physchem colouring ──────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    sc = ax2.scatter(
        emb[:, 0], emb[:, 1],
        c=moments, cmap="viridis", s=5, alpha=0.6, linewidths=0, rasterized=True,
    )
    plt.colorbar(sc, ax=ax2, label="Hydrophobic moment / residue")
    ax2.set_title("UMAP of cVAE μ — hydrophobic moment", fontsize=11)
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")
    out2 = repo_path("docs", "figures", "umap_moment.png")
    fig2.savefig(str(out2), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
