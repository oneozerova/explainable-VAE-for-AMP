"""t-SNE of cVAE μ: one-vs-rest per label + physicochemical coloring.

Saves:
  docs/figures/tsne_one_vs_rest.png  — 2×3 panel, one activity label each
  docs/figures/tsne_physchem.png     — 2×2 panel: hydrophobic moment / net charge /
                                       hydrophobicity / length

Usage::

    python3 scripts/plot_tsne_latent.py
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from amp_vae.config import CONDITION_COLUMNS, DEFAULT_CVAE_CHECKPOINT, DEFAULT_VOCAB_PATH
from amp_vae.data.tokenizer import load_vocab_bundle
from amp_vae.evaluation.concept_vectors import build_condition_dataset, collect_latent_embeddings
from amp_vae.data.schema import SEQUENCE_COLUMN
from amp_vae.evaluation.physchem import hydrophobic_moment_per_residue, net_charge, hydrophobicity
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
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"device={device}")

    vocab = load_vocab_bundle(str(repo_path(DEFAULT_VOCAB_PATH)))
    model = load_cvae_checkpoint(str(repo_path(DEFAULT_CVAE_CHECKPOINT)), vocab, device=device)

    train_idx, val_idx, test_idx = load_splits()
    all_idx = np.concatenate([train_idx, val_idx, test_idx])

    ds_all, cols = build_condition_dataset(
        str(repo_path("data", "processed", "master_dataset.csv")),
        vocab,
        indices=all_idx.tolist(),
    )
    mu_all, cond_all = collect_latent_embeddings(model, ds_all, device=device)
    mu = mu_all.cpu().numpy().astype(np.float32)
    cond = cond_all.cpu().numpy().astype(np.int32)
    print(f"mu={mu.shape}  cond={cond.shape}")

    # sequences for physchem
    frame = pd.read_csv(str(repo_path("data", "processed", "master_dataset.csv")))
    frame_sub = frame.iloc[all_idx.tolist()].reset_index(drop=True)
    frame_sub = frame_sub.dropna(subset=[SEQUENCE_COLUMN]).reset_index(drop=True)
    cond_cols_present = [c for c in CONDITION_COLUMNS if c in frame_sub.columns]
    frame_sub = frame_sub.dropna(subset=cond_cols_present).reset_index(drop=True)
    n = min(len(mu), len(frame_sub))
    mu = mu[:n]
    cond = cond[:n]
    seqs = frame_sub[SEQUENCE_COLUMN].astype(str).tolist()[:n]

    print("Running t-SNE (perplexity=50)...")
    emb = TSNE(n_components=2, perplexity=50, random_state=42, n_jobs=-1, verbose=1).fit_transform(mu)
    print(f"embedding={emb.shape}  range=[{emb.min():.2f},{emb.max():.2f}]")

    # ── figure 1: one-vs-rest per activity label ───────────────────────────────
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
        ax.set_xlabel("t-SNE 1", fontsize=8)
        ax.set_ylabel("t-SNE 2", fontsize=8)
        ax.tick_params(labelsize=7)
        leg = ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
        for h in leg.legend_handles:
            h.set_sizes([20])

    fig.suptitle("t-SNE of cVAE μ — one-vs-rest per label", fontsize=12)
    plt.tight_layout()
    out1 = repo_path("docs", "figures", "tsne_one_vs_rest.png")
    Path(out1).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out1), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out1}")

    # ── figure 2: physicochemical coloring ────────────────────────────────────
    moments = np.array([hydrophobic_moment_per_residue(s) for s in seqs], dtype=np.float32)
    charges = np.array([net_charge(s) for s in seqs], dtype=np.float32)
    hydros = np.array([hydrophobicity(s) for s in seqs], dtype=np.float32)
    lengths = np.array([len(s) for s in seqs], dtype=np.float32)

    moments = np.nan_to_num(moments, nan=0.0)
    charges = np.nan_to_num(charges, nan=0.0)
    hydros = np.nan_to_num(hydros, nan=0.0)

    props = [
        (moments, "Hydrophobic moment / residue", "viridis"),
        (charges, "Net charge",                   "RdBu_r"),
        (hydros,  "Mean hydrophobicity (KD)",      "viridis"),
        (lengths, "Sequence length",               "plasma"),
    ]
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (vals, label, cmap) in zip(axes2.flat, props):
        sc = ax.scatter(
            emb[:, 0], emb[:, 1],
            c=vals, cmap=cmap, s=4, alpha=0.6, linewidths=0, rasterized=True,
        )
        plt.colorbar(sc, ax=ax, label=label, shrink=0.85)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("t-SNE 1", fontsize=8)
        ax.set_ylabel("t-SNE 2", fontsize=8)
        ax.tick_params(labelsize=7)

    fig2.suptitle("t-SNE of cVAE μ — physicochemical properties", fontsize=12)
    plt.tight_layout()
    out2 = repo_path("docs", "figures", "tsne_physchem.png")
    fig2.savefig(str(out2), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
