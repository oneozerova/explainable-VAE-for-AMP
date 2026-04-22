"""Latent-space ablation: z vs [z ‖ c] vs [z ‖ c×W].

2×3 figure (rows = UMAP / t-SNE; cols = z alone / [z ‖ c] / [z ‖ c×W]).
Tests whether label structure emerges when condition c is added,
and whether artificially upweighting c forces clustering.

Usage::

    python3 scripts/plot_latent_ablation.py [--c-weight 5.0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch
import umap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from amp_vae.config import CONDITION_COLUMNS, DEFAULT_CVAE_CHECKPOINT, DEFAULT_VOCAB_PATH
from amp_vae.data.tokenizer import load_vocab_bundle
from amp_vae.evaluation.concept_vectors import build_condition_dataset, collect_latent_embeddings
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
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
UMAP_KWARGS = dict(n_neighbors=15, min_dist=0.05, n_components=2, random_state=42, metric="euclidean", verbose=True)


def umap_fit_transform(X_train, X_all):
    reducer = umap.UMAP(**UMAP_KWARGS)
    reducer.fit(X_train)
    return reducer.transform(X_all)


def tsne_fit_transform(X):
    return TSNE(n_components=2, perplexity=50, random_state=42, n_jobs=-1, verbose=1).fit_transform(X)


def scatter_panel(ax, emb, cond, cols, title, xlabel="dim 1", ylabel="dim 2"):
    ax.scatter(emb[:, 0], emb[:, 1], c="#cccccc", s=2, alpha=0.2, linewidths=0, rasterized=True)
    for k, col in enumerate(cols):
        mask = cond[:, k] == 1
        if not mask.any():
            continue
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            c=COLORS[k], s=4, alpha=0.6, linewidths=0, rasterized=True,
            label=f"{LABEL_DISPLAY.get(col, col)} (n={mask.sum()})",
        )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="upper right", framealpha=0.7, markerscale=2)


def make_zc(z_s, c_s, weight):
    """[z_s || c_s×weight] — both already standardized; weight directly controls
    relative contribution of c dims vs z dims in Euclidean distance."""
    return np.concatenate([z_s, c_s * weight], axis=1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--c-weight", type=float, default=5.0, help="upweight factor for c in 3rd column")
    p.add_argument("--tsne-only", action="store_true", help="save 1×3 t-SNE figure for blogpost in addition to full 2×3")
    p.add_argument("--two-panel", action="store_true", help="save 1×2 t-SNE figure ([z||c] and [z||c×W] only, no z-alone) for blogpost")
    return p.parse_args()


def main():
    args = parse_args()
    W = args.c_weight

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
    n_train = len(train_idx)

    ds_all, cols = build_condition_dataset(
        str(repo_path("data", "processed", "master_dataset.csv")),
        vocab, indices=all_idx.tolist(),
    )
    mu_all, cond_all = collect_latent_embeddings(model, ds_all, device=device)
    z = mu_all.cpu().numpy().astype(np.float32)
    c = cond_all.cpu().numpy().astype(np.float32)
    print(f"z={z.shape}  c={c.shape}")

    # Standardize z and c independently so weight is not undone by a joint scaler
    scaler_z = StandardScaler().fit(z[:n_train])
    z_s = scaler_z.transform(z)
    scaler_c = StandardScaler().fit(c[:n_train])
    c_s = scaler_c.transform(c)
    zc1_s = make_zc(z_s, c_s, weight=1.0)   # equal contribution per dim
    zcW_s = make_zc(z_s, c_s, weight=W)     # c dims upweighted ×W

    # ── UMAP ─────────────────────────────────────────────────────────────────
    print("UMAP: z alone...")
    emb_umap_z  = umap_fit_transform(z_s[:n_train],   z_s)
    print("UMAP: [z || c]...")
    emb_umap_zc = umap_fit_transform(zc1_s[:n_train], zc1_s)
    print(f"UMAP: [z || c×{W}]...")
    emb_umap_zcW = umap_fit_transform(zcW_s[:n_train], zcW_s)

    # ── t-SNE ────────────────────────────────────────────────────────────────
    print("t-SNE: z alone...")
    emb_tsne_z   = tsne_fit_transform(z_s)
    print("t-SNE: [z || c]...")
    emb_tsne_zc  = tsne_fit_transform(zc1_s)
    print(f"t-SNE: [z || c×{W}]...")
    emb_tsne_zcW = tsne_fit_transform(zcW_s)

    # ── figure 2×3 ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    cond_int = c.astype(int)

    scatter_panel(axes[0, 0], emb_umap_z,   cond_int, cols, "UMAP — z alone (32-dim)",       "UMAP 1", "UMAP 2")
    scatter_panel(axes[0, 1], emb_umap_zc,  cond_int, cols, "UMAP — [z ‖ c] (38-dim)",       "UMAP 1", "UMAP 2")
    scatter_panel(axes[0, 2], emb_umap_zcW, cond_int, cols, f"UMAP — [z ‖ c×{W}] (38-dim)",  "UMAP 1", "UMAP 2")

    scatter_panel(axes[1, 0], emb_tsne_z,   cond_int, cols, "t-SNE — z alone (32-dim)",       "dim 1", "dim 2")
    scatter_panel(axes[1, 1], emb_tsne_zc,  cond_int, cols, "t-SNE — [z ‖ c] (38-dim)",       "dim 1", "dim 2")
    scatter_panel(axes[1, 2], emb_tsne_zcW, cond_int, cols, f"t-SNE — [z ‖ c×{W}] (38-dim)",  "dim 1", "dim 2")

    fig.suptitle(
        f"Latent ablation: z alone vs [z ‖ c] vs [z ‖ c×{W}]\n"
        "Condition c upweighted to test whether activity clusters are latent or absent",
        fontsize=12,
    )
    plt.tight_layout()

    out = repo_path("docs", "figures", "latent_ablation_z_vs_zc.png")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    if args.two_panel:
        fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
        scatter_panel(axes3[0], emb_tsne_zc,  cond_int, cols, "[z ‖ c] equal weight (38-dim)",  "dim 1", "dim 2")
        scatter_panel(axes3[1], emb_tsne_zcW, cond_int, cols, f"[z ‖ c×{W}] (38-dim)",          "dim 1", "dim 2")
        fig3.suptitle(
            f"Adding condition vector to t-SNE: equal weight → c×{W}\n"
            "Label clusters emerge only when c dominates pairwise distances",
            fontsize=11,
        )
        plt.tight_layout()
        out3 = repo_path("docs", "figures", "tsne_ablation_2panel.png")
        fig3.savefig(str(out3), dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print(f"saved {out3}")

    if args.tsne_only:
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
        scatter_panel(axes2[0], emb_tsne_z,   cond_int, cols, "t-SNE — z alone (32-dim)",            "dim 1", "dim 2")
        scatter_panel(axes2[1], emb_tsne_zc,  cond_int, cols, "t-SNE — [z ‖ c] equal weight (38-dim)", "dim 1", "dim 2")
        scatter_panel(axes2[2], emb_tsne_zcW, cond_int, cols, f"t-SNE — [z ‖ c×{W}] (38-dim)",        "dim 1", "dim 2")
        fig2.suptitle(
            f"t-SNE ablation: z alone → [z ‖ c] → [z ‖ c×{W}]\n"
            "Activity structure exists in the data; z encodes style, c routes activity",
            fontsize=11,
        )
        plt.tight_layout()
        out2 = repo_path("docs", "figures", "tsne_ablation_cweight.png")
        fig2.savefig(str(out2), dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"saved {out2}")


if __name__ == "__main__":
    main()
