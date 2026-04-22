from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from amp_vae.config import (
    DEFAULT_CVAE_CHECKPOINT,
    DEFAULT_VOCAB_PATH,
    GENERATION_CONDITION_COLUMNS,
)
from amp_vae.data.tokenizer import load_vocab_bundle
from amp_vae.evaluation.concept_vectors import (
    build_condition_dataset,
    collect_latent_embeddings,
    compute_concept_vectors,
    concept_vector_cosine,
    concept_vector_walk,
)
from amp_vae.evaluation.physchem import properties_frame
from amp_vae.inference.generate import generation_condition_vector
from amp_vae.models.cvae import load_cvae_checkpoint
from amp_vae.utils.paths import repo_path


def parse_condition_spec(spec: str) -> dict[str, int]:
    values = {name: 0 for name in GENERATION_CONDITION_COLUMNS}
    if not spec:
        return values
    for item in spec.split(","):
        if not item.strip():
            continue
        key, value = item.split("=", 1)
        values[key.strip()] = int(value.strip())
    return values


def parse_alphas(spec: str) -> list[float]:
    return [float(x) for x in spec.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Compute concept vectors in the cVAE latent space.")
    parser.add_argument("--data", default=str(repo_path("data", "processed", "master_dataset.csv")))
    parser.add_argument("--vocab", default=str(repo_path(DEFAULT_VOCAB_PATH)))
    parser.add_argument("--checkpoint", default=str(repo_path(DEFAULT_CVAE_CHECKPOINT)))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--walk-alphas", default="-2,-1,0,1,2")
    parser.add_argument("--walk-seed", type=int, default=42)
    parser.add_argument("--walk-condition", default="is_anti_gram_positive=1")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--vectors-out", default=str(repo_path("docs", "tables", "concept_vectors.csv")))
    parser.add_argument("--cosine-out", default=str(repo_path("docs", "tables", "concept_vector_cosine.csv")))
    parser.add_argument("--walk-out", default=str(repo_path("docs", "tables", "concept_vector_walk.csv")))
    parser.add_argument("--pca-fig", default=str(repo_path("docs", "figures", "concept_vectors_pca.png")))
    parser.add_argument("--cosine-fig", default=str(repo_path("docs", "figures", "concept_vectors_cosine.png")))
    return parser.parse_args()


def _sample_z(seed: int, latent_dim: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    z = torch.randn(latent_dim, generator=gen)
    return z.to(device)


def _plot_pca(mu: np.ndarray, vectors: dict[str, dict], label_names: list[str], out_path: str) -> None:
    pca = PCA(n_components=2)
    mu_2d = pca.fit_transform(mu)
    origin_2d = pca.transform(np.zeros((1, mu.shape[1])))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(mu_2d[:, 0], mu_2d[:, 1], s=4, alpha=0.25, color="#888888", label="mu(x)")
    colors = plt.cm.tab10(np.linspace(0, 1, len(label_names)))
    for color, name in zip(colors, label_names):
        v = vectors[name]["vector"].cpu().numpy().reshape(1, -1)
        v_2d = pca.transform(v) - origin_2d
        ax.arrow(
            0.0,
            0.0,
            float(v_2d[0, 0]),
            float(v_2d[0, 1]),
            head_width=0.15,
            head_length=0.2,
            color=color,
            length_includes_head=True,
            label=name,
        )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title("Concept vectors (arrows from dataset mean) on PCA of mu")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_cosine(cosine: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cosine.values, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
    ax.set_xticks(range(len(cosine.columns)))
    ax.set_yticks(range(len(cosine.index)))
    ax.set_xticklabels(cosine.columns, rotation=45, ha="right")
    ax.set_yticklabels(cosine.index)
    for i in range(cosine.shape[0]):
        for j in range(cosine.shape[1]):
            ax.text(j, i, f"{cosine.values[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")
    ax.set_title("Cosine similarity between concept vectors")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    vocab = load_vocab_bundle(args.vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_cvae_checkpoint(args.checkpoint, vocab, device=device)

    # Train-only: centroids must not see val/test rows.
    from amp_vae.utils.splits import load_splits

    train_idx, _, _ = load_splits()
    dataset, condition_cols = build_condition_dataset(args.data, vocab, indices=train_idx)
    print(f"dataset_size={len(dataset)} condition_cols={condition_cols}")

    mu, conditions = collect_latent_embeddings(model, dataset, batch_size=args.batch_size, device=device)
    print(f"mu_shape={tuple(mu.shape)} conditions_shape={tuple(conditions.shape)}")

    concepts = compute_concept_vectors(mu, conditions, condition_cols)

    latent_dim = mu.size(1)
    vector_rows = []
    for name in condition_cols:
        info = concepts[name]
        row = {
            "label": name,
            "magnitude": info["magnitude"],
            "n_pos": info["n_pos"],
            "n_neg": info["n_neg"],
        }
        v_np = info["vector"].cpu().numpy()
        for i in range(latent_dim):
            row[f"dim_{i}"] = float(v_np[i])
        vector_rows.append(row)
    vectors_frame = pd.DataFrame(vector_rows)
    Path(args.vectors_out).parent.mkdir(parents=True, exist_ok=True)
    vectors_frame.to_csv(args.vectors_out, index=False)
    print(f"wrote {args.vectors_out}")

    vec_tensors = {name: concepts[name]["vector"] for name in condition_cols}
    cosine = concept_vector_cosine(vec_tensors)
    Path(args.cosine_out).parent.mkdir(parents=True, exist_ok=True)
    cosine.to_csv(args.cosine_out)
    print(f"wrote {args.cosine_out}")

    _plot_pca(mu.cpu().numpy(), concepts, condition_cols, args.pca_fig)
    print(f"wrote {args.pca_fig}")
    _plot_cosine(cosine, args.cosine_fig)
    print(f"wrote {args.cosine_fig}")

    alphas = parse_alphas(args.walk_alphas)
    walk_cond_map = parse_condition_spec(args.walk_condition)
    walk_cond = generation_condition_vector(**walk_cond_map)
    z0 = _sample_z(args.walk_seed, model.latent_dim, device)

    walk_rows = []
    for name in condition_cols:
        v = concepts[name]["vector"]
        steps = concept_vector_walk(
            model,
            vocab=vocab,
            v=v,
            z0=z0,
            alphas=alphas,
            condition=walk_cond,
            temperature=args.temperature,
            device=device,
        )
        seqs = [s for _, s in steps]
        props = properties_frame(seqs)
        for (alpha, seq), (_, prop_row) in zip(steps, props.iterrows()):
            walk_rows.append({
                "label": name,
                "alpha": float(alpha),
                "sequence": seq,
                "length": int(prop_row["length"]),
                "net_charge": float(prop_row["net_charge"]),
                "hydrophobicity": float(prop_row["hydrophobicity"]),
                "hydrophobic_moment": float(prop_row["hydrophobic_moment"]),
            })
    walk_frame = pd.DataFrame(walk_rows)
    Path(args.walk_out).parent.mkdir(parents=True, exist_ok=True)
    walk_frame.to_csv(args.walk_out, index=False)
    print(f"wrote {args.walk_out}")

    print("concept vector magnitudes:")
    for name in condition_cols:
        info = concepts[name]
        print(f"  {name}: |v|={info['magnitude']:.4f} n_pos={info['n_pos']} n_neg={info['n_neg']}")

    pairs = []
    names = list(condition_cols)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((names[i], names[j], float(cosine.values[i, j])))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    print("top 3 cosine pairs by |cos|:")
    for a, b, c in pairs[:3]:
        print(f"  {a} vs {b}: cos={c:+.4f}")


if __name__ == "__main__":
    main()
