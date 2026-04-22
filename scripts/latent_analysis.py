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

from amp_vae.config import (
    DEFAULT_CVAE_CHECKPOINT,
    DEFAULT_VOCAB_PATH,
    GENERATION_CONDITION_COLUMNS,
)
from amp_vae.data.tokenizer import load_vocab_bundle
from amp_vae.evaluation.concept_vectors import (
    build_condition_dataset,
    collect_latent_embeddings,
)
from amp_vae.evaluation.latent_analysis import (
    PROPERTY_KEYS,
    active_units,
    dim_traversal,
    summarize_dim_effects,
)
from amp_vae.evaluation.physchem import properties_frame
from amp_vae.inference.generate import generation_condition_vector
from amp_vae.models.cvae import load_cvae_checkpoint
from amp_vae.utils.paths import repo_path


PROP_COL_MAP = {
    "charge": "net_charge",
    "hydro": "hydrophobicity",
    "moment": "hydrophobic_moment",
    "length": "length",
}


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


def parse_args():
    parser = argparse.ArgumentParser(description="Latent-dim activity and traversal analysis for the cVAE.")
    parser.add_argument("--data", default=str(repo_path("data", "processed", "master_dataset.csv")))
    parser.add_argument("--vocab", default=str(repo_path(DEFAULT_VOCAB_PATH)))
    parser.add_argument("--checkpoint", default=str(repo_path(DEFAULT_CVAE_CHECKPOINT)))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--n-steps", type=int, default=9)
    parser.add_argument("--span-sigma", type=float, default=2.0)
    parser.add_argument("--samples-per-step", type=int, default=3)
    parser.add_argument("--z-seed", type=int, default=42)
    parser.add_argument("--traversal-condition", default="is_anti_gram_positive=1")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k-dims", type=int, default=8)
    parser.add_argument("--active-out", default=str(repo_path("docs", "tables", "latent_active_units.csv")))
    parser.add_argument("--interp-out", default=str(repo_path("docs", "tables", "latent_dim_interpretation.csv")))
    parser.add_argument("--traversal-out", default=str(repo_path("docs", "tables", "latent_dim_traversal.csv")))
    parser.add_argument("--active-fig", default=str(repo_path("docs", "figures", "latent_active_units_bar.png")))
    parser.add_argument("--traversal-fig", default=str(repo_path("docs", "figures", "latent_dim_traversal.png")))
    return parser.parse_args()


def _sample_z(seed: int, latent_dim: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    z = torch.randn(latent_dim, generator=gen)
    return z.to(device)


def _plot_active_bar(active_frame: pd.DataFrame, threshold: float, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    dims = active_frame["dim"].to_numpy()
    variances = active_frame["variance"].to_numpy()
    active_mask = active_frame["is_active"].to_numpy()
    colors = ["#1f77b4" if a else "#b0b0b0" for a in active_mask]
    ax.bar(dims, variances, color=colors)
    ax.axhline(float(threshold), color="red", linewidth=1.0, linestyle="--", label=f"threshold={threshold:g}")
    ax.set_xlabel("latent dimension")
    ax.set_ylabel("Var(mu_i)")
    ax.set_title("Active-unit variance per latent dimension")
    ax.set_xticks(dims)
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_traversal_heatmap(interp_frame: pd.DataFrame, out_path: str) -> None:
    if interp_frame.empty:
        return
    dims = interp_frame["dim"].astype(int).to_numpy()
    cols = [f"delta_{k}" for k in PROPERTY_KEYS]
    matrix = np.abs(interp_frame[cols].to_numpy().astype(np.float64))
    col_max = matrix.max(axis=0)
    col_max = np.where(col_max > 0, col_max, 1.0)
    normed = matrix / col_max

    fig, ax = plt.subplots(figsize=(6.5, max(3.5, 0.45 * len(dims) + 1.5)))
    im = ax.imshow(normed, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(PROPERTY_KEYS)))
    ax.set_xticklabels(list(PROPERTY_KEYS))
    ax.set_yticks(range(len(dims)))
    ax.set_yticklabels([f"dim {d}" for d in dims])
    for i in range(normed.shape[0]):
        for j in range(normed.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if normed[i, j] > 0.5 else "black",
            )
    ax.set_title("Latent-dim traversal: |delta| per property (cells annotated with raw delta)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="normalized |delta| (per column)")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _aggregate_traversal(
    dim: int,
    steps: list[tuple[float, list[str]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    traversal_rows = []
    summary_rows = []
    for alpha, seqs in steps:
        props = properties_frame(seqs)
        row = {"dim": int(dim), "alpha": float(alpha), "n_samples": int(len(seqs))}
        summary = {"alpha": float(alpha)}
        for key, col in PROP_COL_MAP.items():
            values = props[col].astype(float).to_numpy() if col in props.columns else np.array([])
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                mean_val = float("nan")
                std_val = float("nan")
            else:
                mean_val = float(np.mean(finite))
                std_val = float(np.std(finite))
            row[f"mean_{key}"] = mean_val
            row[f"std_{key}"] = std_val
            summary[f"mean_{key}"] = mean_val
        traversal_rows.append(row)
        summary_rows.append(summary)
    return pd.DataFrame(traversal_rows), pd.DataFrame(summary_rows)


def main():
    args = parse_args()
    vocab = load_vocab_bundle(args.vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_cvae_checkpoint(args.checkpoint, vocab, device=device)

    # Train-only: active-units + per-dim std(mu) on train rows.
    from amp_vae.utils.splits import load_splits

    train_idx, _, _ = load_splits()
    dataset, condition_cols = build_condition_dataset(args.data, vocab, indices=train_idx)
    print(f"dataset_size={len(dataset)} condition_cols={condition_cols}")

    mu, _ = collect_latent_embeddings(model, dataset, batch_size=args.batch_size, device=device)
    print(f"mu_shape={tuple(mu.shape)}")

    active_frame = active_units(mu, threshold=args.threshold)
    Path(args.active_out).parent.mkdir(parents=True, exist_ok=True)
    active_frame.to_csv(args.active_out, index=False)
    print(f"wrote {args.active_out}")
    _plot_active_bar(active_frame, threshold=args.threshold, out_path=args.active_fig)
    print(f"wrote {args.active_fig}")

    mu_std_per_dim = mu.float().std(dim=0, unbiased=False).cpu().numpy()

    active_sorted = active_frame[active_frame["is_active"]].sort_values("variance", ascending=False)
    if int(args.top_k_dims) > 0:
        selected_dims = active_sorted["dim"].astype(int).tolist()[: int(args.top_k_dims)]
    else:
        selected_dims = active_sorted["dim"].astype(int).tolist()
    print(f"n_active={len(active_sorted)} traversing_dims={selected_dims}")

    cond_map = parse_condition_spec(args.traversal_condition)
    condition = generation_condition_vector(**cond_map)
    z0 = _sample_z(args.z_seed, model.latent_dim, device)

    traversal_frames: list[pd.DataFrame] = []
    summary_map: dict[int, pd.DataFrame] = {}
    for dim in selected_dims:
        steps = dim_traversal(
            model,
            vocab=vocab,
            dim=int(dim),
            z0=z0,
            condition=condition,
            n_steps=args.n_steps,
            span_sigma=args.span_sigma,
            temperature=args.temperature,
            mu_std=float(mu_std_per_dim[int(dim)]),
            samples_per_step=args.samples_per_step,
            device=device,
        )
        traversal_frame, summary_frame = _aggregate_traversal(int(dim), steps)
        traversal_frames.append(traversal_frame)
        summary_map[int(dim)] = summary_frame

    traversal_all = (
        pd.concat(traversal_frames, ignore_index=True)
        if traversal_frames
        else pd.DataFrame(columns=[
            "dim", "alpha", "n_samples",
            "mean_charge", "std_charge",
            "mean_hydro", "std_hydro",
            "mean_moment", "std_moment",
            "mean_length", "std_length",
        ])
    )
    Path(args.traversal_out).parent.mkdir(parents=True, exist_ok=True)
    traversal_all.to_csv(args.traversal_out, index=False)
    print(f"wrote {args.traversal_out}")

    interp_frame = summarize_dim_effects(mu, summary_map)
    Path(args.interp_out).parent.mkdir(parents=True, exist_ok=True)
    interp_frame.to_csv(args.interp_out, index=False)
    print(f"wrote {args.interp_out}")

    _plot_traversal_heatmap(interp_frame, args.traversal_fig)
    print(f"wrote {args.traversal_fig}")

    n_active = int(active_frame["is_active"].sum())
    print(f"summary: n_active={n_active}/{len(active_frame)}")
    if not interp_frame.empty:
        abs_cols = [f"delta_{k}" for k in PROPERTY_KEYS]
        magnitudes = interp_frame[abs_cols].abs().max(axis=1)
        ranked = interp_frame.assign(_mag=magnitudes).sort_values("_mag", ascending=False)
        print("top 3 dims by largest |property delta|:")
        for _, row in ranked.head(3).iterrows():
            print(
                f"  dim {int(row['dim'])}: driven={row['property_driven_most']} "
                f"delta_charge={row['delta_charge']:+.3f} delta_hydro={row['delta_hydro']:+.3f} "
                f"delta_moment={row['delta_moment']:+.3f} delta_length={row['delta_length']:+.3f}"
            )


if __name__ == "__main__":
    main()
