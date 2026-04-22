from __future__ import annotations

import argparse
import sys
import time
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
    CONDITION_COLUMNS,
    DEFAULT_CVAE_CHECKPOINT,
    DEFAULT_VOCAB_PATH,
)
from amp_vae.data.schema import SEQUENCE_COLUMN
from amp_vae.data.tokenizer import VocabBundle, load_vocab_bundle, tokenize_sequence
from amp_vae.evaluation.concept_vectors import (
    build_condition_dataset,
    collect_latent_embeddings,
)
from amp_vae.evaluation.physchem import properties_frame
from amp_vae.evaluation.projection import (
    fit_pca,
    fit_umap,
    load_parametric_tsne,
    project_parametric_tsne,
    select_real_pairs,
)
from amp_vae.inference.generate import (
    align_condition_to_model,
    decode_from_z_cond,
    generate_batch,
    generation_condition_vector,
)
from amp_vae.models.cvae import load_cvae_checkpoint
from amp_vae.utils.paths import repo_path


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate projection figures with interpolation overlays.")
    parser.add_argument("--data", default=str(repo_path("data", "processed", "master_dataset.csv")))
    parser.add_argument("--vocab", default=str(repo_path(DEFAULT_VOCAB_PATH)))
    parser.add_argument("--checkpoint", default=str(repo_path(DEFAULT_CVAE_CHECKPOINT)))
    parser.add_argument("--tsne-checkpoint", default=str(repo_path("models", "parametric_tsne.pt")))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=9)
    parser.add_argument("--samples-per-step", type=int, default=1)
    parser.add_argument("--n-paths", type=int, default=6)
    parser.add_argument("--pair-seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--fig-dir", default=str(repo_path("docs", "figures")))
    parser.add_argument("--label-a", default="is_anti_gram_positive")
    parser.add_argument("--label-b", default="is_anti_gram_negative")
    parser.add_argument("--n-generated-per-class", type=int, default=200)
    return parser.parse_args()


@torch.no_grad()
def interpolate_mu_path(
    model,
    vocab: VocabBundle,
    mu_start: np.ndarray,
    mu_end: np.ndarray,
    condition: list[float],
    n_steps: int,
    samples_per_step: int,
    temperature: float,
    device: torch.device,
    condition_end: list[float] | None = None,
) -> tuple[np.ndarray, list[tuple[float, list[str]]]]:
    """Decode a straight-line latent interpolation start -> end.

    When ``condition_end`` is provided, the conditioning vector is also linearly
    interpolated (``c_alpha = (1-alpha) * condition + alpha * condition_end``)
    so the endpoint is decoded under the endpoint's condition rather than under
    the start label. The cVAE accepts continuous conditions because the
    conditioning vector is concatenated with ``z`` (no argmax/one-hot
    requirement), so linear blends of one-hot vectors are valid inputs.
    """
    aligned_start = align_condition_to_model(condition, model)
    cond_start_t = torch.tensor(aligned_start, dtype=torch.float32, device=device)
    if condition_end is None:
        cond_end_t = cond_start_t
    else:
        aligned_end = align_condition_to_model(condition_end, model)
        cond_end_t = torch.tensor(aligned_end, dtype=torch.float32, device=device)
    z_s = torch.tensor(mu_start, dtype=torch.float32, device=device).view(-1)
    z_e = torch.tensor(mu_end, dtype=torch.float32, device=device).view(-1)
    alphas = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
    path_points = np.zeros((n_steps, z_s.shape[0]), dtype=np.float32)
    decoded: list[tuple[float, list[str]]] = []
    for i, a in enumerate(alphas.tolist()):
        z_mix = (1.0 - a) * z_s + a * z_e
        cond_alpha = (1.0 - a) * cond_start_t + a * cond_end_t
        path_points[i] = z_mix.detach().cpu().numpy()
        z_batch = z_mix.unsqueeze(0).repeat(samples_per_step, 1)
        cond_batch = cond_alpha.unsqueeze(0).repeat(samples_per_step, 1)
        seqs = decode_from_z_cond(
            model,
            z_batch,
            cond_batch,
            vocab=vocab,
            max_gen_len=vocab.max_len,
            temperature=temperature,
        )
        decoded.append((float(a), seqs))
    return path_points, decoded


def frame_limits(points_2d: np.ndarray, extra_points: np.ndarray | None = None, pad_frac: float = 0.05):
    lo = np.percentile(points_2d, 1, axis=0)
    hi = np.percentile(points_2d, 99, axis=0)
    if extra_points is not None and extra_points.size > 0:
        lo = np.minimum(lo, extra_points.min(axis=0))
        hi = np.maximum(hi, extra_points.max(axis=0))
    span = hi - lo
    lo = lo - pad_frac * span
    hi = hi + pad_frac * span
    return (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1]))


def plot_single_path(
    background_2d: np.ndarray,
    path_2d: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    import matplotlib.cm as cm

    n_steps = len(path_2d)
    alphas_norm = np.linspace(0.0, 1.0, n_steps)
    cmap = cm.get_cmap("plasma")
    step_colors = [cmap(a) for a in alphas_norm]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(
        background_2d[:, 0],
        background_2d[:, 1],
        s=3,
        alpha=0.15,
        color="#aaaacc",
        linewidths=0,
        rasterized=True,
    )
    # draw connecting line first (behind markers)
    ax.plot(path_2d[:, 0], path_2d[:, 1], "-", color="#888888", linewidth=1.2, zorder=3)
    # draw each step with colormap (alpha=0 → purple, alpha=1 → yellow)
    for i, (x, y, c) in enumerate(zip(path_2d[:, 0], path_2d[:, 1], step_colors)):
        ax.scatter(x, y, s=120, color=c, edgecolor="black", linewidths=0.8, zorder=5)
        ax.annotate(f"α={alphas_norm[i]:.2f}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7, zorder=6)
    xlim, ylim = frame_limits(background_2d, path_2d)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # colorbar for alpha progression
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("α (interpolation step)", fontsize=8)

    ax.text(0.02, 0.98, f"μ cloud n={background_2d.shape[0]}", transform=ax.transAxes,
            fontsize=8, va="top", color="#888888")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_multi_paths(
    background_2d: np.ndarray,
    paths_2d: list[np.ndarray],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    ax.scatter(
        background_2d[:, 0],
        background_2d[:, 1],
        s=3,
        alpha=0.2,
        color="#8888aa",
        linewidths=0,
        rasterized=True,
        label=f"mu cloud (n={background_2d.shape[0]})",
    )
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(paths_2d), 1)))
    all_path = np.concatenate(paths_2d, axis=0) if paths_2d else np.zeros((0, 2), dtype=np.float32)
    for i, path_2d in enumerate(paths_2d):
        c = colors[i % len(colors)]
        ax.plot(path_2d[:, 0], path_2d[:, 1], "-o", color=c, markersize=5, linewidth=1.3, label=f"path {i + 1}")
        ax.scatter([path_2d[0, 0]], [path_2d[0, 1]], marker="s", s=70, color=c, edgecolor="black", zorder=5)
        ax.scatter([path_2d[-1, 0]], [path_2d[-1, 1]], marker="*", s=140, color=c, edgecolor="black", zorder=5)
    xlim, ylim = frame_limits(background_2d, all_path)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_tsne_properties(tsne_2d: np.ndarray, sequences: list[str], out_path: Path) -> None:
    props = properties_frame(sequences)
    cols = [("net_charge", "Net charge"), ("hydrophobicity", "Hydrophobicity (KD)"), ("hydrophobic_moment", "Hydrophobic moment")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, (col, label) in zip(axes, cols):
        values = props[col].to_numpy(dtype=float)
        sc = ax.scatter(tsne_2d[:, 0], tsne_2d[:, 1], c=values, cmap="viridis", s=4, alpha=0.7, linewidths=0, rasterized=True)
        ax.set_title(label)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        fig.colorbar(sc, ax=ax, shrink=0.85)
    fig.suptitle("Parametric t-SNE of mu coloured by physicochemical properties")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_tsne_real_latents(tsne_2d: np.ndarray, conditions: np.ndarray, cond_cols: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    any_pos = (conditions >= 0.5).any(axis=1)
    argmax = np.argmax(conditions, axis=1)
    labels = np.where(any_pos, argmax, -1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(cond_cols)))
    other_mask = labels == -1
    if other_mask.any():
        ax.scatter(tsne_2d[other_mask, 0], tsne_2d[other_mask, 1], s=4, alpha=0.3, color="#bbbbbb", linewidths=0, rasterized=True, label="Other")
    for k, name in enumerate(cond_cols):
        mask = labels == k
        if not mask.any():
            continue
        ax.scatter(tsne_2d[mask, 0], tsne_2d[mask, 1], s=5, alpha=0.7, color=colors[k], linewidths=0, rasterized=True, label=name)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Parametric t-SNE of real mu (coloured by dominant activity)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def encode_sequences(model, vocab: VocabBundle, sequences: list[str], conditions: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.train(False)
    mus: list[np.ndarray] = []
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start : start + batch_size]
        cond_chunk = conditions[start : start + batch_size]
        token_batch: list[list[int]] = []
        length_batch: list[int] = []
        for seq in chunk:
            tokens, length = tokenize_sequence(seq, vocab)
            token_batch.append(tokens)
            length_batch.append(length)
        tokens_t = torch.tensor(token_batch, dtype=torch.long, device=device)
        lengths_t = torch.tensor(length_batch, dtype=torch.long, device=device)
        cond_t = torch.tensor(cond_chunk, dtype=torch.float32, device=device)
        mu, _ = model.encoder(tokens_t, lengths_t, cond_t)
        mus.append(mu.detach().cpu().numpy())
    return np.concatenate(mus, axis=0) if mus else np.zeros((0, model.latent_dim), dtype=np.float32)


def plot_tsne_real_vs_generated(
    real_tsne: np.ndarray,
    gen_tsne_by_label: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    ax.scatter(real_tsne[:, 0], real_tsne[:, 1], s=3, alpha=0.2, color="#999999", linewidths=0, rasterized=True, label=f"real mu (n={real_tsne.shape[0]})")
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(gen_tsne_by_label), 1)))
    for i, (name, pts) in enumerate(gen_tsne_by_label.items()):
        ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.7, color=colors[i], edgecolor="black", linewidths=0.3, label=f"generated: {name} (n={pts.shape[0]})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Parametric t-SNE: real vs generated mu")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.pair_seed)
    np.random.seed(args.pair_seed)
    rng = np.random.default_rng(args.pair_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    vocab = load_vocab_bundle(args.vocab)
    model = load_cvae_checkpoint(args.checkpoint, vocab, device=device)
    tsne_model = load_parametric_tsne(args.tsne_checkpoint, device=device)

    # Train-only background for projection.
    from amp_vae.utils.splits import load_splits

    train_idx, _, _ = load_splits()
    dataset, condition_cols = build_condition_dataset(args.data, vocab, indices=train_idx)
    print(f"dataset_size={len(dataset)} condition_cols={condition_cols}")

    mu_t, cond_t = collect_latent_embeddings(model, dataset, batch_size=args.batch_size, device=device)
    mu = mu_t.cpu().numpy().astype(np.float32)
    conditions = cond_t.cpu().numpy().astype(np.float32)
    print(f"mu_shape={mu.shape} conditions_shape={conditions.shape}")

    frame = pd.read_csv(args.data).iloc[list(train_idx)].dropna(subset=[SEQUENCE_COLUMN]).reset_index(drop=True)
    sequences = frame[SEQUENCE_COLUMN].astype(str).tolist()
    if len(sequences) != mu.shape[0]:
        sequences = sequences[: mu.shape[0]]

    t0 = time.time()
    pca = fit_pca(mu)
    print(f"pca fit in {time.time() - t0:.2f}s (explained={pca.explained_variance_ratio_.tolist()})")

    t0 = time.time()
    mu_pca = pca.transform(mu)
    mu_tsne = project_parametric_tsne(tsne_model, mu, device=device)
    print(f"pca+tsne project in {time.time() - t0:.2f}s")

    umap_model = None
    mu_umap = None
    try:
        t0 = time.time()
        umap_model = fit_umap(mu, random_state=args.pair_seed)
        mu_umap = umap_model.transform(mu)
        print(f"umap fit+transform in {time.time() - t0:.2f}s")
    except ImportError as exc:
        print(f"WARNING: umap-learn not available ({exc}); skipping UMAP plots.")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: UMAP failed ({exc}); skipping UMAP plots.")

    pairs = select_real_pairs(mu, conditions, condition_cols, args.label_a, args.label_b, args.n_paths, rng)
    if not pairs:
        raise RuntimeError(f"No pairs found for labels {args.label_a}/{args.label_b}")
    print(f"selected {len(pairs)} pairs for {args.label_a} -> {args.label_b}")

    single_mu_start, single_mu_end, _, _ = pairs[0]
    single_cond = [1.0 if c == args.label_a else 0.0 for c in condition_cols]
    cond_end = [1.0 if c == args.label_b else 0.0 for c in condition_cols]
    path_mu, decoded_single = interpolate_mu_path(
        model,
        vocab,
        single_mu_start,
        single_mu_end,
        single_cond,
        n_steps=args.n_steps,
        samples_per_step=args.samples_per_step,
        temperature=args.temperature,
        device=device,
        condition_end=cond_end,
    )
    print(f"single path mu shape={path_mu.shape}; first seq={decoded_single[0][1][0] if decoded_single[0][1] else ''}")

    path_pca = pca.transform(path_mu)

    pca_xlabel = f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)"
    pca_ylabel = f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)"

    # Fresh sklearn t-SNE on (background + path) jointly so path points get
    # accurate 2D positions — the parametric MLP compresses long-range distances.
    from sklearn.manifold import TSNE as _TSNE
    print("Fitting fresh t-SNE on background + path for interpolation figure...")
    combined_mu = np.concatenate([mu, path_mu], axis=0)
    combined_emb = _TSNE(
        n_components=2, perplexity=50, random_state=42, n_jobs=-1, verbose=1,
    ).fit_transform(combined_mu)
    mu_tsne_fresh = combined_emb[: len(mu)]
    path_tsne_fresh = combined_emb[len(mu) :]

    plot_single_path(
        mu_tsne_fresh,
        path_tsne_fresh,
        title=f"t-SNE with interpolation path ({args.label_a} -> {args.label_b})",
        xlabel="t-SNE 1",
        ylabel="t-SNE 2",
        out_path=fig_dir / "tsne_interpolation_path.png",
    )
    plot_single_path(
        mu_pca,
        path_pca,
        title=f"PCA with interpolation path ({args.label_a} -> {args.label_b})",
        xlabel=pca_xlabel,
        ylabel=pca_ylabel,
        out_path=fig_dir / "pca_interpolation_path.png",
    )
    if mu_umap is not None and umap_model is not None:
        path_umap = umap_model.transform(path_mu)
        plot_single_path(
            mu_umap,
            path_umap,
            title=f"UMAP with interpolation path ({args.label_a} -> {args.label_b})",
            xlabel="UMAP 1",
            ylabel="UMAP 2",
            out_path=fig_dir / "umap_interpolation_path.png",
        )

    paths_mu: list[np.ndarray] = []
    for (mu_s, mu_e, _, _) in pairs:
        p_mu, _ = interpolate_mu_path(
            model,
            vocab,
            mu_s,
            mu_e,
            single_cond,
            n_steps=args.n_steps,
            samples_per_step=args.samples_per_step,
            temperature=args.temperature,
            device=device,
            condition_end=cond_end,
        )
        paths_mu.append(p_mu)

    paths_pca = [pca.transform(p) for p in paths_mu]
    paths_tsne = [project_parametric_tsne(tsne_model, p, device=device) for p in paths_mu]

    plot_multi_paths(
        mu_tsne,
        paths_tsne,
        title=f"Parametric t-SNE with {len(paths_mu)} interpolation paths",
        xlabel="t-SNE 1",
        ylabel="t-SNE 2",
        out_path=fig_dir / "tsne_multiple_interpolation_paths.png",
    )
    plot_multi_paths(
        mu_pca,
        paths_pca,
        title=f"PCA with {len(paths_mu)} interpolation paths",
        xlabel=pca_xlabel,
        ylabel=pca_ylabel,
        out_path=fig_dir / "pca_multiple_interpolation_paths.png",
    )
    if mu_umap is not None and umap_model is not None:
        paths_umap = [umap_model.transform(p) for p in paths_mu]
        plot_multi_paths(
            mu_umap,
            paths_umap,
            title=f"UMAP with {len(paths_mu)} interpolation paths",
            xlabel="UMAP 1",
            ylabel="UMAP 2",
            out_path=fig_dir / "umap_multiple_interpolation_paths.png",
        )

    plot_tsne_properties(mu_tsne, sequences, fig_dir / "tsne_colored_by_properties.png")
    plot_tsne_real_latents(mu_tsne, conditions, list(condition_cols), fig_dir / "tsne_real_latents.png")

    target_labels = ["is_anti_gram_positive", "is_anti_gram_negative", "is_antifungal"]
    gen_tsne_by_label: dict[str, np.ndarray] = {}
    for name in target_labels:
        if name not in CONDITION_COLUMNS:
            continue
        cond_map = {n: 0 for n in CONDITION_COLUMNS}
        cond_map[name] = 1
        cond_vec = generation_condition_vector(**cond_map)
        gen_seqs = generate_batch(
            model,
            vocab,
            condition=cond_vec,
            n_samples=args.n_generated_per_class,
            temperature=args.temperature,
            device=device,
        )
        keep = [s for s in gen_seqs if s]
        if not keep:
            continue
        cond_arr = np.tile(np.array(cond_vec, dtype=np.float32), (len(keep), 1))
        mu_gen = encode_sequences(model, vocab, keep, cond_arr, device=device, batch_size=args.batch_size)
        gen_tsne_by_label[name] = project_parametric_tsne(tsne_model, mu_gen, device=device)

    if gen_tsne_by_label:
        plot_tsne_real_vs_generated(mu_tsne, gen_tsne_by_label, fig_dir / "tsne_real_vs_generated.png")

    print("done.")


if __name__ == "__main__":
    main()
