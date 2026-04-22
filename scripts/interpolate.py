from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import torch

from amp_vae.config import GENERATION_CONDITION_COLUMNS, DEFAULT_CVAE_CHECKPOINT, DEFAULT_VOCAB_PATH
from amp_vae.data.tokenizer import load_vocab_bundle
from amp_vae.evaluation.physchem import properties_frame
from amp_vae.inference.generate import (
    generation_condition_vector,
    interpolate_conditions,
    interpolate_latent,
)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run latent or condition interpolation through the cVAE.")
    parser.add_argument("--vocab", default=str(repo_path(DEFAULT_VOCAB_PATH)))
    parser.add_argument("--checkpoint", default=str(repo_path(DEFAULT_CVAE_CHECKPOINT)))
    parser.add_argument("--mode", choices=["latent", "condition"], default="latent")
    parser.add_argument("--condition", default="is_anti_gram_positive=1")
    parser.add_argument("--cond-start", default="is_anti_gram_positive=1")
    parser.add_argument("--cond-end", default="is_anti_gram_negative=1")
    parser.add_argument("--z-seed-a", type=int, default=42)
    parser.add_argument("--z-seed-b", type=int, default=7)
    parser.add_argument("--n-steps", type=int, default=9)
    parser.add_argument("--samples-per-step", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--output", default=str(repo_path("docs", "tables", "interpolation_path_sequences.csv")))
    parser.add_argument("--canonical-output", default=str(repo_path("docs", "tables", "interpolation_examples.csv")))
    return parser.parse_args()


def _sample_z(seed: int, latent_dim: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    z = torch.randn(latent_dim, generator=gen)
    return z.to(device)


def main():
    args = parse_args()
    vocab = load_vocab_bundle(args.vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_cvae_checkpoint(args.checkpoint, vocab, device=device)

    if args.mode == "latent":
        cond_map = parse_condition_spec(args.condition)
        cond = generation_condition_vector(**cond_map)
        z_start = _sample_z(args.z_seed_a, model.latent_dim, device)
        z_end = _sample_z(args.z_seed_b, model.latent_dim, device)
        steps = interpolate_latent(
            model,
            vocab=vocab,
            z_start=z_start,
            z_end=z_end,
            condition=cond,
            n_steps=args.n_steps,
            temperature=args.temperature,
            samples_per_step=args.samples_per_step,
            device=device,
        )
    else:
        cond_start_map = parse_condition_spec(args.cond_start)
        cond_end_map = parse_condition_spec(args.cond_end)
        cond_start = generation_condition_vector(**cond_start_map)
        cond_end = generation_condition_vector(**cond_end_map)
        z_fixed = _sample_z(args.z_seed_a, model.latent_dim, device)
        steps = interpolate_conditions(
            model,
            vocab=vocab,
            cond_start=cond_start,
            cond_end=cond_end,
            z=z_fixed,
            n_steps=args.n_steps,
            temperature=args.temperature,
            samples_per_step=args.samples_per_step,
            device=device,
        )

    rows = []
    canonical_rows = []
    all_seqs: list[str] = []
    mean_moments: list[float] = []

    for alpha, seqs in steps:
        props = properties_frame(seqs)
        row = {
            "alpha": float(alpha),
            "n_samples": int(len(seqs)),
            "sequences": "|".join(seqs),
            "mean_charge": float(props["net_charge"].mean()),
            "std_charge": float(props["net_charge"].std(ddof=0)),
            "mean_hydro": float(props["hydrophobicity"].mean()),
            "std_hydro": float(props["hydrophobicity"].std(ddof=0)),
            "mean_moment": float(props["hydrophobic_moment"].mean()),
            "std_moment": float(props["hydrophobic_moment"].std(ddof=0)),
            "mean_length": float(props["length"].mean()),
            "std_length": float(props["length"].std(ddof=0)),
        }
        rows.append(row)
        canonical_rows.append({"alpha": float(alpha), "sequence": seqs[0] if seqs else ""})
        all_seqs.extend(seqs)
        mean_moments.append(row["mean_moment"])

    frame = pd.DataFrame(rows)
    canonical = pd.DataFrame(canonical_rows)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.canonical_output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    canonical.to_csv(args.canonical_output, index=False)

    distinct = len({s for s in all_seqs if s})
    moment_std = float(np.nanstd(np.asarray(mean_moments, dtype=np.float64), ddof=0))
    print(f"Mode: {args.mode}")
    print(f"Wrote {len(frame)} alpha rows to {args.output}")
    print(f"Wrote canonical examples to {args.canonical_output}")
    print(f"Distinct sequences across alphas: {distinct}")
    print(f"Hydrophobic-moment std across alpha means: {moment_std:.4f}")


if __name__ == "__main__":
    main()
