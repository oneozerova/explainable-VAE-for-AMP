from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import torch

from amp_vae.config import GENERATION_CONDITION_COLUMNS, DEFAULT_CVAE_CHECKPOINT, DEFAULT_VOCAB_PATH
from amp_vae.data.tokenizer import load_vocab_bundle
from amp_vae.inference.generate import generation_condition_vector, generate_batch
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
    parser = argparse.ArgumentParser(description="Generate peptide candidates from a cVAE checkpoint.")
    parser.add_argument("--vocab", default=str(repo_path(DEFAULT_VOCAB_PATH)))
    parser.add_argument("--checkpoint", default=str(repo_path(DEFAULT_CVAE_CHECKPOINT)))
    parser.add_argument("--condition", default="is_anti_gram_positive=1")
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--output", default=str(repo_path("data", "processed", "generated_candidates.csv")))
    return parser.parse_args()


def main():
    args = parse_args()
    vocab = load_vocab_bundle(args.vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_cvae_checkpoint(args.checkpoint, vocab, device=device)
    cond_map = parse_condition_spec(args.condition)
    cond = generation_condition_vector(**cond_map)
    sequences = generate_batch(
        model,
        vocab=vocab,
        condition=cond,
        n_samples=args.n_samples,
        temperature=args.temperature,
        alpha=args.alpha,
        device=device,
    )
    frame = pd.DataFrame({"Sequence": sequences})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Saved {len(frame)} sequences to {args.output}")


if __name__ == "__main__":
    main()
