"""Batch-generate peptides for AIPAMPDS oracle submission.

Generates one batch per condition with a fixed per-condition seed derived from
``--seed``. Records checkpoint SHA-256 and git SHA for every row so the
submission CSV is fully traceable.

Usage::

    python scripts/generate_for_oracle.py \
        --ckpt models/best_cvae.pt \
        --n-per-condition 250 \
        --seed 42 \
        --temperature 0.8 \
        --top-p 0.95 \
        --out data/external/aipampds/submission_YYYYMMDD.csv

Conditions (6-label schema, no antibacterial label):

    gram_pos       is_anti_gram_positive = 1
    gram_neg       is_anti_gram_negative = 1
    antifungal     is_antifungal         = 1
    antiviral      is_antiviral          = 1
    antiparasitic  is_antiparasitic      = 1
    anticancer     is_anticancer         = 1
    unconditional  all zeros

Output CSV columns::

    sequence, condition_name, condition_vector, seed,
    checkpoint_sha256, git_sha, temperature, top_p, top_k, generated_at

Also writes ``<out>_summary.json`` with per-condition counts + compatibility
rate (kept / generated after VALID_AMINO_ACIDS filter).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import torch

from amp_vae.config import (
    CONDITION_COLUMNS,
    DEFAULT_VOCAB_PATH,
    VALID_AMINO_ACIDS,
)
from amp_vae.data.tokenizer import load_vocab_bundle
from amp_vae.inference.generate import generate, is_valid_sequence
from amp_vae.models.cvae import load_cvae_checkpoint
from amp_vae.utils.paths import repo_path


# ---------------------------------------------------------------------------
# Condition definitions. Ordering fixes the per-condition seed offset.
# ---------------------------------------------------------------------------
CONDITIONS: list[tuple[str, dict[str, int]]] = [
    ("gram_pos", {"is_anti_gram_positive": 1}),
    ("gram_neg", {"is_anti_gram_negative": 1}),
    ("antifungal", {"is_antifungal": 1}),
    ("antiviral", {"is_antiviral": 1}),
    ("antiparasitic", {"is_antiparasitic": 1}),
    ("anticancer", {"is_anticancer": 1}),
    ("unconditional", {}),
]


def build_condition_vector(flags: dict[str, int]) -> list[int]:
    return [int(flags.get(col, 0)) for col in CONDITION_COLUMNS]


def compute_checkpoint_sha256(ckpt_path: Path) -> str:
    return hashlib.sha256(Path(ckpt_path).read_bytes()).hexdigest()


def current_git_sha(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception as exc:  # pragma: no cover - defensive
        return f"unknown:{type(exc).__name__}"


def _filter_valid(sequences: Iterable[str]) -> list[str]:
    kept: list[str] = []
    for raw in sequences:
        if not raw:
            continue
        s = str(raw).strip().upper()
        if not s:
            continue
        if not set(s).issubset(VALID_AMINO_ACIDS):
            continue
        # Defer length limits to is_valid_sequence so the CSV matches the
        # project's canonical validity definition.
        if is_valid_sequence(s):
            kept.append(s)
    return kept


def generate_for_condition(
    model,
    vocab,
    condition_vector: list[int],
    n_target: int,
    seed: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    device: torch.device,
    max_attempts: int = 5,
    oversample: float = 2.0,
) -> tuple[list[str], int]:
    """Generate until ``n_target`` valid sequences collected or attempts exhausted.

    Returns ``(kept, total_generated)`` so the caller can compute a compatibility
    rate = ``len(kept) / total_generated``.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    kept: list[str] = []
    seen: set[str] = set()
    total_generated = 0
    batch_n = max(n_target, int(n_target * oversample))

    for _ in range(max_attempts):
        sequences = generate(
            model,
            vocab=vocab,
            condition=condition_vector,
            n_samples=batch_n,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
            generator=generator,
        )
        total_generated += len(sequences)
        for s in _filter_valid(sequences):
            if s in seen:
                continue
            seen.add(s)
            kept.append(s)
            if len(kept) >= n_target:
                break
        if len(kept) >= n_target:
            break

    return kept[:n_target], total_generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", default=str(repo_path("models", "best_cvae.pt")))
    parser.add_argument("--vocab", default=str(repo_path(DEFAULT_VOCAB_PATH)))
    parser.add_argument("--n-per-condition", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0, help="0 disables top-k")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--device", default=None, help="cuda / cpu; autodetect if omitted")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_path = Path(args.ckpt)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out_path.with_name(out_path.stem + "_summary.json")

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[generate_for_oracle] device={device}", flush=True)
    print(f"[generate_for_oracle] ckpt={ckpt_path}", flush=True)

    vocab = load_vocab_bundle(args.vocab)
    model = load_cvae_checkpoint(str(ckpt_path), vocab, device=device)

    ckpt_sha = compute_checkpoint_sha256(ckpt_path)
    git_sha = current_git_sha(ROOT)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    top_k_val = args.top_k if args.top_k and args.top_k > 0 else None
    top_p_val = args.top_p if args.top_p and 0.0 < args.top_p < 1.0 else None

    rows: list[dict] = []
    per_condition_summary: dict[str, dict] = {}

    for idx, (name, flags) in enumerate(CONDITIONS):
        cond_vec = build_condition_vector(flags)
        cond_seed = int(args.seed) + idx
        print(
            f"[generate_for_oracle] condition={name} seed={cond_seed} "
            f"vec={cond_vec} target={args.n_per_condition}",
            flush=True,
        )
        kept, total_generated = generate_for_condition(
            model=model,
            vocab=vocab,
            condition_vector=cond_vec,
            n_target=args.n_per_condition,
            seed=cond_seed,
            temperature=args.temperature,
            top_p=top_p_val,
            top_k=top_k_val,
            device=device,
        )
        compat = (len(kept) / total_generated) if total_generated else 0.0
        per_condition_summary[name] = {
            "requested": args.n_per_condition,
            "kept": len(kept),
            "generated": total_generated,
            "compatibility_rate": compat,
            "condition_vector": cond_vec,
            "seed": cond_seed,
        }
        print(
            f"[generate_for_oracle]  -> kept {len(kept)}/{total_generated} "
            f"(compat={compat:.3f})",
            flush=True,
        )
        cond_vec_str = "[" + ",".join(str(x) for x in cond_vec) + "]"
        for seq in kept:
            rows.append(
                {
                    "sequence": seq,
                    "condition_name": name,
                    "condition_vector": cond_vec_str,
                    "seed": cond_seed,
                    "checkpoint_sha256": ckpt_sha,
                    "git_sha": git_sha,
                    "temperature": args.temperature,
                    "top_p": top_p_val if top_p_val is not None else "",
                    "top_k": top_k_val if top_k_val is not None else "",
                    "generated_at": generated_at,
                }
            )

    frame = pd.DataFrame(
        rows,
        columns=[
            "sequence",
            "condition_name",
            "condition_vector",
            "seed",
            "checkpoint_sha256",
            "git_sha",
            "temperature",
            "top_p",
            "top_k",
            "generated_at",
        ],
    )
    frame.to_csv(out_path, index=False)
    print(f"[generate_for_oracle] wrote {len(frame)} rows to {out_path}", flush=True)

    summary = {
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "git_sha": git_sha,
        "generated_at": generated_at,
        "seed": int(args.seed),
        "temperature": args.temperature,
        "top_p": top_p_val,
        "top_k": top_k_val,
        "n_per_condition": int(args.n_per_condition),
        "condition_columns": list(CONDITION_COLUMNS),
        "conditions": per_condition_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[generate_for_oracle] wrote summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
