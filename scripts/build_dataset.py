from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from amp_vae.config import CONDITION_COLUMNS, DEFAULT_MAX_LEN
from amp_vae.data.preprocess import build_labeled_dataset
from amp_vae.utils.paths import repo_path
from amp_vae.utils.splits import load_or_create_splits


def parse_args():
    parser = argparse.ArgumentParser(description="Build a labeled AMP dataset from raw CSV files.")
    parser.add_argument(
        "--input",
        action="append",
        dest="inputs",
        default=[str(repo_path("data", "raw", "apd6_peptides_raw_data.csv"))],
        help="Input CSV path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        default=str(repo_path("data", "processed", "master_dataset.csv")),
        help="Output CSV path.",
    )
    parser.add_argument("--keep-all-zero-labels", action="store_true", help="Keep rows with no activity labels.")
    parser.add_argument("--no-deduplicate", action="store_true", help="Disable deduplication by sequence.")
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN, help="Max sequence length (chars).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for split RNG.")
    return parser.parse_args()


def main():
    args = parse_args()
    frame = build_labeled_dataset(
        source_paths=args.inputs,
        output_path=args.output,
        drop_all_zero_labels=not args.keep_all_zero_labels,
        deduplicate=not args.no_deduplicate,
        max_len=args.max_len,
    )
    print(f"Saved {len(frame)} rows to {args.output}")

    train, val, test = load_or_create_splits(
        frame,
        seed=args.seed,
        stratify_cols=CONDITION_COLUMNS,
        force=True,
    )
    print(
        f"Split sizes: train={len(train)} val={len(val)} test={len(test)} "
        f"(total={len(train) + len(val) + len(test)} / {len(frame)})"
    )


if __name__ == "__main__":
    main()
