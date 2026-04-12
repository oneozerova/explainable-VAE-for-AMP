from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from amp_vae.data.preprocess import build_labeled_dataset
from amp_vae.utils.paths import repo_path


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
    return parser.parse_args()


def main():
    args = parse_args()
    frame = build_labeled_dataset(
        source_paths=args.inputs,
        output_path=args.output,
        drop_all_zero_labels=not args.keep_all_zero_labels,
        deduplicate=not args.no_deduplicate,
    )
    print(f"Saved {len(frame)} rows to {args.output}")


if __name__ == "__main__":
    main()
