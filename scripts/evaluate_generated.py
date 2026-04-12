from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from amp_vae.data.schema import SEQUENCE_COLUMN
from amp_vae.evaluation.generation_metrics import novelty_rate, uniqueness_rate, validity_rate


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated peptide sequences.")
    parser.add_argument("--generated", required=True, help="CSV with generated sequences.")
    parser.add_argument("--reference", default=None, help="Optional reference CSV with sequences.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def load_sequences(path: str | Path) -> list[str]:
    frame = pd.read_csv(path)
    seq_col = SEQUENCE_COLUMN if SEQUENCE_COLUMN in frame.columns else frame.columns[0]
    return frame[seq_col].astype(str).str.strip().str.upper().tolist()


def main():
    args = parse_args()
    print(f"[evaluate_generated] loading generated sequences: {args.generated}", flush=True)
    generated = load_sequences(args.generated)
    print(f"[evaluate_generated] generated_count={len(generated)}", flush=True)
    reference = load_sequences(args.reference) if args.reference else []
    if args.reference:
        print(f"[evaluate_generated] loading reference sequences: {args.reference}", flush=True)
        print(f"[evaluate_generated] reference_count={len(reference)}", flush=True)
    payload = {
        "n_sequences": len(generated),
        "validity_rate": validity_rate(generated),
        "uniqueness_rate": uniqueness_rate(generated),
        "novelty_rate": novelty_rate(generated, reference) if reference else None,
    }
    print("[evaluate_generated] metrics:", flush=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[evaluate_generated] saved metrics to {args.output}", flush=True)


if __name__ == "__main__":
    main()
