"""Submit generated peptides to the AIPAMPDS oracle and collect scores.

The HTTP contract for ``aipampds.pianlab.team`` is intentionally pluggable:
the actual endpoint / auth / payload shape is unknown to this repo, so the
:class:`AIPAMPDSClient` ships as a stub that must be filled in by whoever runs
the submission. The rest of the script (concurrency, retry, hit-rate
aggregation) is agnostic to the transport.

Usage::

    python scripts/submit_aipampds.py \
        --in data/external/aipampds/submission_YYYYMMDD.csv \
        --out data/external/aipampds/scored_YYYYMMDD.csv \
        --concurrency 4 \
        --retry 3 --backoff 2.0

Outputs:
    * ``--out`` CSV: every input column plus ``ecoli_score``, ``saureus_score``,
      ``haemolytic_score``, and any extra fields returned by the oracle.
    * ``aipampds_condition_metrics_<suffix>.csv``: per-condition hit metrics
      computed with the uniform formula documented in REPRODUCIBILITY.md.

Uniform hit-rate formula (fixes pipeline_issues.md P0 #19):

    Gram+         hit = saureus_score > 0.5
    Gram-         hit = ecoli_score   > 0.5
    Other         hit = max(ecoli, saureus) > 0.5
    Non-haem      non_haem = haemolytic_score < 0.5
    Safe hit      safe_hit = hit AND non_haem

Unconditional rows get ``NA`` for ``all_requested_hit_rate`` because there is
no target label to hit against.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd


HIT_THRESHOLD = 0.5
HAEM_THRESHOLD = 0.5

# Mapping from submission condition name to the scoring rule used for the
# per-condition hit-rate. "bactericidal" means max(ecoli, saureus) > 0.5.
CONDITION_HIT_RULES: dict[str, str] = {
    "gram_pos": "saureus",
    "gram_neg": "ecoli",
    "antifungal": "bactericidal",
    "antiviral": "bactericidal",
    "antiparasitic": "bactericidal",
    "anticancer": "bactericidal",
    "unconditional": "na",
}


class AIPAMPDSClient:
    """HTTP client for ``aipampds.pianlab.team``.

    Contract
    --------
    ``submit_peptide(sequence: str) -> dict`` must return a dict containing at
    least::

        {
            "ecoli_score": float,
            "saureus_score": float,
            "haemolytic_score": float,
            ...any extra fields the oracle returns...
        }

    TODO(user): fill in the real endpoint. See
    https://aipampds.pianlab.team/ for the current API. Until then this stub
    raises ``NotImplementedError`` so that no stale, fabricated scores make
    their way into the output CSV.
    """

    def __init__(self, base_url: str = "https://aipampds.pianlab.team", timeout: float = 60.0):
        self.base_url = base_url
        self.timeout = timeout

    def submit_peptide(self, sequence: str) -> dict[str, Any]:
        raise NotImplementedError(
            "AIPAMPDSClient.submit_peptide is a stub. Fill in the real HTTP "
            "call against aipampds.pianlab.team (see TODO in AIPAMPDSClient). "
            "Contract: return a dict with ecoli_score, saureus_score, "
            "haemolytic_score keys (floats)."
        )


def _submit_with_retry(
    client: AIPAMPDSClient,
    sequence: str,
    retries: int,
    backoff: float,
) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            return client.submit_peptide(sequence)
        except NotImplementedError:
            # Non-retryable: bubble up unchanged so users plug in the client.
            raise
        except Exception as exc:  # pragma: no cover - transport-level errors
            last_exc = exc
            if attempt >= retries:
                break
            sleep_for = backoff ** attempt
            print(
                f"[submit_aipampds] attempt {attempt} for '{sequence[:20]}...' "
                f"failed ({type(exc).__name__}: {exc}); sleeping {sleep_for:.1f}s",
                flush=True,
            )
            time.sleep(sleep_for)
    assert last_exc is not None
    raise last_exc


def score_dataframe(
    frame: pd.DataFrame,
    client: AIPAMPDSClient,
    concurrency: int,
    retries: int,
    backoff: float,
) -> pd.DataFrame:
    sequences: list[str] = frame["sequence"].astype(str).tolist()
    results: list[dict[str, Any] | None] = [None] * len(sequences)

    def _task(index: int) -> tuple[int, dict[str, Any]]:
        return index, _submit_with_retry(client, sequences[index], retries, backoff)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(_task, i): i for i in range(len(sequences))}
        for future in as_completed(futures):
            idx, payload = future.result()
            results[idx] = payload

    # Build score columns, preserving any extra keys the oracle returned.
    extra_keys: list[str] = []
    seen_keys: set[str] = set()
    for payload in results:
        if payload is None:
            continue
        for key in payload:
            if key in ("ecoli_score", "saureus_score", "haemolytic_score"):
                continue
            if key not in seen_keys:
                seen_keys.add(key)
                extra_keys.append(key)

    scored = frame.copy()
    scored["ecoli_score"] = [r.get("ecoli_score") if r else None for r in results]
    scored["saureus_score"] = [r.get("saureus_score") if r else None for r in results]
    scored["haemolytic_score"] = [r.get("haemolytic_score") if r else None for r in results]
    for key in extra_keys:
        scored[key] = [r.get(key) if r else None for r in results]
    return scored


def _row_hit(rule: str, ecoli: float | None, saureus: float | None) -> bool | None:
    if rule == "saureus":
        return None if saureus is None else bool(saureus > HIT_THRESHOLD)
    if rule == "ecoli":
        return None if ecoli is None else bool(ecoli > HIT_THRESHOLD)
    if rule == "bactericidal":
        if ecoli is None or saureus is None:
            return None
        return bool(max(ecoli, saureus) > HIT_THRESHOLD)
    return None  # unconditional / unknown


def compute_condition_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition, group in scored.groupby("condition_name", sort=False):
        rule = CONDITION_HIT_RULES.get(str(condition), "bactericidal")
        ecoli = group["ecoli_score"]
        saureus = group["saureus_score"]
        haem = group["haemolytic_score"]

        hit_flags: list[bool] = []
        for e, s in zip(ecoli, saureus):
            h = _row_hit(rule, None if pd.isna(e) else float(e), None if pd.isna(s) else float(s))
            if h is not None:
                hit_flags.append(h)

        non_haem_flags = [bool(float(h) < HAEM_THRESHOLD) for h in haem if not pd.isna(h)]

        if rule == "na":
            hit_rate: float | str = "NA"
            safe_hit_rate: float | str = "NA"
        else:
            if hit_flags:
                hit_rate = sum(hit_flags) / len(hit_flags)
            else:
                hit_rate = "NA"
            # safe_hit aligns haemolytic flags to rows that had a hit decision.
            paired = [
                (_row_hit(rule, None if pd.isna(e) else float(e), None if pd.isna(s) else float(s)),
                 None if pd.isna(h) else bool(float(h) < HAEM_THRESHOLD))
                for e, s, h in zip(ecoli, saureus, haem)
            ]
            safe = [hit and nh for hit, nh in paired if hit is not None and nh is not None]
            safe_hit_rate = (sum(safe) / len(safe)) if safe else "NA"

        non_haem_rate = (sum(non_haem_flags) / len(non_haem_flags)) if non_haem_flags else "NA"

        rows.append(
            {
                "condition_name": condition,
                "hit_rule": rule,
                "n_generated": int(len(group)),
                "n_scored": int(ecoli.notna().sum()),
                "hit_rate": hit_rate,
                "non_haemolytic_rate": non_haem_rate,
                "safe_hit_rate": safe_hit_rate,
                "mean_ecoli_score": float(ecoli.mean()) if ecoli.notna().any() else "NA",
                "mean_saureus_score": float(saureus.mean()) if saureus.notna().any() else "NA",
                "mean_haemolytic_score": float(haem.mean()) if haem.notna().any() else "NA",
            }
        )
    return pd.DataFrame(rows)


HEADER_COMMENT_LINES = [
    "# AIPAMPDS condition metrics",
    "# Uniform hit-rate formula (see REPRODUCIBILITY.md):",
    "#   gram_pos      -> hit = saureus_score > 0.5",
    "#   gram_neg      -> hit = ecoli_score   > 0.5",
    "#   other         -> hit = max(ecoli, saureus) > 0.5",
    "#   non_haem      -> haemolytic_score < 0.5",
    "#   safe_hit      -> hit AND non_haem",
    "# unconditional rows write NA (no target label to hit against).",
]


def write_condition_metrics(metrics: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        for line in HEADER_COMMENT_LINES:
            fh.write(line + "\n")
        writer = csv.writer(fh)
        writer.writerow(list(metrics.columns))
        for _, row in metrics.iterrows():
            writer.writerow(row.tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="input_csv", required=True)
    parser.add_argument("--out", dest="output_csv", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--backoff", type=float, default=2.0)
    parser.add_argument(
        "--base-url",
        default="https://aipampds.pianlab.team",
        help="Override AIPAMPDS base URL (for staging / offline tests).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_csv)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(in_path)
    required = {"sequence", "condition_name"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

    client = AIPAMPDSClient(base_url=args.base_url)
    print(
        f"[submit_aipampds] submitting {len(frame)} sequences with "
        f"concurrency={args.concurrency} retry={args.retry} backoff={args.backoff}",
        flush=True,
    )
    scored = score_dataframe(frame, client, args.concurrency, args.retry, args.backoff)
    scored.to_csv(out_path, index=False)
    print(f"[submit_aipampds] wrote scored CSV to {out_path}", flush=True)

    metrics = compute_condition_metrics(scored)
    # Suffix matches the submission file so paired CSVs can be linked.
    stem = in_path.stem
    suffix = stem.split("_", 1)[1] if "_" in stem else stem
    metrics_path = out_path.parent / f"aipampds_condition_metrics_{suffix}.csv"
    write_condition_metrics(metrics, metrics_path)
    print(f"[submit_aipampds] wrote condition metrics to {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
