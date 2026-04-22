"""Generate AIPAMPDS oracle figures and condition metrics CSV.

Reads:
  data/external/aipampds/aipampds_results.csv    — raw AIPAMPDS output
  data/external/aipampds/submission_*.csv        — generated sequences with condition labels
                                                   (uses most recent submission_*.csv)

Writes:
  docs/figures/aipampds_hit_rates.png
  docs/figures/aipampds_score_scatter.png
  data/external/aipampds/aipampds_condition_metrics.csv

Hit formula (from REPRODUCIBILITY.md):
  Gram+         hit = S.aureus_Score > 0.5
  Gram-         hit = E.coli_Score > 0.5
  Other         hit = max(E.coli, S.aureus) > 0.5
  Non-haem      non_haem = Hemolytic_Score < 0.5
  Safe hit      safe_hit = hit AND non_haem
  Unconditional gets NA for hit

Usage::

    python3 scripts/plot_aipampds.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from amp_vae.utils.paths import repo_path


LABEL_DISPLAY = {
    "gram_pos":      "Gram+",
    "gram_neg":      "Gram−",
    "antifungal":    "Antifungal",
    "antiviral":     "Antiviral",
    "antiparasitic": "Antiparasitic",
    "anticancer":    "Anticancer",
    "unconditional": "Unconditional",
}
CONDITION_ORDER = ["gram_pos", "gram_neg", "antifungal", "antiviral", "antiparasitic", "anticancer", "unconditional"]
COLORS = {
    "gram_pos":      "#4C72B0",
    "gram_neg":      "#55A868",
    "antifungal":    "#C44E52",
    "antiviral":     "#8172B2",
    "antiparasitic": "#CCB974",
    "anticancer":    "#64B5CD",
    "unconditional": "#999999",
}


def requested_hit(row: pd.Series) -> float:
    cond = row["condition_name"]
    if cond == "gram_pos":
        return float(row["S.aureus_Score"] > 0.5)
    if cond == "gram_neg":
        return float(row["E.coli_Score"] > 0.5)
    if cond == "unconditional":
        return float("nan")
    return float(max(row["E.coli_Score"], row["S.aureus_Score"]) > 0.5)


def load_data() -> pd.DataFrame:
    results_path = repo_path("data", "external", "aipampds", "aipampds_results.csv")
    results = pd.read_csv(str(results_path))

    # find most recent submission CSV
    sub_dir = Path(repo_path("data", "external", "aipampds"))
    subs = sorted(sub_dir.glob("submission_*.csv"), reverse=True)
    if not subs:
        raise FileNotFoundError("No submission_*.csv found in data/external/aipampds/")
    submission = pd.read_csv(str(subs[0]))
    print(f"Using submission: {subs[0].name}  ({len(submission)} rows)")

    merged = results.merge(
        submission[["sequence", "condition_name"]],
        left_on="Sequence", right_on="sequence", how="inner",
    )
    merged = merged.drop_duplicates(subset=["Sequence", "condition_name"])
    print(f"Merged: {len(merged)} rows")
    merged["requested_hit"] = merged.apply(requested_hit, axis=1)
    merged["amp_hit"]    = (merged["AMP_Score"] > 0.5).astype(float)
    merged["non_hemo"]   = (merged["Hemolytic_Score"] < 0.5).astype(float)
    merged["safe_hit"]   = ((merged["AMP_Score"] > 0.5) & (merged["Hemolytic_Score"] < 0.5)).astype(float)
    return merged


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond in CONDITION_ORDER:
        grp = df[df["condition_name"] == cond]
        if len(grp) == 0:
            continue
        rows.append({
            "condition":             LABEL_DISPLAY.get(cond, cond),
            "condition_key":         cond,
            "n_scored":              len(grp),
            "amp_hit_rate":          round(grp["amp_hit"].mean(), 4),
            "requested_hit_rate":    round(grp["requested_hit"].mean(skipna=True), 4) if cond != "unconditional" else float("nan"),
            "non_hemolytic_rate":    round(grp["non_hemo"].mean(), 4),
            "safe_hit_rate":         round(grp["safe_hit"].mean(), 4),
            "mean_amp_score":        round(grp["AMP_Score"].mean(), 4),
            "mean_e_coli_score":     round(grp["E.coli_Score"].mean(), 4),
            "mean_s_aureus_score":   round(grp["S.aureus_Score"].mean(), 4),
            "mean_hemolytic_score":  round(grp["Hemolytic_Score"].mean(), 4),
        })
    return pd.DataFrame(rows)


def plot_hit_rates(metrics: pd.DataFrame, df: pd.DataFrame) -> None:
    conds = [c for c in CONDITION_ORDER if c in metrics["condition_key"].values]
    display = [LABEL_DISPLAY[c] for c in conds]
    colors  = [COLORS[c] for c in conds]
    n       = len(conds)
    x       = np.arange(n)
    width   = 0.28

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, (col, title) in zip(axes, [
        ("amp_hit_rate",       "AMP hit rate\n(AMP_Score > 0.5)"),
        ("non_hemolytic_rate", "Non-haemolytic rate\n(Hemolytic_Score < 0.5)"),
        ("safe_hit_rate",      "Safe-hit rate\n(AMP AND non-haemolytic)"),
    ]):
        vals = [metrics.loc[metrics["condition_key"]==c, col].values[0] for c in conds]
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(display, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Rate", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.axhline(0.5, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7)

    fig.suptitle("AIPAMPDS oracle scores — generated candidates (current 6-label model)", fontsize=11)
    plt.tight_layout()
    out = repo_path("docs", "figures", "aipampds_hit_rates.png")
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_score_scatter(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # left: E.coli vs S.aureus scatter coloured by condition
    ax = axes[0]
    for cond in CONDITION_ORDER:
        grp = df[df["condition_name"] == cond]
        if len(grp) == 0:
            continue
        ax.scatter(grp["E.coli_Score"], grp["S.aureus_Score"],
                   c=COLORS[cond], s=8, alpha=0.5, linewidths=0, rasterized=True,
                   label=LABEL_DISPLAY[cond])
    ax.axvline(0.5, color="grey", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.axhline(0.5, color="grey", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_xlabel("E. coli score", fontsize=9)
    ax.set_ylabel("S. aureus score", fontsize=9)
    ax.set_title("E. coli vs S. aureus activity", fontsize=10)
    ax.legend(fontsize=7, loc="lower right", markerscale=2)

    # right: hemolytic score distributions per condition
    ax2 = axes[1]
    conds_plotted = [c for c in CONDITION_ORDER if c in df["condition_name"].values]
    positions = list(range(len(conds_plotted)))
    parts = ax2.violinplot(
        [df[df["condition_name"]==c]["Hemolytic_Score"].values for c in conds_plotted],
        positions=positions, showmedians=True, widths=0.7,
    )
    for pc, cond in zip(parts["bodies"], conds_plotted):
        pc.set_facecolor(COLORS[cond])
        pc.set_alpha(0.7)
    ax2.axhline(0.5, color="red", linewidth=0.8, linestyle="--", alpha=0.7, label="threshold")
    ax2.set_xticks(positions)
    ax2.set_xticklabels([LABEL_DISPLAY[c] for c in conds_plotted], rotation=35, ha="right", fontsize=8)
    ax2.set_ylabel("Hemolytic score", fontsize=9)
    ax2.set_title("Haemolytic score distribution", fontsize=10)
    ax2.legend(fontsize=8)

    fig.suptitle("AIPAMPDS oracle — per-sequence scores (current 6-label model)", fontsize=11)
    plt.tight_layout()
    out = repo_path("docs", "figures", "aipampds_score_scatter.png")
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    df = load_data()
    metrics = compute_metrics(df)
    print()
    print(metrics[["condition","n_scored","amp_hit_rate","requested_hit_rate","non_hemolytic_rate","safe_hit_rate"]].to_string(index=False))

    out_csv = repo_path("data", "external", "aipampds", "aipampds_condition_metrics.csv")
    metrics.to_csv(str(out_csv), index=False)
    print(f"\nsaved {out_csv}")

    plot_hit_rates(metrics, df)
    plot_score_scatter(df)


if __name__ == "__main__":
    main()
