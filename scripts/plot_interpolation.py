from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import pandas as pd

from amp_vae.utils.paths import repo_path


PANELS = [
    ("mean_charge", "std_charge", "Net charge"),
    ("mean_hydro", "std_hydro", "Hydrophobicity (KD)"),
    ("mean_moment", "std_moment", "Hydrophobic moment"),
    ("mean_length", "std_length", "Length"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot interpolation property trends.")
    parser.add_argument("--input", default=str(repo_path("docs", "tables", "interpolation_path_sequences.csv")))
    parser.add_argument("--output", default=str(repo_path("docs", "figures", "interpolation_property_trends.png")))
    return parser.parse_args()


def main():
    args = parse_args()
    frame = pd.read_csv(args.input)
    alphas = frame["alpha"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (mean_col, std_col, label) in zip(axes.flat, PANELS):
        means = frame[mean_col].to_numpy()
        stds = frame[std_col].to_numpy()
        ax.errorbar(alphas, means, yerr=stds, marker="o", capsize=3, linewidth=1.5)
        ax.set_xlabel("alpha")
        ax.set_ylabel(label)
        ax.set_title(label + " vs alpha")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
