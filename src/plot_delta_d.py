"""Visualise effect size changes between baseline and expert probability conditions.

Generates a bar chart of the absolute Cohen's *d* values for each dependent
variable in Study-3 (baseline) and Study-6 (expert) scenarios, and
annotates the change Δd.

The figure is saved to results/figures/delta_d_exp6.png and the relative path
is printed to stdout for easy reference.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main(baseline: int = 3, expert: int = 6) -> None:
    table_path = ROOT / "results" / "tables" / f"exp{expert}_vs_exp{baseline}.csv"
    if not table_path.exists():
        raise FileNotFoundError("Run stats_expert_vs_baseline.py first to generate the table with matching studies.")

    df = pd.read_csv(table_path)

    dvs = df["DV"].tolist()
    d_no = df["d_no"].abs().values
    d_exp = df["d_exp"].abs().values

    x = np.arange(len(dvs))
    width = 0.35

    plt.style.use("seaborn-v0_8-pastel")  # modern seaborn palette
    fig, ax = plt.subplots(figsize=(10, 5))

    bars1 = ax.bar(x - width / 2, d_no, width, label="Baseline (Study-3)")
    bars2 = ax.bar(x + width / 2, d_exp, width, label="Expert (Study-6)")

    # Annotation with Δd
    for i, (b1, b2) in enumerate(zip(bars1, bars2)):
        delta = d_no[i] - d_exp[i]
        height = max(b1.get_height(), b2.get_height())
        ax.text(
            b1.get_x() + width,  # centre between bars
            height + 0.04,
            f"Δd={delta:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(dvs, rotation=45, ha="right")
    ax.set_ylabel("|Cohen's d|")
    ax.set_title("Effect size reduction with expert probability")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()

    out_path = FIG_DIR / f"delta_d_exp{expert}_vs_exp{baseline}.png"
    fig.savefig(out_path, dpi=300)
    print(out_path.relative_to(ROOT))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Delta d plot between two studies.")
    parser.add_argument("--baseline", type=int, default=3)
    parser.add_argument("--expert", type=int, default=6)
    args = parser.parse_args()
    main(baseline=args.baseline, expert=args.expert) 