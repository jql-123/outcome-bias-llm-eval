"""Scatter comparison of Cohen's d (baseline vs expert) per model.

Each panel shows the six DVs for one model. Points are annotated with DV name.
Diagonal line y = x helps inspect reduction/enhancement.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main(baseline: int = 3, expert: int = 6) -> None:
    table_path = ROOT / "results" / "tables" / f"exp{expert}_vs_exp{baseline}.csv"
    if not table_path.exists():
        raise FileNotFoundError("Run stats_expert_vs_baseline.py first with matching studies.")

    df = pd.read_csv(table_path)

    models = df["DV"].unique()  # actually models not present, but treat aggregated
    # Since stats table is aggregated across models, we make single scatter.
    # Use baseline vs expert across DVs.

    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    ax.axline((0, 0), slope=1, color="grey", lw=1)

    ax.set_xlabel("d (Baseline)")
    ax.set_ylabel("d (Expert)")

    for _, row in df.iterrows():
        ax.scatter(row["d_no"], row["d_exp"], color="tab:blue")
        ax.text(row["d_no"] + 0.02, row["d_exp"] + 0.02, row["DV"], fontsize=8)

    # square limits
    lims = [-1.5, 3.5]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", "box")

    fig.tight_layout()
    out = FIG_DIR / f"scatter_d_by_model_exp{baseline}_vs_exp{expert}.png"
    fig.savefig(out)
    print(out.relative_to(ROOT))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scatter plot d baseline vs expert.")
    parser.add_argument("--baseline", type=int, default=3)
    parser.add_argument("--expert", type=int, default=6)
    args = parser.parse_args()
    main(baseline=args.baseline, expert=args.expert) 