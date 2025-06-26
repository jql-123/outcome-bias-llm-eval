"""Forest plot of Cohen's d (baseline vs expert) per model.

Reads the stats CSV produced by stats_expert_vs_baseline.py and renders a
horizontal forest plot for each model with 6 DVs.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = ROOT / "results" / "tables" / "exp6_vs_baseline.csv"
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT = FIG_DIR / "forest_d_by_model.png"

PALETTE = {"Baseline": "tab:blue", "Expert": "tab:green"}


def parse_ci(ci_str: str) -> tuple[float, float]:
    """Turn '(x, y)' string into floats; returns (nan, nan) on failure."""
    try:
        low, high = ci_str.strip("() ").split(",")
        return float(low), float(high)
    except Exception:
        return float("nan"), float("nan")


def main() -> None:
    if not TABLE_PATH.exists():
        raise FileNotFoundError("Run stats_expert_vs_baseline.py first.")

    df = pd.read_csv(TABLE_PATH)

    # Extract CI strings into numeric columns ----------------------------------
    df[["d_no_low", "d_no_high"]] = df["CI_d_no"].apply(lambda s: pd.Series(parse_ci(s)))
    df[["d_exp_low", "d_exp_high"]] = df["CI_d_exp"].apply(lambda s: pd.Series(parse_ci(s)))

    models = ["Baseline", "Expert"]  # conceptual groups for plotting order
    dvs = df["DV"].tolist()

    # Prepare plot --------------------------------------------------------------
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 4 + 0.4 * len(dvs)), sharex=True)
    ax = axes

    y_positions = list(range(len(dvs)))
    ax.axvline(0, color="black", lw=1)
    ax.set_ylim(-1, len(dvs))
    ax.set_yticks(y_positions)
    ax.set_yticklabels(dvs)
    ax.set_xlabel("Cohen's d")
    ax.set_xlim(-1.5, 3.5)

    # Plot baseline and expert --------------------------------------------------
    for i, row in df.iterrows():
        y = y_positions[i]
        # baseline
        ax.errorbar(
            row["d_no"],
            y + 0.15,
            xerr=[[row["d_no"] - row["d_no_low"]], [row["d_no_high"] - row["d_no"]]],
            fmt="o",
            color=PALETTE["Baseline"],
            label="Baseline" if i == 0 else "",
        )
        # expert
        ax.errorbar(
            row["d_exp"],
            y - 0.15,
            xerr=[[row["d_exp"] - row["d_exp_low"]], [row["d_exp_high"] - row["d_exp"]]],
            fmt="o",
            color=PALETTE["Expert"],
            label="Expert" if i == 0 else "",
        )

    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT, dpi=300)
    print(OUT.relative_to(ROOT))


if __name__ == "__main__":
    main() 