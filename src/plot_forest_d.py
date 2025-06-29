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
# table path will be set via CLI args

PALETTE = {"Baseline": "tab:blue", "Expert": "tab:green"}


def parse_ci(ci_str: str) -> tuple[float, float]:
    """Turn '(x, y)' string into floats; returns (nan, nan) on failure."""
    try:
        low, high = ci_str.strip("() ").split(",")
        return float(low), float(high)
    except Exception:
        return float("nan"), float("nan")


def main(baseline: int = 3, expert: int = 6) -> None:
    table_path = ROOT / "results" / "tables" / f"exp{expert}_vs_exp{baseline}.csv"
    if not table_path.exists():
        raise FileNotFoundError(f"Stats table {table_path.name} not found – run stats_expert_vs_baseline.py first with matching studies.")

    df = pd.read_csv(table_path)

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
    out = ROOT / "results" / "figures" / f"forest_d_by_model_exp{baseline}_vs_exp{expert}.png"
    fig.savefig(out, dpi=300)
    print(out.relative_to(ROOT))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Forest plot per model for two studies.")
    parser.add_argument("--baseline", type=int, default=3)
    parser.add_argument("--expert", type=int, default=6)
    args = parser.parse_args()
    main(baseline=args.baseline, expert=args.expert) 