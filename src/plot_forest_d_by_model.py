"""Forest plot of Cohen's d (baseline vs expert) **per model**.

This script recomputes effect sizes directly from the cleaned data so it works
for any pair of experiments, not only 3 vs 6.  For every model it plots the
six dependent variables (DVs) with separate points/CI bars for the baseline
and expert studies.

Usage
-----
python src/plot_forest_d_by_model.py                # defaults baseline=3, expert=6
python src/plot_forest_d_by_model.py --baseline 2 --expert 5
python src/plot_forest_d_by_model.py --baseline 3 --expert 6 --scenario flood
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import argparse  # noqa: E402

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "results" / "clean" / "data.csv"
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DVS = [
    "P_post",
    "GR_post",
    "Reckless",
    "Negligent",
    "Blame",
    "Punish",
]

PALETTE = {"Baseline": "tab:blue", "Expert": "tab:green"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def d_ci(d: float, n1: int, n2: int) -> tuple[float, float]:
    """Hedges & Olkin 95 % CI for Cohen's d."""
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan
    se = np.sqrt((n1 + n2) / (n1 * n2) + (d**2) / (2 * (n1 + n2 - 2)))
    delta = 1.96 * se  # 95 % CI
    return d - delta, d + delta


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if a.empty or b.empty:
        return np.nan
    pooled_var = (a.var(ddof=1) + b.var(ddof=1)) / 2
    if pooled_var == 0:
        return 0.0
    return (a.mean() - b.mean()) / np.sqrt(pooled_var)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_plot(model: str, df: pd.DataFrame, base_study: int, exp_study: int, suffix: str):
    base = df[df["study"] == base_study].copy()
    expert = df[df["study"] == exp_study].copy()

    rows = []
    for dv in DVS:
        n_bad = base[(base["outcome"] == "bad") & (base["model"] == model)][dv]
        n_neu = base[(base["outcome"] == "neutral") & (base["model"] == model)][dv]
        e_bad = expert[(expert["outcome"] == "bad") & (expert["model"] == model)][dv]
        e_neu = expert[(expert["outcome"] == "neutral") & (expert["model"] == model)][dv]

        d0 = cohens_d(n_bad, n_neu)
        d0_low, d0_high = d_ci(d0, len(n_bad), len(n_neu))
        d1 = cohens_d(e_bad, e_neu)
        d1_low, d1_high = d_ci(d1, len(e_bad), len(e_neu))

        rows.append([dv, d0, d0_low, d0_high, d1, d1_low, d1_high])

    plot_df = pd.DataFrame(rows, columns=["DV", "d_base", "d_base_low", "d_base_high", "d_exp", "d_exp_low", "d_exp_high"])

    # Figure -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 3 + 0.4 * len(DVS)))

    y_positions = np.arange(len(DVS))
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["DV"].tolist())
    ax.set_xlabel("Cohen's d")

    # Points & CIs ------------------------------------------------------------
    for i, row in plot_df.iterrows():
        y = y_positions[i]
        # baseline
        ax.errorbar(
            row["d_base"],
            y + 0.15,
            xerr=[[row["d_base"] - row["d_base_low"]], [row["d_base_high"] - row["d_base"]]],
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

    ax.set_ylim(-1, len(DVS))
    ax.legend(frameon=False, loc="upper right")
    title = f"{model}  –  exp{base_study} vs exp{exp_study}"
    ax.set_title(title)
    fig.tight_layout()

    out_path = FIG_DIR / f"forest_d_{model}_{suffix}.png"
    fig.savefig(out_path, dpi=300)
    print(out_path.relative_to(ROOT))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Forest plot of Cohen's d per model for two experiments.")
    parser.add_argument("--baseline", type=int, default=3, help="Baseline study number (e.g., 3)")
    parser.add_argument("--expert", type=int, default=6, help="Expert / comparison study number (e.g., 6)")
    parser.add_argument("--scenario", type=str, default=None, help="Optional scenario prefix filter (e.g., 'flood')")
    args = parser.parse_args()

    if not DATA_PATH.exists():
        raise FileNotFoundError("Clean data not found – run experiments first.")

    df = pd.read_csv(DATA_PATH)

    # Outcome tag -------------------------------------------------------------
    df["outcome"] = np.where(df["condition"].astype(str).str.endswith("good"), "neutral", "bad")

    if args.scenario:
        df = df[df["condition"].str.startswith(args.scenario)]
        if df.empty:
            raise ValueError("No data after scenario filter.")

    models = sorted(df["model"].unique())
    suffix = f"exp{args.baseline}_vs_exp{args.expert}"

    for m in models:
        make_plot(m, df, args.baseline, args.expert, suffix)


if __name__ == "__main__":
    main() 