"""Scatter comparison of Cohen's d (baseline vs expert) for *each* LLM model.

For every model present in the cleaned data this script computes Cohen's *d*
for the difference between *bad* and *neutral* outcomes in:
    • Study-3  (baseline – no expert cue)
    • Study-6  (expert-probability cue)
The results are visualised as side-by-side scatter plots (one panel per
model).  Each point is a dependent variable (DV) labelled accordingly and the
45° line helps to see whether the expert cue weakened (below diagonal) or
strengthened (above) the anchoring effect.

Usage
-----
python src/plot_scatter_d_by_model.py            # all scenarios
python src/plot_scatter_d_by_model.py --scenario flood  # single scenario only
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import ttest_ind  # noqa: E402
import argparse  # noqa: E402

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "results" / "clean" / "data.csv"
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = FIG_DIR / "scatter_d_by_model.png"

DVS = [
    "P_post",
    "GR_post",
    "Reckless",
    "Negligent",
    "Blame",
    "Punish",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cohens_d(a: pd.Series, b: pd.Series) -> float:
    """Return Cohen's d for two independent samples (Welch)."""
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

def main(scenario: str | None = None) -> None:  # noqa: C901 (keep flat for clarity)
    if not DATA_PATH.exists():
        raise FileNotFoundError("Clean data not found – run experiments first.")

    df = pd.read_csv(DATA_PATH)

    # Optional scenario filter ------------------------------------------------
    if scenario:
        df = df[df["condition"].str.startswith(scenario)]
        if df.empty:
            raise ValueError(f"No rows for scenario prefix '{scenario}'.")

    # Derive outcome column ---------------------------------------------------
    df["outcome"] = np.where(df["condition"].astype(str).str.endswith("good"), "neutral", "bad")

    models = sorted(df["model"].unique())
    if not models:
        raise ValueError("No model data found in cleaned CSV.")

    # Prepare figure ----------------------------------------------------------
    n_mod = len(models)
    fig, axes = plt.subplots(1, n_mod, figsize=(4 * n_mod, 4), dpi=150, sharex=True, sharey=True)
    if n_mod == 1:
        axes = [axes]  # force iterable

    for ax, model in zip(axes, models, strict=True):
        base = df[(df["study"] == 3) & (df["model"] == model)]
        expert = df[(df["study"] == 6) & (df["model"] == model)]

        xs, ys = [], []
        for dv in DVS:
            n_bad = base[base["outcome"] == "bad"][dv]
            n_neu = base[base["outcome"] == "neutral"][dv]
            e_bad = expert[expert["outcome"] == "bad"][dv]
            e_neu = expert[expert["outcome"] == "neutral"][dv]
            d_base = cohens_d(n_bad, n_neu)
            d_exp = cohens_d(e_bad, e_neu)
            xs.append(d_base)
            ys.append(d_exp)

        # Scatter ---------------------------------------------------------
        ax.axline((0, 0), slope=1, color="grey", lw=1)
        ax.scatter(xs, ys, color="tab:blue")

        # Annotate DV labels (offset to reduce overlap) -------------------
        for x, y, label in zip(xs, ys, DVS, strict=True):
            ax.text(x + 0.05, y + 0.05, label, fontsize=8)

        ax.set_title(model)
        ax.set_xlabel("d (Baseline)")
        if ax is axes[0]:
            ax.set_ylabel("d (Expert)")

    # Square limits across all axes ------------------------------------------
    all_vals = xs + ys  # last iteration values; fine because axes share data range
    lims = [min(all_vals + [0]) - 1.0, max(all_vals + [0]) + 1.0]
    for ax in axes:
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal", "box")

    fig.tight_layout()
    fig.savefig(OUT_PATH)
    print(OUT_PATH.relative_to(ROOT))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scatter plot of Cohen's d baseline vs expert for each model.")
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Filter to a single scenario prefix (e.g. 'flood'); omit for all scenarios.",
    )
    args = parser.parse_args()
    main(scenario=args.scenario) 