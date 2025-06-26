"""Plot all post-outcome dependent variables with Cohen's d annotations.

For each model, the script shows a barplot of six DVs (objective/subjective
probability rescaled to 1-7, recklessness, negligence, blame, punishment) for
neutral vs bad outcomes.  Above each neutral/bad pair the Cohen's d effect size
is annotated.

Usage
-----
python src/plot_dvs_with_effects.py --frame juror
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / "results" / "clean" / "data.csv"
FIG_DIR = ROOT / "results" / "figures"

# Original column names in the CSV
DV_RAW = [
    "P_post",
    "GR_post",
    "Reckless",
    "Negligent",
    "Blame",
    "Punish",
]

# Pretty labels for plotting / order on the x-axis
DV_LABELS = {
    "P_post": "Obj. probability",
    "GR_post": "Subj. probability",
    "Reckless": "Recklessness",
    "Negligent": "Negligence",
    "Blame": "Blame",
    "Punish": "Punishment",
}
ORDER = list(DV_LABELS.values())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    if not CLEAN_PATH.exists():
        raise FileNotFoundError("Clean data not found. Run experiments first.")
    return pd.read_csv(CLEAN_PATH)


def derive_outcome(condition: str) -> str:
    neutral = {"flood_good", "traffic_good"}
    return "neutral" if condition in neutral else "bad"


def cohens_d(x: pd.Series, y: pd.Series) -> float:
    """Compute Cohen's d for two independent samples (unpaired)."""
    x = x.dropna()
    y = y.dropna()
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    pooled = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    if pooled == 0:
        return np.nan
    return (x.mean() - y.mean()) / pooled

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(frame: str = "juror") -> None:
    df = load_data()

    # Filter by framing
    df = df[df["frame"] == frame]
    n_rows = len(df)

    if n_rows == 0:
        raise ValueError(f"No rows found for frame='{frame}'.")

    # Rescale probabilities 0-100 -> 1-7
    for col in ("P_post", "GR_post"):
        if col in df.columns:
            df[col] = (df[col] / 100.0) * 6 + 1

    # Outcome column
    df = df.copy()
    df["outcome"] = df["condition"].apply(derive_outcome)  # type: ignore[call-arg]

    # Long form for plotting
    df_long = df.melt(
        id_vars=["model", "outcome"],
        value_vars=DV_RAW,
        var_name="DV",
        value_name="score",
    )
    df_long["DV"] = df_long["DV"].map(DV_LABELS)  # type: ignore[arg-type]

    # Compute Cohen's d per model & DV
    d_table: dict[str, dict[str, float]] = {}
    for model, g in df_long.groupby("model"):
        d_table[model] = {}
        for dv, gdv in g.groupby("DV"):
            neutral = gdv[gdv["outcome"] == "neutral"]["score"]
            bad = gdv[gdv["outcome"] == "bad"]["score"]
            d_table[model][dv] = cohens_d(neutral, bad)

    # Plot
    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=df_long,
        kind="bar",
        x="DV",
        y="score",
        hue="outcome",
        col="model",
        order=ORDER,
        errorbar="se",
        height=4,
        aspect=1.2,
        palette="pastel",
    )

    g.set_axis_labels("", "Mean (1-7 scale)")
    g.set_titles("{col_name}")
    g.set_xticklabels(rotation=45)

    # Annotate Cohen's d
    for ax in g.axes.flatten():
        if ax is None:
            continue
        model = ax.get_title()
        # Bars are grouped by outcome then DV; easier to compute where to annotate
        for idx, dv in enumerate(ORDER):
            d_val = d_table.get(model, {}).get(dv, np.nan)
            # find max bar height for this dv (two bars)
            bars = [p for p in ax.patches if int(p.get_x() + p.get_width() / 2) == idx]
            if not bars:
                continue
            y_max = max(bar.get_height() for bar in bars)
            ax.text(
                idx,
                y_max + 0.15,
                f"d={d_val:.2f}" if not np.isnan(d_val) else "",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Overall title
    g.fig.suptitle(f"{frame.capitalize()} framing (n={n_rows})", y=1.05, fontsize=14)
    plt.tight_layout()

    # Save
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / f"all_dvs_with_d_{frame}.png"
    g.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot DVs with Cohen's d annotations.")
    parser.add_argument(
        "--frame",
        type=str,
        default="juror",
        help="Frame to filter on (juror or experiment).",
    )
    args = parser.parse_args()
    main(frame=args.frame) 