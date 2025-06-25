"""Plot all dependent variables (post-outcome) by model.

This script loads the cleaned experiment results, filters by framing, reshapes
the data into long form, and produces a faceted barplot that contrasts the
*neutral* versus *bad* outcome conditions for each LLM model.

Usage
-----
python src/plot_all_dvs_by_model.py [--frame juror]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / "results" / "clean" / "data.csv"
FIG_DIR = ROOT / "results" / "figures"

DV_COLS = [
    "P_post",
    "GR_post",
    "Reckless",
    "Negligent",
    "Blame",
    "Punish",
]
ORDER = [
    "Objective probability",
    "Subjective probability",
    "Recklessness",
    "Negligence",
    "Blame",
    "Punishment",
]

# Mapping for pretty DV labels
DV_LABELS = {
    "P_post": "Objective probability",
    "GR_post": "Subjective probability",
    "Reckless": "Recklessness",
    "Negligent": "Negligence",
    "Blame": "Blame",
    "Punish": "Punishment",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    if not CLEAN_PATH.exists():
        raise FileNotFoundError("Clean data not found. Run experiments first.")
    return pd.read_csv(CLEAN_PATH)


def derive_outcome(condition: str) -> str:
    """Return 'neutral' or 'bad' based on condition key."""
    neutral_set = {"flood_good", "traffic_good"}
    return "neutral" if condition in neutral_set else "bad"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(frame: str = "juror") -> None:
    df = load_data()

    # Filter by framing
    df = df[df["frame"] == frame]

    n_rows = len(df)

    # Outcome column & scaling
    df = df.copy()
    # Convert probability scales 0-100 -> 1-7 for plotting
    for col in ("P_post", "GR_post"):
        if col in df.columns:
            df[col] = (df[col] / 100.0) * 6 + 1

    df["outcome"] = df["condition"].apply(derive_outcome)  # type: ignore[call-arg]

    # Long form
    df_long = df.melt(
        id_vars=["model", "outcome"],
        value_vars=DV_COLS,
        var_name="DV",
        value_name="score",
    )
    df_long["DV"] = df_long["DV"].map(DV_LABELS)  # type: ignore[arg-type]

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

    g.set_axis_labels("", "Mean")
    g.set_titles("{col_name}")

    # Add overall title with frame and sample size
    g.fig.suptitle(f"{frame.capitalize()} framing (n={n_rows})", y=1.05, fontsize=14)
    g.set_xticklabels(rotation=45)
    plt.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / f"all_dvs_by_model_{frame}.png"
    g.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot all DVs by model (faceted barplot).")
    parser.add_argument(
        "--frame",
        type=str,
        default="juror",
        help="Frame to filter on (e.g., 'juror' or 'experiment').",
    )
    args = parser.parse_args()
    main(frame=args.frame) 