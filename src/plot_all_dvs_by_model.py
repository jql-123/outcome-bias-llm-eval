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
import pandas as pd
from plot_utils import plot_by_model_generic

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
    out_fname = f"all_dvs_exp3_{frame}.png"
    plot_by_model_generic(df, study=3, frame=frame, out_filename=out_fname)


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