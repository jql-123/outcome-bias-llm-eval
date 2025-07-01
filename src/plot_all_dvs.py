#!/usr/bin/env python
"""Create two multi‐panel bar charts of outcome bias (all DVs) by model.

1. Baseline (Study 1) → results/figures/all_dvs_exp1_juror.png
   Title: "Outcome bias at baseline (no expert cue)"
2. Expert cue (Study 5) → results/figures/all_dvs_exp5_expert_juror.png
   Title: "Outcome bias after expert probability cue"

The script relies on the generic plotting helper in plot_utils.py.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from plot_utils import plot_by_model_generic

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CLEAN_CSV = ROOT / "results" / "clean" / "data.csv"

DISPLAY_NAMES = {
    "gpt4o": "gpt-4o",
    "o1mini": "gpt-o1-mini",
    "o4mini": "gpt-o4-mini",
    "sonnet4": "sonnet-4",
    "deepseekr1": "deepseek-r1",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pretty_df() -> pd.DataFrame:
    if not CLEAN_CSV.exists():
        raise SystemExit("Clean data CSV not found – run experiments first.")
    df = pd.read_csv(CLEAN_CSV)
    # Map to pretty display names but keep in column 'model'
    df["model"] = df["model"].map(DISPLAY_NAMES).fillna(df["model"])
    return df

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_pretty_df()

    # ---------------- Baseline (Study 1) ----------------
    plot_by_model_generic(
        df,
        study=1,
        frame="juror",
        out_filename="all_dvs_exp1_juror.png",
        title="Outcome bias at baseline (no expert cue)",
        show_frame=False,
    )

    # ---------------- Expert cue (Study 5) --------------
    plot_by_model_generic(
        df,
        study=5,
        frame="juror",
        out_filename="all_dvs_exp5_expert_juror.png",
        title="Outcome bias after expert probability cue",
        show_frame=False,
    )


if __name__ == "__main__":
    main() 