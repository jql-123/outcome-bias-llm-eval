"""Plot expert-probability experiment results (study 5 or 6) with standardized style."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from plot_utils import plot_by_model_generic

ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / "results" / "clean" / "data.csv"
FIG_DIR = ROOT / "results" / "figures"


def load_data() -> pd.DataFrame:
    if not CLEAN_PATH.exists():
        raise FileNotFoundError("Need clean data – run experiments first.")
    return pd.read_csv(CLEAN_PATH)


def derive_outcome(condition: str) -> str:
    return "neutral" if condition.endswith("good") else "bad"


def main(frame: str = "juror", study: int = 6) -> None:
    df = load_data()

    out_fname = f"all_dvs_exp{study}_expert_{frame}.png"
    plot_by_model_generic(
        df,
        study=study,
        frame=frame,
        out_filename=out_fname,
        good_suffix="expert_good",
        bad_suffix="expert_bad",
        title="Expert probability",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot expert-probability experiment results.")
    parser.add_argument("--frame", type=str, default="juror", help="Frame filter (juror or experiment)")
    parser.add_argument("--study", type=int, default=6, help="Study number (1, 3, 5 or 6)")
    args = parser.parse_args()
    main(frame=args.frame, study=args.study) 