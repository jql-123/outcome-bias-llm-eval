"""Compute Welch-t and Cohen's d for Experiment 6 (expert-probability anchor)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import ttest_ind

ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / "results" / "clean" / "data.csv"
TABLE_DIR = ROOT / "results" / "tables"

DV_COLS = [
    "P_post",
    "GR_post",
    "Reckless",
    "Negligent",
    "Blame",
    "Punish",
]


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    return (a.mean() - b.mean()) / (((a.var(ddof=1) + b.var(ddof=1)) / 2) ** 0.5)


def main() -> None:
    if not CLEAN_PATH.exists():
        raise FileNotFoundError("Clean data not found – run experiments first.")

    df = pd.read_csv(CLEAN_PATH)
    df6 = df[df["study"] == 6]
    if df6.empty:
        raise ValueError("No study 6 data present in clean CSV.")

    rows = []
    for model, g in df6.groupby("model"):
        neutral = g[g["outcome"] == "neutral"]
        bad = g[g["outcome"] == "bad"]
        for dv in DV_COLS:
            d = cohens_d(neutral[dv], bad[dv])
            t_stat, p_val = ttest_ind(neutral[dv], bad[dv], equal_var=False)
            rows.append([model, dv, d, p_val])

    out_df = pd.DataFrame(rows, columns=["model", "DV", "cohens_d", "p_value"])
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLE_DIR / "exp6_stats.csv"
    out_df.to_csv(out_path, index=False)

    pd.set_option("display.precision", 3)
    print(out_df.to_string(index=False))
    print(f"Saved stats to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main() 