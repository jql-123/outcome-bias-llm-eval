#!/usr/bin/env python
"""Create a downsized csv with ≤3 random rows per model×condition for Study 1 juror-frame. 

Usage (from project root)::
    python -m src.sample_baseline_rows [--n 3] [--seed 42]

The script reads results/clean/data.csv and writes
results/clean/data_3per_condition.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import random

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "results" / "clean" / "data.csv"
OUT_CSV = ROOT / "results" / "clean" / "data_3per_condition.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample ≤N rows per model×condition from Study 1.")
    parser.add_argument("--n", type=int, default=3, help="Maximum rows to keep per model×condition")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible sampling")
    args = parser.parse_args()

    if not DATA_CSV.exists():
        raise SystemExit(f"Input file not found: {DATA_CSV}")

    rng = random.Random(args.seed)
    df = pd.read_csv(DATA_CSV)

    # Keep only baseline rows.
    subset = df[(df["study"] == 1) & (df["frame"] == "juror")]
    if subset.empty:
        raise SystemExit("No rows match study==1 & frame=='juror'.")

    # Group by model & condition and draw up to N samples.
    def _sampler(group: pd.DataFrame) -> pd.DataFrame:
        if len(group) <= args.n:
            return group
        return group.sample(n=args.n, random_state=rng.randint(0, 1_000_000))

    sampled = subset.groupby(["model", "condition"], group_keys=False).apply(_sampler).reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(sampled)} rows → {OUT_CSV} (seed={args.seed})")


if __name__ == "__main__":
    main() 