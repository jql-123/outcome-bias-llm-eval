#!/usr/bin/env python
"""Tabulate how many rows we have for each (study, model, condition, outcome).

Example::

    python -m src.count_completions  
    # or specify a different csv  
    python -m src.count_completions --csv results/clean/data.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description="Count completions per scenario/outcome for each model and study.")
    p.add_argument("--csv", type=Path, default=Path("results/clean/data.csv"), help="Path to data CSV file")
    args = p.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)

    # Group by (study, model, condition, outcome) and count rows
    grouped = (
        df.groupby(["study", "model", "condition", "outcome"], observed=True)  # type: ignore[arg-type]
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    print("- Rows per study / model / condition (bad | neutral) -\n")
    print(grouped.to_string())
    print("\nTotal rows:", len(df))


if __name__ == "__main__":
    main() 