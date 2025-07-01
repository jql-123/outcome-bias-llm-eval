#!/usr/bin/env python
"""Post-process percent_reduction.csv, adding QC flags.

Flags:
• flag_reversal – percent reduction > 100 (effect reversed in expert cue)
• flag_CI       – 95 % CI crosses ±100-% bound

Writes results/tables/percent_reduction_clean.csv
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

IN_CSV = Path("results/tables/percent_reduction.csv")
OUT_CSV = Path("results/tables/percent_reduction_clean.csv")

if not IN_CSV.exists():
    raise SystemExit(f"Input file not found: {IN_CSV}")

df = pd.read_csv(IN_CSV)

# Ensure expected column names regardless of earlier script versions
perc_col = "percent_reduction" if "percent_reduction" in df.columns else "perc_red"
ci_low_col = "ci_low" if "ci_low" in df.columns else "CI_low"
ci_high_col = "ci_high" if "ci_high" in df.columns else "CI_high"

# Rename for consistency
if perc_col != "perc_red":
    df = df.rename(columns={perc_col: "perc_red"})
if ci_low_col != "CI_low":
    df = df.rename(columns={ci_low_col: "CI_low"})
if ci_high_col != "CI_high":
    df = df.rename(columns={ci_high_col: "CI_high"})

# Flags -------------------------------------------------------------
df["flag_reversal"] = df["perc_red"] > 100
# CI bound outside ±100
df["flag_CI"] = (df["CI_low"] < -100) | (df["CI_high"] > 100)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"Cleaned percent-reduction table → {OUT_CSV}") 