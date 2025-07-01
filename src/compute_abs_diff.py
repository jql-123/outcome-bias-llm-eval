#!/usr/bin/env python
"""Compute |d_baseline| − |d_expert| for each model × DV with 95 % CI.

Reads results/clean/data.csv and writes results/tables/abs_diff.csv.
"""
from __future__ import annotations

from pathlib import Path
import argparse
from typing import Any, List
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "results" / "clean" / "data.csv"
OUT_CSV = ROOT / "results" / "tables" / "abs_diff.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

DVS = [
    "P_post",
    "GR_post",
    "Reckless",
    "Negligent",
    "Blame",
    "Punish",
]


def cohens_d(a: Any, b: Any) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if a.empty or b.empty:
        return np.nan
    pooled_var = (a.var(ddof=1) + b.var(ddof=1)) / 2
    if pooled_var == 0:
        return 0.0
    return (a.mean() - b.mean()) / np.sqrt(pooled_var)


def bootstrap_abs_diff(
    b_bad: pd.Series,
    b_neu: pd.Series,
    e_bad: pd.Series,
    e_neu: pd.Series,
    n_iter: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    if (len(b_bad) < 2 or len(b_neu) < 2 or len(e_bad) < 2 or len(e_neu) < 2):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    stats: List[float] = []
    for _ in range(n_iter):
        bb_bad = b_bad.sample(frac=1, replace=True, random_state=rng.integers(0, 1e9))
        bb_neu = b_neu.sample(frac=1, replace=True, random_state=rng.integers(0, 1e9))
        ee_bad = e_bad.sample(frac=1, replace=True, random_state=rng.integers(0, 1e9))
        ee_neu = e_neu.sample(frac=1, replace=True, random_state=rng.integers(0, 1e9))
        d_base = cohens_d(bb_bad, bb_neu)
        d_expt = cohens_d(ee_bad, ee_neu)
        if np.isfinite(d_base) and np.isfinite(d_expt):
            stats.append(abs(d_base) - abs(d_expt))
    if not stats:
        return np.nan, np.nan
    return tuple(np.percentile(stats, [2.5, 97.5]).astype(float))


def main() -> None:
    p = argparse.ArgumentParser(description="Absolute difference of effect size (|d_baseline| - |d_expert|)")
    p.add_argument("--baseline", type=int, default=1, help="Baseline study number")
    p.add_argument("--expert", type=int, default=5, help="Expert study number")
    p.add_argument("--csv", type=Path, default=DATA_CSV, help="Input CSV file")
    args = p.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)

    base = df[df["study"] == args.baseline].copy()
    exp = df[df["study"] == args.expert].copy()

    base["Outcome"] = np.where(base["condition"].str.endswith("good"), "neutral", "bad")
    exp["Outcome"] = np.where(exp["condition"].str.endswith("good"), "neutral", "bad")

    rows: list[list[Any]] = []
    for model, base_m in base.groupby("model", observed=True):
        exp_m = exp[exp["model"] == model]
        if exp_m.empty:
            continue
        for dv in DVS:
            b_bad = base_m[base_m["Outcome"] == "bad"][dv]
            b_neu = base_m[base_m["Outcome"] == "neutral"][dv]
            e_bad = exp_m[exp_m["Outcome"] == "bad"][dv]
            e_neu = exp_m[exp_m["Outcome"] == "neutral"][dv]

            d_base = cohens_d(b_bad, b_neu)
            d_expt = cohens_d(e_bad, e_neu)
            diff = abs(d_base) - abs(d_expt) if (np.isfinite(d_base) and np.isfinite(d_expt)) else np.nan
            ci_low, ci_high = bootstrap_abs_diff(b_bad, b_neu, e_bad, e_neu)
            rows.append([model, dv, d_base, d_expt, diff, ci_low, ci_high])

    out = pd.DataFrame(rows, columns=["model", "DV", "d_baseline", "d_expert", "abs_diff", "CI_low", "CI_high"])
    out.to_csv(OUT_CSV, index=False)
    print(f"Absolute-difference table saved → {OUT_CSV}")


if __name__ == "__main__":
    main() 