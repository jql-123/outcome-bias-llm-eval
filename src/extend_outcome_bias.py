#!/usr/bin/env python
"""Extended outcome-bias analysis.

1) Model-specific Outcome×Expert interaction
2) Percent-reduction of effect size from baseline → expert cue

Usage::

    python -m src.extend_outcome_bias [--baseline 1] [--expert 5]

Outputs are written to results/tables/
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "results" / "clean" / "data.csv"
OUT_DIR = ROOT / "results" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DVS: list[str] = [
    "P_post",
    "GR_post",
    "Reckless",
    "Negligent",
    "Blame",
    "Punish",
]

# Re-use a Cohen's d helper copied from stats_expert_vs_baseline ----------------

def cohens_d(a: Any, b: Any) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if a.empty or b.empty:
        return np.nan
    pooled_var = (a.var(ddof=1) + b.var(ddof=1)) / 2
    if pooled_var == 0:
        return 0.0
    return (a.mean() - b.mean()) / np.sqrt(pooled_var)

# ---------------------------------------------------------------------------
# 1) Model-specific Outcome×Expert interaction
# ---------------------------------------------------------------------------

def interaction_analysis(df: pd.DataFrame, baseline: int, expert: int) -> pd.DataFrame:
    df = df[df["study"].isin([baseline, expert])].copy()
    df["Expert"] = (df["study"] == expert).astype(int)
    df["Outcome"] = np.where(df["condition"].str.endswith("good"), "neutral", "bad")

    rows: list[list[Any]] = []

    for model_key, sub_df in df.groupby("model", observed=True):
        for dv in DVS:
            formula = f"{dv} ~ C(Outcome) * Expert"
            try:
                res = smf.ols(formula, data=sub_df).fit()
            except Exception:
                # Not enough variance / observations
                rows.append([model_key, dv, np.nan, np.nan, np.nan, np.nan, np.nan])
                continue

            # Interaction term name is outcome[T.bad]:Expert by default
            term = "C(Outcome)[T.bad]:Expert"
            if term not in res.params.index:
                # fall back to whatever colon pattern exists
                term_candidates = [idx for idx in res.params.index if ":" in idx]
                if term_candidates:
                    term = term_candidates[0]
                else:
                    rows.append([model_key, dv, np.nan, np.nan, np.nan, np.nan, np.nan])
                    continue
            beta = res.params[term]
            t_val = res.tvalues[term]
            p_val = res.pvalues[term]
            ci_low, ci_high = res.conf_int(alpha=0.05).loc[term]
            rows.append([model_key, dv, beta, t_val, p_val, ci_low, ci_high])

    return pd.DataFrame(
        rows,
        columns=[
            "model",
            "DV",
            "beta_interaction",
            "t_value",
            "p_value",
            "ci_low",
            "ci_high",
        ],
    )

# ---------------------------------------------------------------------------
# 2) Percent-reduction metric
# ---------------------------------------------------------------------------

def bootstrap_percent_reduction(
    n_bad_base: pd.Series,
    n_neu_base: pd.Series,
    e_bad: pd.Series,
    e_neu: pd.Series,
    n_iter: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    if (len(n_bad_base) < 2 or len(n_neu_base) < 2 or len(e_bad) < 2 or len(e_neu) < 2):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    stats: List[float] = []
    for _ in range(n_iter):
        bs_b_bad = n_bad_base.sample(frac=1, replace=True, random_state=rng.integers(0, 1e9))
        bs_b_neu = n_neu_base.sample(frac=1, replace=True, random_state=rng.integers(0, 1e9))
        bs_e_bad = e_bad.sample(frac=1, replace=True, random_state=rng.integers(0, 1e9))
        bs_e_neu = e_neu.sample(frac=1, replace=True, random_state=rng.integers(0, 1e9))
        d_base = cohens_d(bs_b_bad, bs_b_neu)
        d_exp = cohens_d(bs_e_bad, bs_e_neu)
        if np.isfinite(d_base) and d_base != 0:
            perc = 100 * (1 - abs(d_exp) / abs(d_base))
            stats.append(perc)
    if not stats:
        return np.nan, np.nan
    low, high = np.percentile(stats, [2.5, 97.5])
    return float(low), float(high)


def percent_reduction(df: pd.DataFrame, baseline: int, expert: int) -> pd.DataFrame:
    base = df[df["study"] == baseline].copy()
    exp = df[df["study"] == expert].copy()

    # Add outcome column
    base["Outcome"] = np.where(base["condition"].str.endswith("good"), "neutral", "bad")
    exp["Outcome"] = np.where(exp["condition"].str.endswith("good"), "neutral", "bad")

    rows: list[list[Any]] = []
    for model_key in df["model"].unique():
        base_m = base[base["model"] == model_key]
        exp_m = exp[exp["model"] == model_key]
        if base_m.empty or exp_m.empty:
            continue
        for dv in DVS:
            b_bad = base_m[base_m["Outcome"] == "bad"][dv]
            b_neu = base_m[base_m["Outcome"] == "neutral"][dv]
            e_bad = exp_m[exp_m["Outcome"] == "bad"][dv]
            e_neu = exp_m[exp_m["Outcome"] == "neutral"][dv]

            d_base = cohens_d(b_bad, b_neu)
            d_exp = cohens_d(e_bad, e_neu)
            if np.isfinite(d_base) and d_base != 0:
                perc_red = 100 * (1 - abs(d_exp) / abs(d_base))
            else:
                perc_red = np.nan
            ci_low, ci_high = bootstrap_percent_reduction(b_bad, b_neu, e_bad, e_neu)
            rows.append([model_key, dv, d_base, d_exp, perc_red, ci_low, ci_high])

    return pd.DataFrame(
        rows,
        columns=[
            "model",
            "DV",
            "d_baseline",
            "d_expert",
            "percent_reduction",
            "ci_low",
            "ci_high",
        ],
    )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Extended outcome-bias analysis")
    parser.add_argument("--baseline", type=int, default=1, help="Baseline study number (default 1)")
    parser.add_argument("--expert", type=int, default=5, help="Expert study number (default 5)")
    parser.add_argument("--csv", type=Path, default=DATA_CSV, help="Input CSV path")
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"Input CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)

    # 1) interaction test ------------------------------------------------------
    inter_df = interaction_analysis(df, baseline=args.baseline, expert=args.expert)
    out_inter = OUT_DIR / "model_interactions.csv"
    inter_df.to_csv(out_inter, index=False)
    print(f"Model-specific interaction table saved → {out_inter}")

    # 2) percent reduction -----------------------------------------------------
    perc_df = percent_reduction(df, baseline=args.baseline, expert=args.expert)
    out_perc = OUT_DIR / "percent_reduction.csv"
    perc_df.to_csv(out_perc, index=False)
    print(f"Percent-reduction table saved → {out_perc}")


if __name__ == "__main__":
    main() 