"""Compare baseline (Study 3, flood) with expert-probability condition (Study 6, flood).

This script reproduces the Experiment-6 moderation analysis from the human
study: it contrasts the *neutral* (good-outcome) versus *bad* outcome
conditions in the original anchor/outcome design (no probability
stabilisation) against the expert-probability variant.

The script outputs two artefacts
1. A raw numbers CSV written to results/tables/exp6_vs_baseline.csv
2. A publication-ready LaTeX table printed to stdout.

Usage
-----
# All scenarios (default)
python src/stats_expert_vs_baseline.py

# Single scenario only, e.g. flood
python src/stats_expert_vs_baseline.py --scenario flood
"""
from __future__ import annotations

from pathlib import Path

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import argparse

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "results" / "clean" / "data.csv"
TABLE_DIR = ROOT / "results" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = TABLE_DIR / "exp6_vs_baseline.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Hedges & Olkin standard error for Cohen's d -----------------------------

def d_ci(d: float, n1: int, n2: int, alpha: float = 0.05) -> tuple[float, float]:
    """Return (lower, upper) CI for Cohen's *d* using Hedges & Olkin SE."""
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan
    se = np.sqrt((n1 + n2) / (n1 * n2) + (d**2) / (2 * (n1 + n2 - 2)))
    z = 1.96  # 95 %
    delta = z * se
    return d - delta, d + delta

def cohens_d(a: Any, b: Any) -> float:
    """Return Cohen's *d* for two independent samples.

    Both *a* and *b* are converted to numeric and NaNs dropped. If both
    samples have zero variance the function returns 0.0 to avoid division
    by zero.
    """
    a = pd.to_numeric(a, errors="coerce").dropna()  # type: ignore[attr-defined]
    b = pd.to_numeric(b, errors="coerce").dropna()  # type: ignore[attr-defined]
    if a.empty or b.empty:
        return np.nan
    pooled_var = (a.var(ddof=1) + b.var(ddof=1)) / 2  # pooled variance
    if pooled_var == 0:
        return 0.0
    return (a.mean() - b.mean()) / np.sqrt(pooled_var)


def ci95(a: Any, b: Any) -> tuple[float, float]:
    """95 % confidence interval (Welch) for the difference of means."""
    a = pd.to_numeric(a, errors="coerce").dropna()  # type: ignore[attr-defined]
    b = pd.to_numeric(b, errors="coerce").dropna()  # type: ignore[attr-defined]
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    diff = a.mean() - b.mean()
    delta = 1.96 * se  # 95 % CI
    return diff - delta, diff + delta


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main(scenario: str | None = None) -> None:  # noqa: C901  (keep flat for clarity)
    if not DATA_PATH.exists():
        raise FileNotFoundError("Cleaned data not found – run experiments first.")

    df = pd.read_csv(DATA_PATH)

    # Optional scenario filter ----------------------------------------------------
    if scenario:
        df = df[df["condition"].str.startswith(scenario)]

    # Baseline (Study-3) and expert (Study-6) subsets -----------------------------
    base = df[df["study"] == 3].copy()
    expert = df[df["study"] == 6].copy()

    # Outcome column: neutral (good suffix) or bad --------------------------------
    for sub in (base, expert):
        sub["outcome"] = np.where(
            sub["condition"].astype(str).str.endswith("good"),  # type: ignore[attr-defined]
            "neutral",
            "bad",
        )

    dvs = [
        "P_post",
        "GR_post",
        "Reckless",
        "Negligent",
        "Blame",
        "Punish",
    ]

    rows: list[list[object]] = []

    for dv in dvs:
        # ---------------------------------------------------------------------
        # Baseline stats
        # ---------------------------------------------------------------------
        n_bad = base[base["outcome"] == "bad"][dv]
        n_neu = base[base["outcome"] == "neutral"][dv]
        t0, p0 = ttest_ind(n_bad, n_neu, equal_var=False, nan_policy="omit")
        d0 = cohens_d(n_bad, n_neu)
        ci0_low_d, ci0_high_d = d_ci(d0, len(n_bad), len(n_neu))

        # ---------------------------------------------------------------------
        # Expert stats
        # ---------------------------------------------------------------------
        e_bad = expert[expert["outcome"] == "bad"][dv]
        e_neu = expert[expert["outcome"] == "neutral"][dv]
        t1, p1 = ttest_ind(e_bad, e_neu, equal_var=False, nan_policy="omit")
        d1 = cohens_d(e_bad, e_neu)
        ci1_low_d, ci1_high_d = d_ci(d1, len(e_bad), len(e_neu))

        # ---------------------------------------------------------------------
        # Moderation / interaction test (optional)
        # ---------------------------------------------------------------------
        # Compute per-subject deltas (bad minus neutral) without sampling ---------
        # Merge so we only pair within the same scenario & model (balanced view)
        def _prepare_delta(sub_df: pd.DataFrame) -> pd.Series:
            pivot = sub_df.pivot_table(
                index=["model", "condition"],
                columns="outcome",
                values=dv,
                aggfunc="mean",
            )
            if {"bad", "neutral"}.issubset(pivot.columns):
                return pivot["bad"] - pivot["neutral"]
            return pd.Series(dtype=float)

        delta_base = _prepare_delta(base)  # type: ignore[arg-type]
        delta_expert = _prepare_delta(expert)  # type: ignore[arg-type]

        if not delta_base.empty and not delta_expert.empty:
            t_mod, p_mod = ttest_ind(delta_base, delta_expert, equal_var=False, nan_policy="omit")
        else:
            t_mod, p_mod = np.nan, np.nan

        # -----------------------------------------------------------------
        # Delta-d bootstrap CI
        # -----------------------------------------------------------------
        def _bootstrap_delta(n_iter: int = 1000) -> tuple[float, float]:
            if len(n_bad) < 2 or len(n_neu) < 2 or len(e_bad) < 2 or len(e_neu) < 2:
                return np.nan, np.nan
            rng = np.random.default_rng(42)
            deltas = []
            for _ in range(n_iter):
                bs_n_bad = n_bad.sample(frac=1, replace=True, random_state=rng.integers(0, 1e9))
                bs_n_neu = n_neu.sample(frac=1, replace=True, random_state=rng.integers(0, 1e9))
                bs_e_bad = e_bad.sample(frac=1, replace=True, random_state=rng.integers(0, 1e9))
                bs_e_neu = e_neu.sample(frac=1, replace=True, random_state=rng.integers(0, 1e9))
                d_bs_no = cohens_d(bs_n_bad, bs_n_neu)
                d_bs_exp = cohens_d(bs_e_bad, bs_e_neu)
                deltas.append(d_bs_no - d_bs_exp)
            low, high = np.percentile(deltas, [2.5, 97.5])
            return low, high

        delta_d = d0 - d1
        delta_low, delta_high = _bootstrap_delta()

        rows.append(
            [
                dv,
                t0,
                p0,
                f"({ci0_low_d:.2f}, {ci0_high_d:.2f})",
                d0,
                f"({ci0_low_d:.2f}, {ci0_high_d:.2f})",
                t1,
                p1,
                f"({ci1_low_d:.2f}, {ci1_high_d:.2f})",
                d1,
                f"({ci1_low_d:.2f}, {ci1_high_d:.2f})",
                p_mod,
                delta_d,
                f"({delta_low:.2f}, {delta_high:.2f})",
            ]
        )

    columns = [
        "DV",
        "t_no",
        "p_no",
        "CI_no",
        "d_no",
        "CI_d_no",
        "t_exp",
        "p_exp",
        "CI_exp",
        "d_exp",
        "CI_d_exp",
        "p_mod",
        "delta_d",
        "CI_delta",
    ]
    out_df = pd.DataFrame(rows, columns=columns)  # type: ignore[arg-type]

    # Save raw numbers -----------------------------------------------------------
    out_df.to_csv(OUT_CSV, index=False)

    # Print LaTeX table ----------------------------------------------------------
    latex_table = out_df.to_latex(index=False, float_format="%.2f", escape=False)
    print(latex_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare baseline (Study 3) vs expert probability (Study 6).")
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Filter to a single scenario key (e.g. 'flood'); omit for all scenarios.",
    )
    args = parser.parse_args()
    main(scenario=args.scenario) 