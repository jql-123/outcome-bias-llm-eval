"""Two-way ANOVA: Outcome (bad vs neutral) × Expert cue (baseline vs expert-probability).

This script replicates the key interaction test from Kneer & Skoczeń's
Experiment-6 design using the LLM data. It focuses on the *flood* scenario in
Study-3 (no expert cue) and Study-6 (expert-probability cue).

For each dependent variable it fits an OLS model

    DV ~ Outcome * Expert

and extracts the F-statistic, p-value, and partial eta-squared of the
interaction term.  The results are written to
`results/tables/anova_interactions.csv` and a brief significance report is
printed to stdout.

Usage
-----
python src/anova_expert_interaction.py
"""
from __future__ import annotations

from pathlib import Path

from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Compatibility patch for SciPy ≥1.11 where _lazywhere was removed
# ---------------------------------------------------------------------------
try:
    from scipy._lib._util import _lazywhere  # type: ignore
except ImportError:  # pragma: no cover – provide stub so statsmodels imports
    import types, sys

    def _lazywhere(cond: Any, arrays: Any, f, fillvalue=np.nan):  # type: ignore
        """Simple replacement for removed scipy._lib._util._lazywhere."""
        cond = np.asarray(cond, dtype=bool)
        out = np.full(cond.shape, fillvalue, dtype=float)
        if cond.any():
            out[cond] = f(*[np.asarray(a)[cond] for a in arrays])
        return out

    shim = types.ModuleType("scipy._lib._util")
    shim._lazywhere = _lazywhere  # type: ignore
    sys.modules["scipy._lib._util"] = shim

import statsmodels.api as sm
import statsmodels.formula.api as smf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "results" / "clean" / "data.csv"
OUT_DIR = ROOT / "results" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "anova_interactions.csv"

DVS = [
    "P_post",
    "GR_post",
    "Reckless",
    "Negligent",
    "Blame",
    "Punish",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def partial_eta_sq(ss_effect: float, ss_error: float) -> float:
    """Return partial eta-squared given effect and residual sums of squares."""
    if ss_effect + ss_error == 0:
        return np.nan
    return ss_effect / (ss_effect + ss_error)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Clean data not found – run experiments first.")

    df = pd.read_csv(DATA_PATH)

    # Keep flood scenario from studies 3 and 6 only --------------------------------
    df = df[df["condition"].str.startswith("flood")]
    df = df[df["study"].isin([3, 6])].copy()

    # Factors
    df["Expert"] = (df["study"] == 6).astype(int)
    df["Outcome"] = np.where(df["condition"].str.endswith("good"), "neutral", "bad")

    rows: list[list[object]] = []

    for dv in DVS:
        formula = f"{dv} ~ C(Outcome) * Expert"
        model = smf.ols(formula, data=df).fit()
        aov = sm.stats.anova_lm(model, typ=2)

        # Interaction row key depends on statsmodels naming; build string explicitly
        interaction_key = "C(Outcome):Expert"
        if interaction_key not in aov.index:
            # Fallback for other naming (unlikely)
            interaction_key = [idx for idx in aov.index if ":" in idx][0]

        ss_effect = aov.loc[interaction_key, "sum_sq"]
        f_val = aov.loc[interaction_key, "F"]
        p_val = aov.loc[interaction_key, "PR(>F)"]
        ss_error = aov.loc["Residual", "sum_sq"]
        eta = partial_eta_sq(ss_effect, ss_error)

        rows.append([dv, f_val, p_val, eta])

    out_df = pd.DataFrame(rows, columns=["DV", "F_interaction", "p_interaction", "eta_sq_partial"])
    out_df.to_csv(OUT_CSV, index=False)

    print(out_df.to_string(index=False, float_format="{:.3f}".format))

    sig = out_df[out_df["p_interaction"] < 0.05]["DV"].tolist()
    if sig:
        print("\nSignificant interactions (p < .05):", ", ".join(sig))
    else:
        print("\nNo DV shows a significant Outcome × Expert interaction (p < .05).")


if __name__ == "__main__":
    main() 