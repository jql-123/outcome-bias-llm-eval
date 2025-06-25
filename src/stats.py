from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / "results" / "clean" / "data.csv"


# -----------------------------------------------------------------------------
# Effect size helpers
# -----------------------------------------------------------------------------


def cohens_d(x, y, paired=False):
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()
    if paired:
        diff = x - y
        return diff.mean() / diff.std(ddof=1)
    else:
        nx, ny = len(x), len(y)
        dof = nx + ny - 2
        pooled = (((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / dof) ** 0.5
        return (x.mean() - y.mean()) / pooled


def load_data():
    if not CLEAN_PATH.exists():
        raise FileNotFoundError("No clean data found. Run experiments first.")
    return pd.read_csv(CLEAN_PATH)


def welch_t(a, b):
    return stats.ttest_ind(a, b, equal_var=False)


def paired_t(a, b):
    return stats.ttest_rel(a, b)


# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------


def main():
    df = load_data()

    rows = []
    for study in [2, 3]:
        df_study = df[df["study"] == study]
        paired = study == 2
        for model, g in df_study.groupby("model"):
            if study == 3:
                # good vs bad across two domains separately
                for domain in ["flood", "traffic"]:
                    good = g[g["condition"] == f"{domain}_good"]
                    bad = g[g["condition"] == f"{domain}_bad"]
                    for dv in ["BL", "PU"]:
                        d = cohens_d(good[dv], bad[dv], paired=False)
                        t_stat, p_val = welch_t(good[dv], bad[dv])
                        rows.append([study, model, domain, dv, d, p_val])
            else:  # study 2, paired within prompt
                for domain in ["flood", "traffic"]:
                    within = g[g["condition"] == f"{domain}_within"]
                    # split by ending lines (good vs bad) not available so we treat risk diff; for simplicity use P vs GR maybe; we skip
                    for dv in ["BL", "PU"]:
                        # placeholder: compute mean & std vs early vs late? Not available.
                        d, p_val = float("nan"), float("nan")
                        rows.append([study, model, domain, dv, d, p_val])

    summary = pd.DataFrame(rows, columns=["study", "model", "domain", "DV", "d", "p_value"])
    pd.set_option("display.precision", 3)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main() 