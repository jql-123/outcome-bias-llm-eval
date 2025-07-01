#!/usr/bin/env python
"""Bar plot of outcome bias (baseline) for gpt-4o only.

Reads results/tables/exp1_baseline.csv and saves
results/figures/gpt-4o_exp1_baseline.png
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_ind

DISPLAY_NAMES = {
    "gpt4o":    "gpt-4o",
    "o1mini":   "gpt-o1-mini",
    "o4mini":   "gpt-o4-mini",
    "sonnet4":  "sonnet-4",
    "deepseekr1": "deepseek-r1"
}

BASELINE_CSV = Path("results/tables/exp1_baseline.csv")
CLEAN_CSV = Path("results/clean/data.csv")
OUT_PNG = Path("results/figures/gpt-4o_exp1_baseline.png")
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load & preprocess data
# ---------------------------------------------------------------------------

# Load data
if BASELINE_CSV.exists():
    df = pd.read_csv(BASELINE_CSV)
else:
    # Derive baseline from clean CSV (study 1, juror frame)
    if not CLEAN_CSV.exists():
        raise SystemExit("Neither exp1_baseline.csv nor clean data found.")
    clean_df = pd.read_csv(CLEAN_CSV)
    df = clean_df[(clean_df["study"] == 1) & (clean_df["frame"] == "juror")].copy()

df = df[df["model"] == "gpt4o"].copy()
df["model_pretty"] = df["model"].map(DISPLAY_NAMES)

# Derive outcome (neutral vs bad) from condition suffix
df["Outcome"] = df["condition"].apply(lambda c: "neutral" if c.endswith("good") else "bad")

# Explicit DV columns and pretty labels
DV_COLS = ["P_post", "GR_post", "Reckless", "Negligent", "Blame", "Punish"]
PRETTY = {
    "P_post": "Obj. probability",
    "GR_post": "Subj. probability",
    "Reckless": "Recklessness",
    "Negligent": "Negligence",
    "Blame": "Blame",
    "Punish": "Punishment",
}
dv_cols = [c for c in DV_COLS if c in df.columns]

# Scale probability columns to 1–7
for col in ("P_post", "GR_post"):
    if col in df.columns:
        df[col] = df[col] * 7 / 100.0

# Melt for seaborn (include outcome)
plot_df = df.melt(
    id_vars=["Outcome"], value_vars=dv_cols, var_name="DV", value_name="score"
)
plot_df["DV_pretty"] = plot_df["DV"].map(PRETTY)

# Desired order
order_pretty = [PRETTY[c] for c in DV_COLS]

# Colour palette consistent with multi-panel
PALETTE = {"neutral": "#4C72B0", "bad": "#DD8452"}
BAD_COLOR = PALETTE["bad"]

sns.set(style="whitegrid", context="talk")

fig, ax = plt.subplots(figsize=(10, 6))
Hue_order = ["neutral", "bad"]
sns.barplot(
    data=plot_df,
    x="DV_pretty",
    y="score",
    hue="Outcome",
    hue_order=Hue_order,
    palette=PALETTE,
    errorbar="ci",
    ax=ax,
)

ax.set_ylim(0, 7)
ax.set_ylabel("Mean (1–7 scale)", fontweight="bold")
ax.set_xlabel("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=11, fontweight="bold")

ax.set_title("Outcome bias at baseline – gpt-4o", fontsize=15, fontweight="bold")
ax.legend(title=None, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

# ---------------------------------------------------------------------------
# Annotate Cohen's d above bad outcome bars
# ---------------------------------------------------------------------------

def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1, s2 = a.var(ddof=1), b.var(ddof=1)
    s_pool = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
    return (a.mean() - b.mean()) / (s_pool**0.5) if s_pool > 0 else float("nan")

# Compute d per DV
d_values = {}
for raw_dv in DV_COLS:
    if raw_dv not in df.columns:
        continue
    neu = df[df["Outcome"] == "neutral"][raw_dv]
    bad = df[df["Outcome"] == "bad"][raw_dv]
    d_values[PRETTY[raw_dv]] = cohens_d(neu, bad)

# Annotate using predictable patch order (neutral then bad per DV)
patches = ax.patches
bars_per_dv = len(Hue_order)
for i, dv_label in enumerate(order_pretty):
    bad_patch = patches[i * bars_per_dv + Hue_order.index("bad")]
    d_val = d_values.get(dv_label)
    if d_val is None or pd.isna(d_val):
        continue
    ax.text(
        bad_patch.get_x() + bad_patch.get_width() / 2,
        bad_patch.get_height() + 0.2,
        f"d={d_val:.2f}",
        ha="center",
        va="bottom",
        fontsize=8,
        color="black",
    )

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
print(f"Figure saved → {OUT_PNG}")
plt.close(fig) 