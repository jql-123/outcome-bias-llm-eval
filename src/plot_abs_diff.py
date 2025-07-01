#!/usr/bin/env python
"""Bar chart of |d_baseline| − |d_expert| metric with 95 % CI.

Run compute_abs_diff.py first.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

IN_CSV = Path("results/tables/abs_diff.csv")
SIG_CSV = Path("results/tables/model_interactions.csv")
OUT_PNG = Path("results/figures/abs_diff_bar.png")
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

if not IN_CSV.exists():
    raise SystemExit("Run src/compute_abs_diff.py first to generate abs_diff.csv")

df = pd.read_csv(IN_CSV)

name_map = {
    "P_post": "Obj. probability",
    "GR_post": "Subj. probability",
    "Reckless": "Recklessness",
    "Negligent": "Negligence",
    "Blame": "Blame",
    "Punish": "Punishment",
}

df["DV_pretty"] = df["DV"].map(name_map)
order_DV = list(name_map.values())

# Ordering for hue (models)
default_models = ["gpt4o", "sonnet4", "deepseekr1", "o1mini", "o4mini"]
hue_order = [m for m in default_models if m in df["model"].unique()] + [
    m for m in df["model"].unique() if m not in default_models
]

df["DV_pretty"] = pd.Categorical(df["DV_pretty"], categories=order_DV, ordered=True)
df["model"] = pd.Categorical(df["model"], categories=hue_order, ordered=True)

df = df.sort_values(["model", "DV_pretty"]).reset_index(drop=True)

sns.set(style="whitegrid", context="talk")

# Build bar plot
g = sns.catplot(
    data=df,
    kind="bar",
    x="DV_pretty",
    y="abs_diff",
    hue="model",
    order=order_DV,
    hue_order=hue_order,
    ci=None,
    height=6,
    aspect=1.6,
    legend_out=False,
    width=0.7,
)

# Iterate bars skipping legend rectangles
bars = [p for p in g.ax.patches if p.get_width() > 0]
assert len(bars) == len(df)

# Significant Outcome × Expert interaction set --------------------------------
sig_set: set[tuple[str, str]] = set()
if SIG_CSV.exists():
    sig_df = pd.read_csv(SIG_CSV)
    p_col = "p" if "p" in sig_df.columns else (
        "p_value" if "p_value" in sig_df.columns else sig_df.columns[sig_df.columns.str.contains("p", case=False)][0]
    )
    sig_set = {
        (row["model"], row["DV"]) for _, row in sig_df.iterrows() if row[p_col] < 0.05
    }

for idx, row in df.iterrows():
    bar = bars[idx]
    x = bar.get_x() + bar.get_width() / 2
    y = row["abs_diff"]
    g.ax.errorbar(
        x,
        y,
        yerr=[[y - row["CI_low"]], [row["CI_high"] - y]],
        fmt="none",
        capsize=3,
        lw=1,
        ecolor="black",
    )

    # Significance asterisk
    if (row["model"], row["DV"]) in sig_set:
        # place star just beyond the CI whisker
        star_y = row["CI_high"] + 0.2 if y >= 0 else row["CI_low"] - 0.2
        g.ax.text(
            x,
            star_y,
            "*",
            ha="center",
            va="bottom" if y >= 0 else "top",
            fontsize=12,
        )

# Horizontal reference line at 0 (no change)
g.ax.axhline(0, color="black", lw=1)

g.ax.set_ylabel("|d| baseline – |d| expert", fontweight="bold")
g.ax.set_xlabel("", fontweight="bold")

# Interpretation legend text at bottom-left inside axes
g.ax.text(
    0.01,
    -0.18,
    "Δ|d| > 0  →  bias reduced     Δ|d| < 0  →  bias grew",
    transform=g.ax.transAxes,
    ha="left",
    va="top",
    fontsize=10,
)

# Remove default x-axis label and rotate tick labels
g.ax.set_xlabel("")
plt.xticks(rotation=20, fontsize=11, fontweight="bold")

g.add_legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

# Concise figure title
g.fig.suptitle(
    "Absolute change in outcome bias across models",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)

g.fig.tight_layout()
g.savefig(OUT_PNG, dpi=300)
print(f"Figure saved → {OUT_PNG}") 