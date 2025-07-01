#!/usr/bin/env python
"""Grouped bar chart of percent reduction with QC flags.

Reads percent_reduction_clean.csv and saves figure to
results/figures/perc_reduction_bar.png
"""
from __future__ import annotations

import matplotlib
# Use non-GUI backend for compatibility on headless servers
matplotlib.use("Agg")  # type: ignore

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

IN_CSV = Path("results/tables/percent_reduction_clean.csv")
OUT_PNG = Path("results/figures/perc_reduction_bar.png")
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

SIG_CSV = Path("results/tables/model_interactions.csv")

if not IN_CSV.exists():
    raise SystemExit("Run clean_percent_reduction.py first.")

# Clean percent-reduction table
df = pd.read_csv(IN_CSV)

# Significant interaction table ------------------------------------------------
if SIG_CSV.exists():
    sig_df = pd.read_csv(SIG_CSV)
    # Normalise column names
    if "p_value" in sig_df.columns:
        p_col = "p_value"
    elif "p" in sig_df.columns:
        p_col = "p"
    else:
        p_col = sig_df.columns[sig_df.columns.str.contains("p", case=False)][0]
    sig_set = {
        (row["model"], row["DV"])
        for _, row in sig_df.iterrows()
        if row[p_col] < 0.05
    }
else:
    sig_set = set()

# Shorter pretty DV names ----------------------------------------------
name_map = {
    "P_post": "Obj. probability",
    "GR_post": "Subj. probability",
    "Reckless": "Recklessness",
    "Negligent": "Negligence",
    "Blame": "Blameworthiness",
    "Punish": "Punishment",
}

df["DV_pretty"] = df["DV"].map(name_map)

# Desired order ---------------------------------------------------------------
order_DV = list(name_map.values())

# Optional explicit model order for consistency in plots
default_hue_order = [
    "gpt4o",
    "sonnet4",
    "deepseekr1",
    "o1mini",
    "o4mini",
]
hue_order = [m for m in default_hue_order if m in df["model"].unique()] + [
    m for m in df["model"].unique() if m not in default_hue_order
]

# Apply categorical ordering
df["DV_pretty"] = pd.Categorical(df["DV_pretty"], categories=order_DV, ordered=True)
df["model"] = pd.Categorical(df["model"], categories=hue_order, ordered=True)

# Sort to match Seaborn's drawing order: model (hue) first, then DV category.
# Seaborn iterates hue groups sequentially across x categories.
df = df.sort_values(["model", "DV_pretty"]).reset_index(drop=True)

sns.set(style="whitegrid", context="talk")

# Use a slightly smaller bar width (0.7) to leave space between groups
g = sns.catplot(
    data=df,
    kind="bar",
    x="DV_pretty",
    y="perc_red",
    hue="model",
    order=order_DV,
    hue_order=hue_order,
    ci=None,
    height=6,
    aspect=1.6,
    legend_out=False,
    width=0.7,
)

# Iterate over visible bars only (skip zero-size legend rectangles) ----------
bars = [p for p in g.ax.patches if p.get_width() > 0]  # legend patches have width=0

# Ensure counts match
assert len(bars) == len(df), (
    "Mismatch between number of bars and dataframe rows after filtering legend patches."
)

# Now loop in the sorted df order which matches bar order (hue-first)
for idx, row in df.iterrows():
    bar = bars[idx]
    # Coordinates for error bar / text
    x = bar.get_x() + bar.get_width() / 2
    y = row["perc_red"]

    # Cap whiskers to within plotting limits (set later)
    capped_low = max(-50, row["CI_low"])
    capped_high = min(150, row["CI_high"])
    g.ax.errorbar(
        x,
        y,
        yerr=[[y - capped_low], [capped_high - y]],
        fmt="none",
        capsize=3,
        lw=1,
        ecolor="black",
    )

    # Grey flag colouring
    if row["flag_reversal"] or row["flag_CI"]:
        bar.set_color("lightgrey")
        bar.set_edgecolor("darkgrey")

    # Significance asterisk just above whisker
    if (row["model"], row["DV"]) in sig_set:
        star_y = min(capped_high + 5, 150 - 2)  # keep inside axis
        g.ax.text(x, star_y, "*", ha="center", va="bottom", fontsize=12)

# Axis limits and reference lines ---------------------------------------------
g.ax.set_ylim(-50, 150)
g.ax.axhline(0, color="black", lw=1)
g.ax.axhline(100, ls="--", lw=1, color="grey")

g.ax.set_xlabel("")
g.ax.set_ylabel("Percent reduction of outcome effect (%)")

# Slightly smaller x-tick font and rotation
plt.xticks(rotation=20, fontsize=11)

# Legend outside plot
g.add_legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

g.fig.tight_layout()
g.savefig(OUT_PNG, dpi=300)
print(f"Figure saved → {OUT_PNG}") 