"""Utility functions for standardized flood-only bar plots with effect-size annotations."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind

# ---------------------------------------------------------------------------
# Constants (shared across all plots)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures"

DV_COLS: list[str] = [
    "P_post",
    "GR_post",
    "Reckless",
    "Negligent",
    "Blame",
    "Punish",
]
DV_LABELS: dict[str, str] = {
    "P_post": "Obj. probability",
    "GR_post": "Subj. probability",
    "Reckless": "Recklessness",
    "Negligent": "Negligence",
    "Blame": "Blame",
    "Punish": "Punishment",
}
ORDER: Sequence[str] = [DV_LABELS[c] for c in DV_COLS]

# Palette: use first two default seaborn colors, map consistently
PASTEL_PALETTE = sns.color_palette("pastel")  # softer hues
COLOR_MAP = {
    "neutral": PASTEL_PALETTE[0],  # light blue
    "bad": PASTEL_PALETTE[1],      # light orange
}

HUE_ORDER = ["neutral", "bad"]  # neutral left, bad right for both bars & legend

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cohens_d(x: pd.Series, y: pd.Series) -> float:
    """Compute Cohen's d for two independent samples."""
    return (x.mean() - y.mean()) / (((x.var(ddof=1) + y.var(ddof=1)) / 2) ** 0.5)


def _prepare_flood_df(
    df_raw: pd.DataFrame,
    study: int,
    frame: str,
    good_cond: str,
    bad_cond: str,
) -> pd.DataFrame:
    """Filter to specified flood-good / flood-bad conditions and rescale probabilities."""
    df = df_raw[
        (df_raw["study"] == study)
        & (df_raw["frame"] == frame)
        & (df_raw["condition"].isin([good_cond, bad_cond]))
    ].copy()

    # Rescale probabilities 0–100 to 1–7
    for col in ("P_post", "GR_post"):
        if col in df.columns:
            df[col] = (df[col] / 100.0) * 6 + 1

    # Outcome column (neutral vs bad)
    df["outcome"] = df["condition"].apply(lambda c: "neutral" if c == good_cond else "bad")
    return df


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_flood_by_model(
    df_raw: pd.DataFrame,
    study: int,
    frame: str,
    out_filename: str,
    good_cond: str = "flood_good",
    bad_cond: str = "flood_bad",
    title_suffix: str | None = None,
) -> None:
    """Create standardized barplot for flood conditions.

    Parameters
    ----------
    df_raw : pd.DataFrame
        The full cleaned data table (e.g., loaded from results/clean/data.csv).
    study : int
        Study number (experiment index) to filter on.
    frame : str
        "juror" or "experiment" framing.
    out_filename : str
        File name (not path) for the saved PNG in results/figures/.
    """
    df = _prepare_flood_df(df_raw, study, frame, good_cond, bad_cond)
    n_rows = len(df)
    if n_rows == 0:
        raise ValueError(f"No rows for study={study}, frame={frame}, flood conditions only.")

    # Long form
    df_long = df.melt(
        id_vars=["model", "outcome"],
        value_vars=DV_COLS,
        var_name="DV",
        value_name="score",
    )
    df_long["DV"] = df_long["DV"].map(DV_LABELS)  # type: ignore[arg-type]

    # Compute stats table
    stats_rows = []
    for model, g_mod in df.groupby("model"):
        for raw_dv, pretty_dv in DV_LABELS.items():
            neutral = g_mod[g_mod["outcome"] == "neutral"][raw_dv]
            bad = g_mod[g_mod["outcome"] == "bad"][raw_dv]
            d_val = cohens_d(neutral, bad)
            _, p_val = ttest_ind(neutral, bad, equal_var=False)
            stats_rows.append([model, pretty_dv, d_val, p_val])
    stats_df = pd.DataFrame(stats_rows, columns=["model", "DV", "d", "p"])

    # Plot
    g = sns.catplot(
        data=df_long,
        kind="bar",
        x="DV",
        y="score",
        hue="outcome",
        hue_order=HUE_ORDER,
        palette=COLOR_MAP,
        col="model",
        order=ORDER,
        errorbar="se",
        height=4,
        aspect=1.2,
        legend=False,
    )

    g.set_axis_labels("", "Mean (1–7 scale)")
    g.set_titles("{col_name}")
    g.set_xticklabels(rotation=45)

    # Annotate stats
    for ax in g.axes.flatten():
        model = ax.get_title()
        stats_mod = stats_df[stats_df["model"] == model]
        for idx, dv_label in enumerate(ORDER):
            entry = stats_mod[stats_mod["DV"] == dv_label]
            if entry.empty:
                continue
            d_val = entry.iloc[0]["d"]
            p_val = entry.iloc[0]["p"]
            sig = "*" if p_val < 0.05 else "ns"
            label = f"d={d_val:.2f}, {sig}"
            # Find bar tops for this DV group
            bars = [p for p in ax.patches if int(p.get_x() + p.get_width() / 2) == idx]
            if not bars:
                continue
            y_max = max(b.get_height() for b in bars)
            ax.text(idx, y_max + 0.2, label, ha="center", va="bottom", fontsize=9)

    # Legend (top right) using custom patches to ensure colours are correct
    from matplotlib.patches import Patch  # local import to avoid heavy dep if unused

    legend_handles = [
        Patch(facecolor=COLOR_MAP["neutral"], label="Neutral"),
        Patch(facecolor=COLOR_MAP["bad"], label="Bad"),
    ]
    g.fig.legend(handles=legend_handles, loc="upper right", fontsize=11)

    title = f"Experiment {study} – Flood only – {frame.capitalize()} framing"
    if title_suffix:
        title += f" – {title_suffix}"
    title += f" (N={n_rows})"

    g.fig.suptitle(title, y=1.06, fontsize=14)
    plt.tight_layout()

    # Save
    out_path = FIG_DIR / out_filename
    g.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {out_path.relative_to(ROOT)}")
    plt.close(g.fig)


# ---------------------------------------------------------------------------
# Generic plot function (any scenarios, uses suffixes to define outcomes)
# ---------------------------------------------------------------------------

def plot_by_model_generic(
    df_raw: pd.DataFrame,
    study: int,
    frame: str,
    out_filename: str,
    good_suffix: str = "good",
    bad_suffix: str = "bad",
    title: str | None = None,
    show_frame: bool = True,
) -> None:
    """Plot all scenarios for a study, grouping by outcome via suffix."""
    df = df_raw[(df_raw["study"] == study) & (df_raw["frame"] == frame)].copy()
    # Determine outcome via suffix
    def _outcome(cond: str) -> str:
        return "neutral" if cond.endswith(good_suffix) else "bad"

    if df.empty:
        raise ValueError("No data for requested study/frame.")

    for col in ("P_post", "GR_post"):
        if col in df.columns:
            df[col] = (df[col] / 100.0) * 6 + 1

    df["outcome"] = df["condition"].apply(_outcome)

    df_long = df.melt(
        id_vars=["model", "outcome"],
        value_vars=DV_COLS,
        var_name="DV",
        value_name="score",
    )
    df_long["DV"] = df_long["DV"].map(DV_LABELS)

    # stats
    stats_rows = []
    for model, g_mod in df.groupby("model"):
        for raw_dv, pretty_dv in DV_LABELS.items():
            neutral = g_mod[g_mod["outcome"] == "neutral"][raw_dv]
            bad = g_mod[g_mod["outcome"] == "bad"][raw_dv]
            d_val = cohens_d(neutral, bad)
            _, p_val = ttest_ind(neutral, bad, equal_var=False)
            stats_rows.append([model, pretty_dv, d_val, p_val])

    stats_df = pd.DataFrame(stats_rows, columns=["model", "DV", "d", "p"])

    g = sns.catplot(
        data=df_long,
        kind="bar",
        x="DV",
        y="score",
        hue="outcome",
        hue_order=HUE_ORDER,
        palette=COLOR_MAP,
        col="model",
        order=ORDER,
        errorbar="se",
        height=4,
        aspect=1.2,
        legend=False,
    )
    g.set_axis_labels("", "Mean (1–7 scale)")
    g.set_titles("{col_name}")
    g.set_xticklabels(rotation=45)

    # annotate
    for ax in g.axes.flatten():
        model = ax.get_title()
        stats_mod = stats_df[stats_df["model"] == model]
        for idx, dv_label in enumerate(ORDER):
            entry = stats_mod[stats_mod["DV"] == dv_label]
            if entry.empty:
                continue
            d_val = entry.iloc[0]["d"]
            p_val = entry.iloc[0]["p"]
            sig = "*" if p_val < 0.05 else "ns"
            label = f"d={d_val:.2f}, {sig}"
            bars = [p for p in ax.patches if int(p.get_x() + p.get_width() / 2) == idx]
            if not bars:
                continue
            y_max = max(b.get_height() for b in bars)
            ax.text(idx, y_max + 0.2, label, ha="center", va="bottom", fontsize=9)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=COLOR_MAP["neutral"], label="Neutral"),
        Patch(facecolor=COLOR_MAP["bad"], label="Bad"),
    ]
    g.fig.legend(handles=legend_handles, loc="upper right", fontsize=11)

    n_rows = len(df)
    title_main = title if title else f"Experiment {study}"
    suffix = f" – {frame.capitalize()} framing" if show_frame else ""
    g.fig.suptitle(f"{title_main}{suffix} (N={n_rows})", y=1.06, fontsize=14)

    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / out_filename
    g.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {out_path.relative_to(ROOT)}")
    plt.close(g.fig) 