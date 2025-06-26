"""Plot Experiment 6 expert-probability results with Cohen's d annotations."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind

ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / "results" / "clean" / "data.csv"
TABLE_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"

DV_COLS = [
    "P_post",
    "GR_post",
    "Reckless",
    "Negligent",
    "Blame",
    "Punish",
]
DV_LABELS = {
    "P_post": "Obj. probability",
    "GR_post": "Subj. probability",
    "Reckless": "Recklessness",
    "Negligent": "Negligence",
    "Blame": "Blame",
    "Punish": "Punishment",
}
ORDER = list(DV_LABELS.values())
INV_LABELS = {v: k for k, v in DV_LABELS.items()}


def load_data() -> pd.DataFrame:
    if not CLEAN_PATH.exists():
        raise FileNotFoundError("Need clean data – run experiments first.")
    return pd.read_csv(CLEAN_PATH)


def derive_outcome(condition: str) -> str:
    return "neutral" if condition.endswith("good") else "bad"


def main(frame: str = "juror") -> None:
    df = load_data()
    df = df[(df["study"] == 6) & (df["frame"] == frame)]
    if df.empty:
        raise ValueError("No data for Experiment 6 and frame specified.")

    # rescale probabilities
    for col in ("P_post", "GR_post"):
        df[col] = (df[col] / 100.0) * 6 + 1

    df = df.copy()
    df["outcome"] = df["condition"].apply(derive_outcome)

    df_long = df.melt(
        id_vars=["model", "outcome"],
        value_vars=DV_COLS,
        var_name="DV",
        value_name="score",
    )
    df_long["DV"] = df_long["DV"].map(DV_LABELS)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Compute Cohen's d and Welch t for annotations
    # ------------------------------------------------------------------

    def cohens_d(a: pd.Series, b: pd.Series) -> float:
        return (a.mean() - b.mean()) / (((a.var(ddof=1) + b.var(ddof=1)) / 2) ** 0.5)

    def sig_label(p: float, d: float) -> str:
        return f"d={d:.2f}*" if p < 0.05 else f"d={d:.2f}, ns"

    stats_rows = []
    for model, g_mod in df.groupby("model"):
        for raw_dv in DV_COLS:
            neutral = g_mod[g_mod["outcome"] == "neutral"][raw_dv]
            bad = g_mod[g_mod["outcome"] == "bad"][raw_dv]
            d_val = cohens_d(neutral, bad)
            _, p_val = ttest_ind(neutral, bad, equal_var=False)
            label = sig_label(p_val, d_val)
            stats_rows.append([model, raw_dv, d_val, p_val, label])

    stats_df = pd.DataFrame(stats_rows, columns=["model", "DV", "d_value", "p_value", "label"])

    # Save stats table
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    stats_out = TABLE_DIR / "exp6_dstats.csv"
    stats_df.to_csv(stats_out, index=False)

    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=df_long,
        kind="bar",
        x="DV",
        y="score",
        hue="outcome",
        col="model",
        order=ORDER,
        errorbar="se",
        height=4,
        aspect=1.2,
        palette="pastel",
    )

    g.set_axis_labels("", "Mean (1-7 scale)")
    g.set_titles("{col_name}")
    g.set_xticklabels(rotation=45)

    # annotate d
    for ax in g.axes.flatten():
        if ax is None:
            continue
        model = ax.get_title()
        stats_mod = stats_df[stats_df["model"] == model]
        for idx, dv_label in enumerate(ORDER):
            raw_dv = INV_LABELS[dv_label]
            entry = stats_mod[stats_mod["DV"] == raw_dv]
            if entry.empty:
                continue
            label = entry.iloc[0]["label"]
            # max bar height for this dv group
            bars = [p for p in ax.patches if int(p.get_x() + p.get_width() / 2) == idx]
            if not bars:
                continue
            y_max = max(b.get_height() for b in bars)
            ax.text(idx, y_max + 0.2, label, ha="center", va="bottom", fontsize=9)

    g.fig.suptitle(f"Experiment 6 – {frame.capitalize()} framing", y=1.05, fontsize=14)
    plt.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / "expert_by_model.png"
    g.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Experiment 6 expert results.")
    parser.add_argument("--frame", type=str, default="juror", help="Frame filter (juror or experiment)")
    args = parser.parse_args()
    main(frame=args.frame) 