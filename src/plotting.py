from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / "results" / "clean" / "data.csv"
FIG_DIR = ROOT / "results" / "figures"

sns.set_theme(style="whitegrid")


def load_data():
    if not CLEAN_PATH.exists():
        raise FileNotFoundError("Need clean data. Run experiments first.")
    return pd.read_csv(CLEAN_PATH)


def barplot(df: pd.DataFrame, study: int, dv: str, fname: str):
    df_study = df[df["study"] == study]
    order = sorted(df_study["condition"].unique())
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=df_study,
        x="condition",
        y=dv,
        hue="model",
        ci="se",
        order=order,
    )
    ax.set_title(f"Study {study} – {dv}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / fname, dpi=300)
    plt.close()


def main():
    df = load_data()
    for study in [2, 3]:
        for dv in ["BL", "PU"]:
            barplot(df, study, dv, f"{dv.lower()}_study{study}.png")
    print(f"Plots saved to {FIG_DIR}")


if __name__ == "__main__":
    main() 