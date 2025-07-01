import argparse, subprocess, sys
from pathlib import Path
import pandas as pd
import json, yaml

CSV_PATH = Path("results/clean/data.csv")
ROOT = Path(__file__).resolve().parents[1]
PARTS_PATH = ROOT / "data" / "vignette_parts.json"

TARGET_N = 3  # desired rows per model × condition


def main():
    parser = argparse.ArgumentParser(description="Check data.csv and fill missing completions for given models.")
    parser.add_argument("--models", nargs="+", required=True, help="Model keys to check (e.g. o1mini o4mini)")
    parser.add_argument("--study", type=int, default=1)
    parser.add_argument("--frame", default="juror")
    args = parser.parse_args()

    if not CSV_PATH.exists():
        sys.exit("data.csv not found – run experiments first.")

    df = pd.read_csv(CSV_PATH)

    # Build the full expected condition list for Study 1 (or fallback).
    if args.study == 1 and PARTS_PATH.exists():
        with open(PARTS_PATH, "r", encoding="utf-8") as f:
            parts = json.load(f)
        all_conditions = [f"{scen}_{suf}" for scen in parts.keys() for suf in ("good", "bad")]
    else:
        all_conditions = sorted(df[df.study == args.study].condition.unique())

    if len(all_conditions) != 20 and 'parts' in locals():
        expected_set = {f"{k}_{s}" for k in parts.keys() for s in ("good","bad")}
        missing_set = expected_set - set(all_conditions)
        print(f"[WARN] Expected 20 conditions, found {len(all_conditions)}. Missing: {sorted(missing_set)}")

    for model in args.models:
        missing = []
        sub = df[(df.model == model) & (df.study == args.study)]
        count_series = sub.groupby("condition").size()
        for cond in all_conditions:
            have = int(count_series.get(cond, 0))
            if have < TARGET_N:
                missing.extend([cond] * (TARGET_N - have))
        if not missing:
            print(f"✓ {model} already has {TARGET_N} rows for every condition.")
            continue

        print(f"→ {model} is missing {len(missing)} rows across {len(set(missing))} conditions. Running...")
        # Run run_experiment.py with --conds list and appropriate --n=1
        cmd = [
            sys.executable,
            "src/run_experiment.py",
            "--study", str(args.study),
            "--models", model,
            "--frame", args.frame,
            "--n", "1",  # one completion per listed condition occurrence
            "--conds",
            *missing,
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main() 