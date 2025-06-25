from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from . import model_api, parser
from . import prompts

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_RAW = ROOT / "results" / "raw_calls"
RESULTS_CLEAN = ROOT / "results" / "clean"


def load_config():
    with open(ROOT / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_vignettes():
    with open(DATA_DIR / "vignettes.json", "r", encoding="utf-8") as f:
        return json.load(f)


def save_raw(run_id: str, model: str, condition: str, completions):
    RESULTS_RAW.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_RAW / f"{run_id}_{model}_{condition}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for c in completions:
            f.write(json.dumps({"text": c}, ensure_ascii=False) + "\n")


def append_clean(run_id: str, model: str, study: int, condition: str, completions):
    rows = []
    for text in completions:
        try:
            P, GR, RA, BL, PU = parser.parse_numbers(text)
        except Exception:
            continue  # skip unparsable
        rows.append(
            {
                "run_id": run_id,
                "model": model,
                "study": study,
                "condition": condition,
                "P": P,
                "GR": GR,
                "RA": RA,
                "BL": BL,
                "PU": PU,
            }
        )

    RESULTS_CLEAN.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_CLEAN / "data.csv"
    df = pd.DataFrame(rows)
    header = not out_path.exists()
    df.to_csv(out_path, mode="a", index=False, header=header)


def main():
    parser_cli = argparse.ArgumentParser(description="Run Kneer & Skoczeń replication.")
    parser_cli.add_argument("--study", type=int, choices=[2, 3], required=True)
    parser_cli.add_argument("--models", nargs="+", required=True, help="Model aliases as in config.yaml section 'models'.")
    parser_cli.add_argument("--n", type=int, default=None, help="Number of completions per condition (overrides config).")

    args = parser_cli.parse_args()

    cfg = load_config()
    vignettes = load_vignettes()

    study = args.study
    models = args.models

    if study == 3:
        conditions = [
            "flood_good",
            "flood_bad",
            "traffic_good",
            "traffic_bad",
        ]
    else:  # study 2
        conditions = [
            "flood_within",
            "traffic_within",
        ]

    run_id = uuid.uuid4().hex[:8]

    n_samples = args.n if args.n is not None else cfg.get("n_samples", 100)

    for model in models:
        for cond in tqdm(conditions, desc=f"{model}"):
            vignette = vignettes[cond]
            messages = [
                {"role": "system", "content": prompts.SYSTEM_PROMPT},
                {"role": "user", "content": prompts.build_user_prompt(cond, vignette)},
            ]
            completions = model_api.generate(model, messages, n=n_samples)
            save_raw(run_id, model, cond, completions)
            append_clean(run_id, model, study, cond, completions)


if __name__ == "__main__":
    main() 