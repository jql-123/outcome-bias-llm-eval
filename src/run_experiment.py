from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
import datetime

import pandas as pd
import yaml
from tqdm import tqdm

from . import model_api, parser, prompts

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


def split_intro(full_text: str) -> str:
    """Return intro part (before outcome sentence)."""
    for marker in ("As during", "As usual", "It just so happens"):
        idx = full_text.find(marker)
        if idx != -1:
            return full_text[: idx].strip()
    # fallback: return first 80%?? just return full
    return full_text


def save_row(row: dict):
    RESULTS_CLEAN.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_CLEAN / "data.csv"
    df = pd.DataFrame([row])
    header = not out_path.exists()
    df.to_csv(out_path, mode="a", index=False, header=header)


def log_conversation(model, condition, step, messages, response, run_id):
    log_dir = ROOT / "results" / "conversations"
    log_dir.mkdir(parents=True, exist_ok=True)
    fname = log_dir / f"{run_id}_{model}_{condition}_{step}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump({
            "model": model,
            "condition": condition,
            "step": step,
            "messages": messages,
            "response": response,
            "timestamp": datetime.datetime.now().isoformat(),
        }, f, ensure_ascii=False, indent=2)


def main():
    parser_cli = argparse.ArgumentParser(description="Run Kneer & Skoczeń replication.")
    parser_cli.add_argument("--study", type=int, choices=[2, 3], required=True)
    parser_cli.add_argument("--models", nargs="+", required=True, help="Model aliases as in config.yaml section 'models'.")
    parser_cli.add_argument("--n", type=int, default=None, help="Number of completions per condition (overrides config).")
    parser_cli.add_argument("--frame", type=str, default="juror", choices=["juror", "participant"], help="System prompt framing for Step B (juror or participant).")

    args = parser_cli.parse_args()

    cfg = load_config()
    vignettes = load_vignettes()

    study = args.study
    models = args.models
    frame = args.frame

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
            full_text = vignettes[cond]
            intro_text = split_intro(full_text)

            for _ in range(n_samples):
                # Step A: anchor
                anchor_prompt = prompts.build_anchor_prompt(cond, intro_text)
                msgs_a = [
                    {"role": "system", "content": prompts.SYSTEM_2NUM},
                    {"role": "user", "content": anchor_prompt},
                ]
                anchor_resp = model_api.generate(model, msgs_a, n=1)[0]
                try:
                    P_anchor, GR_anchor = parser.parse_two(anchor_resp)
                except Exception:
                    continue  # skip if malformed

                # For anchor step
                log_conversation(model, cond, "anchor", msgs_a, anchor_resp, run_id)

                # Step B: post-outcome
                post_prompt = prompts.build_post_prompt(
                    cond, full_text, P_anchor, GR_anchor
                )
                system_6 = prompts.get_system_6(frame)
                msgs_b = [
                    {"role": "system", "content": system_6},
                    {"role": "user", "content": post_prompt},
                ]
                dv_resp = model_api.generate(model, msgs_b, n=1)[0]
                try:
                    (
                        P_post,
                        GR_post,
                        reckless,
                        negligent,
                        blame,
                        punish,
                    ) = parser.parse_six(dv_resp)
                except Exception:
                    continue

                # For post-outcome step
                log_conversation(model, cond, "post", msgs_b, dv_resp, run_id)

                row = {
                    "run_id": run_id,
                    "model": model,
                    "study": study,
                    "condition": cond,
                    "frame": frame,
                    "P_anchor": P_anchor,
                    "GR_anchor": GR_anchor,
                    "P_post": P_post,
                    "GR_post": GR_post,
                    "Reckless": reckless,
                    "Negligent": negligent,
                    "Blame": blame,
                    "Punish": punish,
                }
                save_row(row)


if __name__ == "__main__":
    main() 