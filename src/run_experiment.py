from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
import datetime
import sys

# ---------------------------------------------------------------------------
# Flexible imports so the script can be executed with `python src/run_experiment.py`
# or via `python -m src.run_experiment`.
# ---------------------------------------------------------------------------

try:
    # When run as a module (python -m src.run_experiment)
    from . import model_api, parser, prompts  # type: ignore
except ImportError:  # pragma: no cover – fallback for direct script execution
    ROOT_DIR = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT_DIR))
    from src import model_api, parser, prompts  # type: ignore

import random

import pandas as pd
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_RAW = ROOT / "results" / "raw_calls"
RESULTS_CLEAN = ROOT / "results" / "clean"

EXPERT_COND = {
    "flood_expert_good": "neutral",
    "flood_expert_bad": "bad",
    # traffic expert conditions disabled for now
}


def load_config():
    with open(ROOT / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_vignettes():
    with open(DATA_DIR / "vignettes.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_parts():
    parts_path = DATA_DIR / "vignette_parts.json"
    if parts_path.exists():
        with open(parts_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


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
    parser_cli = argparse.ArgumentParser(description="Run replication experiments (Studies 2,3,5,6).")
    parser_cli.add_argument("--study", type=int, choices=[2, 3, 5, 6], required=True)
    parser_cli.add_argument("--models", nargs="+", required=True, help="Model aliases as in config.yaml section 'models'.")
    parser_cli.add_argument("--n", type=int, default=None, help="Number of completions per condition (overrides config).")
    parser_cli.add_argument("--total", type=int, default=None, help="Total completions per model (across all conditions). Overrides --n if provided.")
    parser_cli.add_argument("--frame", type=str, default="juror", choices=["juror", "experiment"], help="System prompt framing for Step B (juror or experiment).")

    args = parser_cli.parse_args()

    cfg = load_config()
    vignettes = load_vignettes()
    parts = load_parts()

    study = args.study
    models = args.models
    frame = args.frame

    if study == 5:
        conditions = list(EXPERT_COND.keys())
    elif study == 3:
        # Only flood scenarios for now; traffic ignored
        conditions = [
            "flood_good",
            "flood_bad",
        ]
    elif study == 2:
        conditions = [
            "flood_within",
        ]
    elif study == 6:  # two-step expert probability
        conditions = list(EXPERT_COND.keys())
    else:
        raise ValueError(f"Unknown study: {study}")

    run_id = uuid.uuid4().hex[:8]

    # Determine how many completions to generate
    total_n = args.total  # may be None
    per_cond_n_raw = args.n if args.n is not None else cfg.get("n_samples", 100)
    per_cond_n: int = int(per_cond_n_raw if per_cond_n_raw is not None else 100)

    for model in models:
        if total_n is not None:
            # Build randomized condition sequence with roughly even distribution
            base = total_n // len(conditions)
            remainder = total_n % len(conditions)
            cond_sequence: list[str] = []
            for c in conditions:
                cond_sequence.extend([c] * base)
            # distribute remainder
            if remainder:
                cond_sequence.extend(random.sample(conditions, remainder))
            random.shuffle(cond_sequence)
        else:
            cond_sequence = []
            for c in conditions:
                cond_sequence.extend([c] * per_cond_n)

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def process_condition(cond):
            # ---------------- Study 5 (single-step expert) -----------------
            if study == 5:
                # Expert probability – single-step prompt
                text = vignettes[cond]
                prompt = prompts.build_expert_prompt(cond, text)
                system_6 = prompts.get_system_6(frame)
                msgs = [
                    {"role": "system", "content": system_6},
                    {"role": "user", "content": prompt},
                ]
                try:
                    dv_resp = model_api.generate(model, msgs, n=1)[0]
                    (
                        P_post,
                        GR_post,
                        reckless,
                        negligent,
                        blame,
                        punish,
                    ) = parser.parse_six(dv_resp)
                except Exception:
                    return None

                outcome = EXPERT_COND[cond]

                log_conversation(model, cond, "expert", msgs, dv_resp, run_id)

                row = {
                    "run_id": run_id,
                    "model": model,
                    "study": study,
                    "condition": cond,
                    "outcome": outcome,
                    "frame": frame,
                    "P_post": P_post,
                    "GR_post": GR_post,
                    "Reckless": reckless,
                    "Negligent": negligent,
                    "Blame": blame,
                    "Punish": punish,
                }
                return row

            # ---------------- Study 6 (two-step expert) --------------------
            if study == 6:
                seg = parts.get("flood", {})
                intro_text = seg.get("intro", "")
                expert_line = seg.get("expert_phrase", "")
                outcome_text = seg.get("bad_outcome" if cond.endswith("bad") else "good_outcome", "")

                # Step A – anchor (intro only)
                anchor_prompt = prompts.build_anchor_prompt(cond, intro_text)
                msgs_a = [
                    {"role": "system", "content": prompts.get_system_2(frame)},
                    {"role": "user", "content": anchor_prompt},
                ]
                try:
                    anchor_resp = model_api.generate(model, msgs_a, n=1)[0]
                    P_anchor, GR_anchor = parser.parse_two(anchor_resp)
                except Exception:
                    return None

                log_conversation(model, cond, "anchor", msgs_a, anchor_resp, run_id)

                # Step B – intro + expert + outcome + reminder + 6 Qs
                body = f"{intro_text}\n\n{expert_line}\n\n{outcome_text}"
                post_prompt = prompts.build_post_prompt_from_body(
                    cond, body, P_anchor, GR_anchor
                )
                msgs_b = [
                    {"role": "system", "content": prompts.get_system_6(frame)},
                    {"role": "user", "content": post_prompt},
                ]
                try:
                    dv_resp = model_api.generate(model, msgs_b, n=1)[0]
                    (
                        P_post,
                        GR_post,
                        reckless,
                        negligent,
                        blame,
                        punish,
                    ) = parser.parse_six(dv_resp)
                except Exception:
                    return None

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
                return row

            # ---------------- Studies 2 & 3 -------------------------------
            full_text = vignettes[cond]
            intro_text = split_intro(full_text)

            # Step A: anchor
            anchor_prompt = prompts.build_anchor_prompt(cond, intro_text)
            system_2 = prompts.get_system_2(frame)
            msgs_a = [
                {"role": "system", "content": system_2},
                {"role": "user", "content": anchor_prompt},
            ]
            try:
                anchor_resp = model_api.generate(model, msgs_a, n=1)[0]
                P_anchor, GR_anchor = parser.parse_two(anchor_resp)
            except Exception:
                return None  # skip if malformed

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
            try:
                dv_resp = model_api.generate(model, msgs_b, n=1)[0]
                (
                    P_post,
                    GR_post,
                    reckless,
                    negligent,
                    blame,
                    punish,
                ) = parser.parse_six(dv_resp)
            except Exception:
                return None

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
            return row

        # Use ThreadPoolExecutor to parallelize over conditions
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_cond = {executor.submit(process_condition, cond): cond for cond in cond_sequence}
            for future in tqdm(as_completed(future_to_cond), total=len(future_to_cond), desc=f"{model}"):
                row = future.result()
                if row is not None:
                    save_row(row)


if __name__ == "__main__":
    main() 