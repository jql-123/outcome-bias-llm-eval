# Bias_LLMEval: Replicating Kneer & Skoczeń (2023) with Large Language Models

This repository re-runs Experiment 2 (within-subjects) and Experiment 3 (between-subjects) from *Kneer & Skoczeń, 2023* on four large language models:

| Alias | Model name |
|-------|------------|
| nano  | `gpt-4o-mini` |
| haiku | `claude-3-haiku-20240307` |
| deepseek | `deepseek-ai/deepseek-llm-chat` |

## 1 . Quick start

```bash
# 1. Install the fast Python package manager **uv**
#    https://github.com/astral-sh/uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create an isolated environment (./.venv) and install deps
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 3. Add your API keys
cp .env.example .env  # then edit the file

# 4. Build the vignettes JSON (parses the four DOCX files)
python -m src.build_vignettes

# 5. Run an experiment (100 samples / model by default)
python -m src.run_experiment --study 3 --models nano haiku deepseek

# 6. Re-run the analysis & plots
python -m src.stats
python -m src.plotting
```

Expected number of LLM calls for the full replication (`n_samples=100`) is roughly **2 400** (6 prompts × 4 models × 100 samples), which currently costs ≈ **USD 70–100** depending on token usage.

## 2 . Project layout

```
.
├── data/
│   ├── raw/                  # original Qualtrics DOCX exports (copied automatically)
│   └── vignettes.json        # clean machine-readable texts (auto-generated)
├── results/
│   ├── raw_calls/            # 1 JSONL per (model,condition)
│   ├── clean/                # aggregated CSV (one row per completion)
│   └── figures/              # barplots for blame & punishment
├── src/                      # all Python source code
│   ├── build_vignettes.py    # parses DOCX → JSON
│   ├── prompt_templates.py   # system & user prompts
│   ├── model_api.py          # unified LLM interface
│   ├── run_experiment.py     # CLI entry-point
│   ├── parser.py             # extract 5 numbers
│   ├── stats.py              # effect sizes & t-tests
│   └── plotting.py           # visualisation
├── config.yaml               # n_samples, temperature, top_p, model mapping
├── requirements.txt
└── .env.example              # template for API keys
```

## 3 . Configuration

All hyper-parameters live in `config.yaml`. Override any value via CLI flags or by editing the file.

```yaml
n_samples: 100
max_tokens: 600
top_p: 0.95
models:
  nano:
    id: gpt-4o-mini
    provider: openai
    temperature: 1.0
  haiku:
    id: claude-3-haiku-20240307
    provider: anthropic
    temperature: 0.9
  deepseek:
    id: deepseek-ai/deepseek-llm-chat
    provider: huggingface
    temperature: 0.9
```

## 4 . Re-creating the raw DOCX exports

The repository already contains the four original Qualtrics DOCX exports under `Qualtrics Exports/` in the project root. `src.build_vignettes` copies them into `data/raw/` on first run, so you don't have to move any files manually.

## 5 . Licence

This work is released under the MIT licence. The original experiment material is reproduced here under fair-use for research purposes.

## Expert vs Baseline (Flood) Analysis

To reproduce the moderation analysis that contrasts the *no-probability-stabilising* baseline (Study 3) with the *expert-probability* condition (Study 6) for the flood scenario run:

```bash
# Generate CSV + publication-ready LaTeX table
python src/stats_expert_vs_baseline.py

# Optional visualisation of |d| reductions
python src/plot_delta_d.py
```

The raw numbers are written to `results/tables/exp6_vs_baseline.csv` and the
figure (if requested) is saved to
`results/figures/delta_d_exp6.png`. This mirrors the human paper's Experiment-6 comparison. 

## Analysis Workflow

```bash
python src/clean_percent_reduction.py   # add QC flags to percent_reduction.csv
python src/plot_percent_reduction.py    # grouped bar chart with error bars & flags
```

## Quick reproduction: Study 1 vs Study 5

The commands below assume you have installed the dependencies (see **Setup** section) and exported your API keys in a local `.env`.

```bash
# 1) Run or resume the two studies (juror framing)
python src/check_and_fill.py --models gpt4o sonnet4 deepseekr1 o1mini o4mini --study 1 --frame juror
python src/check_and_fill.py --models gpt4o sonnet4 deepseekr1 o1mini o4mini --study 5 --frame juror
```
Each script call tops-up missing completions so you can re-run it safely; the raw responses land in `results/raw_calls/` and the parsed rows in `results/clean/data.csv`.

---
### Analysis & figures

```bash
# 2) Extended outcome-bias analysis
a) effect-size tables + percent-reduction
python -m src.extend_outcome_bias          # outputs results/tables/*.csv

b) absolute |d| difference metric + plot
a) compute
python -m src.compute_abs_diff             # -> results/tables/abs_diff.csv
b) plot
python src/plot_abs_diff.py                # -> results/figures/abs_diff_bar.png

# 3) Multi-panel bar charts (baseline vs expert)
python src/plot_all_dvs.py                 # → results/figures/all_dvs_exp1_juror.png
                                           #   results/figures/all_dvs_exp5_expert_juror.png

# 4) Single-panel baseline for GPT-4o
python src/plot_single_baseline_gpt4o.py   # → results/figures/gpt-4o_exp1_baseline.png
```

---
### Two-way ANOVA (Outcome × Expert)

To replicate the interaction statistics used for significance flags:

```bash
python -m src.anova_expert_interaction     # writes results/tables/anova_interactions.csv
```

The table lists the β coefficient, *t*-value and *p*-value for the Outcome × Expert term for every model and dependent variable; any row with *p* < .05 is marked with an asterisk in `plot_abs_diff.py`. 

## Models evaluated

The experiments reported in this repository use five publicly-available chat models:

| internal id | display name |
|-------------|--------------|
| `gpt4o`     | **gpt-4o** |
| `sonnet4`   | **sonnet-4** |
| `deepseekr1`| **deepseek-r1** |
| `o1mini`    | **gpt-o1-mini** |
| `o4mini`    | **gpt-o4-mini** |

All models are accessed through their official APIs; no proprietary weights are included in this repository.