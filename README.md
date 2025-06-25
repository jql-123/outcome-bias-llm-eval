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