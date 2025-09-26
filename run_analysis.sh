#!/usr/bin/env bash

# -----------------------------------------------------------------------------
# Paper Reproduction Analysis Script
# -----------------------------------------------------------------------------
# This script runs the essential analysis pipeline to reproduce results from
# "Outcome Bias in Large Language Models and the Limits of Probability Anchoring"
#
# Key outputs:
#   1) Table A1: Outcome effect sizes (baseline vs expert cue)
#   2) Table A2: Outcome × Expert ANOVA results
#   3) Figure: Absolute reduction in bias
#   4) Figure: All dependent variables comparison
#
# All output figures are written to   results/figures/
# All output tables  are written to   results/tables/
# -----------------------------------------------------------------------------

# ----------------------------
# User-configurable variables
# ----------------------------
FRAME="juror"        # "juror" or "experiment"
BASELINE_STUDY=1     # baseline study number
EXPERT_STUDY=5       # comparison study number
SCENARIO=""         # e.g. "flood" to restrict stats_expert_vs_baseline; leave empty for all

# ---------------------------------------------------------------------------
# Parse command-line overrides (e.g., --baseline 1 --expert 5 --frame experiment)
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline) BASELINE_STUDY="$2"; shift 2 ;;
    --expert)   EXPERT_STUDY="$2";   shift 2 ;;
    --frame)    FRAME="$2";          shift 2 ;;
    --scenario) SCENARIO="$2";       shift 2 ;;
    *) echo "Unknown arg $1"; exit 1 ;;
  esac
done

# ----------------------------
# Helper – exit on error
# ----------------------------
set -e  # abort on any failure

# Pretty echo helper
log() {
  echo -e "\n\033[1;34m➤ $1\033[0m"
}

# -----------------------------------------------------------------------------
# 1) Table A1: Outcome effect sizes at baseline and after expert cue
# -----------------------------------------------------------------------------
log "Generating Table A1: Outcome effect sizes (baseline vs expert)"
if [[ -n "${SCENARIO}" ]]; then
  python src/stats_expert_vs_baseline.py --baseline "${BASELINE_STUDY}" --expert "${EXPERT_STUDY}" --scenario "${SCENARIO}"
else
  python src/stats_expert_vs_baseline.py --baseline "${BASELINE_STUDY}" --expert "${EXPERT_STUDY}"
fi

# -----------------------------------------------------------------------------
# 2) Table A2: Outcome × Expert ANOVA results (per variable, pooled across models)
# -----------------------------------------------------------------------------
log "Generating Table A2: Outcome × Expert ANOVA results"
python src/anova_expert_interaction.py --baseline "${BASELINE_STUDY}" --expert "${EXPERT_STUDY}"

# -----------------------------------------------------------------------------
# 3) Core analysis tables and effect sizes
# -----------------------------------------------------------------------------
log "Computing extended outcome bias analysis"
python -m src.extend_outcome_bias --baseline "${BASELINE_STUDY}" --expert "${EXPERT_STUDY}"

log "Computing absolute effect size differences"
python -m src.compute_abs_diff

# -----------------------------------------------------------------------------
# 4) Paper figures
# -----------------------------------------------------------------------------
log "Generating paper figures"
python src/plot_abs_diff.py
python src/plot_all_dvs.py

log "\nPaper reproduction complete!"
log "Tables: results/tables/ (exp6_vs_baseline.csv, anova_interactions.csv)"
log "Figures: results/figures/ (abs_diff_bar.png, all_dvs_*.png)" 