#!/usr/bin/env bash

# -----------------------------------------------------------------------------
# Analysis & Plotting Convenience Script
# -----------------------------------------------------------------------------
# This script runs the full post-processing pipeline after new experiment API
# calls. It will:
#   1) Plot baseline (Study-3) bar charts
#   2) Plot expert-probability (Study-6) bar charts
#   3) Compute baseline vs expert stats and create tables
#   4) Draw forest, scatter, and Δd visualisations
#   5) Run the two-way ANOVA interaction test
#   6) Produce per-model dstats for Study-6
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
# 1) Baseline bar chart (Study-${BASELINE_STUDY})
# -----------------------------------------------------------------------------
log "Plotting baseline bar chart (Study ${BASELINE_STUDY}, frame='${FRAME}')"
python src/plot_all_dvs_by_model.py --frame "${FRAME}" --study "${BASELINE_STUDY}"

# -----------------------------------------------------------------------------
# 2) Expert-probability bar chart (Study-${EXPERT_STUDY})
# -----------------------------------------------------------------------------
log "Plotting expert bar chart (Study ${EXPERT_STUDY}, frame='${FRAME}')"
python src/plot_expert.py --frame "${FRAME}" --study "${EXPERT_STUDY}"

# -----------------------------------------------------------------------------
# 3) Baseline vs Expert statistics table (exp6_vs_baseline.csv)
# -----------------------------------------------------------------------------
log "Computing baseline vs expert stats table"
if [[ -n "${SCENARIO}" ]]; then
  python src/stats_expert_vs_baseline.py --baseline "${BASELINE_STUDY}" --expert "${EXPERT_STUDY}" --scenario "${SCENARIO}"
else
  python src/stats_expert_vs_baseline.py --baseline "${BASELINE_STUDY}" --expert "${EXPERT_STUDY}"
fi

# -----------------------------------------------------------------------------
# 4) Forest, scatter, and Δd plots using the stats table
# -----------------------------------------------------------------------------
log "Drawing forest, scatter, and Δd plots"
python src/plot_forest_d.py --baseline "${BASELINE_STUDY}" --expert "${EXPERT_STUDY}"
python src/plot_scatter_d.py --baseline "${BASELINE_STUDY}" --expert "${EXPERT_STUDY}"
python src/plot_delta_d.py --baseline "${BASELINE_STUDY}" --expert "${EXPERT_STUDY}"

# Per–model scatter
python src/plot_scatter_d_by_model.py

# Per–model forest
python src/plot_forest_d_by_model.py --baseline "${BASELINE_STUDY}" --expert "${EXPERT_STUDY}"

# -----------------------------------------------------------------------------
# 5) Outcome × Expert two-way ANOVA interaction (anova_interactions.csv)
# -----------------------------------------------------------------------------
log "Running two-way ANOVA interaction analysis"
python src/anova_expert_interaction.py --baseline "${BASELINE_STUDY}" --expert "${EXPERT_STUDY}"

# -----------------------------------------------------------------------------
# 6) Study-6 per-model dstats (exp6_stats.csv)
# -----------------------------------------------------------------------------
log "Computing dstats for Study-${EXPERT_STUDY} (expert-probability)"
python src/stats_expert.py

log "\nAll analyses complete!  Figures → results/figures,  Tables → results/tables" 