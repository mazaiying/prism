#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

cd "${PRISM_ROOT}"
export PRISM_FIG_OUT="${PRISM_FIG_OUT:-${PRISM_ROOT}/figures/generated}"
mkdir -p "${PRISM_FIG_OUT}"

echo "[PRISM] Regenerating architecture diagrams -> ${PRISM_FIG_OUT}"
python3 figures/draw_sigmod_industry_figs.py

echo "[PRISM] Regenerating experimental evaluation plots -> ${PRISM_FIG_OUT}"
python3 figures/plot_prism_experiments.py

echo "[PRISM] Done. Generated figures are in ${PRISM_FIG_OUT}"
