#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../scripts/env.sh"

cd "${PRISM_ROOT}"
MODE="${1:-4080}"

echo "[Strict Equal-Recall Sweep] Running mode=${MODE} across 9 core datasets..."
python3 bench/upgrade_e1_e2_e9_e10_strict.py --mode "${MODE}"
python3 bench/verify_all_paper_numbers.py
