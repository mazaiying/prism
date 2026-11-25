#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../scripts/env.sh"

cd "${PRISM_ROOT}"

echo "[E14 Official VecFlow] Running VecFlow-GPU baseline evaluation across datasets..."
python3 bench/e14_official_vecflow.py --dataset all
