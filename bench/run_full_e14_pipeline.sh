#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../scripts/env.sh"

cd "${PRISM_ROOT}"

echo "[E14 Official Baselines] Running DiskANN/ACORN/HNSW + PRISM dual-GPU + formatting..."
python3 bench/e14_official_baselines.py
python3 bench/e14_official_diskann_cpp.py
python3 bench/e14_prism_both_gpus.py --gpu_tag "${PRISM_GPU_TAG:-PRISM}"
python3 bench/verify_and_format_e14.py
