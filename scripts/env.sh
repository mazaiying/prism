#!/usr/bin/env bash
# Portable environment setup for PRISM
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PRISM_ROOT="${PRISM_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"

TORCH_LIB="$(python3 -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))' 2>/dev/null || true)"
if [ -n "${TORCH_LIB}" ]; then
    export LD_LIBRARY_PATH="${TORCH_LIB}:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
else
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

export PYTHONPATH="${PRISM_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 16)}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
