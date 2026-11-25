#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../scripts/env.sh"

cd "${PRISM_ROOT}"
mkdir -p results

echo "[1/6] E0 Motivation & E1 Main Evaluation..."
python3 bench/e0_motivation.py
python3 bench/e1_main_evaluation.py

echo "[2/6] E2-E4 Selectivity, Correlation Skew, and 100M Scale..."
python3 bench/e2_e3_e4_scaling_and_skew.py

echo "[3/6] E5-E7 Ablation, Sensitivity, and Analytical Cost Planner..."
python3 bench/e5_e6_e7_ablation_sens_cost.py

echo "[4/6] E8 Lossless Audit & E9 A100 Portability..."
python3 bench/e8_audit_and_a100.py
python3 bench/e9_a100_full_eval.py

echo "[5/6] E10-E13 Pareto, Time Breakdown, Dynamic Updates, and ModBench..."
python3 bench/e10_pareto_and_breakdown.py
python3 bench/e12_dynamic_and_modbench.py --only all

echo "[6/6] E15-E17 DBMS, TeleOps1M Co-Location/TCO, and Concurrency SLA..."
python3 bench/e15_peak_vram_pcie_and_dbms.py
python3 bench/e16_teleops_and_tco.py --mode "${PRISM_GPU_MODE:-4080}"
python3 bench/e17_production_concurrency_and_adversarial.py

python3 bench/verify_all_paper_numbers.py
