# PRISM: Cross-Modal GPU-Host Vector-SQL Execution for Cloud-Network AIOps at China Unicom

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.x%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![SIGMOD 2027](https://img.shields.io/badge/SIGMOD%202027-Industrial%20Track-purple.svg)](#citation)

Official source code, custom CUDA kernels, baseline harnesses, verification suite, and raw experimental JSON artifacts for the paper:

> **PRISM: Cross-Modal GPU-Host Vector-SQL Execution for Cloud-Network AIOps at China Unicom**  
> Zaiying Ma, Yi Yue, Wencong Yang, and Feng Zhang\*  
> *(SIGMOD 2027 Industrial Track)*

Repository: <https://github.com/mazaiying/prism>

---

## Key Highlights

- **GPU-Host Tile Storage**: Stores compacted 64-bit SLAP relational attributes, per-tile Min-Max / Cat64 headers, and quantized FP16 + ULP-safe residual sketches on GPU VRAM, while keeping full-precision base vectors in pinned host Unified Virtual Addressing (UVA) memory—fitting up to 100M vectors inside a single 16 GB edge GPU.
- **Warp-Compressed Early-Stop**: Packs surviving tuple `(Cid, Row)` pairs into shared-memory rings across clusters to eliminate SIMT branch divergence, and evaluates progressive 4-block ULP ball lower bounds to prune 90%--99% of distance checks and PCIe host fetches without recall loss.
- **Three-Mode Cost Planner & Tombstone-Free Updates**: Automatically selects among M0 (direct filtered host/cluster scan), M1 (fused predicate + ULP sketch stream), and M2 (warp-compacted progressive early-stop), while supporting out-of-distribution `insert_batch` and tombstone-free swap-pop `delete_rows` with zero leaked deleted IDs.
- **Verified Production Results**: Across 9 core benchmarks at strict equal recall ($R@10 \ge 0.95$), PRISM achieves **17.58x** geometric-mean speedup over `cuVS-IVF-Flat`, **39.30x** (**37.12x** on the equal-recall passing subset) over `cuVS-CAGRA`, **10.38x** over `Host-Offload PreFilter`, and **42.80x** over 48-thread `Official ACORN-1`, with 0 lossless audit mismatches across 72/72 E1+E13, 30/30 E2, and 8/8 E9 A100 evaluation cells.

---

## Repository Structure

```text
prism/
├── src/                        # Core PRISM CUDA kernels & Python orchestration
│   ├── __init__.py
│   ├── ext_loader.py           # PyBind11 CUDA JIT extension builder (load_prism_ext)
│   ├── prism_kernels.cu        # Custom CUDA kernels (SLAP filter, ring compaction, 4-block ULP pruning, UVA fetch)
│   ├── prism_engine.py         # PrismEngine index build, search planner (M0/M1/M2), dynamic insert/delete
│   ├── oracle.py               # Exact filtered ground truth & slot-by-slot lossless auditor
│   └── hybrid_datasets.py      # Loaders & SQL workload generators for 11 benchmarks
├── baselines/                  # CPU/GPU baseline wrappers & C++ PyBind11 extensions
│   ├── cpu_graph_ext.cpp
│   ├── filtered_baselines.py
│   └── official_fanns_ext.cpp
├── bench/                      # Benchmark scripts (E0-E17), strict sweeps, and audit tools
│   ├── verify_all_paper_numbers.py
│   ├── run_master_pipeline.sh
│   ├── run_strict_all_pipeline.sh
│   ├── run_full_e14_pipeline.sh
│   └── run_vecflow_all.sh
├── tests/                      # CUDA lossless & dynamic update verification tests
│   ├── test_prism_lossless.py
│   └── test_prism_dynamic_and_stages.py
├── results/                    # 21 verified experimental JSON outputs reported in the paper
├── figures/                    # Paper figure & diagram generation scripts
│   ├── draw_sigmod_industry_figs.py
│   └── plot_prism_experiments.py
├── scripts/                    # Portable environment & figure regeneration helpers
│   ├── env.sh
│   └── reproduce_figures.sh
├── LICENSE
├── requirements.txt
└── .gitignore
```

---

## Quick Verification (No GPU Required)

You can immediately audit all published paper numbers and regenerate all 13 publication figures directly from `results/*.json` on any CPU machine (Linux/macOS):

```bash
python3 -m pip install -r requirements.txt

# 1. Verify all equal-recall assertions, Tables 4 & 7, and E15 metrics from results/*.json
python3 bench/verify_all_paper_numbers.py

# 2. Regenerate all 13 publication figures (PDF + PNG) into figures/generated/
bash scripts/reproduce_figures.sh
```

Generated figures include:
- `fig1_paradigm_comparison.pdf`
- `fig2_aiops_bottleneck_flow.pdf`
- `fig3_ulp_ball_pipeline.pdf`
- `fig4_arch_and_layout.pdf`
- `fig5_matrix_walkthrough.pdf`
- `fig5_e1_main_qps.pdf`
- `fig6_e10_pareto_logloss.pdf`
- `fig6_e2_selectivity.pdf`
- `fig7_e3_e4_skew_and_100m.pdf`
- `fig8_e5_ablation.pdf`
- `fig9_e6_sensitivity.pdf`
- `fig10_e7_e9_cost_and_a100.pdf`
- `fig11_teleops_and_tco.pdf`

---

## GPU Build & Unit Tests

PRISM JIT-compiles `src/prism_kernels.cu` via PyTorch `cpp_extension` (`load_prism_ext` in `src/ext_loader.py`) targeting your visible NVIDIA GPU compute capability (tested on RTX 4080 `sm_89` and A100 `sm_80`).

### Environment Configuration

Configure paths with environment variables or source `scripts/env.sh`:

```bash
source scripts/env.sh
```

Optional environment overrides:
- `PRISM_ROOT`: Repository root (defaults to parent directory of `src/` or `scripts/`)
- `CUDA_HOME`: CUDA toolkit path (defaults to `/usr/local/cuda`)
- `PRISM_DATA_DIR`: Dataset root directory (defaults to `${PRISM_ROOT}/data`)
- `PRISM_ATTR_CACHE`: Cached attribute/embedding directory (defaults to `${PRISM_ROOT}/attr_cache`)
- `CONVLOG_DIR`: Directory containing `distractors_1m.parquet` for `convlog1m` (defaults to `${PRISM_ROOT}/data/convlog1m`)
- `DISKANN_BIN_DIR`: Built DiskANN binaries directory (`build_memory_index`, `build_stitched_index`, `search_memory_index`) for E14/E16
- `PRISM_FIG_OUT`: Output folder for figure scripts (defaults to `figures/generated`)

### Running Adversarial Lossless & Dynamic Tests

On a CUDA machine:

```bash
source scripts/env.sh
pytest tests/ -v
```

- `tests/test_prism_lossless.py`: Verifies IEEE-754 ULP-safe lower bounds and SQL predicate semantics across outlier injections, duplicate vectors, vacuum queries, and uint8/fp32 storage across Regimes 1--3.
- `tests/test_prism_dynamic_and_stages.py`: Verifies tombstone-free `delete_rows`, out-of-distribution `insert_batch` waves, 4-stage timing decomposition, and 100% lossless agreement against exact ground truth.

---

## Dataset Preparation & Full Benchmark Reproduction

`src/hybrid_datasets.py` exposes `load_hybrid_dataset(name, rho_corr=0.75, max_queries=1000, seed=2027)` for 11 benchmarks:

| Dataset Name | Scale (`N x D`) | Expected Files under `${PRISM_DATA_DIR}` |
| :--- | :--- | :--- |
| `teleops1m` | 1,000,000 x 384D | `teleops_raw/{openstack.txt,openssh.txt,hpc.txt,hadoop.txt,zookeeper.txt,apache.txt,linux.txt}` |
| `convlog1m` | 1,000,000 x 1024D | `${CONVLOG_DIR}/distractors_1m.parquet` (or `data/convlog1m/distractors_1m.parquet`) |
| `modbench` | 134,556 x 384D | Local Hugging Face cache of `ucberkeley-dlab/measuring-hate-speech` (`.arrow`) or `attr_cache/modbench_real_384d.npz` |
| `bge1024`, `gist1m`, `msong`, `sift1m`, `deep10m`, `sift10m`, `t2i10m` | 994K--10M x 96D--1024D | `<dataset>/{base.npy, query.npy}` |
| `sift100m` | 100,000,000 x 128D | `sift100m/{base.100M.u8bin, query.npy (or query.public.10K.u8bin)}` |

To re-run the full benchmark pipelines on an RTX 4080 or A100 node:

```bash
# Core pipeline (E0-E13, E15-E17)
bash bench/run_master_pipeline.sh

# Strict equal-recall R95/R99 sweep (4080 or a100)
bash bench/run_strict_all_pipeline.sh 4080

# Official CPU/GPU C++ baselines (DiskANN, ACORN, HNSW, VecFlow)
bash bench/run_full_e14_pipeline.sh
bash bench/run_vecflow_all.sh
```

---

## Experimental Results Inventory (`results/*.json`)

| Result File | Corresponding Paper Experiment / Artifact |
| :--- | :--- |
| `e0_motivation.json` | E0 motivation measurements (selectivity collapse & memory/PCIe wall) |
| `e1_main_evaluation.json` | Main RTX 4080 throughput & recall tables across W1--W4 (R95 & R99) |
| `e2_selectivity_sweep.json` | Selectivity sweep & M0/M1/M2 regime crossover |
| `e3_correlation_skew.json` | Correlation skew robustness |
| `e4_sift100m_scale.json` | SIFT100M build & 16 GB edge scaling evaluation |
| `e5_ablation.json` | Step-by-step architectural & progressive slice ablation |
| `e6_sensitivity.json` | Parameter sensitivity ($B$, Top-$K$, $M$ blocks, quantile clip) |
| `e7_cost_model.json` | Analytical planner accuracy & Oracle comparison |
| `e8_lossless_summary.json` | Cross-dataset exact lossless agreement summary |
| `e9_a100_portability.json` | NVIDIA A100 80 GB portability & SIFT100M A100 scaling |
| `e10_pareto_qps_recall.json` | QPS-vs-Recall Pareto curves & log-lossless verification |
| `e11_time_breakdown.json` | 4-stage query & index-build time breakdown |
| `e12_dynamic_updates.json`, `e12_dynamic_updates_a100.json` | Streaming inserts/deletes & OOD drift resilience (4080 & A100) |
| `e13_modbench_eval.json`, `e13_modbench_eval_a100.json` | Real multi-modal ModBench evaluation (4080 & A100) |
| `e14_official_baselines.json`, `e14_official_vecflow.json` | Official DiskANN, ACORN, HNSW, and VecFlow comparisons |
| `e15_peak_vram_pcie_and_dbms.json` | Peak VRAM, PCIe traffic reduction, & DuckDB 48T SQL baseline |
| `e16_teleops_and_tco.json` | China Unicom TeleOps1M co-location & TCO CAPEX analysis |
| `e17_production_sla_and_audit.json` | Production concurrency QoS, RCA accuracy, & adversarial audit |

---

## Citation

If you find PRISM useful in your research or production deployments, please cite:

```bibtex
@inproceedings{ma2027prism,
  title     = {PRISM: Cross-Modal GPU-Host Vector-SQL Execution for Cloud-Network AIOps at China Unicom},
  author    = {Ma, Zaiying and Yue, Yi and Yang, Wencong and Zhang, Feng},
  booktitle = {Proceedings of the 2027 International Conference on Management of Data (SIGMOD '27 Industrial Track)},
  year      = {2027}
}
```

## License

This project is released under the [MIT License](LICENSE).
