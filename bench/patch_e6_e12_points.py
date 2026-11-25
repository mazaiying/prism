#!/usr/bin/env python3
"""
Targeted hardware refresh on Server 167:
  1. Refresh `modbench` in `e12_dynamic_updates.json` (GPU 0) or `e12_dynamic_updates_a100.json` (GPU 1)
     at R95-calibrated `nprobe = 192` so all 4 datasets in E12 satisfy Recall@10 >= 0.95.
  2. Add `convlog1m` (1024D, M in {1, 2, 4, 8}) to `e6_sensitivity.json` (`m_blocks`) on GPU 0.
"""
import argparse
import gc
import json
import os
import sys
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.hybrid_datasets import load_hybrid_dataset, generate_sql_workload
from src.prism_engine import PrismEngine
from bench.e12_dynamic_and_modbench import run_e12_dynamic_updates


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def timed_ms(fn, warmup=2, repeats=7):
    for _ in range(warmup):
        fn()
    lats = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        lats.append((time.perf_counter() - t0) * 1000.0)
    s = sorted(lats)
    return sum(s[1:-1]) / len(s[1:-1])


def refresh_e12_modbench(json_path: str):
    if os.path.exists(json_path):
        d = json.load(open(json_path))
        if "modbench" in d.get("datasets", {}):
            del d["datasets"]["modbench"]
            with open(json_path, "w") as f:
                json.dump(d, f, indent=2)
    run_e12_dynamic_updates(json_path)


def add_convlog1m_mblocks(e6_path: str):
    d = json.load(open(e6_path))
    ds = load_hybrid_dataset("convlog1m", max_queries=256)
    q_256 = ds.query_vectors[:256]
    p_256 = generate_sql_workload(ds, "W3_compound", num_queries=256, selectivity=0.05)
    m_rows = []
    for mv in [1, 2, 4, 8]:
        em = PrismEngine(nlist=1024, M_blocks=mv, use_sign_bit=True, sketch_mode=2, host_pinned_exact=True)
        b_m = em.build(ds, cache_kmeans=True)
        p_ms = timed_ms(lambda: em.search(q_256, p_256, k=10, nprobe=32, regime_override=2))
        _, _, tel_m = em.search(q_256, p_256, k=10, nprobe=32, regime_override=2, record_telemetry=True)
        row = {
            "M_blocks": mv,
            "qps": (256 * 1000.0) / p_ms,
            "vram_mb": b_m["total_gpu_vram_mb"],
            "sketch_prune_pct": tel_m["overall_sketch_prune_pct"],
        }
        print("  [ConvLog1M M sweep]", row, flush=True)
        m_rows.append(row)
        del em
        free_gpu()
    d["m_blocks"]["convlog1m"] = m_rows
    with open(e6_path, "w") as f:
        json.dump(d, f, indent=2)
    print(f"[Saved updated E6 JSON] {e6_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gpu0", "gpu1"], required=True)
    args = ap.parse_args()
    res_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    if args.mode == "gpu0":
        refresh_e12_modbench(os.path.join(res_dir, "e12_dynamic_updates.json"))
        add_convlog1m_mblocks(os.path.join(res_dir, "e6_sensitivity.json"))
    else:
        refresh_e12_modbench(os.path.join(res_dir, "e12_dynamic_updates_a100.json"))
