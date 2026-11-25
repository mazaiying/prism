#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Official Microsoft C++ Filtered-DiskANN (`microsoft/DiskANN` `cpp_main` —
`apps/build_memory_index` + `apps/search_memory_index` with 48 OpenMP threads, WWW'23).
Evaluates `WL1_single_label` (`col2 == label`) across `modbench`, `sift1m`, `gist1m`, `convlog1m`
at Equal-Recall (`Recall@10 >= 0.95`, 7-run trimmed mean of 48-thread C++ OpenMP QPS)
and updates `results/e14_official_baselines.json`.
"""

from __future__ import annotations
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.hybrid_datasets import load_hybrid_dataset
from src.oracle import compute_exact_filtered_gt, compute_filtered_recall_at_k
from bench.e14_official_baselines import make_label_workloads


def write_fbin(path: str, arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    n, d = arr.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<II", n, d))
        arr.tofile(f)


def read_u32_bin(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        data = np.fromfile(f, dtype=np.uint32, count=n * d)
    return data.reshape(n, d).astype(np.int64)


def read_f32_bin(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        data = np.fromfile(f, dtype=np.float32, count=n * d)
    return data.reshape(n, d)


def parse_diskann_qps(stdout: str, L_target: int) -> float:
    for line in stdout.splitlines():
        m = re.match(r"^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", line)
        if m and int(m.group(1)) == L_target:
            return float(m.group(2))
    raise RuntimeError(f"Could not parse L={L_target} from DiskANN output:\n{stdout}")


def main():
    prism_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    bin_dir = os.path.expanduser(
        os.environ.get("DISKANN_BIN_DIR", os.path.join(prism_root, "third_party", "DiskANN", "build", "apps"))
    )
    build_bin = os.path.join(bin_dir, "build_memory_index")
    search_bin = os.path.join(bin_dir, "search_memory_index")
    out_path = os.path.join(prism_root, "results", "e14_official_baselines.json")

    res = json.load(open(out_path)) if os.path.exists(out_path) else {"datasets": {}}
    n_val, n_test = 64, 256
    n_tot = n_val + n_test

    for ds_name in ["modbench", "sift1m", "gist1m", "convlog1m"]:
        print(f"\n=== Official C++ Microsoft Filtered-DiskANN: {ds_name} ===", flush=True)
        ds = load_hybrid_dataset(ds_name, max_queries=n_tot)
        xb = np.ascontiguousarray(ds.base_vectors, dtype=np.float32)
        xq_val = np.ascontiguousarray(ds.query_vectors[:n_val], dtype=np.float32)
        xq_test = np.ascontiguousarray(ds.query_vectors[n_val:n_tot], dtype=np.float32)

        preds_wl1, q_lbl_a, _, _, _ = make_label_workloads(ds, n_tot)
        gi, gd, _ = compute_exact_filtered_gt(ds, preds_wl1, k=10)
        gvi, gvd = gi[:n_val], gd[:n_val]
        gti, gtd = gi[n_val:n_tot], gd[n_val:n_tot]

        tmp_d = tempfile.mkdtemp(prefix=f"da_cpp_{ds_name}_")
        try:
            base_fbin = os.path.join(tmp_d, "base.fbin")
            qval_fbin = os.path.join(tmp_d, "qval.fbin")
            qtst_fbin = os.path.join(tmp_d, "qtst.fbin")
            lbl_file = os.path.join(tmp_d, "base_labels.txt")
            qval_lbl = os.path.join(tmp_d, "qval_labels.txt")
            qtst_lbl = os.path.join(tmp_d, "qtst_labels.txt")
            idx_pref = os.path.join(tmp_d, "idx")
            res_pref = os.path.join(tmp_d, "res")

            write_fbin(base_fbin, xb)
            write_fbin(qval_fbin, xq_val)
            write_fbin(qtst_fbin, xq_test)
            with open(lbl_file, "w") as f:
                f.write("\n".join(str(int(v) + 1) for v in ds.col2) + "\n")
            with open(qval_lbl, "w") as f:
                f.write("\n".join(str(int(v) + 1) for v in q_lbl_a[:n_val]) + "\n")
            with open(qtst_lbl, "w") as f:
                f.write("\n".join(str(int(v) + 1) for v in q_lbl_a[n_val:n_tot]) + "\n")

            t0 = time.perf_counter()
            subprocess.run(
                [
                    build_bin,
                    "--data_type", "float",
                    "--dist_fn", "l2",
                    "--data_path", base_fbin,
                    "--index_path_prefix", idx_pref,
                    "-R", "32",
                    "-L", "64",
                    "--FilteredLbuild", "64",
                    "--alpha", "1.2",
                    "-T", "48",
                    "--label_file", lbl_file,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            build_s = time.perf_counter() - t0

            # Calibrate L on Q_val
            L_grid = [16, 32, 64, 128, 256, 512]
            subprocess.run(
                [
                    search_bin,
                    "--data_type", "float",
                    "--dist_fn", "l2",
                    "--index_path_prefix", idx_pref,
                    "--result_path", res_pref + "_val",
                    "--query_file", qval_fbin,
                    "--query_filters_file", qval_lbl,
                    "-K", "10",
                    "-T", "48",
                    "-L", *[str(x) for x in L_grid],
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            best_L, best_rv = L_grid[-1], 0.0
            for Lv in L_grid:
                pi = read_u32_bin(f"{res_pref}_val_{Lv}_idx_uint32.bin")
                pd = read_f32_bin(f"{res_pref}_val_{Lv}_dists_float.bin")
                rv = compute_filtered_recall_at_k(pi, gvi, gvd, pd, k=10)
                if rv >= best_rv:
                    best_rv, best_L = rv, Lv
                if rv >= 0.95:
                    break

            # Measure 7 runs on Q_test inside a single index load (`-L best_L` repeated or 7 invocations? Wait:
            # Passing `-L best_L best_L ...` (9 times: 2 warmup + 7 timed!) inside ONE `search_memory_index` process
            # avoids index disk load time and measures pure in-memory 48-thread OpenMP search QPS!)
            L_seq = [str(best_L)] * 9
            proc = subprocess.run(
                [
                    search_bin,
                    "--data_type", "float",
                    "--dist_fn", "l2",
                    "--index_path_prefix", idx_pref,
                    "--result_path", res_pref + "_tst",
                    "--query_file", qtst_fbin,
                    "--query_filters_file", qtst_lbl,
                    "-K", "10",
                    "-T", "48",
                    "-L", *L_seq,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            qps_runs = []
            for line in proc.stdout.splitlines():
                m = re.match(r"^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", line)
                if m and int(m.group(1)) == best_L:
                    qps_runs.append(float(m.group(2)))
            timed = sorted(qps_runs[2:])[1:-1] if len(qps_runs) >= 7 else qps_runs
            mean_qps = float(np.mean(timed))
            pi_t = read_u32_bin(f"{res_pref}_tst_{best_L}_idx_uint32.bin")
            pd_t = read_f32_bin(f"{res_pref}_tst_{best_L}_dists_float.bin")
            rec_t = compute_filtered_recall_at_k(pi_t, gti, gtd, pd_t, k=10)

            print(f"  [Official C++ Filtered-DiskANN | {ds_name}] build={build_s:.2f}s, L={best_L}, QPS={mean_qps:.1f}, R@10={rec_t:.4f}", flush=True)
            res = json.load(open(out_path)) if os.path.exists(out_path) else res
            ds_ent = res["datasets"].setdefault(ds_name, {"N": ds.N, "D": ds.D, "build": {}, "workloads": {}})
            ds_ent["build"]["Official_DiskANN"] = {"build_s": round(build_s, 3), "repo": "microsoft/DiskANN C++ (48T OpenMP)"}
            ds_ent["workloads"].setdefault("WL1_single_label", {})["Official_DiskANN"] = {
                "qps": round(mean_qps, 2),
                "recall10": round(rec_t, 4),
                "latency_ms": round((n_test * 1000.0) / mean_qps, 3),
                "L_search": int(best_L),
                "build_s": round(build_s, 3),
            }
            with open(out_path, "w") as f:
                json.dump(res, f, indent=2)
        finally:
            shutil.rmtree(tmp_d, ignore_errors=True)


if __name__ == "__main__":
    main()
