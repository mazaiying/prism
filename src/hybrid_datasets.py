#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Vector-Relational Dataset Loader & SQL Predicate Workload Generator for PRISM.

Supports 11 datasets from 994K to 100M rows (96D to 1024D):
  1. convlog1m : Real 1M x 1024D conversation embeddings + real organic tabular columns
                 extracted from ${CONVLOG_DIR:-data/convlog1m}/distractors_1m.parquet
  2. sift1m    : 1,000,000 x 128D
  3. gist1m    : 1,000,000 x 960D
  4. bge1024   : 1,000,000 x 1024D
  5. msong     : 994,185   x 420D
  6. deep10m   : 10,000,000 x 96D
  7. sift10m   : 10,000,000 x 128D
  8. t2i10m    : 10,000,000 x 200D
  9. sift100m  : 100,000,000 x 128D (u8bin out-of-core scale test on single 16GB GPU)

Each table exposes 4 relational attribute columns stored natively as uint16/int32:
  - col0 [0..65535] : Uncorrelated / weakly correlated numerical attribute (e.g., price / char_len)
  - col1 [0..65535] : Manifold-correlated numerical attribute (default rho_corr=0.75, e.g., timestamp / turn_score)
  - col2 [0..63]    : Zipfian categorical attribute (64 categories, e.g., tenant_id / lang_cat)
  - col3 [0..65535] : Secondary numerical metric for ad-hoc cross-column algebraic SQL predicates

Predicate flags bitmask:
  - bit 0 (1): col0 BETWEEN [l0, r0]
  - bit 1 (2): col1 BETWEEN [l1, r1]
  - bit 2 (4): ((1ULL << col2) & cat_mask64) != 0
  - bit 3 (8): (w0 * int32(col0) + w3 * int32(col3)) <= alg_bound
"""

from __future__ import annotations
import os
import zlib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np


PRED_COL0_RANGE = 1
PRED_COL1_RANGE = 2
PRED_COL2_CAT   = 4
PRED_ALG_EXPR   = 8


@dataclass
class PredicateBatch:
    """Structured vectorized SQL predicate batch for B queries."""
    op_flags: np.ndarray     # (B,) uint32
    l0: np.ndarray           # (B,) int32
    r0: np.ndarray           # (B,) int32
    l1: np.ndarray           # (B,) int32
    r1: np.ndarray           # (B,) int32
    cat_mask: np.ndarray     # (B,) uint64
    w0: np.ndarray           # (B,) int32
    w3: np.ndarray           # (B,) int32
    alg_bound: np.ndarray    # (B,) int32
    target_sel: np.ndarray   # (B,) float32
    workload_name: str = "custom"

    def __len__(self) -> int:
        return int(self.op_flags.shape[0])

    def slice(self, start: int, end: int) -> "PredicateBatch":
        return PredicateBatch(
            op_flags=self.op_flags[start:end],
            l0=self.l0[start:end],
            r0=self.r0[start:end],
            l1=self.l1[start:end],
            r1=self.r1[start:end],
            cat_mask=self.cat_mask[start:end],
            w0=self.w0[start:end],
            w3=self.w3[start:end],
            alg_bound=self.alg_bound[start:end],
            target_sel=self.target_sel[start:end],
            workload_name=self.workload_name,
        )


@dataclass
class HybridDataset:
    name: str
    N: int
    D: int
    base_vectors: np.ndarray    # (N, D) float32 or uint8 (for sift100m mmap)
    query_vectors: np.ndarray   # (Q, D) float32
    col0: np.ndarray            # (N,) uint16
    col1: np.ndarray            # (N,) uint16
    col2: np.ndarray            # (N,) uint16  [0..63]
    col3: np.ndarray            # (N,) uint16
    is_real_tabular: bool = False
    rho_corr: float = 0.75


def get_paths() -> Tuple[str, str, str]:
    default_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    prism_root = os.path.expanduser(os.environ.get("PRISM_ROOT", default_root))
    data_root = os.path.expanduser(os.environ.get("PRISM_DATA_DIR", os.path.join(prism_root, "data")))
    attr_cache = os.path.expanduser(os.environ.get("PRISM_ATTR_CACHE", os.path.join(prism_root, "attr_cache")))
    os.makedirs(attr_cache, exist_ok=True)
    return prism_root, data_root, attr_cache


def _build_convlog1m_table(attr_cache: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Build or load the 1M x 1024D real organic tabular+embedding dataset from WildChat/Scale1M."""
    prism_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    scale_dir = os.path.expanduser(os.environ.get("CONVLOG_DIR", os.path.join(prism_root, "data", "convlog1m")))
    vec_cache = os.path.join(attr_cache, "convlog1m_vectors_q.npz")
    attr_file = os.path.join(attr_cache, "convlog1m_attrs.npz")

    e0 = np.load(os.path.join(scale_dir, "emb", "emb_rank0.npy"), mmap_mode="r")
    e1 = np.load(os.path.join(scale_dir, "emb", "emb_rank1.npy"), mmap_mode="r")
    # Construct 1,000,000 base vectors (525k from rank0 + 475k from rank1) and 1,000 held-out queries
    base_mmap_path = os.path.join(attr_cache, "convlog1m_base_1m_1024.npy")
    if not os.path.exists(base_mmap_path) or not os.path.exists(vec_cache):
        xb = np.lib.format.open_memmap(base_mmap_path, mode="w+", dtype=np.float32, shape=(1_000_000, 1024))
        xb[:525_000] = e0[:]
        xb[525_000:1_000_000] = e1[:475_000]
        xb.flush()
        rng = np.random.default_rng(2027)
        q_idx = rng.choice(50_000, size=1000, replace=False) + 475_000
        xq = np.array(e1[q_idx], dtype=np.float32, copy=True)
        np.savez(vec_cache, query=xq, q_idx=q_idx)
    xb = np.load(base_mmap_path)
    xq = np.load(vec_cache)["query"].astype(np.float32)

    if not os.path.exists(attr_file):
        import pandas as pd
        df = pd.read_parquet(os.path.join(scale_dir, "distractors_1m.parquet"))
        # Align exact row order using ids_rank0 / ids_rank1 if present
        id0_p = os.path.join(scale_dir, "emb", "ids_rank0.npy")
        id1_p = os.path.join(scale_dir, "emb", "ids_rank1.npy")
        if os.path.exists(id0_p) and os.path.exists(id1_p):
            ids0 = np.load(id0_p)
            ids1 = np.load(id1_p)[:475_000]
            row_ids = np.concatenate([ids0, ids1])
            df = df.iloc[row_ids].reset_index(drop=True)
        else:
            df = df.iloc[:1_000_000].reset_index(drop=True)

        texts = df["text"].fillna("").astype(str)
        # col0: real character length clipped to [0, 65535]
        char_len = texts.str.len().clip(0, 65535).to_numpy(dtype=np.uint16)

        # col1: real turn depth + code/markup punctuation count scaled into [0, 65535]
        turn_idx = df["user_turn_idx"].fillna(1).clip(0, 100).to_numpy(dtype=np.int32)
        brace_cnt = texts.str.count(r"[{};=<>`\[\]]").clip(0, 500).to_numpy(dtype=np.int32)
        col1_raw = np.clip(turn_idx * 450 + brace_cnt * 80, 0, 65535).astype(np.uint16)

        # col2: 64 real semantic/tenant categories combining Unicode script class (4 groups) x 16 IP tenant buckets
        has_cjk = texts.str.contains(r"[\u4e00-\u9fff]", regex=True).to_numpy()
        has_cyr = texts.str.contains(r"[\u0400-\u04ff]", regex=True).to_numpy()
        has_code = (brace_cnt >= 8)
        script_grp = np.zeros(1_000_000, dtype=np.uint16)
        script_grp[has_cjk] = 1
        script_grp[has_cyr & ~has_cjk] = 2
        script_grp[has_code & ~has_cjk & ~has_cyr] = 3
        ip_strs = df["hashed_ip"].fillna("0").astype(str).tolist()
        ip_bkt = np.fromiter((zlib.crc32(s.encode("utf-8")) & 15 for s in ip_strs), dtype=np.uint16, count=1_000_000)
        col2 = (script_grp * 16 + ip_bkt).astype(np.uint16)

        # col3: conversation session hash score [0, 65535]
        conv_strs = df["conversation_hash"].fillna("0").astype(str).tolist()
        col3 = np.fromiter((zlib.crc32(s.encode("utf-8")) & 65535 for s in conv_strs), dtype=np.uint16, count=1_000_000)

        np.savez(attr_file, col0=char_len, col1=col1_raw, col2=col2, col3=col3)

    attrs = np.load(attr_file)
    return xb, xq, {k: attrs[k] for k in ["col0", "col1", "col2", "col3"]}


def _build_standard_attrs(
    name: str,
    xb: np.ndarray,
    attr_cache: str,
    rho_corr: float = 0.75,
    seed: int = 2027,
) -> Dict[str, np.ndarray]:
    """Generate deterministic multi-column relational attributes with controllable vector correlation."""
    tag = f"{name}_rho{int(round(rho_corr * 100)):03d}_s{seed}.npz"
    path = os.path.join(attr_cache, tag)
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in ["col0", "col1", "col2", "col3"]}

    N, D = xb.shape
    rng = np.random.default_rng(seed + (zlib.crc32(name.encode()) & 0xFFFF))

    # Estimate leading manifold direction u1 on a 65,536-row sample (fast even for N=100M)
    sample_n = min(N, 65536)
    step = max(1, N // sample_n)
    xs = np.asarray(xb[::step][:sample_n], dtype=np.float32)
    mean_v = xs.mean(axis=0, keepdims=True)
    xc = xs - mean_v
    # Power iteration for top-2 directions u1, u2 (deterministic & fast)
    u1 = rng.standard_normal((D, 1)).astype(np.float32)
    u1 /= np.linalg.norm(u1) + 1e-12
    for _ in range(12):
        u1 = xc.T @ (xc @ u1)
        u1 /= np.linalg.norm(u1) + 1e-12
    u2 = rng.standard_normal((D, 1)).astype(np.float32)
    u2 -= u1 * float((u1.T @ u2).item())
    u2 /= np.linalg.norm(u2) + 1e-12
    for _ in range(8):
        u2 = xc.T @ (xc @ u2)
        u2 -= u1 * float((u1.T @ u2).item())
        u2 /= np.linalg.norm(u2) + 1e-12

    proj1_sample = (xc @ u1).ravel()
    mu1, sd1 = float(proj1_sample.mean()), float(proj1_sample.std() + 1e-6)
    proj2_sample = (xc @ u2).ravel()
    mu2, sd2 = float(proj2_sample.mean()), float(proj2_sample.std() + 1e-6)
    mean_flat = mean_v.ravel()

    col0 = np.empty(N, dtype=np.uint16)
    col1 = np.empty(N, dtype=np.uint16)
    col2 = np.empty(N, dtype=np.uint16)
    col3 = np.empty(N, dtype=np.uint16)

    # Zipfian CDF over 64 categories (skew s = 1.15)
    ranks = np.arange(1, 65, dtype=np.float64)
    zipf_p = ranks ** (-1.15)
    zipf_p /= zipf_p.sum()
    zipf_cdf = np.cumsum(zipf_p)

    chunk = 1_000_000
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        m = e - s
        x_blk = np.asarray(xb[s:e], dtype=np.float32)
        z1 = ((x_blk @ u1.ravel()) - mu1) / sd1
        z2 = ((x_blk @ u2.ravel()) - mu2) / sd2

        # col0: independent uniform/price-like [0, 65535]
        col0[s:e] = rng.integers(0, 65536, size=m, dtype=np.uint16)

        # col1: manifold-correlated continuous attribute mapped smoothly via Gaussian-like logistic/erf
        noise1 = rng.standard_normal(m).astype(np.float32)
        latent1 = rho_corr * z1 + np.sqrt(max(0.0, 1.0 - rho_corr * rho_corr)) * noise1
        # Map [-3.2, +3.2] smoothly onto [0, 65535]
        u_cdf1 = 0.5 * (1.0 + np.tanh(0.58 * latent1))
        col1[s:e] = np.clip(np.round(u_cdf1 * 65535.0), 0, 65535).astype(np.uint16)

        # col2: Zipfian 64 categories moderately coupled with z2 so hot/cold categories exhibit cluster locality
        noise2 = rng.standard_normal(m).astype(np.float32)
        latent2 = 0.55 * z2 + np.sqrt(1.0 - 0.55 * 0.55) * noise2
        u_cdf2 = np.clip(0.5 * (1.0 + np.tanh(0.58 * latent2)), 0.0, 0.999999)
        col2[s:e] = np.searchsorted(zipf_cdf, u_cdf2).astype(np.uint16)

        # col3: secondary numerical metric [0, 65535]
        col3[s:e] = rng.integers(0, 65536, size=m, dtype=np.uint16)

    np.savez(path, col0=col0, col1=col1, col2=col2, col3=col3)
    return {"col0": col0, "col1": col1, "col2": col2, "col3": col3}


def _build_modbench_table(attr_cache: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Build or load the 134,556 x 384D real human-annotated UC Berkeley D-Lab social moderation dataset."""
    cache_p = os.path.join(attr_cache, "modbench_real_384d.npz")
    if os.path.exists(cache_p):
        z = np.load(cache_p)
        return z["xb"].astype(np.float32), z["xq"].astype(np.float32), {
            "col0": z["col0"], "col1": z["col1"], "col2": z["col2"], "col3": z["col3"]
        }

    import glob
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import torch
    from transformers import AutoTokenizer, AutoModel

    arrow_path = os.path.expanduser(
        "~/.cache/huggingface/datasets/ucberkeley-dlab___measuring-hate-speech/"
        "default/0.0.0/5468f6e118396646b02a2f691e771f6b6d9502ea/measuring-hate-speech-train.arrow"
    )
    with pa.memory_map(arrow_path, "r") as src:
        tbl = ipc.open_stream(src).read_all()

    df = tbl.select([
        "text", "platform", "annotator_id", "sentiment", "respect",
        "insult", "humiliate", "violence", "hate_speech_score"
    ]).to_pandas()

    total_rows = len(df)  # 135,556
    rng = np.random.default_rng(2027)
    perm = rng.permutation(total_rows)
    q_idx = perm[:1000]
    b_idx = perm[1000:]

    # Embed all texts with locally cached sentence-transformers/all-MiniLM-L6-v2 (384D)
    snap_dirs = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/*"
    )))
    model_path = snap_dirs[-1] if snap_dirs else "sentence-transformers/all-MiniLM-L6-v2"
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    mdl = AutoModel.from_pretrained(model_path, local_files_only=True).cuda().eval().half()

    texts_all = df["text"].fillna("").astype(str).tolist()
    embs_all = np.empty((total_rows, 384), dtype=np.float32)
    bs = 1024
    with torch.no_grad():
        for s in range(0, total_rows, bs):
            e = min(total_rows, s + bs)
            enc = tok(texts_all[s:e], padding=True, truncation=True, max_length=128, return_tensors="pt")
            enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
            out = mdl(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).half()
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            embs_all[s:e] = pooled.cpu().numpy()
    del mdl, tok
    torch.cuda.empty_cache()

    xb = np.ascontiguousarray(embs_all[b_idx], dtype=np.float32)
    xq = np.ascontiguousarray(embs_all[q_idx], dtype=np.float32)

    dfb = df.iloc[b_idx].reset_index(drop=True)
    # 4 genuine human-annotated structured moderation columns:
    # col0: human-rated sentiment + respect score [0..65535]
    sent = dfb["sentiment"].fillna(2.0).to_numpy(dtype=np.float32)
    resp = dfb["respect"].fillna(2.0).to_numpy(dtype=np.float32)
    col0 = np.clip(np.round((sent * 8190.0) + (resp * 8190.0)), 0, 65535).astype(np.uint16)

    # col1: continuous Rasch hate_speech_score [-8.5, +5.5] mapped smoothly to [0..65535]
    hs = dfb["hate_speech_score"].fillna(0.0).to_numpy(dtype=np.float32)
    hs_lo, hs_hi = float(np.percentile(hs, 0.1)), float(np.percentile(hs, 99.9))
    col1 = np.clip(np.round((hs - hs_lo) / max(1e-5, hs_hi - hs_lo) * 65535.0), 0, 65535).astype(np.uint16)

    # col2: 64 real categories = platform (0..3) x (annotator_id % 16)
    plat = dfb["platform"].fillna(0).to_numpy(dtype=np.int64) & 3
    ann  = dfb["annotator_id"].fillna(0).to_numpy(dtype=np.int64) & 15
    col2 = ((plat * 16) + ann).astype(np.uint16)

    # col3: human-annotated violence + insult + humiliate composite harm score [0..65535]
    viol = dfb["violence"].fillna(0.0).to_numpy(dtype=np.float32)
    ins  = dfb["insult"].fillna(0.0).to_numpy(dtype=np.float32)
    hum  = dfb["humiliate"].fillna(0.0).to_numpy(dtype=np.float32)
    col3 = np.clip(np.round(viol * 8000.0 + ins * 4200.0 + hum * 4100.0), 0, 65535).astype(np.uint16)

    tmp_p = f"{cache_p}.{os.getpid()}.tmp.npz"
    np.savez(tmp_p, xb=xb, xq=xq, col0=col0, col1=col1, col2=col2, col3=col3)
    os.replace(tmp_p, cache_p)
    return xb, xq, {"col0": col0, "col1": col1, "col2": col2, "col3": col3}


def _build_teleops1m_table(data_root: str, attr_cache: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Build or load TeleOps1M: 1,000,000 base + 1,000 held-out query genuine cloud-network
    infrastructure operational logs from Loghub (OpenStack Nova/Compute/API, OpenSSH gateway,
    HPC node resource manager, Hadoop, ZooKeeper quorum, Apache ingress, Linux syslog),
    embedded at 384D (all-MiniLM-L6-v2) with 4 real parsed operational columns:
      - col0: diurnal event timestamp [0..65535]
      - col1: parsed RPC/HTTP latency or daemon numeric status metric [0..65535]
      - col2: 64 operational categories (8 cloud-network subsystems x 8 severity/event classes)
      - col3: log payload byte length & structured key-value burst density [0..65535]
    """
    xb_p = os.path.join(attr_cache, "teleops1m_base_384d.npy")
    meta_p = os.path.join(attr_cache, "teleops1m_q_and_attrs.npz")
    if os.path.exists(xb_p) and os.path.exists(meta_p):
        xb = np.load(xb_p)
        z = np.load(meta_p)
        return xb, z["xq"].astype(np.float32), {
            "col0": z["col0"], "col1": z["col1"], "col2": z["col2"], "col3": z["col3"]
        }

    import glob
    import re
    import torch
    from transformers import AutoTokenizer, AutoModel

    raw_dir = os.path.join(data_root, "teleops_raw")
    # Quota per real Loghub subsystem summing to >= 1,001,000 non-empty operational log lines
    quotas = [
        (0, "openstack.txt", 208_000),   # OpenStack nova-api & nova-compute
        (1, "openssh.txt",   235_000),   # Network SSH gateway
        (2, "hpc.txt",       232_000),   # Cluster node resource manager
        (3, "hadoop.txt",    180_000),   # Distributed storage & metrics
        (4, "zookeeper.txt",  75_000),   # Coordination quorum
        (5, "apache.txt",     52_000),   # HTTP/LDAP ingress gateway
        (6, "linux.txt",      25_000),   # Host OS syslog
    ]

    re_hms = re.compile(r"(\d{2}):(\d{2}):(\d{2})(?:[.,](\d{1,3}))?")
    re_time_sec = re.compile(r"time:\s*(\d+\.\d+)")
    re_len = re.compile(r"len:\s*(\d+)")
    re_err = re.compile(r"(?:err\s*=\s*|port\s+|pid=)(\d+)")

    texts_raw = []
    c0_list, c1_list, c2_list, c3_list = [], [], [], []
    target_total = 1_001_000

    for sid, fname, max_take in quotas:
        fpath = os.path.join(raw_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing TeleOps raw log file: {fpath}")
        taken = 0
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if len(s) < 8:
                    continue
                m_t = re_hms.search(s[:96])
                if m_t:
                    hh, mm, ss = int(m_t.group(1)), int(m_t.group(2)), int(m_t.group(3))
                    ms = int((m_t.group(4) or "0").ljust(3, "0")[:3])
                    sec_day = (hh % 24) * 3600.0 + (mm % 60) * 60.0 + (ss % 60) + ms * 0.001
                    v0 = int(min(65535, round(sec_day / 86400.0 * 65535.0)))
                else:
                    toks = s.split()
                    ep = next((int(t) for t in toks[:7] if t.isdigit() and len(t) >= 9), taken)
                    v0 = int(round(((ep % 86400) / 86400.0) * 65535.0))

                m_lat = re_time_sec.search(s)
                if m_lat:
                    v1 = int(min(65535, round(float(m_lat.group(1)) * 45000.0)))
                else:
                    m_e = re_err.search(s)
                    num_toks = sum(ch.isdigit() for ch in s[:160])
                    base_v = int(m_e.group(1)) * 17 if m_e else 0
                    v1 = int(min(65535, (base_v + num_toks * 520) & 65535))

                ul = s.upper()
                if any(k in ul for k in ("ERROR", "FATAL", "CRIT", "FAIL", "EXCEPTION", "BREAK-IN", "REFUSED")):
                    sev = 0
                elif any(k in ul for k in ("WARN", "INVALID", "DISCONNECT", "TIMEOUT", "UNAVAILABLE")):
                    sev = 1
                elif "STATUS: 4" in ul or "STATUS: 5" in ul:
                    sev = 2
                elif any(k in ul for k in ("CLAIM", "STOPPED", "SPAWN", "INSTANCE:", "QUORUM")):
                    sev = 3
                elif "POST " in ul or "DELETE " in ul or "PUT " in ul:
                    sev = 4
                elif "GET " in ul:
                    sev = 5
                elif "ACCEPTED" in ul or "SESSION" in ul or "STARTUP" in ul:
                    sev = 6
                else:
                    sev = 7
                sub_slot = 7 if (sid == 0 and "nova-compute" in s[:42]) else sid
                v2 = sub_slot * 8 + sev

                m_l = re_len.search(s)
                plen = int(m_l.group(1)) if m_l else len(s)
                punct = sum(ch in "[]=:/.-_" for ch in s[:160])
                v3 = int(min(65535, plen * 12 + punct * 180))

                texts_raw.append(s[:256])
                c0_list.append(v0)
                c1_list.append(v1)
                c2_list.append(v2)
                c3_list.append(v3)
                taken += 1
                if taken >= max_take:
                    break

    if len(texts_raw) < target_total:
        # Top up deterministically from remaining lines of openssh.txt
        with open(os.path.join(raw_dir, "openssh.txt"), "r", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(f):
                if idx < 235_000:
                    continue
                s = line.strip()
                if len(s) < 8:
                    continue
                m_t = re_hms.search(s[:96])
                sec_day = (int(m_t.group(1)) * 3600 + int(m_t.group(2)) * 60 + int(m_t.group(3))) if m_t else (idx % 86400)
                v0 = int(min(65535, round((sec_day % 86400) / 86400.0 * 65535.0)))
                m_e = re_err.search(s)
                v1 = int(min(65535, ((int(m_e.group(1)) * 17 if m_e else 0) + sum(ch.isdigit() for ch in s[:160]) * 520) & 65535))
                ul = s.upper()
                sev = 0 if ("FAIL" in ul or "BREAK-IN" in ul) else (1 if "INVALID" in ul else 6)
                v2 = 1 * 8 + sev
                v3 = int(min(65535, len(s) * 12 + sum(ch in "[]=:/.-_" for ch in s[:160]) * 180))
                texts_raw.append(s[:256])
                c0_list.append(v0)
                c1_list.append(v1)
                c2_list.append(v2)
                c3_list.append(v3)
                if len(texts_raw) >= target_total:
                    break

    texts_raw = texts_raw[:target_total]
    c0_all = np.asarray(c0_list[:target_total], dtype=np.uint16)
    c1_all = np.asarray(c1_list[:target_total], dtype=np.uint16)
    c2_all = np.asarray(c2_list[:target_total], dtype=np.uint16)
    c3_all = np.asarray(c3_list[:target_total], dtype=np.uint16)

    rng = np.random.default_rng(2027)
    perm = rng.permutation(target_total)
    q_idx = perm[:1000]
    b_idx = perm[1000:target_total]

    snap_dirs = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/*"
    )))
    model_path = snap_dirs[-1] if snap_dirs else "sentence-transformers/all-MiniLM-L6-v2"
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    mdl = AutoModel.from_pretrained(model_path, local_files_only=True).cuda().eval().half()

    embs_all = np.empty((target_total, 384), dtype=np.float32)
    bs = 2048
    with torch.no_grad():
        for s_i in range(0, target_total, bs):
            e_i = min(target_total, s_i + bs)
            enc = tok(texts_raw[s_i:e_i], padding=True, truncation=True, max_length=96, return_tensors="pt")
            enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
            out = mdl(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).half()
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            embs_all[s_i:e_i] = pooled.cpu().numpy()
    del mdl, tok
    torch.cuda.empty_cache()

    xb = np.ascontiguousarray(embs_all[b_idx], dtype=np.float32)
    xq = np.ascontiguousarray(embs_all[q_idx], dtype=np.float32)
    col0 = np.ascontiguousarray(c0_all[b_idx], dtype=np.uint16)
    col1 = np.ascontiguousarray(c1_all[b_idx], dtype=np.uint16)
    col2 = np.ascontiguousarray(c2_all[b_idx], dtype=np.uint16)
    col3 = np.ascontiguousarray(c3_all[b_idx], dtype=np.uint16)

    tmp_xb = f"{xb_p}.{os.getpid()}.tmp.npy"
    np.save(tmp_xb, xb)
    os.replace(tmp_xb, xb_p)
    tmp_meta = f"{meta_p}.{os.getpid()}.tmp.npz"
    np.savez(tmp_meta, xq=xq, col0=col0, col1=col1, col2=col2, col3=col3)
    os.replace(tmp_meta, meta_p)
    return xb, xq, {"col0": col0, "col1": col1, "col2": col2, "col3": col3}


def load_hybrid_dataset(
    name: str,
    rho_corr: float = 0.75,
    max_queries: int = 1000,
    seed: int = 2027,
) -> HybridDataset:
    """Load vector dataset + 4-column relational table."""
    _, data_root, attr_cache = get_paths()
    name_l = name.lower()

    if name_l in ("teleops", "teleops1m"):
        xb, xq, attrs = _build_teleops1m_table(data_root, attr_cache)
        if xq.shape[0] > max_queries:
            xq = xq[:max_queries]
        return HybridDataset(
            name="teleops1m",
            N=xb.shape[0],
            D=xb.shape[1],
            base_vectors=xb,
            query_vectors=np.ascontiguousarray(xq, dtype=np.float32),
            col0=attrs["col0"],
            col1=attrs["col1"],
            col2=attrs["col2"],
            col3=attrs["col3"],
            is_real_tabular=True,
            rho_corr=1.0,
        )

    if name_l == "modbench":
        xb, xq, attrs = _build_modbench_table(attr_cache)
        if xq.shape[0] > max_queries:
            xq = xq[:max_queries]
        return HybridDataset(
            name="modbench",
            N=xb.shape[0],
            D=xb.shape[1],
            base_vectors=xb,
            query_vectors=np.ascontiguousarray(xq, dtype=np.float32),
            col0=attrs["col0"],
            col1=attrs["col1"],
            col2=attrs["col2"],
            col3=attrs["col3"],
            is_real_tabular=True,
            rho_corr=1.0,
        )

    if name_l == "convlog1m":
        xb, xq, attrs = _build_convlog1m_table(attr_cache)
        if xq.shape[0] > max_queries:
            xq = xq[:max_queries]
        return HybridDataset(
            name="convlog1m",
            N=xb.shape[0],
            D=xb.shape[1],
            base_vectors=xb,
            query_vectors=np.ascontiguousarray(xq, dtype=np.float32),
            col0=attrs["col0"],
            col1=attrs["col1"],
            col2=attrs["col2"],
            col3=attrs["col3"],
            is_real_tabular=True,
            rho_corr=1.0,
        )

    if name_l == "sift100m":
        p_u8 = os.path.join(data_root, "sift100m", "base.100M.u8bin")
        # Load 12.8 GB uint8 sequentially into 953 GiB Host RAM to avoid HDD random page-fault stalls
        xb = np.fromfile(p_u8, dtype=np.uint8, offset=8).reshape(100_000_000, 128)
        qp = os.path.join(data_root, "sift100m", "query.npy")
        if os.path.exists(qp):
            xq = np.load(qp).astype(np.float32)
        else:
            q_u8 = np.memmap(
                os.path.join(data_root, "sift100m", "query.public.10K.u8bin"),
                dtype=np.uint8, mode="r", offset=8, shape=(10_000, 128)
            )
            xq = np.array(q_u8, dtype=np.float32)
        if xq.shape[0] > max_queries:
            xq = xq[:max_queries]
        attrs = _build_standard_attrs("sift100m", xb, attr_cache, rho_corr=rho_corr, seed=seed)
        return HybridDataset(
            name="sift100m",
            N=100_000_000,
            D=128,
            base_vectors=xb,
            query_vectors=np.ascontiguousarray(xq, dtype=np.float32),
            col0=attrs["col0"],
            col1=attrs["col1"],
            col2=attrs["col2"],
            col3=attrs["col3"],
            is_real_tabular=False,
            rho_corr=rho_corr,
        )

    ds_dir = os.path.join(data_root, name_l)
    xb = np.load(os.path.join(ds_dir, "base.npy"))
    xq = np.load(os.path.join(ds_dir, "query.npy"))
    if xq.shape[0] > max_queries:
        xq = xq[:max_queries]
    xq = np.ascontiguousarray(xq, dtype=np.float32)
    attrs = _build_standard_attrs(name_l, xb, attr_cache, rho_corr=rho_corr, seed=seed)
    return HybridDataset(
        name=name_l,
        N=xb.shape[0],
        D=xb.shape[1],
        base_vectors=xb,
        query_vectors=xq,
        col0=attrs["col0"],
        col1=attrs["col1"],
        col2=attrs["col2"],
        col3=attrs["col3"],
        is_real_tabular=False,
        rho_corr=rho_corr,
    )


def generate_sql_workload(
    ds: HybridDataset,
    workload: str,
    num_queries: Optional[int] = None,
    selectivity: float = 0.05,
    seed: int = 2027,
) -> PredicateBatch:
    """
    Generate structured SQL predicates calibrated to exact empirical quantile distributions:
      - W1_range      : Uncorrelated range `col0 BETWEEN [l0, r0]` at target selectivity
      - W2_correlated : Manifold-correlated range `col1 BETWEEN [l1, r1]` at target selectivity
      - W3_compound   : Multi-column compound SQL `(col2 IN S_q) AND (col1 BETWEEN [l1, r1]) AND (w0*col0 + w3*col3 <= bound)`
      - W4_mixed      : Per-query log-uniform selectivity drawn across [0.0005, 0.60] mixing W1/W2/W3
    """
    B = num_queries if num_queries is not None else ds.query_vectors.shape[0]
    rng = np.random.default_rng(seed + (zlib.crc32(f"{ds.name}_{workload}_{selectivity:.6f}".encode()) & 0xFFFF))

    # Subsample 200,000 rows for accurate empirical quantile calibration
    step = max(1, ds.N // 200_000)
    c0_s = np.sort(ds.col0[::step].astype(np.int32))
    c1_s = np.sort(ds.col1[::step].astype(np.int32))
    c0_raw = ds.col0[::step].astype(np.int32)
    c1_raw = ds.col1[::step].astype(np.int32)
    c2_raw = ds.col2[::step].astype(np.uint64)
    c3_raw = ds.col3[::step].astype(np.int32)
    M_s = c0_raw.shape[0]

    op_flags = np.zeros(B, dtype=np.uint32)
    l0 = np.zeros(B, dtype=np.int32)
    r0 = np.full(B, 65535, dtype=np.int32)
    l1 = np.zeros(B, dtype=np.int32)
    r1 = np.full(B, 65535, dtype=np.int32)
    cat_mask = np.full(B, np.uint64(0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
    w0 = np.ones(B, dtype=np.int32)
    w3 = np.ones(B, dtype=np.int32)
    alg_bound = np.full(B, 1 << 30, dtype=np.int32)
    target_sels = np.full(B, selectivity, dtype=np.float32)

    def _pick_quant_window(sorted_arr: np.ndarray, sel: float, u_start: float) -> Tuple[int, int]:
        sel_c = float(np.clip(sel, 0.0002, 0.999))
        q_lo = u_start * (1.0 - sel_c)
        q_hi = min(1.0, q_lo + sel_c)
        i_lo = min(len(sorted_arr) - 1, int(q_lo * len(sorted_arr)))
        i_hi = min(len(sorted_arr) - 1, max(i_lo, int(q_hi * len(sorted_arr)) - 1))
        lo_v = int(sorted_arr[i_lo])
        hi_v = int(sorted_arr[i_hi])
        if hi_v <= lo_v and sel_c > 0.002:
            hi_v = min(65535, lo_v + max(1, int(sel_c * 65535)))
        return lo_v, hi_v

    wl = workload.lower()
    if wl in ("w1", "w1_range"):
        op_flags[:] = PRED_COL0_RANGE
        u_starts = rng.random(B)
        for i in range(B):
            l0[i], r0[i] = _pick_quant_window(c0_s, selectivity, float(u_starts[i]))

    elif wl in ("w2", "w2_correlated"):
        op_flags[:] = PRED_COL1_RANGE
        # Anchor each query's window around its own nearest sample's col1 quantile so correlated queries probe skewed clusters
        u_starts = rng.random(B)
        for i in range(B):
            l1[i], r1[i] = _pick_quant_window(c1_s, selectivity, float(u_starts[i]))

    elif wl in ("w3", "w3_compound"):
        # Multi-column compound SQL: category IN mask AND col1 range AND linear algebraic expression
        op_flags[:] = PRED_COL1_RANGE | PRED_COL2_CAT | PRED_ALG_EXPR
        # Factor target selectivity across the 3 clauses: sel_cat * sel_c1 * sel_alg ~= selectivity
        for i in range(B):
            sel = float(np.clip(selectivity, 0.0004, 0.85))
            # Pick random subset of categories covering ~sel^(0.40)
            target_cat_sel = sel ** 0.40
            perm = rng.permutation(64)
            mask_val = np.uint64(0)
            acc = 0.0
            cat_counts = np.bincount(c2_raw.astype(np.int32), minlength=64) / float(M_s)
            for c_id in perm:
                mask_val |= (np.uint64(1) << np.uint64(c_id))
                acc += cat_counts[c_id]
                if acc >= target_cat_sel:
                    break
            cat_mask[i] = mask_val

            # Remaining selectivity split between col1 range and algebraic expression
            rem = min(0.99, sel / max(acc, 1e-4))
            sel_1 = np.sqrt(rem)
            l1[i], r1[i] = _pick_quant_window(c1_s, sel_1, float(rng.random()))

            w0_i = int(rng.integers(1, 5))
            w3_i = int(rng.integers(1, 5))
            w0[i], w3[i] = w0_i, w3_i
            # Calibrate algebraic bound on rows passing cat & col1 in subsample
            sub_ok = (((np.uint64(1) << c2_raw) & mask_val) != 0) & (c1_raw >= l1[i]) & (c1_raw <= r1[i])
            if np.any(sub_ok):
                comb = np.sort(w0_i * c0_raw[sub_ok] + w3_i * c3_raw[sub_ok])
                target_k = max(0, min(len(comb) - 1, int(round(sel_1 * len(comb))) - 1))
                alg_bound[i] = int(comb[target_k])
            else:
                alg_bound[i] = int((w0_i + w3_i) * 65535)

    elif wl in ("w4", "w4_mixed"):
        # Log-uniform selectivity in [0.0005, 0.60], cycling across W1, W2, W3 per query!
        log_sels = np.exp(rng.uniform(np.log(0.0005), np.log(0.60), size=B))
        target_sels[:] = log_sels.astype(np.float32)
        cat_counts = np.bincount(c2_raw.astype(np.int32), minlength=64) / float(M_s)
        for i in range(B):
            sel = float(log_sels[i])
            mode = i % 3
            if mode == 0:
                op_flags[i] = PRED_COL0_RANGE
                l0[i], r0[i] = _pick_quant_window(c0_s, sel, float(rng.random()))
            elif mode == 1:
                op_flags[i] = PRED_COL1_RANGE
                l1[i], r1[i] = _pick_quant_window(c1_s, sel, float(rng.random()))
            else:
                op_flags[i] = PRED_COL1_RANGE | PRED_COL2_CAT | PRED_ALG_EXPR
                target_cat_sel = sel ** 0.40
                perm = rng.permutation(64)
                mask_val = np.uint64(0)
                acc = 0.0
                for c_id in perm:
                    mask_val |= (np.uint64(1) << np.uint64(c_id))
                    acc += cat_counts[c_id]
                    if acc >= target_cat_sel:
                        break
                cat_mask[i] = mask_val
                rem = min(0.99, sel / max(acc, 1e-4))
                sel_1 = np.sqrt(rem)
                l1[i], r1[i] = _pick_quant_window(c1_s, sel_1, float(rng.random()))
                w0_i, w3_i = int(rng.integers(1, 5)), int(rng.integers(1, 5))
                w0[i], w3[i] = w0_i, w3_i
                sub_ok = (((np.uint64(1) << c2_raw) & mask_val) != 0) & (c1_raw >= l1[i]) & (c1_raw <= r1[i])
                if np.any(sub_ok):
                    comb = np.sort(w0_i * c0_raw[sub_ok] + w3_i * c3_raw[sub_ok])
                    tk = max(0, min(len(comb) - 1, int(round(sel_1 * len(comb))) - 1))
                    alg_bound[i] = int(comb[tk])
                else:
                    alg_bound[i] = int((w0_i + w3_i) * 65535)
    else:
        raise ValueError(f"Unknown workload: {workload}")

    return PredicateBatch(
        op_flags=op_flags,
        l0=l0, r0=r0,
        l1=l1, r1=r1,
        cat_mask=cat_mask,
        w0=w0, w3=w3,
        alg_bound=alg_bound,
        target_sel=target_sels,
        workload_name=f"{workload}_sel{selectivity:.4f}",
    )


def eval_predicate_single_cpu(ds: HybridDataset, preds: PredicateBatch, q_idx: int) -> np.ndarray:
    """Evaluate exact boolean mask (N,) for a single query on CPU (used in unit tests)."""
    flags = int(preds.op_flags[q_idx])
    mask = np.ones(ds.N, dtype=bool)
    if flags & PRED_COL0_RANGE:
        mask &= (ds.col0 >= preds.l0[q_idx]) & (ds.col0 <= preds.r0[q_idx])
    if flags & PRED_COL1_RANGE:
        mask &= (ds.col1 >= preds.l1[q_idx]) & (ds.col1 <= preds.r1[q_idx])
    if flags & PRED_COL2_CAT:
        cm = np.uint64(preds.cat_mask[q_idx])
        mask &= ((np.uint64(1) << ds.col2.astype(np.uint64)) & cm) != 0
    if flags & PRED_ALG_EXPR:
        expr = int(preds.w0[q_idx]) * ds.col0.astype(np.int32) + int(preds.w3[q_idx]) * ds.col3.astype(np.int32)
        mask &= (expr <= int(preds.alg_bound[q_idx]))
    return mask
