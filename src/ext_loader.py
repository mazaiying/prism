#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast Direct NVCC / PyBind11 Extension Loader for PRISM CUDA C++17 Kernels.

IMPORTANT: Never pass `--use_fast_math` so IEEE-754 `rn` / `ru` rounding modes
(`__double2float_ru`, `fmaf`, `sqrtf`) remain strictly compliant with Theorem 1.
"""

from __future__ import annotations
import hashlib
import importlib.util
import os
import subprocess
import sysconfig
import torch
from torch.utils import cpp_extension

_CACHED_MOD = None


def load_prism_ext(force_rebuild: bool = False):
    global _CACHED_MOD
    if _CACHED_MOD is not None and not force_rebuild:
        return _CACHED_MOD

    src_dir = os.path.dirname(os.path.abspath(__file__))
    cu_path = os.path.join(src_dir, "prism_kernels.cu")
    with open(cu_path, "rb") as f:
        code_bytes = f.read()

    # Detect current visible GPU compute capability (sm_89 on RTX 4080, sm_80 on A100)
    major, minor = torch.cuda.get_device_capability(0)
    arch_flag = f"sm_{major}{minor}"
    sig = hashlib.sha256(code_bytes + arch_flag.encode()).hexdigest()[:12]

    prism_root = os.path.expanduser(
        os.environ.get("PRISM_ROOT", os.path.abspath(os.path.join(src_dir, "..")))
    )
    build_dir = os.path.join(prism_root, "build_ext")
    os.makedirs(build_dir, exist_ok=True)
    mod_name = f"prism_cuda_{arch_flag}_{sig}"
    so_path = os.path.join(build_dir, f"{mod_name}.so")

    if force_rebuild or not os.path.exists(so_path):
        cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        nvcc = os.path.join(cuda_home, "bin", "nvcc")
        if not os.path.exists(nvcc):
            nvcc = "nvcc"

        py_inc = sysconfig.get_path("include")
        inc_Flags = [f"-I{py_inc}"] + [f"-I{p}" for p in cpp_extension.include_paths()]
        lib_Flags = [f"-L{p}" for p in cpp_extension.library_paths()]
        # Check C++11 ABI flag matching installed torch
        abi = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0

        cmd = [
            nvcc,
            "-O3",
            "-std=c++17",
            f"-arch={arch_flag}",
            "--shared",
            "-Xcompiler", "-fPIC",
            f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
            f"-DTORCH_EXTENSION_NAME={mod_name}",
            "-DTORCH_API_INCLUDE_EXTENSION_H",
            *inc_Flags,
            cu_path,
            *lib_Flags,
            "-lc10", "-lc10_cuda", "-ltorch_cpu", "-ltorch_cuda", "-ltorch", "-ltorch_python",
            "-o", so_path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"NVCC compilation failed:\nCMD: {' '.join(cmd)}\nSTDERR:\n{res.stderr}")

    spec = importlib.util.spec_from_file_location(mod_name, so_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    _CACHED_MOD = mod
    return mod
