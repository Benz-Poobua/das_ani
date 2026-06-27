"""
:module: src/eval.py
:author: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Stress-test the CC pipeline (Scalability, Complexity, Fidelity)
          and the preprocessing backends (hybrid / pure_torch / pure_numpy).

Experiments (each can be skipped from the CLI):
- preprocess : per-file timing + fidelity of the three preprocessing
               backends. ``pure_numpy`` is the correctness benchmark;
               ``hybrid`` (scipy filters + torch normalization) should match
               it to float32 round-off on CPU and GPU; ``pure_torch``
               (rFFT-domain bandpass) is the GPU-tailored approximation whose
               deviation this experiment quantifies.
- scaling    : strong scaling of the CC engine over worker counts, with
               speedup S(p), parallel efficiency E(p), and an Amdahl
               serial-fraction fit.
- complexity : lag sweep, conventional vs v1, with NCF fidelity metrics.

Publication-grade metrics added on top of wall/IO/CC timing:
- FLOP model + achieved GFLOP/s, arithmetic intensity, achieved bandwidth
  (roofline positioning) for conventional vs v1;
- throughput (NCFs/s) and latency percentiles (per-file CC p50/p95);
- peak memory (host RSS high-water + GPU VRAM);
- strong-scaling speedup / efficiency / Amdahl serial fraction;
- a software-environment record captured into the manifest.

Note on GPU timing: accurate cc_sec/io_sec attribution on CUDA requires the
``torch.cuda.synchronize()`` added to ``cc.process_single_file`` before the CC
timer stops; without it, asynchronous kernel time leaks into the I/O stage.

Outputs:
- benchmark_results.csv
- run_manifest.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from src.ani import (
    PREPROCESS_MODES,
    decimate_raw,
    differentiate_cpu,
    preprocess,
    choose_block_size_v1,
)
from src.cc import process_single_file, _worker_warmup, _get_ingest_cfg, _peak_rss_mb
from src.utils import load_config, get_cfg, load_data, convert_to_numpy, nextpow2
from src.error import rel_frobenius, max_abs_error, cosine_similarity_per_trace

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# CSV helpers
# -----------------------------------------------------------------------------
def checkpoint_csv(results: List[RunResult], csv_path: Path) -> None:
    if not results:
        return
    df = pd.DataFrame([r.__dict__ for r in results])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

# -----------------------------------------------------------------------------
# Resume / manifest helpers
# -----------------------------------------------------------------------------
def _hash_text(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]

def write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

def capture_environment() -> Dict[str, Any]:
    """
    Record the software/hardware environment for reproducibility. Written into
    run_manifest.json so a results CSV can always be tied back to the exact
    library versions and device it was produced on.
    """
    env: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import scipy
        env["scipy"] = scipy.__version__
    except Exception:
        pass
    try:
        env["torch"] = torch.__version__
        env["torch_cuda"] = torch.version.cuda
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["gpu_count"] = torch.cuda.device_count()
    except Exception:
        pass
    try:
        env["git_commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        env["git_commit"] = None
    try:
        env["slurm_cpus_per_task"] = os.environ.get("SLURM_CPUS_PER_TASK")
        env["slurm_mem_per_node"] = os.environ.get("SLURM_MEM_PER_NODE")
    except Exception:
        pass
    return env

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _deepcopy_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(cfg)

def _set_nested(cfg: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    d = cfg
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value

def _pick_benchmark_files(data_root: Path, n_files: int) -> List[Path]:
    files = sorted(data_root.rglob("*.npz"))
    if not files:
        files = sorted(data_root.rglob("*.zarr"))
    if not files:
        raise RuntimeError(f"No valid data files found under {data_root}")
    return files[: max(1, int(n_files))]

def _first_vs_idx_from_cfg(cfg: Dict[str, Any]) -> int:
    first_chan = int(get_cfg(cfg, ["data", "first_chan"], required=True))
    last_chan = int(get_cfg(cfg, ["data", "last_chan"], required=True))
    src_stride = int(get_cfg(cfg, ["data", "src_stride"], 10))
    if last_chan < first_chan:
        raise ValueError("data.last_chan must be >= data.first_chan")
    src_ch_all_num = np.arange(first_chan, last_chan + 1, src_stride, dtype=int)
    return int(src_ch_all_num[0] - first_chan)

def _vs_indices_from_cfg(cfg: Dict[str, Any]) -> np.ndarray:
    first_chan = int(get_cfg(cfg, ["data", "first_chan"], required=True))
    last_chan = int(get_cfg(cfg, ["data", "last_chan"], required=True))
    src_stride = int(get_cfg(cfg, ["data", "src_stride"], 10))
    src_ch_all_num = np.arange(first_chan, last_chan + 1, src_stride, dtype=int)
    return src_ch_all_num - first_chan

def _output_path_for(cfg: Dict[str, Any], file_path: Path, *, vs_idx: int, mode: str) -> Path:
    out_root = Path(get_cfg(cfg, ["paths", "output_root"], required=True)).expanduser().resolve()
    is_auto = bool(get_cfg(cfg, ["xcorr", "auto_cc"], False))
    base_no_ext = file_path.stem if file_path.suffix == '.npz' else file_path.name.replace('.zarr', '')

    if is_auto:
        return out_root / f"{base_no_ext}_auto_{mode}.npy"
    else:
        return out_root / f"{base_no_ext}_cc_{vs_idx:03d}_{mode}.npy"

def _load_npy(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode='r')

# -----------------------------------------------------------------------------
# Metrics (fidelity)
# -----------------------------------------------------------------------------
def fidelity_metrics(v1_arr: np.ndarray, conv_arr: np.ndarray, eps: float = 1e-15) -> Dict[str, float]:
    rel_f = float(rel_frobenius(v1_arr, conv_arr, eps=eps))
    max_a = float(max_abs_error(v1_arr, conv_arr))
    cos = cosine_similarity_per_trace(v1_arr, conv_arr, eps=eps)
    cos_mean = float(np.mean(cos))
    cos_p05 = float(np.percentile(cos, 5))
    return {
        "rel_fro": rel_f,
        "max_abs": max_a,
        "cos_mean": cos_mean,
        "cos_p05": cos_p05,
    }

# -----------------------------------------------------------------------------
# Metrics (cost model: FLOPs, bytes, throughput, scaling)
# -----------------------------------------------------------------------------
def _rfft_flops(L: int) -> float:
    """
    Approximate flop count of one length-L real FFT (or its inverse):
    ~2.5 * L * log2(L), i.e. half of the textbook 5 N log2 N for a complex FFT.
    Used only to form *consistent* GFLOP/s estimates for conventional vs v1; the
    ratio is robust to the constant even if the absolute count is approximate.
    """
    L = int(L)
    if L <= 1:
        return 0.0
    return 2.5 * L * math.log2(L)

def theoretical_cc_cost(
    *,
    mode: str,
    npts_seg: int,
    M: int,
    nch: int,
    nseg: int,
    n_vs: int,
    auto_cc: bool,
    v1_fft_snap_pow2: bool = True,
    v1_fallback: str = "v1_2M",
) -> Tuple[float, float, int]:
    """
    Analytic per-FILE cost of the correlation engine.

    Returns (flops, bytes_moved, n_corr) where n_corr is the number of NCFs
    produced for the file. Mirrors the asymmetric "compute-once, broadcast-many"
    schedule in cc.process_single_file:
      - source spectrum transformed once per virtual source (nseg*nblk rFFTs);
      - receiver spectra transformed for every channel (nch*nseg*nblk rFFTs);
      - one inverse rFFT per receiver after summation;
      - spectral product+accumulate over (nseg, nblk, nfreq).
    """
    npts_seg = int(npts_seg); M = int(M); nch = int(nch); nseg = int(nseg)
    n_vs = 1 if auto_cc else max(1, int(n_vs))
    n_corr = nch * n_vs

    if str(mode).lower() == "v1":
        K, L = choose_block_size_v1(M, fft_snap_pow2=v1_fft_snap_pow2, fallback=v1_fallback)
        nblk = (npts_seg + K - 1) // K
    else:
        L = int(nextpow2(2 * npts_seg - 1))
        nblk = 1
    nfreq = L // 2 + 1

    fwd = n_vs * (nseg * nblk) + n_vs * (nch * nseg * nblk)   # source + receiver rFFTs
    inv = n_vs * nch                                          # one irFFT per receiver
    fft_flops = (fwd + inv) * _rfft_flops(L)
    prod_flops = float(n_vs) * nch * nseg * nblk * nfreq * 8.0  # complex MAC ~ 8 flop

    flops = fft_flops + prod_flops
    bytes_moved = (
        fwd * (L * 4 + nfreq * 8)                  # read real block, write complex spec
        + float(n_vs) * nch * nseg * nblk * nfreq * 8  # read spectra for the product
        + inv * (nfreq * 8 + L * 4)                # irFFT read/write
    )
    return float(flops), float(bytes_moved), int(n_corr)

def amdahl_serial_fraction(ps: Sequence[int], speedups: Sequence[float]) -> float:
    """
    Estimate the Amdahl serial fraction f from (p, speedup) pairs, where
    S(p) = 1 / (f + (1-f)/p). Solves for f at each p>1 and averages.
    """
    fs: List[float] = []
    for p, s in zip(ps, speedups):
        if p > 1 and s > 0:
            f = (1.0 / s - 1.0 / p) / (1.0 - 1.0 / p)
            fs.append(min(max(f, 0.0), 1.0))
    return float(np.mean(fs)) if fs else 0.0

# -----------------------------------------------------------------------------
# Benchmark Runner
# -----------------------------------------------------------------------------
@dataclass
class RunResult:
    experiment: str
    mode: str
    njobs: int
    io_format: str
    n_files: int

    # Timers (medians)
    wall_sec: float
    io_sec: float
    cc_sec: float

    # Robust timers (explicit tracking for redundancy)
    wall_median: Optional[float] = None
    io_median: Optional[float] = None
    cc_median: Optional[float] = None

    max_lag_sec: Optional[float] = None
    window_sec: Optional[float] = None
    ratio_lag_win: Optional[float] = None

    # Error-bar stats for batch wall time
    wall_mean: Optional[float] = None
    wall_std: Optional[float] = None
    wall_p25: Optional[float] = None
    wall_p75: Optional[float] = None
    n_eff: Optional[int] = None

    # Latency percentiles (per-file, within a run)
    cc_p50: Optional[float] = None
    cc_p95: Optional[float] = None
    io_p50: Optional[float] = None
    io_p95: Optional[float] = None

    # Cost model / roofline (per file, from cc_sec)
    flops: Optional[float] = None
    bytes_moved: Optional[float] = None
    gflops: Optional[float] = None              # achieved GFLOP/s
    arith_intensity: Optional[float] = None     # FLOP / byte
    achieved_bw_gbs: Optional[float] = None      # GB/s
    n_corr: Optional[int] = None                 # NCFs produced per file
    nseg_eff: Optional[int] = None

    # Throughput
    ncfs_per_sec: Optional[float] = None
    mbytes_per_sec: Optional[float] = None

    # Strong-scaling derived
    speedup: Optional[float] = None
    efficiency: Optional[float] = None
    amdahl_serial_frac: Optional[float] = None

    # Memory (peak)
    peak_rss_mb: Optional[float] = None
    peak_vram_mb: Optional[float] = None

    # Fidelity
    rel_fro: Optional[float] = None
    max_abs: Optional[float] = None
    cos_mean: Optional[float] = None
    cos_p05: Optional[float] = None
    n_fid_samples: Optional[int] = None

    # Free-form label (e.g. file name for per-file preprocess rows).
    note: Optional[str] = None

def _process_unpack(args):
    return process_single_file(*args)

class BenchmarkRunner:
    def __init__(self, cc_config_path: Path, files: List[Path], out_dir: Path):
        self.base_cfg: Dict[str, Any] = load_config(cc_config_path)
        self.files = files
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.bench_root = (self.out_dir / "bench_outputs").resolve()
        self.bench_root.mkdir(parents=True, exist_ok=True)

        fs_raw = float(get_cfg(self.base_cfg, ["data", "fs_raw"], required=True))
        # Canonical key is ingest.decimation; _get_ingest_cfg also accepts the
        # deprecated preprocess.decimation with a warning.
        dec, _ = _get_ingest_cfg(self.base_cfg)
        self.decimation = int(dec)
        self.fs_proc = fs_raw / float(dec)
        self.vs_idx0 = _first_vs_idx_from_cfg(self.base_cfg)

        # Geometry needed by the FLOP model.
        first_chan = int(get_cfg(self.base_cfg, ["data", "first_chan"], required=True))
        last_chan = int(get_cfg(self.base_cfg, ["data", "last_chan"], required=True))
        self.nch = last_chan - first_chan + 1
        self.n_vs = int(len(_vs_indices_from_cfg(self.base_cfg)))

        # Probe the first file once for sample count (-> nseg). Best-effort.
        self.npts0_proc: Optional[int] = None
        try:
            _, das_array, _, npts, _ = load_data(self.files[0], mmap=True)
            self.npts0_proc = int(npts) // self.decimation
        except Exception as e:
            logger.warning("Could not probe npts for FLOP model: %s", e)

    # --- cost-model helper -------------------------------------------------
    def nseg_for(self, npts_seg: int) -> int:
        if not self.npts0_proc or npts_seg <= 0:
            return 1
        return max(1, self.npts0_proc // int(npts_seg))

    def cost_for(self, *, mode: str, max_lag_sec: float, window_sec: float,
                 auto_cc: bool) -> Tuple[float, float, int, int]:
        """Return (flops, bytes, n_corr, nseg) for one configuration/file."""
        M = int(round(float(max_lag_sec) * self.fs_proc))
        npts_seg = int(round(float(window_sec) * self.fs_proc))
        nseg = self.nseg_for(npts_seg)
        v1_snap = bool(get_cfg(self.base_cfg, ["xcorr", "v1_fft_snap_pow2"], True))
        v1_fb = str(get_cfg(self.base_cfg, ["xcorr", "v1_fallback"], "v1_2M"))
        flops, byts, n_corr = theoretical_cc_cost(
            mode=mode, npts_seg=npts_seg, M=M, nch=self.nch, nseg=nseg,
            n_vs=self.n_vs, auto_cc=auto_cc,
            v1_fft_snap_pow2=v1_snap, v1_fallback=v1_fb,
        )
        return flops, byts, n_corr, nseg

    def _attach_cost(self, r: RunResult, *, mode: str, max_lag_sec: float,
                     window_sec: float, auto_cc: bool) -> None:
        """Fill the FLOP/throughput/roofline fields of a row from its cc_sec."""
        try:
            flops, byts, n_corr, nseg = self.cost_for(
                mode=mode, max_lag_sec=max_lag_sec, window_sec=window_sec, auto_cc=auto_cc
            )
        except Exception as e:
            logger.warning("Cost model failed (%s); skipping FLOP columns.", e)
            return
        r.flops = flops
        r.bytes_moved = byts
        r.n_corr = n_corr
        r.nseg_eff = nseg
        r.arith_intensity = float(flops / byts) if byts > 0 else None
        cc = float(r.cc_sec or 0.0)
        if cc > 0:
            r.gflops = flops / 1e9 / cc
            r.achieved_bw_gbs = byts / 1e9 / cc
            r.ncfs_per_sec = n_corr / cc
            r.mbytes_per_sec = byts / 1e6 / cc

    def _prepare_cfg(self, *, run_id: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        cfg = _deepcopy_cfg(self.base_cfg)
        for k, v in overrides.items():
            _set_nested(cfg, k, v)
        _set_nested(cfg, "paths.output_root", str(self.bench_root / run_id))
        if "perf" in cfg:
            cfg["perf"]["enabled"] = False
        return cfg

    def _run_pool(self, cfg: Dict[str, Any]) -> Dict[str, float]:
        """
        Run one batch over the golden subset and return a stats dict with wall
        time, median + p50/p95 of per-file IO/CC, and peak memory (max across
        files). Replaces the previous (wall, io_med, cc_med) tuple.
        """
        njobs = max(1, int(get_cfg(cfg, ["runtime", "njobs"], 1)))
        mode = str(get_cfg(cfg, ["xcorr", "mode"], "conventional")).lower()

        max_lag_sec = float(get_cfg(cfg, ["xcorr", "max_lag_sec"], 4.0))
        xcorr_seg_sec = float(get_cfg(cfg, ["xcorr", "xcorr_seg_sec"], 8.0))
        if mode == "v1":
            xcorr_seg_sec = float(get_cfg(cfg, ["xcorr", "xcorr_seg_sec_v1"], xcorr_seg_sec))

        M = int(round(max_lag_sec * self.fs_proc))
        npts_seg = int(round(xcorr_seg_sec * self.fs_proc))

        v1_fft_snap_pow2 = bool(get_cfg(cfg, ["xcorr", "v1_fft_snap_pow2"], True))
        v1_fallback = str(get_cfg(cfg, ["xcorr", "v1_fallback"], "v1_2M"))

        slurm_cores = os.environ.get("SLURM_CPUS_PER_TASK")
        ncores = int(slurm_cores) if slurm_cores else (os.cpu_count() or 1)
        threads_per_proc = max(1, ncores // njobs)

        initializer = None
        if npts_seg > 0 and M > 0:
            from functools import partial
            do_compile = bool(get_cfg(cfg, ["runtime", "torch_compile"], False))
            compile_mode = str(get_cfg(cfg, ["runtime", "compile_mode"], "reduce-overhead"))
            initializer = partial(
                _worker_warmup,
                mode=mode,
                npts_seg=npts_seg,
                max_lag_samples=M,
                v1_fft_snap_pow2=v1_fft_snap_pow2,
                v1_fallback=v1_fallback,
                threads_per_proc=threads_per_proc,
                do_compile=do_compile,
                compile_mode=compile_mode,
            )

        io_times: List[float] = []
        cc_times: List[float] = []
        peak_vram: List[float] = []
        peak_rss: List[float] = []

        def _absorb(d: Dict[str, Any]) -> None:
            if "io_sec" in d: io_times.append(float(d["io_sec"]))
            if "cc_sec" in d: cc_times.append(float(d["cc_sec"]))
            if d.get("peak_vram_mb") is not None: peak_vram.append(float(d["peak_vram_mb"]))
            if d.get("peak_rss_mb") is not None: peak_rss.append(float(d["peak_rss_mb"]))

        t0 = time.perf_counter()
        with mp.Pool(processes=njobs, initializer=initializer, maxtasksperchild=1) as pool:
            task_args = [(str(f), cfg) for f in self.files]
            for res in pool.imap_unordered(_process_unpack, task_args, chunksize=1):
                if isinstance(res, dict):
                    _absorb(res)
                elif isinstance(res, (tuple, list)):
                    for item in res:
                        if isinstance(item, dict):
                            _absorb(item)
        t1 = time.perf_counter()

        def _pct(xs: List[float], q: float) -> float:
            return float(np.percentile(xs, q)) if xs else 0.0

        return {
            "wall": float(t1 - t0),
            "io_med": float(np.median(io_times)) if io_times else 0.0,
            "cc_med": float(np.median(cc_times)) if cc_times else 0.0,
            "io_p50": _pct(io_times, 50), "io_p95": _pct(io_times, 95),
            "cc_p50": _pct(cc_times, 50), "cc_p95": _pct(cc_times, 95),
            "peak_vram_mb": float(max(peak_vram)) if peak_vram else 0.0,
            "peak_rss_mb": float(max(peak_rss)) if peak_rss else 0.0,
        }

    def run_batch(self, *, run_id: str, overrides: Dict[str, Any], repeats: int = 2) -> Dict[str, float]:
        repeats = max(1, int(repeats))
        recs: List[Dict[str, float]] = []
        for r in range(repeats):
            recs.append(self._run_pool(self._prepare_cfg(run_id=f"{run_id}_rep{r}", overrides=overrides)))

        # Discard warm-up run (first) for all metrics when repeats > 1.
        use = recs if repeats == 1 else recs[1:]

        def col(key: str) -> np.ndarray:
            return np.asarray([d[key] for d in use], dtype=np.float64)

        arr_w = col("wall")
        return {
            "wall_sec": float(np.median(arr_w)),
            "io_sec": float(np.median(col("io_med"))),
            "cc_sec": float(np.median(col("cc_med"))),
            "wall_median": float(np.median(arr_w)),
            "io_median": float(np.median(col("io_med"))),
            "cc_median": float(np.median(col("cc_med"))),
            "wall_mean": float(np.mean(arr_w)),
            "wall_std": float(np.std(arr_w, ddof=1)) if arr_w.size >= 2 else 0.0,
            "wall_p25": float(np.percentile(arr_w, 25)),
            "wall_p75": float(np.percentile(arr_w, 75)),
            "n_eff": int(arr_w.size),
            # latency percentiles (median across repeats of the per-rep percentile)
            "cc_p50": float(np.median(col("cc_p50"))),
            "cc_p95": float(np.median(col("cc_p95"))),
            "io_p50": float(np.median(col("io_p50"))),
            "io_p95": float(np.median(col("io_p95"))),
            # memory: worst case across repeats
            "peak_vram_mb": float(np.max(col("peak_vram_mb"))),
            "peak_rss_mb": float(np.max(col("peak_rss_mb"))),
        }

    def compare_fidelity_for_last_run(self, *, conv_cfg: Dict[str, Any], v1_cfg: Dict[str, Any]) -> Dict[str, float]:
        test_file = self.files[0]
        vs_idx = int(self.vs_idx0)
        p_conv = _output_path_for(conv_cfg, test_file, vs_idx=vs_idx, mode="conventional")
        p_v1 = _output_path_for(v1_cfg, test_file, vs_idx=vs_idx, mode="v1")

        if not p_conv.exists(): raise FileNotFoundError(f"Missing conventional output for fidelity: {p_conv}")
        if not p_v1.exists(): raise FileNotFoundError(f"Missing v1 output for fidelity: {p_v1}")

        R = _load_npy(p_conv)
        V = _load_npy(p_v1)
        return fidelity_metrics(V, R)

    def compare_fidelity_aggregate(
        self, *, conv_cfg: Dict[str, Any], v1_cfg: Dict[str, Any],
        max_files: int = 3, vs_samples: int = 3,
    ) -> Dict[str, float]:
        """
        Aggregate v1-vs-conventional fidelity across several files and virtual
        sources, instead of a single (file[0], vs0) pair. Returns the mean of
        each metric plus the worst (max) rel_fro / (min) cos_p05, and the sample
        count. Falls back gracefully to whatever outputs exist.
        """
        vs_all = _vs_indices_from_cfg(self.base_cfg)
        if len(vs_all) == 0:
            return {}
        # sample up to vs_samples virtual sources spread across the array
        idx = np.unique(np.linspace(0, len(vs_all) - 1, num=min(vs_samples, len(vs_all))).astype(int))
        vs_sel = [int(vs_all[i]) for i in idx]
        files_sel = self.files[: max(1, int(max_files))]

        rel, mxa, cmean, cp05 = [], [], [], []
        for f in files_sel:
            for vs in vs_sel:
                p_conv = _output_path_for(conv_cfg, f, vs_idx=vs, mode="conventional")
                p_v1 = _output_path_for(v1_cfg, f, vs_idx=vs, mode="v1")
                if not (p_conv.exists() and p_v1.exists()):
                    continue
                try:
                    m = fidelity_metrics(_load_npy(p_v1), _load_npy(p_conv))
                except Exception:
                    continue
                rel.append(m["rel_fro"]); mxa.append(m["max_abs"])
                cmean.append(m["cos_mean"]); cp05.append(m["cos_p05"])

        if not rel:
            # fall back to the single-pair check
            return self.compare_fidelity_for_last_run(conv_cfg=conv_cfg, v1_cfg=v1_cfg)
        return {
            "rel_fro": float(np.max(rel)),       # worst case
            "max_abs": float(np.max(mxa)),
            "cos_mean": float(np.mean(cmean)),
            "cos_p05": float(np.min(cp05)),      # worst case
            "n_fid_samples": int(len(rel)),
        }

    def cleanup(self) -> None:
        if self.bench_root.exists():
            shutil.rmtree(self.bench_root)
            logger.info("Cleaned benchmark outputs: %s", self.bench_root)

# -----------------------------------------------------------------------------
# Experiments
# -----------------------------------------------------------------------------
def _ingest_for_preprocess(cfg: Dict[str, Any], file_path: Path) -> tuple[np.ndarray, float]:
    """
    Load + channel-slice + (optionally) decimate/differentiate one file,
    mirroring the ingest stage of ``cc.process_single_file``, so the
    preprocessing experiment operates on exactly the array the production
    pipeline would see.

    :return: (data_raw float32 (nch, nt), fs_proc)
    """
    fs_raw = float(get_cfg(cfg, ["data", "fs_raw"], required=True))
    first_chan = int(get_cfg(cfg, ["data", "first_chan"], required=True))
    last_chan = int(get_cfg(cfg, ["data", "last_chan"], required=True))
    f2 = float(get_cfg(cfg, ["preprocess", "f2"], 5.0))
    decimation, diff = _get_ingest_cfg(cfg)
    fs_proc = fs_raw / decimation

    _, das_array, _, _, _ = load_data(file_path, mmap=True)
    data_raw = np.asarray(das_array[:], dtype=np.float32)

    nch_expected = last_chan - first_chan + 1
    nch_file = data_raw.shape[0]
    if nch_file == nch_expected:
        pass
    elif nch_file > last_chan:
        data_raw = data_raw[first_chan: last_chan + 1, :]
    else:
        raise ValueError(
            f"{Path(file_path).name}: nch_file={nch_file} < last_chan+1={last_chan + 1}"
        )

    if decimation > 1:
        data_raw = decimate_raw(data_raw, fs_raw, decimation, f2_target=f2)
    if diff:
        data_raw = differentiate_cpu(data_raw, fs_proc)
    return np.ascontiguousarray(data_raw, dtype=np.float32), fs_proc


def run_preprocess_test(
    runner: BenchmarkRunner,
    modes: Sequence[str],
    *,
    repeats: int = 3,
    use_gpu: Optional[bool] = None,
) -> List[RunResult]:
    """
    Time + validate the per-window preprocessing backends on the golden subset.

    For every golden file and every requested backend, runs ``ani.preprocess``
    ``repeats`` times (first repeat discarded as cache / JIT warm-up when
    repeats > 1), records peak host RSS and (on CUDA) peak VRAM, then computes
    fidelity metrics of the result against the ``pure_numpy`` benchmark.

    One CSV row per (file, mode): experiment="preprocess", mode=<backend>,
    note=<file name>.
    """
    logger.info("=== Experiment: Preprocessing backends ===")
    cfg = runner.base_cfg
    if use_gpu is None:
        use_gpu = bool(get_cfg(cfg, ["runtime", "use_gpu"], False))
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    f1 = float(get_cfg(cfg, ["preprocess", "f1"], 1.0))
    f2 = float(get_cfg(cfg, ["preprocess", "f2"], 5.0))
    ram_win_sec = float(get_cfg(cfg, ["preprocess", "ram_win_sec"], 0.0))

    bad = [m for m in modes if m not in PREPROCESS_MODES]
    if bad:
        raise ValueError(f"Unknown preprocess mode(s) {bad}; valid: {PREPROCESS_MODES}")

    io_fmt = "zarr" if str(runner.files[0]).endswith("zarr") else "npz"
    repeats = max(1, int(repeats))
    results: List[RunResult] = []

    for fpath in runner.files:
        try:
            data_raw, fs_proc = _ingest_for_preprocess(cfg, Path(fpath))
        except Exception as e:
            logger.warning("Preprocess eval: skipping %s (%s)", Path(fpath).name, e)
            continue

        # Reference: the legacy all-CPU chain (computed once per file).
        ref = preprocess(data_raw, fs_proc, f1, f2, ram_win_sec, mode="pure_numpy")

        for mode in modes:
            times: List[float] = []
            out: Any = None
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for _ in range(repeats):
                t0 = time.perf_counter()
                out = preprocess(
                    data_raw, fs_proc, f1, f2, ram_win_sec,
                    use_gpu=use_gpu, device=device, mode=mode,
                )
                if isinstance(out, torch.Tensor) and out.device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)

            peak_vram = (
                torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0
            )
            peak_rss = _peak_rss_mb()

            arr = np.asarray(times if repeats == 1 else times[1:], dtype=np.float64)
            out_np = convert_to_numpy(out).astype(np.float32, copy=False)
            fid = fidelity_metrics(out_np, ref)

            results.append(RunResult(
                experiment="preprocess",
                mode=mode,
                njobs=1,
                io_format=io_fmt,
                n_files=1,
                wall_sec=float(np.median(arr)),
                io_sec=0.0,
                cc_sec=0.0,
                wall_median=float(np.median(arr)),
                wall_mean=float(np.mean(arr)),
                wall_std=float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0,
                wall_p25=float(np.percentile(arr, 25)),
                wall_p75=float(np.percentile(arr, 75)),
                n_eff=int(arr.size),
                peak_rss_mb=float(peak_rss),
                peak_vram_mb=float(peak_vram),
                rel_fro=fid["rel_fro"],
                max_abs=fid["max_abs"],
                cos_mean=fid["cos_mean"],
                cos_p05=fid["cos_p05"],
                note=Path(fpath).name,
            ))
            logger.info(
                "[preprocess/%s] %s | device=%s | wall_med=%.3fs | rel_fro=%.3e | "
                "cos_mean=%.6f | rss=%.0fMB | vram=%.0fMB",
                mode, Path(fpath).name, device, float(np.median(arr)),
                fid["rel_fro"], fid["cos_mean"], peak_rss, peak_vram,
            )
        del data_raw, ref
    return results


def run_scaling_test(runner: BenchmarkRunner, cores_list: List[int], *, window_sec: float, repeats: int = 2) -> List[RunResult]:
    logger.info("=== Experiment: Strong scaling ===")
    results: List[RunResult] = []
    max_lag_sec_base = float(get_cfg(runner.base_cfg, ["xcorr", "max_lag_sec"], 4.0))
    auto_cc = bool(get_cfg(runner.base_cfg, ["xcorr", "auto_cc"], False))
    io_fmt = "zarr" if str(runner.files[0]).endswith("zarr") else "npz"

    for mode in ("conventional", "v1"):
        mode_rows: List[RunResult] = []
        for p in cores_list:
            run_id = f"scale_{mode}_p{p}"
            stats = runner.run_batch(
                run_id=run_id,
                overrides={
                    "runtime.njobs": p,
                    "xcorr.mode": mode,
                    "xcorr.xcorr_seg_sec": float(window_sec),
                    "xcorr.xcorr_seg_sec_v1": float(window_sec),
                    "xcorr.max_lag_sec": max_lag_sec_base,
                },
                repeats=repeats,
            )

            row = RunResult(
                experiment="scaling",
                mode=mode,
                njobs=p,
                io_format=io_fmt,
                wall_sec=stats["wall_sec"],
                io_sec=stats["io_sec"],
                cc_sec=stats["cc_sec"],
                wall_median=stats["wall_median"],
                io_median=stats["io_median"],
                cc_median=stats["cc_median"],
                n_files=len(runner.files),
                max_lag_sec=max_lag_sec_base,
                window_sec=float(window_sec),
                ratio_lag_win=float(max_lag_sec_base / window_sec),
                wall_mean=stats["wall_mean"],
                wall_std=stats["wall_std"],
                wall_p25=stats["wall_p25"],
                wall_p75=stats["wall_p75"],
                n_eff=stats["n_eff"],
                cc_p50=stats["cc_p50"], cc_p95=stats["cc_p95"],
                io_p50=stats["io_p50"], io_p95=stats["io_p95"],
                peak_rss_mb=stats["peak_rss_mb"], peak_vram_mb=stats["peak_vram_mb"],
            )
            runner._attach_cost(row, mode=mode, max_lag_sec=max_lag_sec_base,
                                window_sec=float(window_sec), auto_cc=auto_cc)
            mode_rows.append(row)
            logger.info("[%s] p=%d | Wall Med: %.2fs | CC Med: %.2fs | IO Med: %.2fs | "
                        "GFLOP/s: %s | RSS: %.0fMB",
                        mode, p, stats["wall_sec"], stats["cc_sec"], stats["io_sec"],
                        f"{row.gflops:.1f}" if row.gflops else "n/a", stats["peak_rss_mb"])

        # --- strong-scaling derived metrics (speedup / efficiency / Amdahl) ---
        base = next((r for r in mode_rows if r.njobs == min(c for c in cores_list)), None)
        base_t = base.wall_sec if base and base.wall_sec else None
        ps, sps = [], []
        for r in mode_rows:
            if base_t and r.wall_sec and r.wall_sec > 0:
                r.speedup = base_t / r.wall_sec
                r.efficiency = r.speedup / max(1, r.njobs)
                ps.append(r.njobs); sps.append(r.speedup)
        f_ser = amdahl_serial_fraction(ps, sps)
        for r in mode_rows:
            r.amdahl_serial_frac = f_ser
        results.extend(mode_rows)
    return results

def run_complexity_test(runner: BenchmarkRunner, lags_sec: List[float], *, window_sec: float, njobs: int, repeats: int = 2) -> List[RunResult]:
    logger.info("=== Experiment: Complexity sweep (lag) ===")
    results: List[RunResult] = []
    auto_cc = bool(get_cfg(runner.base_cfg, ["xcorr", "auto_cc"], False))
    io_fmt = "zarr" if str(runner.files[0]).endswith("zarr") else "npz"
    window_sec = float(window_sec)

    for lag in lags_sec:
        lag = float(lag)
        if lag <= 0 or lag >= window_sec:
            continue
        ratio = lag / window_sec
        logger.info("Lag=%.3fs, Window=%.3fs, ratio=%.4f", lag, window_sec, ratio)

        conv_id = f"complex_conv_L{lag:g}_W{window_sec:g}_p{njobs}"
        stats_conv = runner.run_batch(
            run_id=conv_id,
            overrides={"runtime.njobs": int(njobs), "xcorr.mode": "conventional", "xcorr.max_lag_sec": float(lag), "xcorr.xcorr_seg_sec": float(window_sec)},
            repeats=repeats,
        )

        v1_id = f"complex_v1_L{lag:g}_W{window_sec:g}_p{njobs}"
        stats_v1 = runner.run_batch(
            run_id=v1_id,
            overrides={"runtime.njobs": int(njobs), "xcorr.mode": "v1", "xcorr.max_lag_sec": float(lag), "xcorr.xcorr_seg_sec": float(window_sec), "xcorr.xcorr_seg_sec_v1": float(window_sec)},
            repeats=repeats,
        )

        conv_cfg = runner._prepare_cfg(run_id=f"{conv_id}_rep{repeats-1}", overrides={"runtime.njobs": int(njobs), "xcorr.mode": "conventional", "xcorr.max_lag_sec": float(lag), "xcorr.xcorr_seg_sec": float(window_sec)})
        v1_cfg = runner._prepare_cfg(run_id=f"{v1_id}_rep{repeats-1}", overrides={"runtime.njobs": int(njobs), "xcorr.mode": "v1", "xcorr.max_lag_sec": float(lag), "xcorr.xcorr_seg_sec": float(window_sec), "xcorr.xcorr_seg_sec_v1": float(window_sec)})

        fid: Dict[str, float] = {}
        try:
            fid = runner.compare_fidelity_aggregate(conv_cfg=conv_cfg, v1_cfg=v1_cfg)
            logger.info("Fidelity (n=%s): rel_fro=%.3e max_abs=%.3e cos_mean=%.6f",
                        fid.get("n_fid_samples"), fid["rel_fro"], fid["max_abs"], fid["cos_mean"])
        except Exception as e:
            logger.warning("Fidelity check failed at lag=%.3f: %s", lag, e)

        row_conv = RunResult(
            experiment="complexity", mode="conventional", njobs=int(njobs), io_format=io_fmt,
            wall_sec=stats_conv["wall_sec"], io_sec=stats_conv["io_sec"], cc_sec=stats_conv["cc_sec"],
            wall_median=stats_conv["wall_median"], io_median=stats_conv["io_median"], cc_median=stats_conv["cc_median"],
            wall_mean=stats_conv["wall_mean"], wall_std=stats_conv["wall_std"],
            wall_p25=stats_conv["wall_p25"], wall_p75=stats_conv["wall_p75"],
            n_eff=stats_conv["n_eff"], n_files=len(runner.files), max_lag_sec=float(lag),
            window_sec=float(window_sec), ratio_lag_win=float(ratio),
            cc_p50=stats_conv["cc_p50"], cc_p95=stats_conv["cc_p95"],
            io_p50=stats_conv["io_p50"], io_p95=stats_conv["io_p95"],
            peak_rss_mb=stats_conv["peak_rss_mb"], peak_vram_mb=stats_conv["peak_vram_mb"],
        )
        runner._attach_cost(row_conv, mode="conventional", max_lag_sec=float(lag),
                            window_sec=float(window_sec), auto_cc=auto_cc)

        row_v1 = RunResult(
            experiment="complexity", mode="v1", njobs=int(njobs), io_format=io_fmt,
            wall_sec=stats_v1["wall_sec"], io_sec=stats_v1["io_sec"], cc_sec=stats_v1["cc_sec"],
            wall_median=stats_v1["wall_median"], io_median=stats_v1["io_median"], cc_median=stats_v1["cc_median"],
            wall_mean=stats_v1["wall_mean"], wall_std=stats_v1["wall_std"],
            wall_p25=stats_v1["wall_p25"], wall_p75=stats_v1["wall_p75"],
            n_eff=stats_v1["n_eff"], n_files=len(runner.files), max_lag_sec=float(lag),
            window_sec=float(window_sec), ratio_lag_win=float(ratio),
            cc_p50=stats_v1["cc_p50"], cc_p95=stats_v1["cc_p95"],
            io_p50=stats_v1["io_p50"], io_p95=stats_v1["io_p95"],
            peak_rss_mb=stats_v1["peak_rss_mb"], peak_vram_mb=stats_v1["peak_vram_mb"],
            rel_fro=fid.get("rel_fro"), max_abs=fid.get("max_abs"),
            cos_mean=fid.get("cos_mean"), cos_p05=fid.get("cos_p05"),
            n_fid_samples=fid.get("n_fid_samples"),
        )
        runner._attach_cost(row_v1, mode="v1", max_lag_sec=float(lag),
                            window_sec=float(window_sec), auto_cc=auto_cc)

        results.append(row_conv)
        results.append(row_v1)

    return results

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Robustness benchmark for CC engines (No plotting dependencies).")
    p.add_argument("--cc_config", type=str, required=True, help="Path to configs/cc.yaml")
    p.add_argument("--outdir", type=str, default="./data/benchmarks", help="Output directory")
    p.add_argument("--n_files", type=int, default=3, help="Number of files in golden subset")
    p.add_argument("--repeats", type=int, default=2, help="Repeats per run (discard first)")
    p.add_argument("--skip_scaling", action="store_true")
    p.add_argument("--skip_complexity", action="store_true")
    p.add_argument("--skip_preprocess", action="store_true",
                   help="Skip the preprocessing-backend experiment")
    p.add_argument("--preprocess_modes", type=str, nargs="*", default=None,
                   help="Preprocess backends to evaluate "
                        "(default: hybrid pure_torch pure_numpy)")
    p.add_argument("--cleanup", action="store_true", help="Delete bench outputs after run")
    p.add_argument("--max_cores", type=int, default=None, help="Max njobs to test")
    p.add_argument("--cores", type=int, nargs="*", default=None, help="Explicit cores list")
    p.add_argument("--window_sec", type=float, default=60.0, help="xcorr_seg_sec for complexity sweep")
    p.add_argument("--njobs_complexity", type=int, default=4, help="njobs to use during complexity sweep")
    p.add_argument("--lags", type=float, nargs="*", default=None, help="Explicit lag list in seconds")
    p.add_argument("--auto_cc", action="store_true", help="Force Auto-Correlation mode")
    return p.parse_args(args=argv)

def main() -> None:
    mp.set_start_method("spawn", force=True)
    args = parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / "benchmark_results.csv"
    manifest_path = outdir / "run_manifest.json"

    cc_cfg_path = Path(args.cc_config)
    cc_cfg = load_config(Path(args.cc_config))

    if args.auto_cc:
        if "xcorr" not in cc_cfg: cc_cfg["xcorr"] = {}
        cc_cfg["xcorr"]["auto_cc"] = True

    data_root = Path(get_cfg(cc_cfg, ["paths", "data_root"], required=True)).expanduser().resolve()
    bench_files = _pick_benchmark_files(data_root, n_files=int(args.n_files))
    logger.info("Golden subset (%d files): %s", len(bench_files), [p.name for p in bench_files])

    if args.cores is not None and len(args.cores) > 0:
        cores_list = [max(1, int(x)) for x in args.cores]
    else:
        slurm_cores = os.environ.get("SLURM_CPUS_PER_TASK")
        total = int(slurm_cores) if slurm_cores else (os.cpu_count() or 4)
        if args.max_cores is not None: total = min(total, int(args.max_cores))
        cores_list = []
        p2 = 1
        while p2 <= total:
            cores_list.append(p2)
            p2 *= 2
        if total not in cores_list: cores_list.append(total)

    lags = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0] if not args.lags else [float(x) for x in args.lags]

    try: cfg_text = cc_cfg_path.read_text()
    except Exception: cfg_text = ""

    preprocess_modes = (
        [str(m).strip().lower() for m in args.preprocess_modes]
        if args.preprocess_modes else list(PREPROCESS_MODES)
    )

    manifest = {
        "cc_config": str(cc_cfg_path.resolve()), "cc_config_hash16": _hash_text(cfg_text),
        "data_root": str(data_root), "golden_subset": [str(p) for p in bench_files],
        "n_files": int(args.n_files), "repeats": int(args.repeats), "cores_list": cores_list,
        "window_sec": float(args.window_sec), "njobs_complexity": int(args.njobs_complexity),
        "lags_sec": lags, "skip_scaling": bool(args.skip_scaling), "skip_complexity": bool(args.skip_complexity),
        "skip_preprocess": bool(args.skip_preprocess), "preprocess_modes": preprocess_modes,
        "is_auto_cc": bool(get_cfg(cc_cfg, ["xcorr", "auto_cc"], False)),
        "environment": capture_environment(),
    }
    write_manifest(manifest_path, manifest)

    runner = BenchmarkRunner(cc_cfg_path, bench_files, outdir)
    if args.auto_cc:
        # BenchmarkRunner re-loads the config from disk, so the --auto_cc
        # override applied to cc_cfg above must be propagated explicitly.
        _set_nested(runner.base_cfg, "xcorr.auto_cc", True)
    results: List[RunResult] = []

    if not args.skip_preprocess:
        results.extend(run_preprocess_test(
            runner, preprocess_modes, repeats=max(3, int(args.repeats)),
        ))
        checkpoint_csv(results, csv_path)

    if not args.skip_scaling:
        results.extend(run_scaling_test(runner, cores_list, window_sec=float(args.window_sec), repeats=int(args.repeats)))
        checkpoint_csv(results, csv_path)

    if not args.skip_complexity:
        results.extend(run_complexity_test(runner, lags_sec=lags, window_sec=float(args.window_sec), njobs=int(args.njobs_complexity), repeats=int(args.repeats)))
        checkpoint_csv(results, csv_path)

    if results:
        df = pd.DataFrame([r.__dict__ for r in results])
        df.to_csv(csv_path, index=False)
        logger.info("Saved results: %s", csv_path)

    if args.cleanup:
        runner.cleanup()

if __name__ == "__main__":
    main()

# Example
# python3 -m src.eval \
#   --cc_config configs/urban_cc.yaml \
#   --outdir data/benchmarks/urban \
#   --n_files 16 \
#   --repeats 4 \
#   --cores 1 2 4 8 16 \
#   --window_sec 60 \
#   --njobs_complexity 1 \
#   --lags 0.5 1 2 3 4 5 6 \
#   --preprocess_modes hybrid pure_torch pure_numpy \
#   --cleanup
