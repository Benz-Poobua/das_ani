"""
:module: src/eval.py
:author: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Stress-test the CC pipeline (Scalability, Complexity, Fidelity, I/O Profiling).

This is the "Test Pilot":
- runs the CC pipeline live with controlled overrides
- supports both Cross-Correlation (structural) and Auto-Correlation (CWI/dv/v) modes
- dynamically compares NPZ vs Zarr if Zarr datasets are present
- measures total wall-clock time, plus inner I/O and CC times
- calculates Amdahl's Law & Universal Scalability Law bounds dynamically
- performs deterministic fidelity checks on one chosen output

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
import multiprocessing as mp
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from scipy.optimize import curve_fit

import numpy as np
import pandas as pd

from src.cc import process_single_file, _worker_warmup
from src.utils import load_config, get_cfg
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
# Amdahl & USL helpers 
# -----------------------------------------------------------------------------
def amdahl_speedup(p: int, s: float) -> float: 
    p = max(1, int(p))
    s = min(max(float(s), 0.0), 1.0)
    return 1.0 / (s + (1.0 - s) / float(p))

def estimate_serial_fraction(p: int, speedup: float) -> float:
    p = int(p)
    if p < 2 or float(speedup) <= 0.0:
        return float("nan")
    denom = (1.0 - 1.0 / float(p))
    if denom <= 0.0:
        return float("nan")
    s = (1.0 / float(speedup) - 1.0 / float(p)) / denom
    return float(min(max(s, 0.0), 1.0))

def fit_amdahl_s(ps: List[int], speedups: List[float]) -> float:
    pairs = [(int(p), float(S)) for p, S in zip(ps, speedups) if int(p) >= 1 and float(S) > 0]
    if len(pairs) < 2:
        return float("nan")

    best_s = 0.0
    best_err = float("inf")

    for step in (1e-2, 1e-3, 1e-4):
        lo = max(0.0, best_s - 5 * step) if best_err < float("inf") else 0.0
        hi = min(1.0, best_s + 5 * step) if best_err < float("inf") else 1.0
        grid = np.arange(lo, hi + step, step, dtype=np.float64)
        for s in grid:
            err = sum((amdahl_speedup(p, float(s)) - Sobs) ** 2 for p, Sobs in pairs)
            if err < best_err:
                best_err = err
                best_s = float(s)

    return float(best_s)

def usl_speedup(p: int, sigma: float, kappa: float) -> float:
    p = float(p)
    if p < 1: return 0.0
    denom = 1.0 + sigma * (p - 1.0) + kappa * p * (p - 1.0)
    return p / denom

def usl_model(p, sigma, kappa):
    p = np.asarray(p, dtype=float)
    denom = 1.0 + sigma * (p - 1.0) + kappa * p * (p - 1.0)
    return np.divide(p, denom, out=np.zeros_like(p), where=denom != 0)

def fit_usl_params(ps: List[int], speedups: List[float]) -> Tuple[float, float]:
    pairs = [(int(p), float(S)) for p, S in zip(ps, speedups) if int(p) >= 1 and float(S) > 0]
    if len(pairs) < 2: return 0.1, 0.0  

    best_params = (0.1, 0.0)
    best_err = float("inf")

    for s in np.linspace(0.0, 0.5, 50):
        for k in np.linspace(0.0, 0.02, 50):
            err = np.sum([(usl_speedup(p, s, k) - Sobs)**2 for p, Sobs in pairs])
            if err < best_err:
                best_err = err
                best_params = (float(s), float(k))
    return best_params

def fit_usl_params_precise(ps: List[int], speedups: List[float]) -> Tuple[float, float]:
    ps_arr = np.array(ps, dtype=float)
    ss_arr = np.array(speedups, dtype=float)
    
    if len(ps_arr) < 3:
        return fit_usl_params(ps, speedups) 
    try:
        popt, _ = curve_fit(usl_model, ps_arr, ss_arr, p0=[0.1, 0.001], bounds=([0.0, 0.0], [1.0, 1.0]))
        return float(popt[0]), float(popt[1])
    except Exception:
        return fit_usl_params(ps, speedups)
    
def predict_max_cores(sigma: float, kappa: float) -> int:
    if kappa <= 0: return 999 
    return int(round(np.sqrt((1.0 - sigma) / kappa)))

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

# -----------------------------------------------------------------------------
# Config / Selection Helpers
# -----------------------------------------------------------------------------
def _deepcopy_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(cfg)

def _set_nested(cfg: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    d = cfg
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict): d[k] = {}
        d = d[k]
    d[keys[-1]] = value

def _pick_benchmark_files(data_root: Path, n_files: int, ext: str = ".npz") -> List[Path]:
    """
    Safely discovers files based on format, preventing .zarr internal chunk reads.
    """
    if ext == ".zarr":
        files = [p for p in data_root.rglob("*.zarr") if p.is_dir()]
    else:
        files = [p for p in data_root.rglob(f"*{ext}") if p.is_file()]
        
    if not files:
        raise RuntimeError(f"No {ext} datasets found under {data_root}")
    return sorted(files)[: max(1, int(n_files))]

def _first_vs_idx_from_cfg(cfg: Dict[str, Any]) -> int:
    first_chan = int(get_cfg(cfg, ["data", "first_chan"], required=True))
    last_chan = int(get_cfg(cfg, ["data", "last_chan"], required=True))
    src_stride = int(get_cfg(cfg, ["data", "src_stride"], 10))
    if last_chan < first_chan:
        raise ValueError("data.last_chan must be >= data.first_chan")
    src_ch_all_num = np.arange(first_chan, last_chan + 1, src_stride, dtype=int)
    return int(src_ch_all_num[0] - first_chan)

def _output_path_for(cfg: Dict[str, Any], file_path: Path, *, vs_idx: int, mode: str) -> Path:
    out_root = Path(get_cfg(cfg, ["paths", "output_root"], required=True)).expanduser().resolve()
    basename = file_path.name
    is_auto = bool(get_cfg(cfg, ["xcorr", "auto_cc"], False))
    basename = basename.replace(".zarr", "").replace(".npz", "")
    
    if is_auto:
        return out_root / f"{basename}_auto_{mode}.npy"
    else:
        return out_root / f"{basename}_cc_{vs_idx:03d}_{mode}.npy"
    
def _load_npy(path: Path) -> np.ndarray:
    return np.load(path)

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
    
    # Timing Components
    wall_sec: float
    io_sec: float     # Average I/O per file
    cc_sec: float     # Average CC compute per file

    max_lag_sec: Optional[float] = None
    window_sec: Optional[float] = None
    ratio_lag_win: Optional[float] = None

    # Stats for Total Wall
    wall_mean: Optional[float] = None
    wall_std: Optional[float] = None
    wall_p25: Optional[float] = None
    wall_p75: Optional[float] = None
    n_eff: Optional[int] = None

    # Scalability Limits (Amdahl / USL)
    amdahl_s: Optional[float] = None
    usl_sigma: Optional[float] = None
    usl_kappa: Optional[float] = None
    max_optimal_cores: Optional[int] = None

    # Fidelity
    rel_fro: Optional[float] = None
    max_abs: Optional[float] = None
    cos_mean: Optional[float] = None
    cos_p05: Optional[float] = None

def _process_unpack(args):
    return process_single_file(*args)

class BenchmarkRunner:
    def __init__(self, cc_config_path: Path, files: List[Path], out_dir: Path, io_format: str, data_root: Path):
        self.base_cfg: Dict[str, Any] = load_config(cc_config_path)
        self.files = files
        self.out_dir = out_dir
        self.io_format = io_format
        self.data_root = data_root
        
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.bench_root = (self.out_dir / "bench_outputs").resolve()
        self.bench_root.mkdir(parents=True, exist_ok=True)

        fs_raw = float(get_cfg(self.base_cfg, ["data", "fs_raw"], required=True))
        dec = int(get_cfg(self.base_cfg, ["preprocess", "decimation"], 1))
        self.fs_proc = fs_raw / float(dec)
        self.vs_idx0 = _first_vs_idx_from_cfg(self.base_cfg)

    def _prepare_cfg(self, *, run_id: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        cfg = _deepcopy_cfg(self.base_cfg)
        for k, v in overrides.items():
            _set_nested(cfg, k, v)
            
        _set_nested(cfg, "paths.data_root", str(self.data_root))
        _set_nested(cfg, "paths.output_root", str(self.bench_root / run_id))

        if "perf" in cfg:
            cfg["perf"]["enabled"] = False
        return cfg

    def _run_pool(self, cfg: Dict[str, Any]) -> Dict[str, float]:
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
            initializer = partial(
                _worker_warmup, mode=mode, npts_seg=npts_seg, max_lag_samples=M,
                v1_fft_snap_pow2=v1_fft_snap_pow2, v1_fallback=v1_fallback, threads_per_proc=threads_per_proc,
            )

        total_io = 0.0
        total_cc = 0.0
        valid_returns = 0

        t0 = time.perf_counter()
        
        with mp.Pool(processes=njobs, initializer=initializer, maxtasksperchild=1) as pool:
            task_args = [(str(f), cfg) for f in self.files]
            
            for res in pool.imap_unordered(_process_unpack, task_args, chunksize=1):
                if isinstance(res, dict):
                    total_io += res.get("io_time", 0.0)
                    total_cc += res.get("cc_time", 0.0)
                    valid_returns += 1

        t1 = time.perf_counter()
        wall_time = float(t1 - t0)

        return {
            "wall_sec": wall_time,
            "io_sec": float(total_io / valid_returns) if valid_returns > 0 else 0.0,
            "cc_sec": float(total_cc / valid_returns) if valid_returns > 0 else 0.0
        }

    def run_batch(self, *, run_id: str, overrides: Dict[str, Any], repeats: int = 2) -> Dict[str, float]:
        repeats = max(1, int(repeats))
        times_wall, times_io, times_cc = [], [], []
        
        for r in range(repeats):
            stats = self._run_pool(self._prepare_cfg(run_id=f"{run_id}_rep{r}", overrides=overrides))
            times_wall.append(stats["wall_sec"])
            times_io.append(stats["io_sec"])
            times_cc.append(stats["cc_sec"])

        # Discard warmup
        if repeats > 1:
            times_wall = times_wall[1:]
            times_io = times_io[1:]
            times_cc = times_cc[1:]

        arr = np.asarray(times_wall, dtype=np.float64)
        return {
            "wall_sec": float(np.median(arr)),          
            "wall_mean": float(np.mean(arr)),
            "wall_std": float(np.std(arr, ddof=1)) if arr.size >= 2 else float("nan"),
            "wall_p25": float(np.percentile(arr, 25)),
            "wall_p75": float(np.percentile(arr, 75)),
            "io_sec": float(np.median(times_io)),
            "cc_sec": float(np.median(times_cc)),
            "n_eff": int(arr.size),
        }

    def compare_fidelity_for_last_run(self, *, conv_cfg: Dict[str, Any], v1_cfg: Dict[str, Any]) -> Dict[str, float]:
        test_file = self.files[0]
        vs_idx = int(self.vs_idx0)

        p_conv = _output_path_for(conv_cfg, test_file, vs_idx=vs_idx, mode="conventional")
        p_v1 = _output_path_for(v1_cfg, test_file, vs_idx=vs_idx, mode="v1")

        if not p_conv.exists(): raise FileNotFoundError(f"Missing conventional output: {p_conv}")
        if not p_v1.exists(): raise FileNotFoundError(f"Missing v1 output: {p_v1}")

        return fidelity_metrics(_load_npy(p_v1), _load_npy(p_conv))

    def cleanup(self) -> None:
        if self.bench_root.exists():
            shutil.rmtree(self.bench_root)
            logger.info("Cleaned benchmark outputs: %s", self.bench_root)


# -----------------------------------------------------------------------------
# Experiments
# -----------------------------------------------------------------------------
def run_scaling_test(runner: BenchmarkRunner, cores_list: List[int], *, window_sec: float, repeats: int = 2) -> List[RunResult]:
    logger.info(f"=== Experiment: Strong scaling [{runner.io_format.upper()}] ===")
    all_results: List[RunResult] = []
    max_lag_sec_base = float(get_cfg(runner.base_cfg, ["xcorr", "max_lag_sec"], 4.0))

    for mode in ("conventional", "v1"):
        mode_results: List[RunResult] = []
        for p in cores_list:
            run_id = f"scale_{runner.io_format}_{mode}_p{p}"
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

            res = RunResult(
                experiment="scaling",
                mode=mode,
                njobs=p,
                io_format=runner.io_format,
                wall_sec=stats["wall_sec"],
                io_sec=stats["io_sec"],
                cc_sec=stats["cc_sec"],
                n_files=len(runner.files),
                max_lag_sec=max_lag_sec_base,
                window_sec=float(window_sec),
                ratio_lag_win=float(max_lag_sec_base / window_sec),
                wall_mean=stats["wall_mean"],
                wall_std=stats["wall_std"],
                wall_p25=stats["wall_p25"],
                wall_p75=stats["wall_p75"],
                n_eff=stats["n_eff"],
            )
            mode_results.append(res)
            logger.info("[%s|%s] p=%d wall=%.2fs (Avg file I/O=%.2fs, Avg file CC=%.2fs)", mode, runner.io_format, p, stats["wall_sec"], stats["io_sec"], stats["cc_sec"])
            
        # ---------------------------------------------------------
        # Dynamically calculate Amdahl's Law and USL Parameters
        # ---------------------------------------------------------
        t1 = next((r.wall_sec for r in mode_results if r.njobs == 1), None)
        if t1 is not None and len(mode_results) > 1:
            ps = [r.njobs for r in mode_results]
            speedups = [t1 / r.wall_sec if r.wall_sec > 0 else 0 for r in mode_results]
            
            s_est = fit_amdahl_s(ps, speedups)
            sigma, kappa = fit_usl_params_precise(ps, speedups)
            opt_cores = predict_max_cores(sigma, kappa)
            
            logger.info(f"  -> Amdahl's Law (Serial Fraction 's'): {s_est:.4f}")
            logger.info(f"  -> Universal Scalability Law (sigma={sigma:.4f}, kappa={kappa:.4f}): Peak Scaling Cores = {opt_cores}")
            
            # Attach to dataclass so it saves to CSV
            for r in mode_results:
                r.amdahl_s = s_est
                r.usl_sigma = sigma
                r.usl_kappa = kappa
                r.max_optimal_cores = opt_cores
                
        all_results.extend(mode_results)

    return all_results

def run_complexity_test(runner: BenchmarkRunner, lags_sec: List[float], *, window_sec: float, njobs: int, repeats: int = 2) -> List[RunResult]:
    logger.info(f"=== Experiment: Complexity sweep (lag) [{runner.io_format.upper()}] ===")
    results: List[RunResult] = []

    for lag in lags_sec:
        lag = float(lag)
        if lag <= 0 or lag >= window_sec: continue
        
        window_sec = float(window_sec)
        ratio = lag / window_sec

        conv_id = f"complex_{runner.io_format}_conv_L{lag:g}_W{window_sec:g}_p{njobs}"
        stats_conv = runner.run_batch(
            run_id=conv_id,
            overrides={
                "runtime.njobs": int(njobs),
                "xcorr.mode": "conventional",
                "xcorr.max_lag_sec": float(lag),
                "xcorr.xcorr_seg_sec": float(window_sec),
            },
            repeats=repeats,
        )
        
        v1_id = f"complex_{runner.io_format}_v1_L{lag:g}_W{window_sec:g}_p{njobs}"
        stats_v1 = runner.run_batch(
            run_id=v1_id,
            overrides={
                "runtime.njobs": int(njobs),
                "xcorr.mode": "v1",
                "xcorr.max_lag_sec": float(lag),
                "xcorr.xcorr_seg_sec": float(window_sec),
                "xcorr.xcorr_seg_sec_v1": float(window_sec),
            },
            repeats=repeats,
        )

        conv_cfg = runner._prepare_cfg(run_id=f"{conv_id}_rep{repeats-1}", overrides={"runtime.njobs": int(njobs), "xcorr.mode": "conventional", "xcorr.max_lag_sec": float(lag), "xcorr.xcorr_seg_sec": float(window_sec)})
        v1_cfg = runner._prepare_cfg(run_id=f"{v1_id}_rep{repeats-1}", overrides={"runtime.njobs": int(njobs), "xcorr.mode": "v1", "xcorr.max_lag_sec": float(lag), "xcorr.xcorr_seg_sec": float(window_sec), "xcorr.xcorr_seg_sec_v1": float(window_sec)})

        fid = {}
        try:
            fid = runner.compare_fidelity_for_last_run(conv_cfg=conv_cfg, v1_cfg=v1_cfg)
        except Exception as e:
            logger.warning("Fidelity check failed at lag=%.3f: %s", lag, e)

        results.append(RunResult(
            experiment="complexity", mode="conventional", njobs=int(njobs), io_format=runner.io_format,
            wall_sec=stats_conv["wall_sec"], io_sec=stats_conv["io_sec"], cc_sec=stats_conv["cc_sec"],
            wall_mean=stats_conv["wall_mean"], wall_std=stats_conv["wall_std"], wall_p25=stats_conv["wall_p25"], wall_p75=stats_conv["wall_p75"],
            n_eff=stats_conv["n_eff"], n_files=len(runner.files), max_lag_sec=float(lag), window_sec=float(window_sec), ratio_lag_win=float(ratio),
        ))
        
        results.append(RunResult(
            experiment="complexity", mode="v1", njobs=int(njobs), io_format=runner.io_format,
            wall_sec=stats_v1["wall_sec"], io_sec=stats_v1["io_sec"], cc_sec=stats_v1["cc_sec"],
            wall_mean=stats_v1["wall_mean"], wall_std=stats_v1["wall_std"], wall_p25=stats_v1["wall_p25"], wall_p75=stats_v1["wall_p75"],
            n_eff=stats_v1["n_eff"], n_files=len(runner.files), max_lag_sec=float(lag), window_sec=float(window_sec), ratio_lag_win=float(ratio),
            rel_fro=fid.get("rel_fro"), max_abs=fid.get("max_abs"), cos_mean=fid.get("cos_mean"), cos_p05=fid.get("cos_p05"),
        ))

    return results

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Robustness & performance benchmark for Cross-Correlation. I/O vs compute profiled.")
    p.add_argument("--cc_config", type=str, required=True, help="Path to configs/cc.yaml")
    p.add_argument("--outdir", type=str, default="./data/benchmarks", help="Output directory")
    p.add_argument("--n_files", type=int, default=3, help="Number of files in golden subset")
    p.add_argument("--repeats", type=int, default=2, help="Repeats per run (discard first)")
    p.add_argument("--skip_scaling", action="store_true")
    p.add_argument("--skip_complexity", action="store_true")
    p.add_argument("--cleanup", action="store_true", help="Delete bench outputs after run")

    p.add_argument("--max_cores", type=int, default=None, help="Max njobs to test (default: cpu_count)")
    p.add_argument("--cores", type=int, nargs="*", default=None, help="Explicit cores list")

    p.add_argument("--window_sec", type=float, default=60.0, help="xcorr_seg_sec for complexity sweep")
    p.add_argument("--njobs_complexity", type=int, default=4, help="njobs to use during complexity sweep")
    p.add_argument("--lags", type=float, nargs="*", default=None, help="Explicit lag list in seconds")

    p.add_argument("--auto_cc", action="store_true", help="Force Auto-Correlation mode.")

    return p.parse_args(args=argv)

def main() -> None:
    mp.set_start_method("spawn", force=True)
    args = parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / "benchmark_results.csv"
    manifest_path = outdir / "run_manifest.json"

    cc_cfg_path = Path(args.cc_config)
    cc_cfg = load_config(cc_cfg_path)

    if args.auto_cc:
        if "xcorr" not in cc_cfg: cc_cfg["xcorr"] = {}
        cc_cfg["xcorr"]["auto_cc"] = True

    data_root_npz = Path(get_cfg(cc_cfg, ["paths", "data_root"], required=True)).expanduser().resolve()
    data_root_zarr = data_root_npz.with_name(f"{data_root_npz.name}_zarr")

    # Construct the formats we will test based on directory existence
    formats_to_test = []
    
    # 1. NPZ (Base)
    bench_files_npz = _pick_benchmark_files(data_root_npz, n_files=int(args.n_files), ext=".npz")
    formats_to_test.append(("npz", bench_files_npz, data_root_npz))
    logger.info("Found NPZ directory. Golden subset: %d files", len(bench_files_npz))

    # 2. Zarr (Dynamic)
    if data_root_zarr.exists():
        bench_files_zarr = _pick_benchmark_files(data_root_zarr, n_files=int(args.n_files), ext=".zarr")
        formats_to_test.append(("zarr", bench_files_zarr, data_root_zarr))
        logger.info("Found Zarr directory. Golden subset: %d folders", len(bench_files_zarr))
    else:
        logger.warning("Zarr directory not found at %s. Evaluating NPZ only.", data_root_zarr)

    # Core List
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

    # Lag list
    lags = [float(x) for x in args.lags] if (args.lags and len(args.lags) > 0) else [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]

    # Write manifest
    try: cfg_text = cc_cfg_path.read_text()
    except Exception: cfg_text = ""

    manifest = {
        "cc_config": str(cc_cfg_path.resolve()),
        "cc_config_hash16": _hash_text(cfg_text),
        "data_root_npz": str(data_root_npz),
        "data_root_zarr_checked": str(data_root_zarr),
        "formats_tested": [f[0] for f in formats_to_test],
        "n_files": int(args.n_files),
        "repeats": int(args.repeats),
        "cores_list": cores_list,
        "window_sec": float(args.window_sec),
        "njobs_complexity": int(args.njobs_complexity),
        "lags_sec": lags,
        "is_auto_cc": bool(get_cfg(cc_cfg, ["xcorr", "auto_cc"], False)), 
    }
    write_manifest(manifest_path, manifest)

    # Execute
    results: List[RunResult] = []
    
    for io_format, files, data_root in formats_to_test:
        runner = BenchmarkRunner(cc_cfg_path, files, outdir, io_format=io_format, data_root=data_root)

        if not args.skip_scaling:
            results.extend(run_scaling_test(runner, cores_list, window_sec=float(args.window_sec), repeats=int(args.repeats)))
            checkpoint_csv(results, csv_path)

        if not args.skip_complexity:
            results.extend(run_complexity_test(runner, lags_sec=lags, window_sec=float(args.window_sec), njobs=int(args.njobs_complexity), repeats=int(args.repeats)))
            checkpoint_csv(results, csv_path)

        if args.cleanup:
            runner.cleanup()

    # Final Output
    if results:
        df = pd.DataFrame([r.__dict__ for r in results])
        df.to_csv(csv_path, index=False)
        logger.info("Saved final results: %s", csv_path)

if __name__ == "__main__":
    main()