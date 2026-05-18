"""
:module: src/eval.py
:author: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Stress-test the CC pipeline (Scalability, Complexity, Fidelity).

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
from typing import Any, Dict, List, Optional, Sequence

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
# Benchmark Runner
# -----------------------------------------------------------------------------
@dataclass
class RunResult:
    experiment: str
    mode: str
    njobs: int
    io_format: str
    n_files: int
    
    # Timers (Now using Medians instead of Means)
    wall_sec: float       
    io_sec: float
    cc_sec: float

    # Robust Timers (Explicit tracking for redundancy)
    wall_median: Optional[float] = None
    io_median: Optional[float] = None
    cc_median: Optional[float] = None

    max_lag_sec: Optional[float] = None
    window_sec: Optional[float] = None
    ratio_lag_win: Optional[float] = None

    # Error bar stats for Batch Wall Time
    wall_mean: Optional[float] = None
    wall_std: Optional[float] = None
    wall_p25: Optional[float] = None
    wall_p75: Optional[float] = None
    n_eff: Optional[int] = None

    # Fidelity
    rel_fro: Optional[float] = None
    max_abs: Optional[float] = None
    cos_mean: Optional[float] = None
    cos_p05: Optional[float] = None

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
        dec = int(get_cfg(self.base_cfg, ["preprocess", "decimation"], 1))
        self.fs_proc = fs_raw / float(dec)
        self.vs_idx0 = _first_vs_idx_from_cfg(self.base_cfg)

    def _prepare_cfg(self, *, run_id: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        cfg = _deepcopy_cfg(self.base_cfg)
        for k, v in overrides.items():
            _set_nested(cfg, k, v)
        _set_nested(cfg, "paths.output_root", str(self.bench_root / run_id))
        if "perf" in cfg:
            cfg["perf"]["enabled"] = False
        return cfg

    def _run_pool(self, cfg: Dict[str, Any]) -> tuple[float, float, float]:
        """
        Runs the batch and returns: (Total Wall Time, MEDIAN IO sec, MEDIAN CC sec)
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
            do_compile   = bool(get_cfg(cfg, ["runtime", "torch_compile"], False))
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

        io_times = []
        cc_times = []

        t0 = time.perf_counter()
        with mp.Pool(processes=njobs, initializer=initializer, maxtasksperchild=1) as pool:
            task_args = [(str(f), cfg) for f in self.files]
            
            for res in pool.imap_unordered(_process_unpack, task_args, chunksize=1):
                if isinstance(res, dict):
                    if "io_sec" in res: io_times.append(res["io_sec"])
                    if "cc_sec" in res: cc_times.append(res["cc_sec"])
                elif isinstance(res, (tuple, list)):
                    for item in res:
                        if isinstance(item, dict):
                            if "io_sec" in item: io_times.append(item["io_sec"])
                            if "cc_sec" in item: cc_times.append(item["cc_sec"])

        t1 = time.perf_counter()
        
        wall_time = float(t1 - t0)
        
        # CHANGED: Now grabbing the Median of the files processed in this batch
        med_io = float(np.median(io_times)) if io_times else 0.0
        med_cc = float(np.median(cc_times)) if cc_times else 0.0

        return wall_time, med_io, med_cc

    def run_batch(self, *, run_id: str, overrides: Dict[str, Any], repeats: int = 2) -> Dict[str, float]:
        repeats = max(1, int(repeats))
        times_w, times_io, times_cc = [], [], []
        
        for r in range(repeats):
            w, io, cc = self._run_pool(self._prepare_cfg(run_id=f"{run_id}_rep{r}", overrides=overrides))
            times_w.append(w)
            times_io.append(io)
            times_cc.append(cc)

        # Discard warmup run for ALL metrics
        arr_w = np.asarray(times_w, dtype=np.float64) if repeats == 1 else np.asarray(times_w[1:], dtype=np.float64)
        arr_io = np.asarray(times_io, dtype=np.float64) if repeats == 1 else np.asarray(times_io[1:], dtype=np.float64)
        arr_cc = np.asarray(times_cc, dtype=np.float64) if repeats == 1 else np.asarray(times_cc[1:], dtype=np.float64)

        return {
            # CHANGED: Standard '_sec' variables are now strictly Medians
            "wall_sec": float(np.median(arr_w)), 
            "io_sec": float(np.median(arr_io)),
            "cc_sec": float(np.median(arr_cc)),
            
            # Explicit Tracking (same as above for backward compatibility)
            "wall_median": float(np.median(arr_w)), 
            "io_median": float(np.median(arr_io)),
            "cc_median": float(np.median(arr_cc)),
            
            "wall_mean": float(np.mean(arr_w)),
            "wall_std": float(np.std(arr_w, ddof=1)) if arr_w.size >= 2 else 0.0,
            "wall_p25": float(np.percentile(arr_w, 25)),
            "wall_p75": float(np.percentile(arr_w, 75)),
            "n_eff": int(arr_w.size),
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

    def cleanup(self) -> None:
        if self.bench_root.exists():
            shutil.rmtree(self.bench_root)
            logger.info("Cleaned benchmark outputs: %s", self.bench_root)

# -----------------------------------------------------------------------------
# Experiments
# -----------------------------------------------------------------------------
def run_scaling_test(runner: BenchmarkRunner, cores_list: List[int], *, window_sec: float, repeats: int = 2) -> List[RunResult]:
    logger.info("=== Experiment: Strong scaling ===")
    results: List[RunResult] = []
    max_lag_sec_base = float(get_cfg(runner.base_cfg, ["xcorr", "max_lag_sec"], 4.0))
    io_fmt = "zarr" if str(runner.files[0]).endswith("zarr") else "npz"

    for mode in ("conventional", "v1"):
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

            results.append(RunResult(
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
            ))
            logger.info("[%s] p=%d | Wall Med: %.2fs | CC Med: %.2fs | IO Med: %.2fs", 
                        mode, p, stats["wall_sec"], stats["cc_sec"], stats["io_sec"])
    return results

def run_complexity_test(runner: BenchmarkRunner, lags_sec: List[float], *, window_sec: float, njobs: int, repeats: int = 2) -> List[RunResult]:
    logger.info("=== Experiment: Complexity sweep (lag) ===")
    results: List[RunResult] = []
    io_fmt = "zarr" if str(runner.files[0]).endswith("zarr") else "npz"

    for lag in lags_sec:
        lag = float(lag)
        if lag <= 0 or lag >= window_sec: continue
        window_sec = float(window_sec)
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
            fid = runner.compare_fidelity_for_last_run(conv_cfg=conv_cfg, v1_cfg=v1_cfg)
            logger.info("Fidelity: rel_fro=%.3e max_abs=%.3e cos_mean=%.6f", fid["rel_fro"], fid["max_abs"], fid["cos_mean"])
        except Exception as e:
            logger.warning("Fidelity check failed at lag=%.3f: %s", lag, e)

        results.append(RunResult(
            experiment="complexity", mode="conventional", njobs=int(njobs), io_format=io_fmt,
            wall_sec=stats_conv["wall_sec"], io_sec=stats_conv["io_sec"], cc_sec=stats_conv["cc_sec"],
            wall_median=stats_conv["wall_median"], io_median=stats_conv["io_median"], cc_median=stats_conv["cc_median"],
            wall_mean=stats_conv["wall_mean"], wall_std=stats_conv["wall_std"], 
            wall_p25=stats_conv["wall_p25"], wall_p75=stats_conv["wall_p75"],
            n_eff=stats_conv["n_eff"], n_files=len(runner.files), max_lag_sec=float(lag), window_sec=float(window_sec), ratio_lag_win=float(ratio)
        ))
        
        results.append(RunResult(
            experiment="complexity", mode="v1", njobs=int(njobs), io_format=io_fmt,
            wall_sec=stats_v1["wall_sec"], io_sec=stats_v1["io_sec"], cc_sec=stats_v1["cc_sec"],
            wall_median=stats_v1["wall_median"], io_median=stats_v1["io_median"], cc_median=stats_v1["cc_median"],
            wall_mean=stats_v1["wall_mean"], wall_std=stats_v1["wall_std"], 
            wall_p25=stats_v1["wall_p25"], wall_p75=stats_v1["wall_p75"],
            n_eff=stats_v1["n_eff"], n_files=len(runner.files), max_lag_sec=float(lag), window_sec=float(window_sec), ratio_lag_win=float(ratio),
            rel_fro=fid.get("rel_fro"), max_abs=fid.get("max_abs"), cos_mean=fid.get("cos_mean"), cos_p05=fid.get("cos_p05")
        ))

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

    manifest = {
        "cc_config": str(cc_cfg_path.resolve()), "cc_config_hash16": _hash_text(cfg_text),
        "data_root": str(data_root), "golden_subset": [str(p) for p in bench_files],
        "n_files": int(args.n_files), "repeats": int(args.repeats), "cores_list": cores_list,
        "window_sec": float(args.window_sec), "njobs_complexity": int(args.njobs_complexity),
        "lags_sec": lags, "skip_scaling": bool(args.skip_scaling), "skip_complexity": bool(args.skip_complexity),
        "is_auto_cc": bool(get_cfg(cc_cfg, ["xcorr", "auto_cc"], False)), 
    }
    write_manifest(manifest_path, manifest)

    runner = BenchmarkRunner(cc_cfg_path, bench_files, outdir)
    results: List[RunResult] = []

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
#   --cc_config configs/offshore_cc.yaml \
#   --outdir data/benchmarks/offshore_test \
#   --n_files 16 \
#   --repeats 4 \
#   --cores 1 2 4 8 16 \
#   --window_sec 204.8 \
#   --njobs_complexity 1 \
#   --lags 5 10 20 30
#   --cleanup