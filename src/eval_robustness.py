"""
:module: src/eval_robustness.py
:author: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Stress-test the CC pipeline (Scalability, Complexity, Fidelity).

This is the "Test Pilot":
- runs the CC pipeline live with controlled overrides
- measures wall-clock time (makespan)
- probes scaling with njobs
- probes crossover with max_lag_sec (and later lag/window ratio)
- performs deterministic fidelity checks on one chosen output (same file, same VS)

Outputs:
- benchmark_results.csv
- run_manifest.json
- plots/*.png
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns

from src.cc import process_single_file, _worker_warmup
from src.utils import load_config, get_cfg
from src.error import rel_frobenius, max_abs_error, cosine_similarity_per_trace

# If you benchmark v1+cpp, optionally pre-load C++ once (prevents compile storm)
try:
    from src.ani import _maybe_load_cpp_extension
except Exception:
    _maybe_load_cpp_extension = None  # type: ignore


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Amdahl helpers
# -----------------------------------------------------------------------------
def amdahl_speedup(p: int, s: float) -> float: 
    """
    Amdahl speedup:
        S(p) = 1 / ( s + (1-s)/p )
    """
    p = max(1, int(p))
    s = float(s)
    s = min(max(s, 0.0), 1.0)
    return 1.0 / (s + (1.0 - s) / float(p))

def estimate_serial_fraction(p: int, speedup: float) -> float:
    """
    Estimate serial fraction from observed speedup:
        s = (1/S - 1/p) / (1 - 1/p)
    Only meaningful for p>=2.
    """
    p = int(p)
    if p < 2:
        return float("nan")
    S = float(speedup)
    if S <= 0.0:
        return float("nan")
    denom = (1.0 - 1.0 / float(p))
    if denom <= 0.0:
        return float("nan")
    s = (1.0 / S - 1.0 / float(p)) / denom
    return float(min(max(s, 0.0), 1.0))

def fit_amdahl_s(ps: List[int], speedups: List[float]) -> float:
    """
    Fit a single s by coarse-to-fine grid search.
    """
    pairs = [(int(p), float(S)) for p, S in zip(ps, speedups) if int(p) >= 1 and float(S) > 0]
    if len(pairs) < 2:
        return float("nan")

    best_s = 0.0
    best_err = float("inf")

    for step in (1e-2, 1e-3, 1e-4):
        if best_err < float("inf"):
            lo = max(0.0, best_s - 5 * step)
            hi = min(1.0, best_s + 5 * step)
        else:
            lo, hi = 0.0, 1.0

        grid = np.arange(lo, hi + step, step, dtype=np.float64)
        for s in grid:
            err = 0.0
            for p, Sobs in pairs:
                Sp = amdahl_speedup(p, float(s))
                err += (Sp - Sobs) ** 2
            if err < best_err:
                best_err = err
                best_s = float(s)

    return float(best_s)

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
    # safer than cfg.copy() because cfg is nested
    return copy.deepcopy(cfg)


def _set_nested(cfg: Dict[str, Any], path: str, value: Any) -> None:
    """
    Set cfg with dotted path like "runtime.njobs" or "xcorr.max_lag_sec".
    Creates intermediate dicts if needed.
    """
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
        raise RuntimeError(f"No .npz files found under {data_root}")
    return files[: max(1, int(n_files))]


def _first_vs_idx_from_cfg(cfg: Dict[str, Any]) -> int:
    """
    Match cc.py logic:
      src_ch_all_num = arange(first_chan, last_chan+1, src_stride)
      src_idx = src_ch_all_num - first_chan  (0-based index into data tensor)
    Return the first src_idx.
    """
    first_chan = int(get_cfg(cfg, ["data", "first_chan"], required=True))
    last_chan = int(get_cfg(cfg, ["data", "last_chan"], required=True))
    src_stride = int(get_cfg(cfg, ["data", "src_stride"], 10))
    if last_chan < first_chan:
        raise ValueError("data.last_chan must be >= data.first_chan")

    src_ch_all_num = np.arange(first_chan, last_chan + 1, src_stride, dtype=int)
    src_idx0 = int(src_ch_all_num[0] - first_chan)
    return src_idx0


def _output_path_for(cfg: Dict[str, Any], file_path: Path, *, vs_idx: int, mode: str) -> Path:
    out_root = Path(get_cfg(cfg, ["paths", "output_root"], required=True)).expanduser().resolve()
    basename = file_path.name
    return out_root / basename.replace(".npz", f"_cc_{vs_idx:03d}_{mode}.npy")


def _load_npy(path: Path) -> np.ndarray:
    return np.load(path)

# -----------------------------------------------------------------------------
# Metrics (fidelity)
# -----------------------------------------------------------------------------
def fidelity_metrics(v1_arr: np.ndarray, conv_arr: np.ndarray, eps: float = 1e-15) -> Dict[str, float]:
    """
    Compute robust waveform-based metrics (NCF matrices).
    """
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
    wall_sec: float
    n_files: int
    max_lag_sec: Optional[float] = None
    window_sec: Optional[float] = None
    ratio_lag_win: Optional[float] = None

    # fidelity (only meaningful for v1 vs conventional comparisons)
    rel_fro: Optional[float] = None
    max_abs: Optional[float] = None
    cos_mean: Optional[float] = None
    cos_p05: Optional[float] = None


class BenchmarkRunner:
    def __init__(self, cc_config_path: Path, files: List[Path], out_dir: Path):
        self.base_cfg: Dict[str, Any] = load_config(cc_config_path)
        self.files = files
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # isolate benchmark outputs here
        self.bench_root = (self.out_dir / "bench_outputs").resolve()
        self.bench_root.mkdir(parents=True, exist_ok=True)

        # derived
        fs_raw = float(get_cfg(self.base_cfg, ["data", "fs_raw"], required=True))
        dec = int(get_cfg(self.base_cfg, ["preprocess", "decimation"], 1))
        self.fs_proc = fs_raw / float(dec)

        self.vs_idx0 = _first_vs_idx_from_cfg(self.base_cfg)

    def _prepare_cfg(self, *, run_id: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        cfg = _deepcopy_cfg(self.base_cfg)

        # apply overrides
        for k, v in overrides.items():
            _set_nested(cfg, k, v)

        # force isolated output root
        _set_nested(cfg, "paths.output_root", str(self.bench_root / run_id))

        # reduce I/O noise
        if "perf" in cfg:
            cfg["perf"]["enabled"] = False

        # IMPORTANT: we keep runlog as-is because cc.py writes runlog unconditionally
        # (If later you add cfg-controlled runlog, we can disable here.)

        return cfg

    def _run_pool(self, cfg: Dict[str, Any]) -> float:
        """
        Run CC on self.files with ProcessPoolExecutor.
        Returns makespan (wall sec).
        """
        njobs = int(get_cfg(cfg, ["runtime", "njobs"], 1))
        njobs = max(1, njobs)

        mode = str(get_cfg(cfg, ["xcorr", "mode"], "conventional")).lower()
        use_cpp = bool(get_cfg(cfg, ["xcorr", "use_cpp"], True))

        # Preload C++ extension ONCE in parent for v1+cpp (matches cc.py main behavior)
        if mode == "v1" and use_cpp and _maybe_load_cpp_extension is not None:
            try:
                _maybe_load_cpp_extension()
            except Exception as e:
                logger.warning("C++ preload failed (continuing): %s", e)

        # compute warmup params (match cc.py main)
        max_lag_sec = float(get_cfg(cfg, ["xcorr", "max_lag_sec"], 4.0))
        xcorr_seg_sec = float(get_cfg(cfg, ["xcorr", "xcorr_seg_sec"], 8.0))
        if mode == "v1":
            xcorr_seg_sec = float(get_cfg(cfg, ["xcorr", "xcorr_seg_sec_v1"], xcorr_seg_sec))

        M = int(round(max_lag_sec * self.fs_proc))
        npts_seg = int(round(xcorr_seg_sec * self.fs_proc))

        v1_fft_snap_pow2 = bool(get_cfg(cfg, ["xcorr", "v1_fft_snap_pow2"], True))
        v1_fallback = str(get_cfg(cfg, ["xcorr", "v1_fallback"], "v1_2M"))

        # threads per proc like your cc.py does
        ncores = os.cpu_count() or 1
        threads_per_proc = max(1, ncores // njobs)

        initializer = None
        # Only warmup if parameters are valid
        if npts_seg > 0 and M > 0:
            from functools import partial
            initializer = partial(
                _worker_warmup,
                mode=mode,
                npts_seg=npts_seg,
                max_lag_samples=M,
                v1_fft_snap_pow2=v1_fft_snap_pow2,
                v1_fallback=v1_fallback,
                use_cpp=use_cpp,
                threads_per_proc=threads_per_proc,
            )

        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=njobs, initializer=initializer) as ex:
            futs = [ex.submit(process_single_file, str(f), cfg) for f in self.files]
            for fut in as_completed(futs):
                _ = fut.result()  # raise any exceptions
        t1 = time.perf_counter()
        return float(t1 - t0)

    def run_batch(self, *, run_id: str, overrides: Dict[str, Any], repeats: int = 2) -> float:
        """
        Run repeats times; discard first as warm-up. Return median of remaining.
        """
        repeats = max(1, int(repeats))
        times: List[float] = []
        for r in range(repeats):
            t = self._run_pool(self._prepare_cfg(run_id=f"{run_id}_rep{r}", overrides=overrides))
            times.append(t)

        if repeats == 1:
            return float(times[0])

        # discard first (warmup), take median of rest
        rest = np.asarray(times[1:], dtype=np.float64)
        return float(np.median(rest))

    def compare_fidelity_for_last_run(
        self,
        *,
        conv_cfg: Dict[str, Any],
        v1_cfg: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Load one deterministic output (first file, first vs) and compare.
        """
        test_file = self.files[0]
        vs_idx = int(self.vs_idx0)

        p_conv = _output_path_for(conv_cfg, test_file, vs_idx=vs_idx, mode="conventional")
        p_v1 = _output_path_for(v1_cfg, test_file, vs_idx=vs_idx, mode="v1")

        if not p_conv.exists():
            raise FileNotFoundError(f"Missing conventional output for fidelity: {p_conv}")
        if not p_v1.exists():
            raise FileNotFoundError(f"Missing v1 output for fidelity: {p_v1}")

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
def run_scaling_test(runner: BenchmarkRunner, cores_list: List[int], *, repeats: int = 2) -> List[RunResult]:
    logger.info("=== Experiment: Strong scaling ===")
    results: List[RunResult] = []

    for mode in ("conventional", "v1"):
        for p in cores_list:
            p = max(1, int(p))
            run_id = f"scale_{mode}_p{p}"

            t = runner.run_batch(
                run_id=run_id,
                overrides={"runtime.njobs": p, "xcorr.mode": mode},
                repeats=repeats,
            )

            results.append(RunResult(
                experiment="scaling",
                mode=mode,
                njobs=p,
                wall_sec=float(t),
                n_files=len(runner.files),
            ))
            logger.info("[%s] p=%d wall=%.2fs", mode, p, t)

    return results

def run_complexity_test(
    runner: BenchmarkRunner,
    lags_sec: List[float],
    *,
    window_sec: float,
    njobs: int,
    repeats: int = 2,
) -> List[RunResult]:
    """
    Sweep lag, run both modes each point, record makespan and fidelity.
    """
    logger.info("=== Experiment: Complexity sweep (lag) ===")
    results: List[RunResult] = []

    for lag in lags_sec:
        lag = float(lag)
        if lag <= 0:
            continue
        if lag >= window_sec:
            logger.warning("Skipping lag=%.3f >= window=%.3f", lag, window_sec)
            continue

        ratio = lag / float(window_sec)
        logger.info("Lag=%.3fs, Window=%.3fs, ratio=%.4f", lag, window_sec, ratio)

        # --- conventional ---
        conv_id = f"complex_conv_L{lag:g}_W{window_sec:g}_p{njobs}"
        t_conv = runner.run_batch(
            run_id=conv_id,
            overrides={
                "runtime.njobs": int(njobs),
                "xcorr.mode": "conventional",
                "xcorr.max_lag_sec": float(lag),
                "xcorr.xcorr_seg_sec": float(window_sec),
            },
            repeats=repeats,
        )

        # --- v1 ---
        v1_id = f"complex_v1_L{lag:g}_W{window_sec:g}_p{njobs}"
        t_v1 = runner.run_batch(
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

        # Build cfg objects pointing to the LAST repeat outputs for fidelity comparison.
        # (run_batch uses run_id_repX; we want the last rep directory)
        conv_cfg = runner._prepare_cfg(run_id=f"{conv_id}_rep{repeats-1}", overrides={
            "runtime.njobs": int(njobs),
            "xcorr.mode": "conventional",
            "xcorr.max_lag_sec": float(lag),
            "xcorr.xcorr_seg_sec": float(window_sec),
        })
        v1_cfg = runner._prepare_cfg(run_id=f"{v1_id}_rep{repeats-1}", overrides={
            "runtime.njobs": int(njobs),
            "xcorr.mode": "v1",
            "xcorr.max_lag_sec": float(lag),
            "xcorr.xcorr_seg_sec": float(window_sec),
            "xcorr.xcorr_seg_sec_v1": float(window_sec),
        })

        # fidelity
        fid: Dict[str, float] = {}
        try:
            fid = runner.compare_fidelity_for_last_run(conv_cfg=conv_cfg, v1_cfg=v1_cfg)
            logger.info(
                "Fidelity: rel_fro=%.3e max_abs=%.3e cos_mean=%.6f cos_p05=%.6f",
                fid["rel_fro"], fid["max_abs"], fid["cos_mean"], fid["cos_p05"]
            )
        except Exception as e:
            logger.warning("Fidelity check failed at lag=%.3f: %s", lag, e)

        results.append(RunResult(
            experiment="complexity",
            mode="conventional",
            njobs=int(njobs),
            wall_sec=float(t_conv),
            n_files=len(runner.files),
            max_lag_sec=float(lag),
            window_sec=float(window_sec),
            ratio_lag_win=float(ratio),
        ))
        results.append(RunResult(
            experiment="complexity",
            mode="v1",
            njobs=int(njobs),
            wall_sec=float(t_v1),
            n_files=len(runner.files),
            max_lag_sec=float(lag),
            window_sec=float(window_sec),
            ratio_lag_win=float(ratio),
            rel_fro=fid.get("rel_fro"),
            max_abs=fid.get("max_abs"),
            cos_mean=fid.get("cos_mean"),
            cos_p05=fid.get("cos_p05"),
        ))

    return results


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_results(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    def _window_label(subdf: pd.DataFrame) -> str:
        """Return a clean title suffix describing window_sec."""
        if "window_sec" not in subdf.columns:
            return ""
        vals = pd.to_numeric(subdf["window_sec"], errors="coerce").dropna().unique()
        vals = np.sort(vals)
        if vals.size == 0:
            return ""
        if vals.size == 1:
            return f"(window={vals[0]:g}s)"
        # multiple windows in same dataframe
        return f"(window in [{vals[0]:g}..{vals[-1]:g}]s, n={vals.size})"
    
    def _add_ratio_top_axis(ax: plt.Axes, *, window_sec: float) -> None:
        """
        Add top x-axis mapping:
            bottom: lag_sec
            top: ratio = lag_sec / window_sec
        Only valid if window_sec is constant for this axes.
        """
        if not np.isfinite(window_sec) or window_sec <= 0:
            return

        def sec_to_ratio(x):
            return np.asarray(x) / float(window_sec)

        def ratio_to_sec(r):
            return np.asarray(r) * float(window_sec)

        secax = ax.secondary_xaxis("top", functions=(sec_to_ratio, ratio_to_sec))
        secax.set_xlabel("lag/window ratio", labelpad=12)
        # nice-ish tick formatting
        secax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
        secax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))

    # -----------------------
    # SCALING PLOTS 
    # -----------------------
    dsc = df[df["experiment"] == "scaling"].copy()
    if not dsc.empty:
        plt.figure(figsize=(9, 5))
        ax = sns.lineplot(data=dsc, x="njobs", y="wall_sec", hue="mode", marker="o")
        ax.set_title("Strong scaling: Wall time vs cores")
        ax.set_xlabel("njobs (processes)")
        ax.set_ylabel("wall time (s)")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(plots_dir / "scaling_wall_time.png", dpi=220)
        plt.close()

        dsc2 = dsc.copy()
        dsc2["speedup"] = np.nan
        dsc2["efficiency"] = np.nan
        dsc2["serial_frac_est"] = np.nan

        for mode in dsc2["mode"].unique():
            sub = dsc2[dsc2["mode"] == mode].sort_values("njobs")
            p0 = int(sub["njobs"].iloc[0])
            t0 = float(sub["wall_sec"].iloc[0])

            for idx, row in sub.iterrows():
                p = int(row["njobs"])
                tp = float(row["wall_sec"])
                S = (t0 / tp) if tp > 0 else float("nan")
                dsc2.loc[idx, "speedup"] = S
                dsc2.loc[idx, "efficiency"] = S / (float(p) / float(p0)) if (p0 > 0 and p > 0) else float("nan")
                dsc2.loc[idx, "serial_frac_est"] = estimate_serial_fraction(p, S)
        
        plt.figure(figsize=(9, 5))
        ax = sns.lineplot(data=dsc2, x="njobs", y="speedup", hue="mode", marker="o")
        ax.set_title("Strong scaling: Speedup vs cores")
        ax.set_xlabel("njobs (processes)")
        ax.set_ylabel("speedup = T1 / Tp")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(plots_dir / "scaling_speedup.png", dpi=220)
        plt.close()

        plt.figure(figsize=(9, 5))
        ax = sns.lineplot(data=dsc2, x="njobs", y="speedup", hue="mode", marker="o")
        ax.set_title("Strong scaling: Speedup vs cores (Amdahl fit)")
        ax.set_xlabel("njobs (processes)")
        ax.set_ylabel("speedup = T1 / Tp")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        for mode in dsc2["mode"].unique():
            sub = dsc2[dsc2["mode"] == mode].sort_values("njobs")
            ps = [int(x) for x in sub["njobs"].tolist()]
            Ss = [float(x) for x in sub["speedup"].tolist()]
            s_fit = fit_amdahl_s(ps, Ss)
            if np.isfinite(s_fit):
                p_grid = sorted(set(ps))
                y_fit = [amdahl_speedup(p, s_fit) for p in p_grid]
                ax.plot(p_grid, y_fit, linestyle="--", linewidth=2, label=f"{mode} Amdahl (s={s_fit:.2f})")

        ax.legend(title="mode", frameon=True, loc="best")
        plt.tight_layout()
        plt.savefig(plots_dir / "scaling_speedup_amdahl.png", dpi=220)
        plt.close()

        plt.figure(figsize=(9, 5))
        ax = sns.lineplot(data=dsc2, x="njobs", y="efficiency", hue="mode", marker="o")
        ax.set_title("Strong scaling: Parallel efficiency vs cores")
        ax.set_xlabel("njobs (processes)")
        ax.set_ylabel("efficiency")
        ax.set_ylim(0.0, 1.05)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(plots_dir / "scaling_efficiency.png", dpi=220)
        plt.close()

        plt.figure(figsize=(9, 5))
        ax = sns.lineplot(data=dsc2, x="njobs", y="serial_frac_est", hue="mode", marker="o")
        ax.set_title("Amdahl diagnostic: Estimated serial fraction vs cores")
        ax.set_xlabel("njobs (processes)")
        ax.set_ylabel("serial fraction (estimated)")
        ax.set_ylim(0.0, 1.0)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(plots_dir / "scaling_serial_fraction.png", dpi=220)
        plt.close()

    # -----------------------
    # COMPLEXITY PLOTS 
    # -----------------------
    dcp_all = df[df["experiment"] == "complexity"].copy()
    if not dcp_all.empty:
        # If multiple window_sec appear, split plots per window to keep ratio axis correct.
        if "window_sec" in dcp_all.columns:
            wvals = pd.to_numeric(dcp_all["window_sec"], errors="coerce").dropna().unique()
            wvals = np.sort(wvals)
            groups = [dcp_all[dcp_all["window_sec"] == w] for w in wvals] if wvals.size > 0 else [dcp_all]
        else:
            groups = [dcp_all]

        for g in groups:
            g = g.copy()
            win_label = _window_label(g)

            # Determine window for ratio axis (only if single-valued)
            w_unique = pd.to_numeric(g.get("window_sec", pd.Series(dtype=float)), errors="coerce").dropna().unique()
            window_for_axis = float(w_unique[0]) if w_unique.size == 1 else None

            # 1) wall time vs lag
            plt.figure(figsize=(9, 5))
            ax = sns.lineplot(data=g, x="max_lag_sec", y="wall_sec", hue="mode", marker="o")
            ax.set_title(f"Complexity sweep: Wall time vs max lag {win_label}".strip(), pad=20)
            ax.set_xlabel("max_lag_sec (s)")
            ax.set_ylabel("wall time (s)")
            if window_for_axis is not None:
                _add_ratio_top_axis(ax, window_sec=window_for_axis)
            plt.tight_layout()

            suffix = f"_W{window_for_axis:g}s" if window_for_axis is not None else ""
            plt.savefig(plots_dir / f"complexity_wall_time_vs_lag{suffix}.png", dpi=220)
            plt.close()

            # 2) speedup vs lag (conv/v1)
            conv = g[g["mode"] == "conventional"][["max_lag_sec", "wall_sec"]].rename(columns={"wall_sec": "t_conv"})
            v1 = g[g["mode"] == "v1"][["max_lag_sec", "wall_sec"]].rename(columns={"wall_sec": "t_v1"})
            m = conv.merge(v1, on="max_lag_sec", how="inner")
            if not m.empty:
                m["speedup_conv_over_v1"] = m["t_conv"] / m["t_v1"]

                plt.figure(figsize=(9, 5))
                ax = sns.lineplot(data=m, x="max_lag_sec", y="speedup_conv_over_v1", marker="o")
                ax.set_title(f"Complexity sweep: Speedup (conv / v1) vs max lag {win_label}".strip(), pad=20)
                ax.set_xlabel("max_lag_sec (s)")
                ax.set_ylabel("speedup")
                if window_for_axis is not None:
                    _add_ratio_top_axis(ax, window_sec=window_for_axis)
                plt.tight_layout()
                plt.savefig(plots_dir / f"complexity_speedup_vs_lag{suffix}.png", dpi=220)
                plt.close()
            
            # 3) fidelity vs lag (v1 rows only)
            v1f = g[g["mode"] == "v1"].copy()
            if "rel_fro" in v1f.columns and v1f["rel_fro"].notna().any():
                plt.figure(figsize=(9, 5))
                ax = sns.lineplot(data=v1f, x="max_lag_sec", y="rel_fro", marker="o")
                ax.set_title(f"Fidelity: rel_fro (v1 vs conv) vs max lag {win_label}".strip(), pad=20)
                ax.set_xlabel("max_lag_sec (s)")
                ax.set_ylabel("rel_fro")
                ax.set_yscale("log")
                if window_for_axis is not None:
                    _add_ratio_top_axis(ax, window_sec=window_for_axis)
                plt.tight_layout()
                plt.savefig(plots_dir / f"fidelity_rel_fro_vs_lag{suffix}.png", dpi=220)
                plt.close()

            if "cos_p05" in v1f.columns and v1f["cos_p05"].notna().any():
                plt.figure(figsize=(9, 5))
                ax = sns.lineplot(data=v1f, x="max_lag_sec", y="cos_p05", marker="o")
                ax.set_title(f"Fidelity: cos_p05 (v1 vs conv) vs max lag {win_label}".strip(), pad=20)
                ax.set_xlabel("max_lag_sec (s)")
                ax.set_ylabel("cos_p05")
                if window_for_axis is not None:
                    _add_ratio_top_axis(ax, window_sec=window_for_axis)
                plt.tight_layout()
                plt.savefig(plots_dir / f"fidelity_cos_p05_vs_lag{suffix}.png", dpi=220)
                plt.close()

    logger.info("Saved plots to %s", plots_dir)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Robustness & performance benchmark for cc.py")
    p.add_argument("--cc_config", type=str, required=True, help="Path to configs/cc.yaml")
    p.add_argument("--outdir", type=str, default="./data/benchmarks", help="Output directory")
    p.add_argument("--n_files", type=int, default=3, help="Number of .npz files in golden subset")
    p.add_argument("--repeats", type=int, default=2, help="Repeats per run (discard first)")
    p.add_argument("--skip_scaling", action="store_true")
    p.add_argument("--skip_complexity", action="store_true")
    p.add_argument("--cleanup", action="store_true", help="Delete bench outputs after run")

    # scaling controls
    p.add_argument("--max_cores", type=int, default=None, help="Max njobs to test (default: cpu_count)")
    p.add_argument("--cores", type=int, nargs="*", default=None, help="Explicit cores list (overrides auto)")

    # complexity controls
    p.add_argument("--window_sec", type=float, default=60.0, help="xcorr_seg_sec for complexity sweep")
    p.add_argument("--njobs_complexity", type=int, default=4, help="njobs to use during complexity sweep")
    p.add_argument("--lags", type=float, nargs="*", default=None, help="Explicit lag list in seconds")

    # resume / plotting
    p.add_argument("--resume", action="store_true", help="If CSV exists, skip running and only re-plot.")
    p.add_argument("--force", action="store_true", help="Force rerun even if CSV exists.")
    p.add_argument("--no_plots", action="store_true", help="Do not generate plots (CSV only).")

    return p.parse_args(args=argv)


def main() -> None:
    mp.set_start_method("spawn", force=True)
    args = parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / "benchmark_results.csv"
    manifest_path = outdir / "run_manifest.json"

    # RESUME: plot-only from existing results
    if args.resume and (not args.force) and csv_path.exists():
        logger.info("[RESUME] Loading existing results: %s", csv_path)
        df = pd.read_csv(csv_path)
        if not args.no_plots:
            plot_results(df, outdir)
        else:
            logger.info("[RESUME] --no_plots set; skipping plot generation.")
        return

    cc_cfg_path = Path(args.cc_config)
    cc_cfg = load_config(Path(args.cc_config))
    data_root = Path(get_cfg(cc_cfg, ["paths", "data_root"], required=True)).expanduser().resolve()

    bench_files = _pick_benchmark_files(data_root, n_files=int(args.n_files))
    logger.info("Golden subset (%d files): %s", len(bench_files), [p.name for p in bench_files])

    # Determine exact cores list now (and write to manifest)
    if args.cores is not None and len(args.cores) > 0:
        cores_list = [max(1, int(x)) for x in args.cores]
    else:
        total = os.cpu_count() or 4
        if args.max_cores is not None:
            total = min(total, int(args.max_cores))
        cores_list = []
        p2 = 1
        while p2 <= total:
            cores_list.append(p2)
            p2 *= 2
        if total not in cores_list:
            cores_list.append(total)

    # Determine exact lags list now (and write to manifest)
    if args.lags is None or len(args.lags) == 0:
        lags = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
    else:
        lags = [float(x) for x in args.lags]

    # Write manifest
    try:
        cfg_text = cc_cfg_path.read_text()
    except Exception:
        cfg_text = ""

    manifest = {
        "cc_config": str(cc_cfg_path.resolve()),
        "cc_config_hash16": _hash_text(cfg_text),
        "data_root": str(data_root),
        "golden_subset": [str(p) for p in bench_files],
        "n_files": int(args.n_files),
        "repeats": int(args.repeats),
        "cores_list": cores_list,
        "window_sec": float(args.window_sec),
        "njobs_complexity": int(args.njobs_complexity),
        "lags_sec": lags,
        "skip_scaling": bool(args.skip_scaling),
        "skip_complexity": bool(args.skip_complexity),
    }
    write_manifest(manifest_path, manifest)
    logger.info("Wrote manifest: %s", manifest_path)

    runner = BenchmarkRunner(cc_cfg_path, bench_files, outdir)
    results: List[RunResult] = []

    # --- scaling ---
    if not args.skip_scaling:
        results.extend(run_scaling_test(runner, cores_list, repeats=int(args.repeats)))

    # --- complexity ---
    if not args.skip_complexity:
        results.extend(
            run_complexity_test(
                runner,
                lags_sec=lags,
                window_sec=float(args.window_sec),
                njobs=int(args.njobs_complexity),
                repeats=int(args.repeats),
            )
        )

    # write results + plots
    if results:
        df = pd.DataFrame([r.__dict__ for r in results])
        df.to_csv(csv_path, index=False)
        logger.info("Saved results: %s", csv_path)

        if not args.no_plots:
            plot_results(df, outdir)
        else:
            logger.info("--no_plots set; skipping plot generation.")

    if args.cleanup:
        runner.cleanup()


if __name__ == "__main__":
    main()


# Example
# python -m src.eval_robustness \
#   --cc_config configs/cc.yaml \
#   --outdir data/benchmarks/final \
#   --n_files 16 \
#   --repeats 1 \
#   --cores 1 2 4 8 16\
#   --window_sec 60 \
#   --njobs_complexity 16 \
#   --lags 0.5 1 2 4 5 10 20

# Re-plotting only from existing CSV:
# python - << 'EOF'
# import pandas as pd
# from pathlib import Path
# from src.eval_robustness import plot_results

# df = pd.read_csv("data/benchmarks/final/benchmark_results.csv")
# plot_results(df, Path("data/benchmarks/final"))
# EOF