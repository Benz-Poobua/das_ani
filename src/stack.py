"""
:module: src/stack.py
:auth: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Stacking engine for DAS Ambient Noise Interferometry.

Description:
This module provides high-performance aggregation of raw Cross-Correlation (NCF) 
and Auto-Correlation (ACF) functions. It is designed to handle long DAS 
deployments by prioritizing I/O efficiency and strict memory boundaries.

Computational Optimizations:
    1. **Streaming Accumulation:** Replaces standard list-based averaging with 
       a running fp64 accumulator. This ensures peak memory usage never exceeds 
       two arrays (Running Sum + Current Load), allowing the stacking of 
       thousands of files on a single CPU core.
    2. **Sliding-Window $O(N)$ Logic:** Implements a boundary-update algorithm for 
       rolling windows. Instead of re-stacking $W$ days for every step, the 
       engine simply adds the entering day and subtracts the leaving day. This 
       reduces computational complexity from $O(N \\cdot W)$ to $O(N)$.
    3. **I/O Caching:** Each base-stack file is loaded exactly once per Virtual 
       Source group, regardless of how many overlapping windows it belongs to, 
       yielding up to a 30x reduction in disk overhead.
    4. **HPC-Ready Multiprocessing:** Exploits the "embarrassingly parallel" 
       nature of Virtual Sources. By dispatching independent (VS, Method) 
       groups to different processes, the engine scales linearly across 
       multi-core Sherlock nodes.
"""
from __future__ import annotations

import argparse
import functools
import logging
import multiprocessing as mp
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from src.utils import timeit, load_config, get_cfg

logger = logging.getLogger(__name__)

# =====================================================
# Filename parsing
# =====================================================
# Matches YYYYMMDD and optionally _HHMMSS at the start of the filename.
_DATETIME_RE = re.compile(r"(\d{8})(?:_(\d{6}))?")
_VS_RE = re.compile(r"_(cc_\d+|auto)")

# Method suffix supports underscores in the method name (e.g. "v1_2M").
# Window unit accepts h, d, or the legacy literal "daily".
_CC_METHOD_RE = re.compile(
    r"_(?:cc_\d+|auto)(?:_(\d+[hd]|daily))?(?:_([A-Za-z][A-Za-z0-9_]*))?\.npy$"
)

# Window string parser: e.g. "1h", "30d".
_WINDOW_RE = re.compile(r"(\d+)([hd])")


def parse_date_vs(path: str | Path) -> Tuple[datetime, int]:
    """Extract (datetime, virtual-source ID) from filename. vs=-1 for auto-CC."""
    name = Path(path).name

    m_date = _DATETIME_RE.match(name)
    if m_date is None:
        raise ValueError(f"Cannot parse datetime from: {name}")

    date_str = m_date.group(1)
    time_str = m_date.group(2) if m_date.group(2) else "000000"
    dt_obj = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")

    m_vs = _VS_RE.search(name)
    if m_vs is None:
        raise ValueError(f"Cannot parse VS/Auto index from: {name}")

    match_str = m_vs.group(1)
    vs = -1 if match_str == "auto" else int(match_str.replace("cc_", ""))

    return dt_obj, vs


def parse_date_vs_method(path: str | Path) -> Tuple[datetime, int, Optional[str]]:
    """
    Extract (datetime, virtual-source ID, cc_method) from filename.

    Backward compatible: returns method=None if no method suffix is present.
    Auto-correlation files use vs_id=-1.

    Examples accepted:
        20210901_000000_cc_080.npy
        20211110_150000_cc_080_conventional.npy        (sub-daily raw CC)
        20211110_150000_auto_conventional.npy          (sub-daily raw auto-CC)
        20211110_150000_auto_1h_conventional.npy       (sub-daily stacked auto-CC)
        20210901_cc_080_30d_v1_2M.npy                  (30-day stack, v1_2M method)
    """
    name = Path(path).name
    dt_obj, vs = parse_date_vs(name)

    m_method = _CC_METHOD_RE.search(name)
    method = m_method.group(2) if m_method else None
    return dt_obj, vs, method


# =====================================================
# Helpers
# =====================================================
def _time_fmt_for(label: str) -> str:
    """
    Pick a strftime format from a window/base label.

    Days → date only (%Y%m%d). Hours → date + time (%Y%m%d_%H%M%S).
    Anything else falls back to the date-only format.
    """
    m = _WINDOW_RE.fullmatch(label.strip().lower())
    if m is None:
        return "%Y%m%d"
    return "%Y%m%d_%H%M%S" if m.group(2) == "h" else "%Y%m%d"


def _output_valid(path: Path, overwrite: bool) -> bool:
    """
    Return True if the output already exists, is loadable, and we shouldn't
    overwrite it. Returns False otherwise (i.e. the caller should proceed
    with computing this output).

    A corrupt half-written .npy from a killed previous run will fail to load
    and be recomputed, instead of silently being skipped.
    """
    if overwrite or not path.exists():
        return False
    try:
        # mmap_mode='r' avoids loading the whole array; we only need the header.
        np.load(path, mmap_mode="r")
        return True
    except Exception as e:
        logger.warning("Existing output %s is unreadable (%s); recomputing.", path, e)
        return False


def _streaming_mean(filelist: Sequence[Path]) -> Tuple[Optional[np.ndarray], int]:
    """
    Computes the elementwise mean of a large sequence of files using a 
    low-memory streaming accumulator.

    **Numerical Theory:**
    When summing thousands of seismic traces, floating-point rounding errors 
    can accumulate, especially if the individual correlation values are small. 
    This function uses a **float64 accumulator** to maintain high precision 
    during the summation before casting the final result back to float32 for storage.

    **Memory Theory:**
    By processing files one by one, the memory footprint remains constant 
    at $2 \times (\text{array size})$ regardless of whether we are stacking 
    10 files or 10,000. This prevents the "Memory Exhaustion" crashes common 
    in naive stacking implementations.

    :param filelist: A sequence of Paths to .npy files.
    :return: A tuple of (stacked_array_float32, count_of_successful_files).
    """
    total: Optional[np.ndarray] = None
    expected_shape: Optional[tuple] = None
    n = 0
    for f in filelist:
        try:
            a = np.load(f)
        except Exception as e:
            logger.warning("Failed to load %s: %s", f, e)
            continue
        if total is None:
            total = a.astype(np.float64, copy=True)
            expected_shape = a.shape
        else:
            if a.shape != expected_shape:
                logger.warning("Shape mismatch in %s: %s vs %s; skipping",
                               f, a.shape, expected_shape)
                continue
            total += a
        n += 1
    if n == 0 or total is None:
        return None, 0
    return (total / n).astype(np.float32), n


# =====================================================
# Workers (top-level for multiprocessing pickling)
# =====================================================
def _base_stack_worker(args: Tuple[Any, ...]) -> Optional[str]:
    """
    Stack one (datetime, vs, method) group → one output file.

    Args tuple:
        ((dt_obj, vs, method), filelist, out_base, base_label, overwrite)
    """
    (dt_obj, vs, method), filelist, out_base, base_label, overwrite = args

    suffix = f"_{method}" if method else ""
    vs_str = "auto" if vs == -1 else f"cc_{vs:03d}"
    time_fmt = _time_fmt_for(base_label)
    outname = f"{dt_obj.strftime(time_fmt)}_{vs_str}_{base_label}{suffix}.npy"
    outpath = Path(out_base) / outname

    if _output_valid(outpath, overwrite):
        return None

    stack, n = _streaming_mean([Path(f) for f in filelist])
    if stack is None:
        return None
    np.save(outpath, stack)
    return outname


def _window_stack_worker(args: Tuple[Any, ...]) -> int:
    """
    Rolling-window stack one (vs, method) group across all its dates.
    Each base file in the group is loaded at most once via an internal cache;
    the running sum is updated incrementally as the window slides.

    Args tuple:
        ((vs, method), items, out_root, window_str, win_val, win_unit,
         step_seconds, min_frac, overwrite)
        where items = list of (datetime, path).
        step_seconds is passed instead of timedelta because timedelta pickles
        cleanly but staying explicit avoids surprises.

    Returns the number of output files written (for logging).
    """
    ((vs, method), items, out_root, window_str, win_val, win_unit,
     step_seconds, min_frac, overwrite) = args

    out_root = Path(out_root)
    step_delta = timedelta(seconds=step_seconds)

    items = sorted(items, key=lambda x: x[0])
    dates = [d for d, _ in items]
    date_to_path: Dict[datetime, Path] = {d: Path(p) for d, p in items}

    suffix = f"_{method}" if method else ""
    vs_str = "auto" if vs == -1 else f"cc_{vs:03d}"
    time_fmt = "%Y%m%d_%H%M%S" if win_unit == "h" else "%Y%m%d"
    required_steps = max(1, int(np.ceil(win_val * float(min_frac))))

    # Running-sum state.
    running_sum: Optional[np.ndarray] = None
    expected_shape: Optional[tuple] = None
    file_cache: Dict[datetime, np.ndarray] = {}
    active_dates: set = set()
    n_outputs = 0

    for end_date in dates:
        start_date = end_date - step_delta * (win_val - 1)

        # Target window membership: the dates that *should* be in this window.
        target_window: set = set()
        for k in range(win_val):
            target = start_date + step_delta * k
            if target in date_to_path:
                target_window.add(target)

        entering = target_window - active_dates
        leaving = active_dates - target_window

        # Subtract leaving first so we can free their cache slots.
        for d in leaving:
            arr = file_cache.pop(d, None)
            if arr is not None and running_sum is not None:
                running_sum -= arr
            active_dates.discard(d)

        # Load + add entering.
        for d in entering:
            try:
                arr = np.load(date_to_path[d]).astype(np.float64)
            except Exception as e:
                logger.warning("Failed to load %s: %s", date_to_path[d], e)
                continue
            if running_sum is None:
                running_sum = arr.copy()
                expected_shape = arr.shape
            else:
                if arr.shape != expected_shape:
                    logger.warning("Shape mismatch %s: %s vs %s; skipping",
                                   date_to_path[d], arr.shape, expected_shape)
                    continue
                running_sum += arr
            file_cache[d] = arr
            active_dates.add(d)

        if len(active_dates) < required_steps or running_sum is None:
            continue

        outname = f"{end_date.strftime(time_fmt)}_{vs_str}_{window_str}{suffix}.npy"
        outpath = out_root / outname

        if _output_valid(outpath, overwrite):
            continue

        stack = (running_sum / len(active_dates)).astype(np.float32)
        np.save(outpath, stack)
        n_outputs += 1

    return n_outputs


# =====================================================
# Public API
# =====================================================
@timeit
def base_stack_ncf(
    raw_root: str | Path,
    out_base: str | Path,
    base_label: str,
    *,
    overwrite: bool = False,
    njobs: int = 1,
) -> None:
    """
    Average raw NCF slices into base stacks per (datetime, vs_id, method).

    Each group is independent, so groups are dispatched across a worker pool
    when njobs > 1. With njobs=1 the loop runs in-process (no pool overhead).
    """
    raw_root = Path(raw_root).expanduser().resolve()
    out_base = Path(out_base).expanduser().resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    all_files = sorted(raw_root.rglob("*.npy"))
    logger.info("Found %d raw NCF slices under %s", len(all_files), raw_root)

    groups: Dict[Tuple[datetime, int, Optional[str]], List[Path]] = {}
    for p in all_files:
        try:
            dt_obj, vs, method = parse_date_vs_method(p)
            groups.setdefault((dt_obj, vs, method), []).append(p)
        except Exception as e:
            logger.warning("Skipping %s: %s", p, e)

    task_args = [
        (key, [str(f) for f in filelist], str(out_base), base_label, overwrite)
        for key, filelist in groups.items()
    ]

    if njobs <= 1 or len(task_args) <= 1:
        for ta in tqdm(task_args, desc=f"Base stack ({base_label})"):
            _base_stack_worker(ta)
    else:
        with mp.get_context("spawn").Pool(processes=int(njobs)) as pool:
            for _ in tqdm(pool.imap_unordered(_base_stack_worker, task_args, chunksize=1),
                          total=len(task_args), desc=f"Base stack ({base_label}, njobs={njobs})"):
                pass


@timeit
def stack_ncf_window(
    base_root: str | Path,
    out_root: str | Path,
    window_str: str,
    *,
    overwrite: bool = False,
    min_frac: float = 0.9,
    njobs: int = 1,
) -> None:
    """
    Sliding-window stacking per (VS, method) using a running-sum accumulator.

    Each base file is loaded at most once per (vs, method) group, instead of
    re-loaded for every overlapping window. A small per-group file cache
    holds only the files currently inside the window; older files are freed
    as soon as they slide out.

    :param min_frac: Minimum fraction of required time steps (0.0 to 1.0).
                     e.g., 0.9 means a 10-day window needs at least 9 daily files.
    :param njobs: Workers across (vs, method) groups. Each group is internally
                  serial (running-sum has data dependencies), but groups are
                  embarrassingly parallel.
    """
    base_root = Path(base_root).expanduser().resolve()
    out_root = Path(out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    base_files = sorted(base_root.glob("*.npy"))
    if not base_files:
        logger.warning("No base stacks in %s", base_root)
        return

    m = _WINDOW_RE.fullmatch(window_str.strip().lower())
    if m is None:
        raise ValueError(f"window_str must match e.g. '2h' or '7d'; got {window_str!r}")
    win_val = int(m.group(1))
    win_unit = m.group(2)
    step_delta = timedelta(days=1) if win_unit == "d" else timedelta(hours=1)
    step_seconds = step_delta.total_seconds()

    # Bucket base files by (vs, method).
    vs_groups: Dict[Tuple[int, Optional[str]], List[Tuple[datetime, Path]]] = {}
    for p in base_files:
        try:
            dt_obj, vs, method = parse_date_vs_method(p)
            vs_groups.setdefault((vs, method), []).append((dt_obj, p))
        except Exception:
            continue

    task_args = [
        ((vs, method),
         [(d, str(p)) for d, p in items],
         str(out_root), window_str, win_val, win_unit,
         step_seconds, float(min_frac), overwrite)
        for (vs, method), items in vs_groups.items()
    ]

    if njobs <= 1 or len(task_args) <= 1:
        for ta in tqdm(task_args, desc=f"{window_str} stacks"):
            _window_stack_worker(ta)
    else:
        with mp.get_context("spawn").Pool(processes=int(njobs)) as pool:
            for _ in tqdm(pool.imap_unordered(_window_stack_worker, task_args, chunksize=1),
                          total=len(task_args), desc=f"{window_str} stacks (njobs={njobs})"):
                pass


# =====================================================
# Config-driven runner
# =====================================================
@dataclass(frozen=True)
class StackPlan:
    raw_root: Path
    stacks_root: Path
    overwrite: bool
    base_stack: str             # e.g., "1d", "1h"
    do_windows: Dict[str, bool] # e.g., {"7d": True, "2h": False}
    min_frac: float
    njobs: int


def build_stack_plan(cfg: Mapping[str, Any]) -> StackPlan:
    raw_root = Path(get_cfg(
        cfg, ["stacking", "raw_root"],
        get_cfg(cfg, ["paths", "output_root"], "./data/ncf_raw")
    )).expanduser()
    stacks_root = Path(get_cfg(cfg, ["stacking", "stacks_root"], "./data/ncf_stacks")).expanduser()
    overwrite = bool(get_cfg(cfg, ["stacking", "overwrite"], False))

    base_stack = str(get_cfg(cfg, ["stacking", "base_stack"], "1d")).strip().lower()
    if _WINDOW_RE.fullmatch(base_stack) is None:
        raise ValueError(f"stacking.base_stack must match (\\d+)([hd]); got {base_stack!r}")

    min_frac = float(get_cfg(cfg, ["stacking", "min_frac"], 0.9))
    if not (0.0 < min_frac <= 1.0):
        raise ValueError(f"stacking.min_frac must be in (0, 1]; got {min_frac}")

    njobs = max(1, int(get_cfg(cfg, ["stacking", "njobs"], 1)))

    windows_cfg = get_cfg(cfg, ["stacking", "windows"], {"7d": True, "15d": True, "30d": True})
    do_windows: Dict[str, bool] = {}
    for k, v in windows_cfg.items():
        if not isinstance(k, str):
            continue
        key = k.strip().lower()
        if _WINDOW_RE.fullmatch(key) is None:
            logger.warning("Ignoring malformed window key %r in stacking.windows", k)
            continue
        do_windows[key] = bool(v)
    if not do_windows:
        do_windows = {"7d": True, "15d": True, "30d": True}

    return StackPlan(
        raw_root=raw_root.resolve(),
        stacks_root=stacks_root.resolve(),
        overwrite=overwrite,
        base_stack=base_stack,
        do_windows=do_windows,
        min_frac=min_frac,
        njobs=njobs,
    )


@timeit
def run_all_stacks_from_config(cfg: Mapping[str, Any]) -> None:
    if not bool(get_cfg(cfg, ["stacking", "enabled"], True)):
        logger.info("Skipping stacking because stacking.enabled=False")
        return

    plan = build_stack_plan(cfg)
    base_dir = plan.stacks_root / plan.base_stack
    base_dir.mkdir(parents=True, exist_ok=True)
    plan.stacks_root.mkdir(parents=True, exist_ok=True)

    # Base stack is mandatory.
    base_stack_ncf(
        plan.raw_root, base_dir, plan.base_stack,
        overwrite=plan.overwrite, njobs=plan.njobs,
    )

    # Sliding windows.
    for win_str, enabled in sorted(plan.do_windows.items()):
        if not enabled:
            continue
        out_dir = plan.stacks_root / win_str
        stack_ncf_window(
            base_dir, out_dir, win_str,
            overwrite=plan.overwrite,
            min_frac=plan.min_frac,
            njobs=plan.njobs,
        )


# =====================================================
# CLI
# =====================================================
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NCF stacking tool (per virtual source)")
    p.add_argument("--config", type=str, required=True, help="Path to config file (.yaml/.yml/.json)")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return p.parse_args(args=argv)


if __name__ == "__main__":
    # Configure logging only when run as a script, so importing stack.py from
    # other modules / notebooks doesn't clobber their logging setup.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled.")

    cfg = load_config(args.config)
    run_all_stacks_from_config(cfg)


# Example:
# python -m src.stack --config configs/cc.yaml --verbose