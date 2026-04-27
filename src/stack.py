"""
:module: src/stack.py
:author: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Stacking workflows for DAS ambient-noise NCFs/ACFs.
          Supports time-agnostic base stacking (e.g., 1h, 1d) and 
          sliding-window stacks (e.g., 2h, 6h, 7d, 30d) for both 
          structural cross-correlation and rapid Coda Wave Interferometry.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import numpy as np

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
from typing import Any, Mapping, Optional, Sequence, Tuple, Dict, List

from src.utils import timeit, load_config, get_cfg

# =====================================================
# Logging
# =====================================================
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =====================================================
# Parsing helper
# =====================================================
# Matches YYYYMMDD and optionally _HHMMSS
_DATETIME_RE = re.compile(r"(\d{8})(?:_(\d{6}))?")
_VS_RE = re.compile(r"_(cc_\d+|auto)")

# Matches method suffix for stacked outputs (Now supports 'h' and 'd'):
#   YYYYMMDD_cc_###_1h[_METHOD].npy
#   YYYYMMDD_cc_###_7d[_METHOD].npy
_CC_METHOD_RE = re.compile(r"_(?:cc_\d+|auto)(?:_(\d+[hd]|daily))?(?:_([A-Za-z0-9]+))?\.npy$")

def parse_date_vs(path: str | Path) -> Tuple[datetime, int]:
    name = Path(path).name

    m_date = _DATETIME_RE.match(name)
    if m_date is None:
        raise ValueError(f"Cannot parse datetime from: {name}")
    
    date_str = m_date.group(1)
    time_str = m_date.group(2) if m_date.group(2) else "000000"
    
    # Return a full datetime object, not just .date()
    dt_obj = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")

    m_vs = _VS_RE.search(name)
    if m_vs is None:
        raise ValueError(f"Cannot parse VS/Auto index from: {name}")
    
    match_str = m_vs.group(1)
    if match_str == "auto":
        vs = -1
    else:
        vs = int(match_str.replace("cc_", ""))

    return dt_obj, vs


def parse_date_vs_method(path: str | Path) -> Tuple[datetime, int, Optional[str]]:
    """
    Extract (datetime, virtual-source ID, cc_method) from filename.

    Backward compatible:
        - If method suffix not present, returns method=None
        - If it is an Auto-Correlation file, vs_id is returned as -1.

    Accepts (examples):
        20210901_000000_cc_080.npy
        20211110_150000_cc_080_conventional.npy  <-- Sub-daily CC Raw
        20211110_150000_auto_conventional.npy    <-- Sub-daily Auto-CC Raw
        20211110_150000_auto_1h_conventional.npy <-- Sub-daily Auto-CC Stacked

    :param path: file path or file name
    :return: (date, vs_id, method) where method may be None
    """
    name = Path(path).name
    dt_obj, vs = parse_date_vs(name)

    m_method = _CC_METHOD_RE.search(name)
    method = None
    if m_method:
        # group(1) is the window (e.g. 1h, 7d), group(2) is the method (e.g. v1)
        method = m_method.group(2)

    return dt_obj, vs, method

# =====================================================
# Stacking core
# =====================================================
@timeit
def base_stack_ncf(raw_root: str | Path, out_base: str | Path, base_label: str, *, overwrite: bool = False) -> None:
    """
    Create base NCF stacks per (datetime, vs_id, method) by averaging raw slices.
    """
    raw_root = Path(raw_root).expanduser().resolve()
    out_base = Path(out_base).expanduser().resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    all_files = sorted(raw_root.rglob("*.npy"))
    logger.info("Found %d raw NCF slices under %s", len(all_files), raw_root)

    # Group by (datetime, VS, method)
    groups: Dict[Tuple[datetime, int, Optional[str]], List[Path]] = {}
    for p in all_files:
        try:
            dt_obj, vs, method = parse_date_vs_method(p)
            # If your base is 1d, you might want to truncate to the day. 
            # If it's 1h, truncate to the hour. For simplicity, we group by exact timestamp here,
            # assuming raw files are already output at the base frequency.
            groups.setdefault((dt_obj, vs, method), []).append(p)
        except Exception as e:
            logger.warning("Skipping %s: %s", p, e)

    # Stack each group
    for (dt_obj, vs, method), filelist in tqdm(groups.items(), desc=f"Base stack ({base_label})"):
        suffix = f"_{method}" if method else ""
        vs_str = "auto" if vs == -1 else f"cc_{vs:03d}"
        
        # Determine time format based on unit
        time_fmt = "%Y%m%d" if "d" in base_label else "%Y%m%d_%H%M%S"
        outname = f"{dt_obj.strftime(time_fmt)}_{vs_str}_{base_label}{suffix}.npy"
        
        outpath = out_base / outname

        if outpath.exists() and not overwrite:
            continue

        arrs = [np.load(f) for f in filelist]
        stack = np.mean(arrs, axis=0).astype(np.float32)
        np.save(outpath, stack)


@timeit
def stack_ncf_window(
    base_root: str | Path,
    out_root: str | Path,
    window_str: str,
    *,
    overwrite: bool = False,
    min_frac: float = 0.9   
    ) -> None:
    """
    Sliding-window stacking per (VS, method) with tolerance for missing data.
    
    Dynamically computes step sizes (hours or days) based on the `window_str` 
    suffix (e.g., '2h', '7d'). 
    
    :param min_frac: Minimum fraction of required time steps (0.0 to 1.0).
                     e.g., 0.9 means a 10-hour window needs at least 9 hourly files.
    """
    base_root = Path(base_root).expanduser().resolve()
    out_root = Path(out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    base_files = sorted(base_root.glob("*.npy"))
    if not base_files:
        logger.warning("No base stacks in %s", base_root)
        return

    # Parse window unit
    m = re.fullmatch(r"(\d+)([hd])", window_str)
    win_val = int(m.group(1))
    win_unit = m.group(2)
    
    # Define step size
    step_delta = timedelta(days=1) if win_unit == 'd' else timedelta(hours=1)

    records: List[Tuple[datetime, int, Optional[str], Path]] = []
    for p in base_files:
        try:
            dt_obj, vs, method = parse_date_vs_method(p)
            records.append((dt_obj, vs, method, p))
        except Exception:
            continue

    vs_groups: Dict[Tuple[int, Optional[str]], List[Tuple[datetime, Path]]] = {}
    for dt_obj, vs, method, p in records:
        vs_groups.setdefault((vs, method), []).append((dt_obj, p))

    for (vs, method), items in tqdm(vs_groups.items(), desc=f"{window_str} stacks"):
        items = sorted(items, key=lambda x: x[0])
        dates = [d for d, _ in items]
        date_to_path = {d: p for d, p in items}

        for end_date in dates:
            # Shift back by (window - 1) steps
            start_date = end_date - (step_delta * (win_val - 1))

            valid_paths = []
            files_found_count = 0
            
            for k in range(win_val):
                target_date = start_date + (step_delta * k)
                if target_date in date_to_path:
                    valid_paths.append(date_to_path[target_date])
                    files_found_count += 1
            
            required_steps = int(np.ceil(win_val * min_frac))
            if files_found_count < required_steps:
                continue

            suffix = f"_{method}" if method else ""
            vs_str = "auto" if vs == -1 else f"cc_{vs:03d}"
            
            time_fmt = "%Y%m%d" if win_unit == 'd' else "%Y%m%d_%H%M%S"
            outname = f"{end_date.strftime(time_fmt)}_{vs_str}_{window_str}{suffix}.npy"
            outpath = out_root / outname

            if outpath.exists() and not overwrite:
                continue

            arrs = [np.load(p) for p in valid_paths]
            stack = np.mean(arrs, axis=0).astype(np.float32)
            np.save(outpath, stack)

# =====================================================
# High-level runner (config-driven)
# =====================================================
@dataclass(frozen=True)
class StackPlan:
    raw_root: Path
    stacks_root: Path
    overwrite: bool
    base_stack: str           # e.g., "1d", "1h"
    do_windows: Dict[str, bool]  # e.g., {"7d": True, "2h": False}

def build_stack_plan(cfg: Mapping[str, Any]) -> StackPlan:
    stacking_enabled = bool(get_cfg(cfg, ["stacking", "enabled"], True))
    if not stacking_enabled:
        logger.info("stacking.enabled is False; stacking stage will be skipped.")
        
    raw_root = Path(get_cfg(cfg, ["stacking", "raw_root"], get_cfg(cfg, ["paths", "output_root"], "./data/ncf_raw"))).expanduser()
    stacks_root = Path(get_cfg(cfg, ["stacking", "stacks_root"], "./data/ncf_stacks")).expanduser()
    overwrite = bool(get_cfg(cfg, ["stacking", "overwrite"], False))

    # NEW: Extract base stack
    base_stack = str(get_cfg(cfg, ["stacking", "base_stack"], "1d")).strip().lower()

    windows_cfg = get_cfg(cfg, ["stacking", "windows"], {"7d": True, "15d": True, "30d": True})
    do_windows: Dict[str, bool] = {}
    
    for k, v in windows_cfg.items():
        if not isinstance(k, str):
            continue
        m = re.fullmatch(r"(\d+)([hd])", k.strip().lower())
        if not m:
            continue
        # Store the exact string key (e.g., '7d', '2h')
        do_windows[k.strip().lower()] = bool(v)

    if not do_windows:
        do_windows = {"7d": True, "15d": True, "30d": True}

    return StackPlan(
        raw_root=raw_root.resolve(),
        stacks_root=stacks_root.resolve(),
        overwrite=overwrite,
        base_stack=base_stack,
        do_windows=do_windows,
    )
    
@timeit
def run_all_stacks_from_config(cfg: Mapping[str, Any]) -> None:
    stacking_enabled = bool(get_cfg(cfg, ["stacking", "enabled"], True))
    if not stacking_enabled:
        logger.info("Skipping stacking because stacking.enabled=False")
        return
        
    plan = build_stack_plan(cfg)

    # Use base_stack (e.g. '1h' or '1d') as the directory name instead of 'daily'
    base_dir = plan.stacks_root / plan.base_stack
    base_dir.mkdir(parents=True, exist_ok=True)
    plan.stacks_root.mkdir(parents=True, exist_ok=True)

    # BASE STACK is mandatory
    base_stack_ncf(plan.raw_root, base_dir, plan.base_stack, overwrite=plan.overwrite)

    # Sliding windows
    for win_str, enabled in sorted(plan.do_windows.items()):
        if not enabled:
            continue
        out_dir = plan.stacks_root / win_str
        stack_ncf_window(base_dir, out_dir, win_str, overwrite=plan.overwrite)

# =====================================================
# CLI
# =====================================================
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NCF stacking tool (per virtual source)")
    p.add_argument("--config", type=str, required=True, help="Path to config file (.yaml/.yml/.json)")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return p.parse_args(args=argv)


if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled.")

    cfg = load_config(args.config)
    run_all_stacks_from_config(cfg)


# Example:
# python -m src.stack --config configs/cc.yaml --verbose