"""
:module: src/stack.py
:author: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Stacking workflows for DAS ambient-noise NCFs.
          Supports daily stacking and sliding-window stacks (7d, 15d, 30d).
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
_DATE_RE = re.compile(r"(\d{8})")
_VS_RE = re.compile(r"_cc_(\d{3})")

# Matches method suffix for stacked outputs:
#   YYYYMMDD_cc_###_daily[_METHOD].npy
#   YYYYMMDD_cc_###_7d[_METHOD].npy
#   YYYYMMDD_cc_###_15d[_METHOD].npy
#   YYYYMMDD_cc_###_30d[_METHOD].npy
# Also works for raw slices like:
#   YYYYMMDD_000000_cc_###[_METHOD].npy   (method captured if present)
_CC_METHOD_RE = re.compile(r"_cc_\d{3}(?:_(?:daily|\d+d))?(?:_([A-Za-z0-9]+))?\.npy$")

def parse_date_vs(path: str | Path) -> Tuple[datetime.date, int]:
    """
    Extract (date, virtual-source ID) from filename.

    Accepts:
        20210901_000000_cc_080.npy
        20210901_cc_080_daily.npy
        20210901_cc_080_7d.npy
        20210901_cc_080_15d.npy
        ...

    :param path: file path or file name
    :return: (date, vs_id)
    """
    name = Path(path).name

    # Extract leading date YYYYMMDD
    m_date = _DATE_RE.match(name)
    if m_date is None:
        raise ValueError(f"Cannot parse date from: {name}")
    date = datetime.strptime(m_date.group(1), "%Y%m%d").date()

    # Extract VS index (### after "_cc_")
    m_vs = _VS_RE.search(name)
    if m_vs is None:
        raise ValueError(f"Cannot parse VS index from: {name}")
    vs = int(m_vs.group(1))

    return date, vs


def parse_date_vs_method(path: str | Path) -> Tuple[datetime.date, int, Optional[str]]:
    """
    Extract (date, virtual-source ID, cc_method) from filename.

    Backward compatible:
        - If method suffix not present, returns method=None

    Accepts (examples):
        20210901_000000_cc_080.npy
        20210901_000000_cc_080_v1.npy
        20210901_000000_cc_080_conventional.npy

        20210901_cc_080_daily.npy
        20210901_cc_080_daily_v1.npy
        20210901_cc_080_daily_conventional.npy

        20210901_cc_080_7d.npy
        20210901_cc_080_7d_v1.npy
        20210901_cc_080_30d_conventional.npy

    :param path: file path or file name
    :return: (date, vs_id, method) where method may be None
    """
    name = Path(path).name

    # Extract date + VS using existing logic (robust)
    date, vs = parse_date_vs(name)

    # Extract optional method suffix (last token after known parts)
    m_method = _CC_METHOD_RE.search(name)
    method = None if (m_method is None or m_method.group(1) is None) else m_method.group(1)

    return date, vs, method

# =====================================================
# Stacking core
# =====================================================
@timeit
def daily_stack_ncf(raw_root: str | Path, out_daily: str | Path, *, overwrite: bool = False) -> None:
    """
    Create daily NCF stacks per (date, vs_id, method) by averaging all raw slices for that day.

    Output naming (method-aware):
        YYYYMMDD_cc_###_daily.npy                  (if method missing)
        YYYYMMDD_cc_###_daily_v1.npy
        YYYYMMDD_cc_###_daily_conventional.npy
    """
    raw_root = Path(raw_root).expanduser().resolve()
    out_daily = Path(out_daily).expanduser().resolve()
    out_daily.mkdir(parents=True, exist_ok=True)

    # Collect all .npy files under ncf_root
    all_files = sorted(raw_root.rglob("*.npy"))
    logger.info("Found %d raw NCF slices under %s", len(all_files), raw_root)

    # Group by (date, VS, method)
    groups: Dict[Tuple[datetime.date, int, Optional[str]], List[Path]] = {}
    for p in all_files:
        try:
            date, vs, method = parse_date_vs_method(p)
            groups.setdefault((date, vs, method), []).append(p)
        except Exception as e:
            logger.warning("Skipping %s: %s", p, e)

    logger.info("Found %d (date, VS, method) groups for daily stacking.", len(groups))

    # Stack each group
    for (date, vs, method), filelist in tqdm(groups.items(), desc="Daily stack"):
        suffix = f"_{method}" if method else ""
        outname = f"{date.strftime('%Y%m%d')}_cc_{vs:03d}_daily{suffix}.npy"
        outpath = out_daily / outname

        if outpath.exists() and not overwrite:
            continue

        # Average all raw pieces for that (date, vs, method)
        arrs = [np.load(f) for f in filelist]
        stack = np.mean(arrs, axis=0).astype(np.float32)

        np.save(outpath, stack)
        logger.info("Saved daily stack: %s", outpath)

@timeit
def stack_ncf_window(
    daily_root: str | Path,
    out_root: str | Path,
    window_days: int,
    *,
    overwrite: bool = False,
    ) -> None:
    """
    Sliding-window stacking per (VS, method).

    For each (VS, method), for each end date D:
        stack all daily files from [D-window_days+1, ..., D]
    Requires full window (exactly window_days daily files).

    Output naming (method-aware):
        YYYYMMDD_cc_###_<Nd>.npy                   (if method missing)
        YYYYMMDD_cc_###_<Nd>_v1.npy
        YYYYMMDD_cc_###_<Nd>_conventional.npy
    """
    daily_root = Path(daily_root).expanduser().resolve()
    out_root = Path(out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    daily_files = sorted(daily_root.glob("*.npy"))
    if not daily_files:
        logger.warning("No daily stacks in %s", daily_root)
        return

    # Collect records (date, vs, method, path)
    records: List[Tuple[datetime.date, int, Optional[str], Path]] = []
    for p in daily_files:
        try:
            date, vs, method = parse_date_vs_method(p)
            records.append((date, vs, method, p))
        except Exception:
            continue

    # Group by (VS, method)
    vs_groups: Dict[Tuple[int, Optional[str]], List[Tuple[datetime.date, Path]]] = {}
    for date, vs, method, p in records:
        vs_groups.setdefault((vs, method), []).append((date, p))

    logger.info("Found %d (VS, method) groups for %dd window stacks.", len(vs_groups), window_days)

    # Process each (VS, method)
    for (vs, method), items in tqdm(vs_groups.items(), desc=f"{window_days}d stacks"):
        # Sort by date
        items = sorted(items, key=lambda x: x[0])
        dates = [d for d, _ in items]

        # Use a date->path lookup for fast window assembly
        date_to_path = {d: p for d, p in items}

        for end_date in dates:
            start_date = end_date - timedelta(days=window_days - 1)

            # Build contiguous window dates
            win_dates = [start_date + timedelta(days=k) for k in range(window_days)]

            # Require full window present
            if not all(d in date_to_path for d in win_dates):
                continue

            suffix = f"_{method}" if method else ""
            outname = f"{end_date.strftime('%Y%m%d')}_cc_{vs:03d}_{window_days}d{suffix}.npy"
            outpath = out_root / outname

            if outpath.exists() and not overwrite:
                continue

            arrs = [np.load(date_to_path[d]) for d in win_dates]
            stack = np.mean(arrs, axis=0).astype(np.float32)

            np.save(outpath, stack)
            logger.info("Saved %dd stack: %s", window_days, outpath)

# =====================================================
# High-level runner (config-driven)
# =====================================================
@dataclass(frozen=True)
class StackPlan:
    raw_root: Path
    stacks_root: Path
    overwrite: bool
    do_windows: Dict[int, bool]  # {7:True, 15:False, 30:True}

def build_stack_plan(cfg: Mapping[str, Any]) -> StackPlan:
    """
    Daily is mandatory and always runs. Windows are controlled via config.
    """
    stacking_enabled = bool(get_cfg(cfg, ["stacking", "enabled"], True))
    if not stacking_enabled:
        # Still return a plan (caller can decide to skip)
        logger.info("stacking.enabled is False; stacking stage will be skipped.")
    raw_root = Path(get_cfg(cfg, ["stacking", "raw_root"], get_cfg(cfg, ["paths", "output_root"], "./data/ncf_raw"))).expanduser()
    stacks_root = Path(get_cfg(cfg, ["stacking", "stacks_root"], "./data/ncf_stacks")).expanduser()
    overwrite = bool(get_cfg(cfg, ["stacking", "overwrite"], False))

    windows_cfg = get_cfg(cfg, ["stacking", "windows"], {"7d": True, "15d": True, "30d": True})
    if not isinstance(windows_cfg, Mapping):
        raise ValueError("stacking.windows must be a mapping like {7d: true, 15d: false, 30d: true}")
    
    # Parse keys like "7d" -> 7
    do_windows: Dict[int, bool] = {}
    for k, v in windows_cfg.items():
        if not isinstance(k, str):
            continue
        m = re.fullmatch(r"(\d+)d", k.strip())
        if not m:
            continue
        days = int(m.group(1))
        do_windows[days] = bool(v)

    # If user provided nothing usable, default to (7,15,30) all True
    if not do_windows:
        do_windows = {7: True, 15: True, 30: True}

    return StackPlan(
        raw_root=raw_root.resolve(),
        stacks_root=stacks_root.resolve(),
        overwrite=overwrite,
        do_windows=do_windows,
    )
    
@timeit
def run_all_stacks_from_config(cfg: Mapping[str, Any]) -> None:
    stacking_enabled = bool(get_cfg(cfg, ["stacking", "enabled"], True))
    if not stacking_enabled:
        logger.info("Skipping stacking because stacking.enabled=False")
        return
    plan = build_stack_plan(cfg)

    daily_dir = plan.stacks_root / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    plan.stacks_root.mkdir(parents=True, exist_ok=True)

    # DAILY is mandatory
    daily_stack_ncf(plan.raw_root, daily_dir, overwrite=plan.overwrite)

    # Sliding windows
    for days, enabled in sorted(plan.do_windows.items()):
        if not enabled:
            continue
        out_dir = plan.stacks_root / f"{days}d"
        stack_ncf_window(daily_dir, out_dir, days, overwrite=plan.overwrite)

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