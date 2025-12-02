"""
:module: src/stack.py
:author: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Stacking workflows for DAS ambient-noise NCFs.
          Supports daily stacking and sliding-window stacks (7d, 15d, 30d).
"""
import os
import json
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta

from utils import timeit, convert_to_numpy

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Helper
# ==============================================================
def _extract_date(fname):
    """
    Extract YYYYMMDD from file name, e.g.:

        20210901_001000_cc_005.npy  →  20210901

    :param fname: filename string
    :type fname: str
    :return: datetime.date object
    """
    base = os.path.basename(fname)
    digits = ''.join([c for c in base if c.isdigit()])
    date_str = digits[:8]
    return datetime.strptime(date_str, "%Y%m%d").date()


# DAILY STACKING
# ==============================================================
@timeit
def daily_stack_ncf(ncf_root, out_daily):
    """
    Create daily-average NCF for each day.

    :param ncf_root: Directory containing raw NCFs (.npy) from cc.py
    :type ncf_root: str
    :param out_daily: Output directory for daily stacks (data/ncf_stacks/daily)
    :type out_daily: str
    """

    os.makedirs(out_daily, exist_ok=True)

    # Collect all files
    all_files = sorted(
        os.path.join(root, f)
        for root, _, files in os.walk(ncf_root)
        for f in files if f.endswith(".npy")
    )

    logger.info(f"Found {len(all_files)} raw NCF slices.")

    # Group by day
    day_dict = {}
    for path in all_files:
        d = _extract_date(path)
        day_dict.setdefault(d, []).append(path)

    logger.info(f"Found {len(day_dict)} days to stack.")

    # Stack day by day
    for d, flist in tqdm(day_dict.items(), desc="Daily stack"):
        arrs = [np.load(f) for f in flist]
        stack = np.mean(arrs, axis=0)

        out_path = os.path.join(out_daily, f"{d.strftime('%Y%m%d')}_daily.npy")
        np.save(out_path, stack)

        logger.info(f"Saved daily stack: {out_path}")

# MULTI-DAY SLIDING WINDOW STACKING
# ==============================================================
@timeit
def stack_ncf_window(daily_root, out_root, window_days):
    """
    Create sliding-window stack (7d, 15d, 30d).

    Example: window_days = 7 → 
        For each day D, stack all daily NCFs from D-6 ... D.

    :param daily_root: Directory with daily stacks (data/ncf_stacks/daily)
    :type daily_root: str
    :param out_root: Directory to store sliding stacks
    :type out_root: str
    :param window_days: window length in days
    :type window_days: int
    """

    os.makedirs(out_root, exist_ok=True)

    # Collect daily stacks
    files = sorted([f for f in os.listdir(daily_root) if f.endswith(".npy")])
    dates = [_extract_date(f) for f in files]

    if len(files) == 0:
        logger.warning(f"No daily stacks found in {daily_root}.")
        return

    for i in tqdm(range(len(files)), desc=f"{window_days}d sliding stack"):
        end_date = dates[i]
        start_date = end_date - timedelta(days=window_days - 1)

        # Collect files inside window
        fwin = [
            os.path.join(daily_root, files[j])
            for j in range(len(files))
            if start_date <= dates[j] <= end_date
        ]

        if len(fwin) == 0:
            continue

        # Load and stack
        arrs = [np.load(p) for p in fwin]
        stack = np.mean(arrs, axis=0)

        out_path = os.path.join(
            out_root,
            f"{end_date.strftime('%Y%m%d')}_{window_days}d.npy"
        )
        np.save(out_path, stack)
        logger.info(f"Saved {window_days}d stack: {out_path}")

# MASTER STACK WORKFLOW (used for CLI)
# ==============================================================
def run_all_stacks(
    raw_root="data/ncf_raw",
    stacks_root="data/ncf_stacks",
    do_daily=True,
    do_7d=True,
    do_15d=True,
    do_30d=True
):
    """
    Full stacking workflow:
        daily → 7d → 15d → 30d
    
    :param raw_root: raw CC directory
    :param stacks_root: where stacks are saved
    """

    daily_dir = os.path.join(stacks_root, "daily")

    if do_daily:
        daily_stack_ncf(raw_root, daily_dir)

    # Multi-window stacks
    if do_7d:
        stack_ncf_window(daily_dir, os.path.join(stacks_root, "7d"), 7)
    if do_15d:
        stack_ncf_window(daily_dir, os.path.join(stacks_root, "15d"), 15)
    if do_30d:
        stack_ncf_window(daily_dir, os.path.join(stacks_root, "30d"), 30)

# CLI
# ==============================================================
def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="NCF stacking tool")

    p.add_argument("--raw_root", type=str, default="data/ncf_raw")
    p.add_argument("--stacks_root", type=str, default="data/ncf_stacks")

    p.add_argument("--no_daily", action="store_true")
    p.add_argument("--no_7d", action="store_true")
    p.add_argument("--no_15d", action="store_true")
    p.add_argument("--no_30d", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all_stacks(
        raw_root=args.raw_root,
        stacks_root=args.stacks_root,
        do_daily=not args.no_daily,
        do_7d=not args.no_7d,
        do_15d=not args.no_15d,
        do_30d=not args.no_30d
    )

# Example
# python -m src.stack \
#     --ncf_root ./data/ncf_raw \
#     --out_root ./data/ncf_stacks \
#     --mode daily

# python -m src.stack \
#     --ncf_root ./data/ncf_stacks/daily \
#     --out_root ./data/ncf_stacks \
#     --mode window \
#     --window_days 7

# python -m src.stack \
#     --ncf_root ./data/ncf_stacks/daily \
#     --out_root ./data/ncf_stacks \
#     --mode multi \
#     --windows 7 15 30

# python -m src.stack \
#     --ncf_root ./data/ncf_raw \
#     --out_root ./data/ncf_stacks \
#     --mode full \
#     --windows 7 15 30
