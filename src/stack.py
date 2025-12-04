"""
:module: src/stack.py
:author: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Stacking workflows for DAS ambient-noise NCFs.
          Supports daily stacking and sliding-window stacks (7d, 15d, 30d).
"""
import re
import os
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta

from src.utils import timeit

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Helper
# ==============================================================
def _parse_date_vs(fname):
    """
    Extract (date, virtual-source ID) from filename.

    Accepts filenames such as:
        20210901_000000_cc_080.npy
        20210901_cc_080_daily.npy
        20210901_cc_080_7d.npy

    :param fname: input filename
    :return: (date: datetime.date, vs_id: int)
    """
    base = os.path.basename(fname)

    # Extract leading date YYYYMMDD
    m_date = re.match(r'(\d{8})', base)
    if m_date is None:
        raise ValueError(f'Cannot parse date from: {base}')
    date = datetime.strptime(m_date.group(1), '%Y%m%d').date()

    # Extract VS index (### after "_cc_")
    m_vs = re.search(r'_cc_(\d{3})', base)
    if m_vs is None:
        raise ValueError(f'Cannot parse VS index from: {base}')
    vs = int(m_vs.group(1))

    return date, vs

# DAILY STACKING: stack over time per (date, VS)
# ==============================================================
@timeit
def daily_stack_ncf(ncf_root, out_daily):
    """
    Create daily NCF stacks *per virtual source index*.

    For each combination of (date, vs_id):
        stack = mean of all segments for that (date, vs_id)

    Output naming:
        YYYYMMDD_cc_###_daily.npy

    :param ncf_root: directory containing raw NCF files (.npy)
    :param out_daily: directory for output daily stacks
    """
    os.makedirs(out_daily, exist_ok=True)

    # Collect all .npy files under ncf_root
    all_files = sorted(
        os.path.join(root, f)
        for root, _, files in os.walk(ncf_root)
        for f in files if f.endswith('.npy')
    )

    logger.info(f'Found {len(all_files)} raw NCF slices.')

    # Group by (date, VS)
    groups = {}
    for path in all_files:
        try:
            date, vs = _parse_date_vs(path)
            groups.setdefault((date, vs), []).append(path)
        except Exception as e:
            logger.warning(f'Skipping {path}: {e}')

    logger.info(f'Found {len(groups)} (date, VS) groups for stacking.')

    # Stack each group
    for (date, vs), filelist in tqdm(groups.items(), desc='Daily stack'):
        arrs = [np.load(f) for f in filelist]
        stack = np.mean(arrs, axis=0)

        outname = f"{date.strftime('%Y%m%d')}_cc_{vs:03d}_daily.npy"
        outpath = os.path.join(out_daily, outname)

        np.save(outpath, stack)
        logger.info(f'Saved daily stack: {outpath}')

# MULTI-DAY SLIDING WINDOW STACKING: per VS separately
# ==============================================================
@timeit
def stack_ncf_window(daily_root, out_root, window_days):
    """
    Sliding-window stacking (7d, 15d, 30d) per VS.

    For each VS:
        For each date D:
            stack all daily NCFs from (D - window_days + 1) ... D

    Output naming:
        YYYYMMDD_cc_###_<window_days>d.npy

    :param daily_root: directory containing daily stacks
    :param out_root: directory for sliding-window stacks
    :param window_days: number of days in the window (e.g., 7, 15, 30)
    """
    os.makedirs(out_root, exist_ok=True)

    # Collect all daily stack files
    daily_files = sorted(
        f for f in os.listdir(daily_root) if f.endswith('.npy')
    )
    if len(daily_files) == 0:
        logger.warning(f'No daily stacks in {daily_root}.')
        return

    # Parse (date, VS)
    records = []
    for f in daily_files:
        try:
            date, vs = _parse_date_vs(f)
            records.append((date, vs, f))
        except:
            continue

    # Group by VS
    vs_groups = {}
    for date, vs, fname in records:
        vs_groups.setdefault(vs, []).append((date, fname))

    logger.info(f'Found {len(vs_groups)} virtual sources for sliding stacks.')

    # Process each VS
    for vs, items in tqdm(vs_groups.items(), desc=f'{window_days}d stacks'):
        # Sort by date
        items = sorted(items, key=lambda x: x[0])
        dates = [d for d, _ in items]
        fnames = [f for _, f in items]

        for i in range(len(items)):
            end_date = dates[i]
            start_date = end_date - timedelta(days=window_days - 1)

            # files inside window
            fwin = [
                os.path.join(daily_root, fnames[j])
                for j in range(len(items))
                if start_date <= dates[j] <= end_date
            ]

            # Require full window (e.g., 7 days)
            if len(fwin) < window_days:
                continue

            arrs = [np.load(p) for p in fwin]
            stack = np.mean(arrs, axis=0)

            outname = (
                f"{end_date.strftime('%Y%m%d')}_cc_{vs:03d}_{window_days}d.npy"
            )
            outpath = os.path.join(out_root, outname)
            np.save(outpath, stack)

            logger.info(f'Saved {window_days}d stack: {outpath}')

# MASTER STACK WORKFLOW (used for CLI)
# ==============================================================
def run_all_stacks(
    raw_root='data/ncf_raw',
    stacks_root='data/ncf_stacks',
    do_daily=True,
    do_7d=True,
    do_15d=True,
    do_30d=True,
):
    """
    Full stacking workflow:
        daily → 7d → 15d → 30d
    All operations are done **per virtual source index**.

    :param raw_root: directory with raw NCFs
    :param stacks_root: base directory for stacks
    """
    daily_dir = os.path.join(stacks_root, 'daily')

    if do_daily:
        daily_stack_ncf(raw_root, daily_dir)

    if do_7d:
        stack_ncf_window(daily_dir, os.path.join(stacks_root, '7d'), 7)
    if do_15d:
        stack_ncf_window(daily_dir, os.path.join(stacks_root, '15d'), 15)
    if do_30d:
        stack_ncf_window(daily_dir, os.path.join(stacks_root, '30d'), 30)

# CLI
# ==============================================================
def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='NCF stacking tool (per virtual source)')

    p.add_argument('--raw_root', type=str, default='data/ncf_raw')
    p.add_argument('--stacks_root', type=str, default='data/ncf_stacks')

    p.add_argument('--no_daily', action='store_true')
    p.add_argument('--no_7d', action='store_true')
    p.add_argument('--no_15d', action='store_true')
    p.add_argument('--no_30d', action='store_true')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_all_stacks(
        raw_root=args.raw_root,
        stacks_root=args.stacks_root,
        do_daily=not args.no_daily,
        do_7d=not args.no_7d,
        do_15d=not args.no_15d,
        do_30d=not args.no_30d,
    )

# Example
# python -m src.stack \
#     --raw_root ./data/ncf_raw \
#     --stacks_root ./data/ncf_stacks 

# python -m src.stack \
#     --raw_root ./data/ncf_raw \
#     --stacks_root ./data/ncf_stacks \
#     --no_daily --no_15d --no_30d