
"""
:module: src/disp_pick.py
:author: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Compute dispersion images (f–v) and pick dispersion curves
          from NCF stacks (daily, 7d, 15d, 30d) per virtual source.

Workflow:
---------
Input folder:  
    data/ncf_stacks/daily/
        20210901_cc_000_daily.npy
        20210901_cc_010_daily.npy
        ...

Output folder:
    results/dispersion/daily/VS_000/
         20210901_cc_000_daily_fv_panel.npy
         20210901_cc_000_daily_f_axis.npy
         20210901_cc_000_daily_v_axis.npy
         20210901_cc_000_daily_pick.npy
         20210901_cc_000_daily_meta.json
"""
import os 
import re
import json
import argparse
import logging
import numpy as np
from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.utils import convert_to_numpy, timeit
from src.disp import compute_dispersion_from_ncf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# HELPER: EXTRACT VIRTUAL SOURCE INDEX FROM FILENAME
def extract_vs_index(fname: str) -> int:
    """
    Extract virtual source index from filenames such as:
        '20210901_cc_000_daily.npy'
        '20210901_cc_010_7d.npy'

    Returns integer (e.g., 0, 10, 20, ...)

    :param fname: filename string
    :return: integer VS index
    """
    base = os.path.basename(fname)
    m = re.search(r'_cc_(\d+)', base)
    if not m:
        raise ValueError(f'Cannot parse VS index from filename: {fname}')
    return int(m.group(1))

# WORKER: PROCESS ONE NCF FILE
# ================================================================================
def process_one_ncf(ncf_path, results_root, stack_window, fs, dx, disp_kwargs):
    """
    Process ONE NCF file → compute dispersion image + pick curve.

    :param ncf_path: Path to NCF .npy file
    :type  ncf_path: str
    :param results_root: Root directory for saving results
    :type  results_root: str
    :param stack_window: Name of stack window ('daily', '7d', '15d', '30d')
    :type  stack_window: str
    :param fs: Sampling frequency (Hz)
    :type  fs: float
    :param dx: Channel spacing (m)
    :type  dx: float
    :param disp_kwargs: Additional keyword arguments for dispersion_curve()
    :type  disp_kwargs: dict
    """
    base = os.path.basename(ncf_path).replace('.npy', '')
    vs_idx = extract_vs_index(base)

    # Output directory structure
    outdir = os.path.join(results_root, stack_window, f'VS_{vs_idx:03d}')
    os.makedirs(outdir, exist_ok=True)

    out_pick = os.path.join(outdir, f'{base}_pick.npy')

    # Skip already processed results
    if os.path.exists(out_pick):
        logger.info(f'[SKIP] Already exists → {out_pick}')
        return None

    logger.info(f'[PROCESS] {base} (VS={vs_idx})')

    # Load NCF
    ncf = np.load(ncf_path)

    # Compute dispersion + pick
    fv_panel, f_axis, v_axis, picks = compute_dispersion_from_ncf(ncf=ncf, fs=fs, dx=dx, **disp_kwargs)

    # Save arrays
    np.save(os.path.join(outdir, f'{base}_fv_panel.npy'), convert_to_numpy(fv_panel))
    np.save(os.path.join(outdir, f'{base}_f_axis.npy'), convert_to_numpy(f_axis))
    np.save(os.path.join(outdir, f'{base}_v_axis.npy'), convert_to_numpy(v_axis))
    np.save(os.path.join(outdir, f'{base}_pick.npy'), picks)

    # Metadata
    meta = {
        'ncf_path': os.path.abspath(ncf_path),
        'vs_index': vs_idx,
        'fs': float(fs),
        'dx': float(dx),
        'stack_window': stack_window,
        'ncf_shape': list(ncf.shape),
        'fv_shape': list(convert_to_numpy(fv_panel).shape),
        'dispersion_params': disp_kwargs,
    }
    with open(os.path.join(outdir, f'{base}_meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)

    return out_pick

# MAIN PIPELINE
# ================================================================================
@timeit 
def main(ncf_root, results_root, stack_window, njobs, fs, dx, disp_kwargs):
    """
    Main workflow for dispersion processing:

    - Scan all NCF .npy files under ncf_root
    - Process each file in parallel
    - Save fv-panel, axes, pick curve, and metadata

    :param ncf_root: Directory containing NCF .npy files
    :type  ncf_root: str
    :param results_root: Root directory for saving dispersion results
    :type  results_root: str
    :param stack_window: Label for output folder ('daily', '7d', '15d', '30d')
    :type  stack_window: str
    :param njobs: Number of parallel workers
    :type  njobs: int
    :param fs: Sampling frequency
    :type  fs: float
    :param dx: Channel spacing (m)
    :type  dx: float
    :param disp_kwargs: KW args for dispersion_curve()
    :type  disp_kwargs: dict
    """
    ncf_root = os.path.expanduser(ncf_root)
    results_root = os.path.expanduser(results_root)

    filelist = sorted(glob(os.path.join(ncf_root, '*.npy')))

    logger.info(f'Found {len(filelist)} NCF files in {ncf_root}')
    logger.info(f'Saving results under {results_root}/{stack_window}/VS_xxx')

    if len(filelist) == 0:
        logger.warning('No NCF files found. Abort.')
        return

    with ProcessPoolExecutor(max_workers=njobs) as ex:
        futures = [
            ex.submit(process_one_ncf, fpath, results_root, stack_window, fs, dx, disp_kwargs)
            for fpath in filelist
        ]

        for fut in tqdm(as_completed(futures), total=len(futures), desc='Dispersion'):
            try:
                fut.result()
            except Exception as e:
                logger.error(f'Error in worker: {e}')

# CLI
# ================================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='Dispersion imaging + picking from stacked NCF files'
    )

    parser.add_argument('--ncf_root', type=str, required=True,
                        help='Folder containing stacked NCF .npy files')
    parser.add_argument('--results_root', type=str, default='results/dispersion',
                        help="Where to store output results")
    parser.add_argument('--stack_window', type=str, default='daily',
                        choices=['daily', '7d', '15d', '30d'],
                        help='Label for output folder')

    parser.add_argument('--fs', type=float, default=250,
                        help='Sampling frequency (Hz)')
    parser.add_argument('--dx', type=float, default=8.16,
                        help='Channel spacing (m)')

    parser.add_argument('--njobs', type=int, default=4,
                        help='Number of parallel workers')

    # Dispersion parameters
    parser.add_argument('--vmin', type=float, default=200)
    parser.add_argument('--vmax', type=float, default=4000)
    parser.add_argument('--dv', type=float, default=10)

    parser.add_argument('--fmin', type=float, default=0.1)
    parser.add_argument('--fmax', type=float, default=50)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    disp_kwargs = dict(
        vmin=args.vmin,
        vmax=args.vmax,
        dv=args.dv,
        fmin=args.fmin,
        fmax=args.fmax,
    )

    main(
        ncf_root=args.ncf_root,
        results_root=args.results_root,
        stack_window=args.stack_window,
        njobs=args.njobs,
        fs=args.fs,
        dx=args.dx,
        disp_kwargs=disp_kwargs,
    )

# Example
# python -m src.disp_pick \
#     --ncf_root ./data/ncf_stacks/daily \
#     --results_root ./results/dispersion \
#     --stack_window daily \
#     --dx 8.16 --fs 250 \
#     --njobs 4 \
#     --vmin 200 --vmax 4000 --dv 10 \ 
#     --fmin 0.1 --fmax 40            

# python -m src.disp_pick \
#     --ncf_root ./data/ncf_stacks/30d \
#     --results_root ./results/dispersion \
#     --stack_window 30d \
#     --njobs 12