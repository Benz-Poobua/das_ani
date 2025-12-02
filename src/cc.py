"""
:module: src/cc.py
:auth: Benz Poobua 
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: DAS ambient noise interferometry (ANI)
          Cross-correlation workflow for NCF generation.
"""
import os
import argparse
import logging
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
sys.path.insert(0, os.path.dirname(__file__))

from utils import (load_data, 
                   convert_to_tensor, 
                   timeit,
                   write_runlog, 
                   gpu_memory, 
                   cpu_memory, 
                   auto_np_pair_chunk, 
                   check_existing_output, 
                   load_resume_state, 
                   save_resume_state)

from ani import preprocess, TorchCrossCorrelation, daily_stack_ncf, stack_ncf_window

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# PROCESS ONE NPZ → MULTI NCF (virtual-source CC)
# =====================================================
@timeit
def process_signal_file(file_path, output_cc, use_gpu=False):
    """
    Process one DAS file and compute cross-correlation (NCF)
    for all virtual source channels.

    :param file_path: Path to the DAS file to process.
    :type file_path: str
    :param output_cc: Directory path where output cross-correlation results will be written.
    :type output_cc: str
    :param use_gpu: Whether to attempt using GPU for cross-correlation.
    :type use_gpu: bool
    :return: Path to the saved output file of this processing run.
    :rtype: str or None
    """
    logger.info(f'Processing file: {file_path}')
    
    write_runlog(f'Started: {file_path}')

    # Set data parameters
    fs_raw      = 250                               # sampling frequency (Hz)
    first_chan  = 399                               # first channel number
    last_chan   = 748                               # last channel number
    nch_expected = last_chan - first_chan + 1       # expected number of channels
    dx = 8.16                                       # spatial sampling interval (m); `chann_len`

    # Set virtual sources
    src_ch_all_num = np.arange(first_chan, last_chan + 1, 10)  # every 10th channel
    logger.info(f'Virtual source channels: {src_ch_all_num}')

    # Convert to array indices (0-based)
    src_ch_all = src_ch_all_num - first_chan

    # Set preprocessing parameters
    decimation  = 1                                 # decimation factor after filtering 
    f1, f2      = 1.0, 40.0                         # bandpass filter corners
    diff        = False                             # whether to differentiate (strain → strain rate) 
    ram_win     = 1.0                               # RAM window in seconds (0 → one-bit)
    min_length  = 60.0                              # minimum segment length in seconds
    min_npts    = int(min_length * fs_raw)
    fs_proc     = fs_raw / decimation

    # Set ambient noise cross-correlation parameters
    is_spectral_whitening   = True
    window_freq             = 0.0                   # 0 → aggressive whitening, else running mean in Hz
    max_lag                 = 1.0                   # time lag in seconds
    xcorr_seg               = 2.0                   # segment length in seconds for CC window
    npts_lag                = int(max_lag * fs_proc)               
    npts_seg                = int(xcorr_seg * fs_proc) 
    cc_out_len              = 2 * npts_lag + 1  # length of lag window

    # Computational parameters
    device      = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')   
    logger.info(f'Using device: {device}')

    # Load data
    # ========================================
    data_dict, data_raw, dt, N, T = load_data(file_path, mmap=True)
    nch, npts = data_raw.shape
    basename = os.path.basename(file_path)

    if nch != nch_expected:
        raise ValueError(f'Data shape mismatch in {file_path}: expected {nch_expected} channels, got {nch}')
    
    if npts < min_npts:
        logger.warning(f'Skipping {file_path} because npts={npts} < min_npts={min_npts}')
        return None
    
    # Preprocess data (on CPU)
    # ========================================
    data_proc = preprocess(data_raw, fs_raw, f1, f2, decimation, diff, ram_win)
    npts_proc = data_proc.shape[1]

    # Compute segmentation
    npts_new = (npts_proc // npts_seg) * npts_seg
    nseg = npts_new // npts_seg
    leftover = npts_proc - npts_new

    # Segmentation consistency check 
    if leftover != 0:
        logger.warning(
            f'[WARN] Preprocessed length {npts_proc} not divisible by '
            f'npts_seg={npts_seg}. Trimming {leftover} samples.')

    if npts_new <= 0 or nseg == 0:
        logger.warning(
            f'[WARN] After preprocessing/segmentation, npts_new={npts_new}, '
            f'nseg={nseg}. Skipping file {file_path} (too short).')
        return None
        
    if npts_new != nseg * npts_seg:
        raise RuntimeError(
            f'[ERROR] Segmentation mismatch: npts_new={npts_new}, '
            f'nseg={nseg}, npts_seg={npts_seg}')
    
    # Trim to exact multiple of npts_seg
    data_proc = data_proc[:, :npts_new]
    flag_mean = nseg
    
    # Decide batch size automatically
    npair_chunk = auto_np_pair_chunk(
        nch=nch, 
        npts_seg=npts_seg, 
        device=device, 
        frac_mem=0.25,  # recommend for 8GB RAM CPU
        min_chunk=64, 
        max_chunk=4096
    ) 
    logger.info(f'Using npair_chunk = {npair_chunk} (auto-selected)')
    write_runlog(f"npair_chunk={npair_chunk} | nch={nch} | npts_seg={npts_seg}")

    # Free GPU memory before heavy ops
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Build CC model
    # ========================================
    model_conf = {
        'is_spectral_whitening': is_spectral_whitening, 
        'whitening_params': (float(fs_proc), float(window_freq), float(f1), float(f2))
    }

    model = TorchCrossCorrelation(**model_conf)

    # Multi-GPU optional (CUDA only)
    multi_gpu = (device.type == 'cuda' and use_gpu and torch.cuda.device_count() > 1)

    if multi_gpu:
        logger.info(f'Using DataParallel over {torch.cuda.device_count()} GPUs.')
        model = nn.DataParallel(model)
        model.to(device)
    else: 
        model.to(device)

    # Optional torch.compile for extra speed (single-device only)
    if not multi_gpu:
        try:
            model = torch.compile(model, mode='max-autotune')
            logger.info('Enabled torch.compile() for TorchCrossCorrelation.')
        except Exception as e:
            logger.warning(f'torch.compile() not available or failed: {e}')
    
    model.eval()
    
    # Convert data to tensor: CPU → pinned → GPU
    # ========================================
    data_tensor = convert_to_tensor(data_proc, device='cpu')    # always CPU first

    # Optional pinned memory (benefits GPU cloud execution)
    if device.type == 'cuda':
        data_tensor = data_tensor.pin_memory()
    
    # Move to final device
    data_tensor = data_tensor.to(device, non_blocking=True)
    logger.debug(f'Data tensor shape: {data_tensor.shape}, device: {data_tensor.device}')

    # Resume state file
    meta_path = os.path.join(output_cc, basename.replace('.npz', '_cc_state.json'))
    completed_src = load_resume_state(meta_path)

    # Loop over virtual sources
    # ========================================
    vs_bar = tqdm(src_ch_all, desc=f'VS {basename}', leave=True)

    for src_idx in vs_bar:

        # Output path
        out_path = os.path.join(output_cc, basename.replace('.npz', f'_cc_{src_idx:03d}.npy'))
        expected_shape = (nch, cc_out_len)

        # Auto-resume check
        if check_existing_output(out_path, expected_shape):
            vs_bar.set_postfix_str(f'skip VS={src_idx}')
            continue

        if src_idx in completed_src:
            vs_bar.set_postfix_str(f'resume-skip VS={src_idx}')
            continue

        vs_bar.set_postfix_str(f'proc VS={src_idx}')
        logger.info(f'[VS] Processing src_idx={src_idx} (abs ch={first_chan+src_idx})')
        write_runlog(f'Start VS {src_idx}: {gpu_memory()} | {cpu_memory()}')

        # Prepare receiver pairs
        pair_ch1 = np.full(nch, src_idx, dtype=int)
        pair_ch2 = np.arange(nch, dtype=int)
        npair   = len(pair_ch1)

        # Chunking to avoid memory overflow 
        nchunk  = int(np.ceil(npair / npair_chunk))
        write_runlog(f'VS {src_idx}: npair={npair}, npair_chunk={npair_chunk}, nchunk={nchunk}')

        # Prepare output array 
        ccall = np.zeros((npair, cc_out_len), dtype=np.float32)

        # Chunked correlation loop (GPU batching)
        chunk_bar = tqdm(range(nchunk), desc=f'VS {src_idx} batches', leave=False)
        
        for ichunk in chunk_bar:
            if ichunk % 10 == 0 or ichunk == nchunk - 1:
                chunk_bar.set_postfix_str(f'{ichunk+1}/{nchunk}')

            start_idx   = npair_chunk * ichunk
            end_idx     = min(start_idx + npair_chunk, npair)
            batch_len = end_idx - start_idx

            ich1 = pair_ch1[start_idx:end_idx]
            ich2 = pair_ch2[start_idx:end_idx]

            # Full-length traces for this batch: (batch_len, npts_new)
            full1 = data_tensor[ich1, :]
            full2 = data_tensor[ich2, :]

            # Reshape into segments: (batch_len * nseg, npts_seg)
            if device.type == 'cuda':
                # GPU → use preallocated buffers + copy_
                data1 = full1.reshape(batch_len * nseg, npts_seg).contiguous()
                data2 = full2.reshape(batch_len * nseg, npts_seg).contiguous()
            else:
                # CPU → zero-copy views (no .copy_ – faster)
                data1 = full1.reshape(batch_len * nseg, npts_seg)
                data2 = full2.reshape(batch_len * nseg, npts_seg)

            # Run CC model
            cc_chunk = model(data1, data2)
            cc_np    = cc_chunk.detach().cpu().numpy()

            # Sum over segments
            # cc_np shape: (batch_len * nseg, full_corr_len)
            cc_sum = cc_np.reshape(batch_len, nseg, -1).sum(axis=1)

            # Extract lag window centered
            lag_start = npts_seg - npts_lag - 1
            lag_end = lag_start + cc_out_len

            ccall[start_idx:end_idx, :] += cc_sum[:, lag_start:lag_end]

            # Log current memory
            if ichunk % 5 == 0 or ichunk == nchunk - 1:
                write_runlog(
                    f"Batch {ichunk+1}/{nchunk} | {gpu_memory('GPU:')} | {cpu_memory('CPU:')}"
                )

            if device.type == 'cuda' and ichunk % 3 == 0:
                torch.cuda.empty_cache()

        # Normalize by number of segments
        ccall /= flag_mean

        # Save
        np.save(out_path, ccall)
        logger.info(f'Saving output to {out_path}')
        write_runlog(f'Completed VS {src_idx}, saved → {out_path}')

        # Update resume state
        completed_src.add(src_idx)
        save_resume_state(meta_path, completed_src)

    return out_path

# MAIN MULTI-FILE EXECUTION
# =====================================================
@timeit
def main(
    data_root='./data/preprocessed', 
    output_root='./data/ncf_raw', 
    njobs=4, 
    use_gpu=False):
    """
    Run ANI workflow across all .npz files in data_root.

    :param data_root: Directory containing preprocessed DAS files (.npz).
    :type data_root: str
    :param output_root: Output directory for generated NCF stacks.
    :type output_root: str
    :param njobs: Number of parallel worker processes (ProcessPoolExecutor).
    :type njobs: int
    :param use_gpu: Whether GPU is used for CC.
    :type use_gpu: bool

    :return: None
    """
    logger.info('\n--- DAS Ambient Noise Processing Workflow ---\n')

    # Expand user (~) and normalize paths
    data_root = os.path.expanduser(data_root)
    output_root = os.path.expanduser(output_root)
    daily_root = os.path.expanduser(daily_root)

    os.makedirs(output_root, exist_ok=True)

    # Collect input files (recursive, .npz only)
    filelist = sorted(
        os.path.join(root, fname)
        for root, _, files in os.walk(data_root)
        for fname in files
        if fname.endswith('.npz'))

    logger.info(f'Found {len(filelist)} files in {data_root}')
    write_runlog(f'Found {len(filelist)} input files in {data_root}.')
    
    # Parallel processing
    Executor = ProcessPoolExecutor
    with Executor(max_workers=njobs) as executor:
        futures = [executor.submit(process_signal_file, fpath, output_root, use_gpu) for fpath in filelist]
        for fut in tqdm(as_completed(futures), total=len(futures), desc='Processing files'):
            try: 
                result = fut.result()
                if result:
                    logger.info(f'Done: {result}')
            except Exception as e:
                logger.error(f'Error processing file: {e}')
                write_runlog(f'Error: {e}')
    
# CLI
# =====================================================
def parse_args():
    """
    Parse command-line arguments for the DAS ambient-noise CC pipeline.

    :return: Namespace of parsed arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description='DAS ambient noise cross-correlation processing pipeline'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='./data/preprocessed',
        help='Root directory containing preprocessed DAS .npz files'
    )
    parser.add_argument(
        '--output_root',
        type=str,
        default='./data/ncf',
        help='Directory where output cross-correlation files (.npy) will be saved'
    )
    parser.add_argument(
        '--njobs',
        type=int,
        default=10,
        help='Number of parallel worker processes'
    )
    parser.add_argument(
        '--use_gpu',
        action='store_true',
        help='If set, attempt to use GPU for cross-correlation when available'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable debug/verbose logging output'
    )
    return parser.parse_args()

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    args = parse_args()

    # Set verbose / debug logging if requested
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug('Verbose logging enabled.')

    main(
        data_root   = args.data_root,
        output_root = args.output_root,
        njobs       = args.njobs,
        use_gpu     = args.use_gpu)

# Example
# python -m src.cc --data_root ./data/preprocessed --output_root ./data/ncf_raw --njobs 4 --use_gpu --verbose