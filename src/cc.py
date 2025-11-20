"""
:module: src/cc.py
:auth: Benz Poobua 
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: This script provides DAS ambient noise processing workflow
"""
import os
import argparse
import logging
import torch
import numpy as np
from torch import nn
from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils import load_data, convert_to_tensor, runtime
from ani import preprocess, TorchCrossCorrelation

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_signal_file(file_path, output_cc, use_gpu=False):
    """
    Main DAS processing workflow for ambient noise cross-correlation.

    :param file_path: Path to the DAS file to process.
    :type file_path: str
    :param output_cc: Directory path where output cross-correlation results will be written.
    :type output_cc: str
    :param use_gpu: Whether to attempt using GPU for cross-correlation.
    :type use_gpu: bool
    :return: Path to the saved output file of this processing run.
    :rtype: str or None
    """
    start_time = runtime()
    logger.info(f'Processing file: {file_path}')

    # Set data parameters
    fs_raw      = 250                               # sampling frequency (Hz)
    first_chan  = 400                               # first channel number
    last_chan   = 749                               # last channel number

    nch_expected = last_chan - first_chan + 1       # expected number of channels
    dx = 8.16                                       # spatial sampling interval (m); `chann_len`

    # Set virtual sources
    src_ch_all_num = np.arange(first_chan, last_chan + 1, 10)  # every 10th channel up to nch_expected
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

    # Computational parameters
    device          = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')   
    logger.info(f'Using device: {device}')
    gpu_ids         = [0]                           # GPU device id, if using GPU
    npair_chunk     = 1250                          # chunk size for pair processing

    # Load data
    # ========================================
    data_dict, data_raw, dt, N, T = load_data(file_path)
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

    # Prepare for cross-correlation
    # ========================================
    npts_new = (npts // npts_seg) * npts_seg
    if npts_new < npts_seg:
        logger.warning(f'File {file_path} too short after segmentation: npts_new = {npts_new}')
        return None
    
    data_proc = data_proc[:, :npts_new]
    nseg = npts_new // npts_seg
    flag_mean = nseg

    # Free GPU memory if any
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Build CC model
    # ========================================
    model_conf = {
        'is_spectral_whitening': is_spectral_whitening, 
        'whitening_params': (fs_proc, window_freq, f1, f2)
    }
    model = TorchCrossCorrelation(**model_conf)
    model = nn.DataParallel(model, device_ids=gpu_ids)
    model.to(device)
    model.eval()

    # Move data to Tensor
    data_tensor = convert_to_tensor(data_proc).to(device)
    logger.debug(f'Data tensor shape: {data_tensor.shape}, device: {data_tensor.device}')

    # Loop over virtual sources
    # ========================================
    for src_idx in src_ch_all:
        logger.info(f'Processing virtual source channel index {src_idx} (channel number {first_chan + src_idx})')
        pair_channel1 = src_idx * np.ones(nch, dtype=int)
        pair_channel2 = np.arange(nch, dtype=int)
        npair   = len(pair_channel1)
        nchunk  = int(np.ceil(npair / npair_chunk))

        # Prepare output array 
        ccall = np.zeros((npair, 2 * npts_lag + 1), dtype=np.float32)

        for ichunk in range(nchunk):
            start_idx   = npair_chunk * ichunk
            end_idx     = min(start_idx + npair_chunk, npair)
            ich1 = pair_channel1[start_idx:end_idx].astype(int)
            ich2 = pair_channel2[start_idx:end_idx].astype(int)

            data1 = data_tensor[ich1, :].reshape(-1, npts_seg)
            data2 = data_tensor[ich2, :].reshape(-1, npts_seg)

            cc_chunk = model(data1, data2)
            cc_np    = cc_chunk.cpu().numpy()

            # Sum over segments
            cc_sum   = np.sum(cc_np.reshape(len(ich1), nseg, -1), axis=1)

            # Extract lag window centered
            lag_start = npts_seg - npts_lag - 1
            lag_end = lag_start + (2 * npts_lag + 1)
            ccall[start_idx:end_idx, :] += cc_sum[:, lag_start:lag_end]

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # Normalize by number of segments
        ccall /= flag_mean

        output_file_tmp = os.path.join(
            output_cc, 
            basename.replace('.npz', f'_cc_{src_idx:03d}.npy')
        )
        logger.info(f'Saving output to {output_file_tmp}')
        np.save(output_file_tmp, ccall)

    elapsed = runtime() - start_time
    logger.info(f'Finished {file_path} in {elapsed:.2f} seconds')
    return output_file_tmp

def main(data_root='./data/preprocessed', output_root='./data/ncf', njobs=8, use_gpu=False):
    """
    Orchestrate the DAS ambient-noise processing workflow across multiple files.

    :param data_root: Root directory containing subfolders of preprocessed DAS `.npz` files.
    :type data_root: str
    :param output_root: Directory where cross-correlation output files will be saved.
    :type output_root: str
    :param njobs: Number of parallel worker processes.
    :type njobs: int
    :return: None
    """
    logger.info('\n--- DAS Ambient Noise Processing Workflow ---\n')

    # Expand user (~) and normalize paths
    data_root = os.path.expanduser(data_root)
    output_root = os.path.expanduser(output_root)

    # Collect input files
    filelist = []
    for subdir, dirs, files in os.walk(data_root):
        for fname in files:
            if fname.endswith('.npz'):
                filelist.append(os.path.join(subdir, fname))
    filelist = sorted(filelist)

    logger.info(f'Found {len(filelist)} files in {data_root}')
    os.makedirs(output_root, exist_ok=True)

    # Parallel processing
    with ProcessPoolExecutor(max_workers=njobs) as executor:
        futures = [executor.submit(process_signal_file, fpath, output_root, use_gpu) for fpath in filelist]
        for fut in tqdm(as_completed(futures), total=len(futures), desc='Processing files'):
            try: 
                result = fut.result()
                logger.debug(f'Completed: {result}')
            except Exception as e:
                logger.error(f'Error processing file: {e}')

    logger.info('All files processed.')

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
    args = parse_args()

    # Set verbose / debug logging if requested
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug('Verbose logging enabled.')

    main(
        data_root   = args.data_root,
        output_root = args.output_root,
        njobs       = args.njobs,
        use_gpu     = args.use_gpu
    )

# Example
# python src/cc.py --data_root ./data/preprocessed --output_root ./data/ncf --njobs 10 --use_gpu --verbose