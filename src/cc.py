"""
:module: src/cc.py
:auth: Benz Poobua 
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: DAS ambient noise interferometry (ANI)
          Cross-correlation workflow for NCF generation.
"""
from __future__ import annotations

import argparse
import json
import logging
import torch
import multiprocessing as mp
import numpy as np

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from torch import nn
from tqdm import tqdm
from typing import Any, Mapping, Optional, Sequence

from src.utils import (
    auto_np_pair_chunk,
    check_existing_output,
    convert_to_tensor,
    cpu_memory,
    gpu_memory,
    load_data,
    load_resume_state,
    save_resume_state,
    timeit,
    write_runlog, 
    load_config, 
    get_cfg
    )

from src.ani import preprocess, TorchCrossCorrelation 

# =====================================================
# Logging
# =====================================================
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =====================================================
# PROCESS ONE NPZ → MULTI NCF (virtual-source CC)
# =====================================================
@timeit
def process_single_file(file_path: str | Path, cfg: Mapping[str, Any]) -> Optional[str]:
    """
    Process one DAS file and compute cross-correlation (NCF) for all virtual sources.

    Config-driven: see configs/cc.yaml

    :param file_path: Input .npz path
    :param cfg: Loaded config mapping
    :return: Path to last saved output, or None if skipped
    """
    in_path = Path(file_path).expanduser().resolve()

    # ---- paths ----
    out_dir = Path(get_cfg(cfg, ["paths", "output_root"], required=True)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- runtime ----
    use_gpu = bool(get_cfg(cfg, ["runtime", "use_gpu"], False))
    mmap = bool(get_cfg(cfg, ["runtime", "mmap"], True))
    frac_mem = float(get_cfg(cfg, ["runtime", "frac_mem"], 0.25))
    min_chunk = int(get_cfg(cfg, ["runtime", "min_chunk"], 64))
    max_chunk = int(get_cfg(cfg, ["runtime", "max_chunk"], 4096))

    do_compile = bool(get_cfg(cfg, ["runtime", "torch_compile"], False))
    compile_mode = str(get_cfg(cfg, ["runtime", "compile_mode"], "max-autotune"))

    # ---- data ----
    fs_raw = float(get_cfg(cfg, ["data", "fs_raw"], required=True))
    first_chan = int(get_cfg(cfg, ["data", "first_chan"], required=True))
    last_chan = int(get_cfg(cfg, ["data", "last_chan"], required=True))
    nch_expected = last_chan - first_chan + 1

    src_stride = int(get_cfg(cfg, ["data", "src_stride"], 10))
    min_length_sec = float(get_cfg(cfg, ["data", "min_length_sec"], 60.0))
    min_npts = int(min_length_sec * fs_raw)

    # (kept for completeness; not used directly in CC loop here)
    _dx = float(get_cfg(cfg, ["data", "dx"], 8.16))

    # ---- preprocess ----
    decimation = int(get_cfg(cfg, ["preprocess", "decimation"], 1))
    f1 = float(get_cfg(cfg, ["preprocess", "f1"], 1.0))
    f2 = float(get_cfg(cfg, ["preprocess", "f2"], 5.0))
    diff = bool(get_cfg(cfg, ["preprocess", "diff"], False))
    ram_win_sec = float(get_cfg(cfg, ["preprocess", "ram_win_sec"], 0.0))
    fs_proc = fs_raw / decimation

    # ---- xcorr ----
    is_spectral_whitening = bool(get_cfg(cfg, ["xcorr", "is_spectral_whitening"], True))
    window_freq_hz = float(get_cfg(cfg, ["xcorr", "window_freq_hz"], 0.0))
    max_lag_sec = float(get_cfg(cfg, ["xcorr", "max_lag_sec"], 4.0))
    xcorr_seg_sec = float(get_cfg(cfg, ["xcorr", "xcorr_seg_sec"], 8.0))

    npts_lag = int(max_lag_sec * fs_proc)
    npts_seg = int(xcorr_seg_sec * fs_proc)
    cc_out_len = 2 * npts_lag + 1

    # ---- virtual sources ----
    src_ch_all_num = np.arange(first_chan, last_chan + 1, src_stride, dtype=int)  # abs channel numbers
    src_ch_all = src_ch_all_num - first_chan  # 0-based indices
    logger.info("Processing file: %s", in_path)
    logger.info("Virtual source channels (abs): %s", src_ch_all_num.tolist())
    write_runlog(f"Started: {in_path}")

    # ---- device ----
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ---- load data ----
    _, data_raw, dt, N, T = load_data(in_path, mmap=mmap)
    nch, npts = data_raw.shape
    basename = in_path.name

    if nch != nch_expected:
        raise ValueError(
            f"Data shape mismatch in {in_path}: expected {nch_expected} channels, got {nch}"
        )

    if npts < min_npts:
        logger.warning("Skipping %s because npts=%d < min_npts=%d", in_path, npts, min_npts)
        return None
    
    # ---- preprocess (CPU, numpy) ----
    data_proc = preprocess(data_raw, fs_raw, f1, f2, decimation, diff, ram_win_sec)
    npts_proc = int(data_proc.shape[1])

    # ---- segmentation ----
    if npts_seg <= 0:
        logger.warning("Invalid npts_seg=%d; skipping %s", npts_seg, in_path)
        return None
    
    npts_new = (npts_proc // npts_seg) * npts_seg
    nseg = npts_new // npts_seg
    leftover = npts_proc - npts_new

    if leftover != 0:
        logger.warning(
            "Preprocessed length %d not divisible by npts_seg=%d. Trimming %d samples.",
            npts_proc,
            npts_seg,
            leftover,
        )

    if npts_new <= 0 or nseg == 0:
        logger.warning("Too short after preprocessing/segmentation; skipping %s", in_path)
        return None
        
    data_proc = data_proc[:, :npts_new]
    flag_mean = int(nseg)
    
    # ---- chunk size ----
    npair_chunk = auto_np_pair_chunk(
        nch=nch, 
        npts_seg=npts_seg, 
        device=device, 
        frac_mem=frac_mem, 
        min_chunk=min_chunk, 
        max_chunk=max_chunk
    ) 
    logger.info("Using npair_chunk=%d (auto-selected)", npair_chunk)
    write_runlog(f"npair_chunk={npair_chunk} | nch={nch} | npts_seg={npts_seg}")

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ---- model ----
    model_conf = {
        'is_spectral_whitening': is_spectral_whitening, 
        'whitening_params': (float(fs_proc), float(window_freq_hz), float(f1), float(f2))
    }

    model: nn.Module = TorchCrossCorrelation(**model_conf)

    multi_gpu = (device.type == 'cuda' and use_gpu and torch.cuda.device_count() > 1)

    if multi_gpu:
        logger.info("Using DataParallel over %d GPUs.", torch.cuda.device_count())
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)
    
    if do_compile and not multi_gpu:
        try:
            model = torch.compile(model, mode=compile_mode)
            logger.info("Enabled torch.compile() mode=%s", compile_mode)
        except Exception as e:
            logger.warning("torch.compile() not available or failed: %s", e)
    
    model.eval()
    
    # ---- data tensor (CPU -> pinned -> device) ----
    data_tensor = convert_to_tensor(data_proc, device=torch.device("cpu"))    # always CPU first
    if device.type == "cuda":
        data_tensor = data_tensor.pin_memory()
    data_tensor = data_tensor.to(device, non_blocking=True)
    
    # ---- resume state ----
    meta_path = out_dir / basename.replace(".npz", "_cc_state.json")
    completed_src = load_resume_state(meta_path)

    last_out: Optional[Path] = None

    # ---- loop over virtual sources ----
    vs_bar = tqdm(src_ch_all, desc=f"VS {basename}", leave=True)

    for src_idx in vs_bar:
        src_idx = int(src_idx)

        out_path = out_dir / basename.replace(".npz", f"_cc_{src_idx:03d}.npy")
        expected_shape = (nch, cc_out_len)

        if check_existing_output(out_path, expected_shape):
            vs_bar.set_postfix_str(f"skip VS={src_idx}")
            last_out = out_path
            continue

        if src_idx in completed_src:
            vs_bar.set_postfix_str(f"resume-skip VS={src_idx}")
            last_out = out_path
            continue

        vs_bar.set_postfix_str(f"proc VS={src_idx}")
        logger.info("[VS] Processing src_idx=%d (abs ch=%d)", src_idx, first_chan + src_idx)
        write_runlog(f"Start VS {src_idx}: {gpu_memory() or ''} | {cpu_memory()}")

        # VS against all receivers
        pair_ch1 = np.full(nch, src_idx, dtype=int)
        pair_ch2 = np.arange(nch, dtype=int)
        npair = int(pair_ch1.size)

        # Chunking to avoid memory overflow 
        nchunk  = int(np.ceil(npair / npair_chunk))
        write_runlog(f"VS {src_idx}: npair={npair}, npair_chunk={npair_chunk}, nchunk={nchunk}")

        # Prepare output array 
        ccall = np.zeros((npair, cc_out_len), dtype=np.float32)

        # Chunked correlation loop (GPU batching)
        chunk_bar = tqdm(range(nchunk), desc=f"VS {src_idx} batches", leave=False)
        
        for ichunk in chunk_bar:
            if ichunk % 10 == 0 or ichunk == nchunk - 1:
                chunk_bar.set_postfix_str(f"{ichunk+1}/{nchunk}")

            start_idx   = npair_chunk * ichunk
            end_idx     = min(start_idx + npair_chunk, npair)
            batch_len   = end_idx - start_idx

            ich1 = pair_ch1[start_idx:end_idx]
            ich2 = pair_ch2[start_idx:end_idx]

            # Full-length traces for this batch: (batch_len, npts_new)
            full1 = data_tensor[ich1, :]
            full2 = data_tensor[ich2, :]

            # Reshape into segments: (batch_len * nseg, npts_seg)
            # Advanced indexing can yield non-contiguous tensors; contiguous helps CUDA kernels.
            if device.type == 'cuda':
                data1 = full1.reshape(batch_len * nseg, npts_seg).contiguous()
                data2 = full2.reshape(batch_len * nseg, npts_seg).contiguous()
            else:
                data1 = full1.reshape(batch_len * nseg, npts_seg)
                data2 = full2.reshape(batch_len * nseg, npts_seg)

            # Run CC model (no autograd)
            with torch.no_grad():
                cc_chunk = model(data1, data2)

            cc_np = cc_chunk.detach().cpu().numpy()

            # Sum over segments
            # cc_np shape: (batch_len * nseg, full_corr_len)
            cc_sum = cc_np.reshape(batch_len, nseg, -1).sum(axis=1)

            # Extract lag window centered
            lag_start = npts_seg - npts_lag - 1
            lag_end = lag_start + cc_out_len

            ccall[start_idx:end_idx, :] += cc_sum[:, lag_start:lag_end]

            # Log memory
            if ichunk % 5 == 0 or ichunk == nchunk - 1:
                write_runlog(
                    f"Batch {ichunk+1}/{nchunk} | {gpu_memory('GPU:') or ''} | {cpu_memory('CPU:')}"
                )

            if device.type == "cuda" and ichunk % 3 == 0:
                torch.cuda.empty_cache()

        # Normalize by number of segments
        ccall /= float(flag_mean)

        # Save
        np.save(out_path, ccall)
        last_out = out_path
        logger.info("Saved output to %s", out_path)
        write_runlog(f"Completed VS {src_idx}, saved → {out_path}")

        # Update resume state
        completed_src.add(src_idx)
        save_resume_state(meta_path, completed_src)

    return str(last_out) if last_out is not None else None

# =====================================================
# MAIN MULTI-FILE EXECUTION
# =====================================================
@timeit
def main(config_path: str | Path) -> None:
    """
    Run ANI workflow across all .npz files under data_root from config.
    """
    cfg = load_config(config_path)
    data_root = Path(get_cfg(cfg, ["paths", "data_root"], required=True)).expanduser().resolve()
    output_root = Path(get_cfg(cfg, ["paths", "output_root"], required=True)).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    njobs = int(get_cfg(cfg, ["runtime", "njobs"], 4))

    filelist = sorted(data_root.rglob("*.npz"))
    logger.info("Found %d files in %s", len(filelist), data_root)
    write_runlog(f"Found {len(filelist)} input files in {data_root}.")

    if not filelist:
        logger.error("No input files found in %s", data_root)
        return

    # Parallel processing
    with ProcessPoolExecutor(max_workers=njobs) as executor:
        futures = [executor.submit(process_single_file, str(fpath), cfg) for fpath in filelist]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            try: 
                result = fut.result()
                if result:
                    logger.info("Done: %s", result)
            except Exception:
                logger.exception("Error processing file")
                write_runlog("Error: see stderr/logs for traceback.")


# =====================================================    
# CLI
# =====================================================
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DAS ambient noise cross-correlation processing pipeline")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (.yaml/.yml/.json)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug/verbose logging output",
    )
    return parser.parse_args(args=argv)

if __name__ == "__main__":
    # Safer for PyTorch + multiprocessing (esp., CUDA)
    mp.set_start_method("spawn", force=True)

    args = parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled.")

    main(args.config)

# Example
# python -m src.cc --config configs/cc.yaml --verbose