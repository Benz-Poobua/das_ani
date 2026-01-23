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
import logging
import os
import time
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
    get_cfg, 
    write_perf_row
    )

from src.ani import preprocess, TorchCrossCorrelation, whiten_per_segment_torch

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

    njobs = int(get_cfg(cfg, ["runtime", "njobs"], 1))
    njobs = max(1, njobs)

    # ---- config ----
    perf_enabled = bool(get_cfg(cfg, ["perf", "enabled"], False))
    perf_out_path = str(get_cfg(cfg, ["perf", "out_path"], "./data/runlogs/perf_cc.csv"))
    log_every_vs = bool(get_cfg(cfg, ["perf", "log_every_vs"], True))
    log_every_chunk = bool(get_cfg(cfg, ["perf", "log_every_chunk"], False))

    # Avoid CPU oversubscription: split cores across processes
    ncores = os.cpu_count() or 1
    threads_per_proc = max(1, ncores // njobs)
    torch.set_num_threads(threads_per_proc)
    logger.info("CPU threads per process: %d (total cores=%d, njobs=%d)",
                threads_per_proc, ncores, njobs)

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

    # ---- mode ----
    mode = str(get_cfg(cfg, ["xcorr", "mode"], "conventional")).lower()
    if mode not in {"conventional", "v1"}:
        raise ValueError(f"xcorr.mode must be 'conventional' or 'v1'; got {mode}")

    if mode == "v1":
        xcorr_seg_sec = float(get_cfg(cfg, ["xcorr", "xcorr_seg_sec_v1"], xcorr_seg_sec))

    npts_lag = int(round(max_lag_sec * fs_proc))
    npts_seg = int(round(xcorr_seg_sec * fs_proc))

    if npts_seg <= 0:
        raise ValueError(f"xcorr_seg_sec too small -> npts_seg={npts_seg}")
    if npts_lag <= 0:
        raise ValueError(f"max_lag_sec too small -> npts_lag={npts_lag}")

    if npts_lag >= npts_seg:
        raise ValueError(
            f"Require max_lag_sec < xcorr_seg_sec so that M < Nseg. "
            f"Got npts_lag={npts_lag}, npts_seg={npts_seg}."
        )

    cc_out_len = 2 * npts_lag + 1

    v1_fft_snap_pow2 = bool(get_cfg(cfg, ["xcorr", "v1_fft_snap_pow2"], True))
    v1_fallback = str(get_cfg(cfg, ["xcorr", "v1_fallback"], "v1_2M"))

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
    
    logger.info(
    "XCORR config | mode=%s | v1_fft_snap_pow2=%s | v1_fallback=%s | "
    "npts_seg=%d | max_lag_samples=%d | out_len=%d",
    mode, v1_fft_snap_pow2, v1_fallback,
    npts_seg, npts_lag, cc_out_len)

    if v1_fallback not in {"v1_2M", "v1_Mp1"}:
        raise ValueError(f"xcorr.v1_fallback must be 'v1_2M' or 'v1_Mp1'; got {v1_fallback}")
    
    # ---- preprocess (CPU, numpy) ----
    data_proc = preprocess(data_raw, fs_raw, f1, f2, decimation, diff, ram_win_sec).astype(np.float32, copy=False)
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
        npts_proc, npts_seg, leftover)

    if npts_new <= 0 or nseg == 0:
        logger.warning("Too short after preprocessing/segmentation; skipping %s", in_path)
        return None
        
    data_proc = data_proc[:, :npts_new]
    flag_mean = int(nseg)

    # Whitening: do it once, per segment, before CC
    prewhiten = bool(is_spectral_whitening)

    nyq_proc = fs_proc / 2.0
    if not (0.0 < f1 < f2 < nyq_proc):
        raise ValueError(f"Whitening band invalid after decimation: f1={f1}, f2={f2}, Nyquist={nyq_proc}")

    if prewhiten:
        logger.info(
            "Whitening (per-segment, torch) | df=fs_proc/npts_seg=%.6f | window=%.3f Hz | band=[%.2f, %.2f] Hz",
            fs_proc / npts_seg, window_freq_hz, f1, f2
        )

        # Convert ONCE to torch on target device
        data_tensor = convert_to_tensor(data_proc, device=torch.device("cpu"))  # torch CPU
        if device.type == "cuda":
            data_tensor = data_tensor.pin_memory().to(device, non_blocking=True)
        else:
            data_tensor = data_tensor.to(device)

        # Whiten in-place pipeline (returns torch tensor on same device)
        data_tensor = whiten_per_segment_torch(
            data_tensor,
            fs_proc=fs_proc,
            npts_seg=npts_seg,
            window_freq_hz=window_freq_hz,
            f1=f1,
            f2=f2,
        )
    else:
        data_tensor = convert_to_tensor(data_proc, device=torch.device("cpu"))
        if device.type == "cuda":
            data_tensor = data_tensor.pin_memory()
        data_tensor = data_tensor.to(device, non_blocking=True)

    # Lag window (compute only after segmentation is valid) 
    if mode == "conventional":
        L_full = 2 * npts_seg - 1           # full CC length
        center = npts_seg - 1               # lag 0 index
        lag_start = center - npts_lag
        lag_end = center + npts_lag + 1     # exclusive

        if lag_start < 0 or lag_end > L_full:
            raise ValueError(
            f"Lag window out of bounds: lag_start={lag_start}, lag_end={lag_end} (exclusive). "
            f"valid indices=[0, {L_full-1}], valid exclusive end <= {L_full} "
            f"(npts_seg={npts_seg}, npts_lag={npts_lag})")
    else:
        lag_start = lag_end = 0 # dummy ints; not used in v1
    
    # ---- chunk size ----
    npair_chunk = auto_np_pair_chunk(
        nch=nch, 
        npts_seg=npts_seg, 
        device=device, 
        frac_mem=frac_mem, 
        min_chunk=min_chunk, 
        max_chunk=max_chunk, 
        nworkers=njobs) 
    
    logger.info("Using npair_chunk=%d (auto-selected)", npair_chunk)
    write_runlog(f"npair_chunk={npair_chunk} | nch={nch} | npts_seg={npts_seg}")

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ---- model ----
    # IMPORTANT: if prewhiten=True, disable whitening inside CC to avoid double whitening
    is_spectral_whitening_cc = False if prewhiten else bool(is_spectral_whitening)
    whitening_params_cc = None if prewhiten else (float(fs_proc), float(window_freq_hz), float(f1), float(f2))

    model_conf = {
        "mode": mode,
        "max_lag_samples": int(npts_lag) if mode == "v1" else None,
        "is_spectral_whitening": is_spectral_whitening_cc,
        "whitening_params": whitening_params_cc,
        "v1_fft_snap_pow2": v1_fft_snap_pow2,
        "v1_fallback": v1_fallback,
        }

    model: nn.Module = TorchCrossCorrelation(**model_conf)

    multi_gpu = (device.type == 'cuda' and use_gpu and torch.cuda.device_count() > 1)

    if multi_gpu:
        logger.info("Using DataParallel over %d GPUs.", torch.cuda.device_count())
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)
    
    if do_compile and (device.type == "cuda") and not multi_gpu:
        try:
            model = torch.compile(model, mode=compile_mode)
            logger.info("Enabled torch.compile() mode=%s", compile_mode)
        except Exception as e:
            logger.warning("torch.compile() not available or failed: %s", e)
    
    model.eval()
        
    # ---- resume state ----
    meta_path = out_dir / basename.replace(".npz", f"_cc_state_{mode}.json")
    completed_src = load_resume_state(meta_path)

    last_out: Optional[Path] = None

    # ---- loop over virtual sources ----
    vs_bar = tqdm(src_ch_all, desc=f"VS {basename}", leave=True)

    for src_idx in vs_bar:
        src_idx = int(src_idx)

        out_path = out_dir / basename.replace(".npz", f"_cc_{src_idx:03d}_{mode}.npy")
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
        npair = nch
        nchunk = int(np.ceil(npair / npair_chunk))
        write_runlog(f"XCORR mode={mode} | npts_seg={npts_seg} | M={npts_lag} | out_len={cc_out_len}")

        # Prepare output array 
        ccall = np.zeros((npair, cc_out_len), dtype=np.float32)

        # Precompute source trace once per VS
        src_trace = data_tensor[src_idx:src_idx+1, :]   # (1, npts_new)

        chunk_bar = tqdm(range(nchunk), desc=f"VS {src_idx} batches", leave=False)

        t_vs0 = time.perf_counter()

        for ichunk in chunk_bar:
            if ichunk % 10 == 0 or ichunk == nchunk - 1:
                chunk_bar.set_postfix_str(f"{ichunk+1}/{nchunk}")

            start_idx   = npair_chunk * ichunk
            end_idx     = min(start_idx + npair_chunk, npair)
            batch_len   = end_idx - start_idx
            
            rcv0, rcv1 = start_idx, end_idx

            full1 = src_trace.expand(batch_len, -1)  # view for both CPU and CUDA
            full2 = data_tensor[rcv0:rcv1, :]

            # data1 must be materialized (expand -> stride-0)
            data1 = full1.contiguous().view(batch_len * nseg, npts_seg)

            # data2 usually already contiguous; avoid copies unless needed
            if not full2.is_contiguous():
                full2 = full2.contiguous()
            data2 = full2.view(batch_len * nseg, npts_seg)

            # Run CC model (no autograd)
            with torch.inference_mode():
                cc_chunk = model(data1, data2)
            
            if mode == "conventional" and cc_chunk.shape[1] != 2 * npts_seg - 1:
                raise RuntimeError(f"conventional output length mismatch: got {cc_chunk.shape[1]}, expected {2*npts_seg-1}")

            cc_sum_t = cc_chunk.reshape(batch_len, nseg, -1).sum(dim=1)  # (batch_len, Lout_full or Lout_v1)

            if mode == "v1" and cc_sum_t.shape[1] != cc_out_len:
                raise RuntimeError(f"v1 output length mismatch: got {cc_sum_t.shape[1]}, expected {cc_out_len}")

            if mode == "conventional":
                # slice on-device first (reduces PCIe + CPU work)
                cc_win = cc_sum_t[:, lag_start:lag_end]
                ccall[start_idx:end_idx, :] += cc_win.cpu().numpy()
            else:
                # v1 already returns (2M+1)
                ccall[start_idx:end_idx, :] += cc_sum_t.cpu().numpy()

            # Log memory
            if ichunk % 5 == 0 or ichunk == nchunk - 1:
                write_runlog(
                    f"Batch {ichunk+1}/{nchunk} | {gpu_memory('GPU:') or ''} | {cpu_memory('CPU:')}"
                )

        # Normalize by number of segments
        ccall /= float(flag_mean)

        # Save
        np.save(out_path, ccall)

        t_vs1 = time.perf_counter()
        if perf_enabled and log_every_vs:
            write_perf_row(
                {
                    "file": basename,
                    "mode": mode,
                    "vs_idx": int(src_idx),
                    "nch": int(nch),
                    "npts_seg": int(npts_seg),
                    "nseg": int(nseg),
                    "max_lag_samples": int(npts_lag),
                    "npair_chunk": int(npair_chunk),
                    "device": str(device),
                    "seconds_vs": float(t_vs1 - t_vs0),
                },
                perf_out_path,
                add_pid_suffix=True,   # keep True with njobs>1
            )

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

    use_gpu = bool(get_cfg(cfg, ["runtime", "use_gpu"], False))
    if use_gpu and torch.cuda.is_available() and njobs > 1:
        logger.warning("use_gpu=True with njobs=%d may oversubscribe the GPU. Consider njobs=1.", njobs)


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