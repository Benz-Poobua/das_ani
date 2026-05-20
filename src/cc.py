"""
:module: src/cc.py
:auth: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: High-performance DAS Ambient Noise Interferometry (ANI) engine for 
          cross-correlation (NCF) and auto-correlation (ACF) generation.

Description:
This module serves as the primary execution engine for generating Noise 
Correlation Functions (NCFs) from massive DAS datasets. It is optimized for 
High-Performance Computing (HPC) environments, featuring hardware-aware 
resource management, asynchronous I/O, and highly parallelized spectral math.

Key Architecture & Performance Features:
    - Asymmetric GPU Pipeline: Implements a "Compute-Once, Broadcast-Many" 
      architecture for Virtual Source (VS) mode. The source spectrum is 
      calculated once and reused across receiver batches, eliminating 
      redundant FFTs.
    - Deferred Device-to-Host Transfer: Cross-correlation sums are accumulated 
      directly in GPU VRAM. Transfers to CPU/Host memory are deferred until the 
      entire Virtual Source is complete, minimizing PCIe bus contention and 
      maximizing GPU kernel utilization.
    - SLURM-Safe Multiprocessing: Implements a "Double-Guard" on GPU visibility. 
      The parent process is shielded from the GPU during setup to prevent 
      premature lock acquisition in SLURM Exclusive-Process mode, ensuring 
      clean initialization of the worker pool.
    - Memory-Budgeted Chunking: Utilizes a heuristic memory model to 
      dynamically scale channel batches based on available VRAM and algorithm 
      complexity (Conventional vs. Zhang 2026 v1), ensuring zero Out-of-Memory 
      (OOM) crashes.
    - Deterministic HPC Threading: Enforces strict control over OMP, MKL, and 
      intra-op threading via SLURM_CPUS_PER_TASK to ensure predictable 
      performance and fair resource sharing on multi-tenant nodes.

Workflow:
    1. Pre-computes whitening and block-sizing parameters.
    2. Spawns a hardware-aware worker pool with pre-warmed PyTorch models.
    3. Processes datasets in parallel, managing auto-resume states and 
       internal profiling (I/O vs. Compute) for performance benchmarking.
"""
from __future__ import annotations

import argparse
import functools
import logging
import os
import time
import gc
import multiprocessing as mp
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Dict

import numpy as np
import torch
import zarr
from torch import nn
from tqdm import tqdm

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
    write_perf_row,
)

from src.ani import (
    preprocess,
    TorchCrossCorrelation,
    whiten_per_segment_torch,
    choose_block_size_v1,
)

logger = logging.getLogger(__name__)

# Global flag to ensure the PyTorch model is only initialized once per worker process.
_WARMED_UP = False


def _set_thread_env(threads: int) -> None:
    """
    Sets environment variables to strictly control the number of threads used by
    underlying C/C++ math libraries (BLAS, OpenMP, MKL, NumExpr).
    """
    threads_str = str(max(1, int(threads)))

    os.environ["OMP_NUM_THREADS"] = threads_str
    os.environ["MKL_NUM_THREADS"] = threads_str
    os.environ["VECLIB_MAXIMUM_THREADS"] = threads_str
    os.environ["NUMEXPR_NUM_THREADS"] = threads_str
    os.environ["OPENBLAS_NUM_THREADS"] = threads_str


def _worker_warmup(
    *,
    mode: str,
    npts_seg: int,
    max_lag_samples: int,
    v1_fft_snap_pow2: bool,
    v1_fallback: str,
    threads_per_proc: int,
    do_compile: bool = False,
    compile_mode: str = "reduce-overhead",
    warmup_nseg: int = 1,
) -> None:
    """
    Initialise the PyTorch correlation model and run two dummy forward passes.
    """
    global _WARMED_UP
    if _WARMED_UP:
        return

    _set_thread_env(threads_per_proc)
    torch.set_num_threads(max(1, threads_per_proc))
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    torch.backends.mkl.enabled = True
    device = torch.device("cpu")

    model = TorchCrossCorrelation(
        mode=mode,
        max_lag_samples=int(max_lag_samples) if mode == "v1" else None,
        is_spectral_whitening=False,
        whitening_params=None,
        v1_fft_snap_pow2=bool(v1_fft_snap_pow2),
        v1_fallback=str(v1_fallback),
    ).to(device)
    model.eval()

    if do_compile:
        try:
            model = torch.compile(model, backend="inductor", mode=compile_mode)
            logger.info("torch.compile() applied in warmup | mode=%s", compile_mode)
        except Exception as e:
            logger.warning("torch.compile() failed in warmup: %s", e)

    Bwarm    = 32
    nseg_w   = max(1, int(warmup_nseg))
    x = torch.zeros((Bwarm, nseg_w, int(npts_seg)), dtype=torch.float32, device=device)
    y = torch.zeros((Bwarm, nseg_w, int(npts_seg)), dtype=torch.float32, device=device)

    # First pass triggers compile/codegen (if enabled). Second confirms steady state.
    with torch.inference_mode():
        _ = model(x, y)
        _ = model(x, y)

    _WARMED_UP = True
    logger.info(
        "Worker warm-up done | mode=%s | npts_seg=%d | nseg=%d | M=%d | threads=%d | compiled=%s",
        mode, npts_seg, nseg_w, max_lag_samples, threads_per_proc, do_compile,
    )


def _unwrap_model(m: nn.Module) -> nn.Module:
    """
    Return the underlying TorchCrossCorrelation through any DataParallel /
    torch.compile wrappers, so that compute_X / compute_Y / combine are
    directly callable.
    """
    if isinstance(m, nn.DataParallel):
        m = m.module
    # torch.compile produces an OptimizedModule whose underlying module is `_orig_mod`.
    inner = getattr(m, "_orig_mod", None)
    if inner is not None:
        m = inner
    return m


@timeit
def process_single_file(file_path: str | Path, cfg: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    t_total_start = time.perf_counter()
    io_time = 0.0
    cc_time = 0.0

    in_path = Path(file_path).expanduser().resolve()
    out_dir = Path(get_cfg(cfg, ["paths", "output_root"], required=True)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    use_gpu = bool(get_cfg(cfg, ["runtime", "use_gpu"], False))
    mmap = bool(get_cfg(cfg, ["runtime", "mmap"], True))
    frac_mem = float(get_cfg(cfg, ["runtime", "frac_mem"], 0.25))
    min_chunk = int(get_cfg(cfg, ["runtime", "min_chunk"], 64))
    max_chunk = int(get_cfg(cfg, ["runtime", "max_chunk"], 4096))
    do_compile = bool(get_cfg(cfg, ["runtime", "torch_compile"], False))
    compile_mode = str(get_cfg(cfg, ["runtime", "compile_mode"], "reduce-overhead"))
    njobs = max(1, int(get_cfg(cfg, ["runtime", "njobs"], 1)))

    perf_enabled = bool(get_cfg(cfg, ["perf", "enabled"], False))
    perf_out_path = str(get_cfg(cfg, ["perf", "out_path"], "./data/runlogs/perf_cc.csv"))
    log_every_vs = bool(get_cfg(cfg, ["perf", "log_every_vs"], True))

    slurm_cores = os.environ.get("SLURM_CPUS_PER_TASK")
    ncores = int(slurm_cores) if slurm_cores else (os.cpu_count() or 1)
    threads_per_proc = max(1, ncores // njobs)
    torch.set_num_threads(threads_per_proc)
    # Note: set_num_interop_threads(1) is set once at worker start in
    # _worker_warmup. PyTorch only honors it before the first parallel work
    # is queued, so re-setting here is a no-op and has been removed.

    fs_raw = float(get_cfg(cfg, ["data", "fs_raw"], required=True))
    first_chan = int(get_cfg(cfg, ["data", "first_chan"], required=True))
    last_chan = int(get_cfg(cfg, ["data", "last_chan"], required=True))
    nch_expected = last_chan - first_chan + 1
    src_stride = int(get_cfg(cfg, ["data", "src_stride"], 10))
    min_length_sec = float(get_cfg(cfg, ["data", "min_length_sec"], 60.0))
    min_npts = int(min_length_sec * fs_raw)

    decimation = int(get_cfg(cfg, ["preprocess", "decimation"], 1))
    f1 = float(get_cfg(cfg, ["preprocess", "f1"], 1.0))
    f2 = float(get_cfg(cfg, ["preprocess", "f2"], 5.0))
    diff = bool(get_cfg(cfg, ["preprocess", "diff"], False))
    ram_win_sec = float(get_cfg(cfg, ["preprocess", "ram_win_sec"], 0.0))
    fs_proc = fs_raw / decimation

    is_spectral_whitening = bool(get_cfg(cfg, ["xcorr", "is_spectral_whitening"], True))
    window_freq_hz = float(get_cfg(cfg, ["xcorr", "window_freq_hz"], 0.0))
    max_lag_sec = float(get_cfg(cfg, ["xcorr", "max_lag_sec"], 4.0))
    xcorr_seg_sec = float(get_cfg(cfg, ["xcorr", "xcorr_seg_sec"], 8.0))
    mode = str(get_cfg(cfg, ["xcorr", "mode"], "conventional")).lower()
    auto_cc = bool(get_cfg(cfg, ["xcorr", "auto_cc"], False))

    whiten_chunk_nch = int(get_cfg(cfg, ["preprocess", "whiten_chunk_nch"], 64))

    if mode == "v1":
        xcorr_seg_sec = float(get_cfg(cfg, ["xcorr", "xcorr_seg_sec_v1"], xcorr_seg_sec))

    npts_lag = int(round(max_lag_sec * fs_proc))
    npts_seg = int(round(xcorr_seg_sec * fs_proc))

    cc_out_len = 2 * npts_lag + 1
    v1_fft_snap_pow2 = bool(get_cfg(cfg, ["xcorr", "v1_fft_snap_pow2"], True))
    v1_fallback = str(get_cfg(cfg, ["xcorr", "v1_fallback"], "v1_2M"))

    src_ch_all_num = np.arange(first_chan, last_chan + 1, src_stride, dtype=int)
    src_ch_all = src_ch_all_num - first_chan

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    # ==========================================================
    # 1. DATA INGESTION
    # ==========================================================
    t_io_start = time.perf_counter()
    _, das_array, dt, npts, _ = load_data(in_path, mmap=mmap)
    data_raw = das_array[:] if isinstance(das_array, zarr.Array) else das_array

    # Channel slicing — handle both pre-sliced and full-cable inputs
    nch_file = data_raw.shape[0]
    if nch_file == nch_expected:
        # File is already sliced to the target range
        pass
    elif nch_file > last_chan:
        # File contains the full cable (or more); slice to the requested channels
        data_raw = data_raw[first_chan : last_chan + 1, :]
    else:
        # File doesn't contain enough channels — skip
        logger.warning(
            "[%s] nch_file=%d < last_chan+1=%d, skipping",
            in_path.name, nch_file, last_chan + 1,
        )
        return None

    nch = data_raw.shape[0]

    io_time += (time.perf_counter() - t_io_start)

    basename = in_path.name.replace('.zarr', '').replace('.npz', '')

    if npts < min_npts:
        return None

    # ==========================================================
    # 2. PREPROCESSING
    # ==========================================================
    data_proc = preprocess(data_raw, fs_raw, f1, f2, decimation, diff, ram_win_sec).astype(np.float32, copy=False)
    npts_proc = int(data_proc.shape[1])

    npts_new = (npts_proc // npts_seg) * npts_seg
    nseg = npts_new // npts_seg
    if npts_new <= 0 or nseg == 0:
        return None

    data_proc = data_proc[:, :npts_new]
    flag_mean = int(nseg)
    prewhiten = bool(is_spectral_whitening)

    if prewhiten:
        data_tensor = convert_to_tensor(data_proc, device=torch.device("cpu"))
        del data_raw, data_proc
        gc.collect()

        data_tensor = data_tensor.pin_memory().to(device, non_blocking=True) if device.type == "cuda" else data_tensor.to(device)
        data_tensor = whiten_per_segment_torch(
            data_tensor, fs_proc=fs_proc, npts_seg=npts_seg,
            window_freq_hz=window_freq_hz, f1=f1, f2=f2,
            chunk_nch=whiten_chunk_nch,
        )
    else:
        data_tensor = convert_to_tensor(data_proc, device=torch.device("cpu"))
        del data_raw, data_proc
        gc.collect()
        data_tensor = data_tensor.pin_memory().to(device, non_blocking=True) if device.type == "cuda" else data_tensor.to(device)

    # ==========================================================
    # 2b. SIZING (v1-aware)
    # ==========================================================
    # Pre-compute v1 block sizing so auto_np_pair_chunk can use it. This makes
    # the chunk-size budget reflect the actual working-set of compute_X/Y for
    # v1 (real x_blocked + y_blocked + complex X + Y) instead of the much
    # smaller conventional output size.
    if mode == "v1":
        v1_K, v1_Lfft = choose_block_size_v1(
            int(npts_lag), fft_snap_pow2=v1_fft_snap_pow2, fallback=v1_fallback
        )
        v1_nblocks = (int(npts_seg) + v1_K - 1) // v1_K
        v1_nfreq   = v1_Lfft // 2 + 1
    else:
        v1_K = v1_Lfft = v1_nblocks = v1_nfreq = None

    npair_chunk = auto_np_pair_chunk(
        nch=nch, npts_seg=npts_seg, device=device,
        frac_mem=frac_mem, min_chunk=min_chunk, max_chunk=max_chunk,
        mode=mode, nseg=int(nseg),
        nblocks=v1_nblocks, Lfft=v1_Lfft, nfreq=v1_nfreq,
        nworkers=njobs,
    )

    is_spectral_whitening_cc = False if prewhiten else bool(is_spectral_whitening)
    whitening_params_cc = None if prewhiten else (float(fs_proc), float(window_freq_hz), float(f1), float(f2))

    # Lag window (conventional only — v1 returns 2M+1 directly).
    lag_start = (npts_seg - 1) - npts_lag if mode == "conventional" else 0
    lag_end = (npts_seg - 1) + npts_lag + 1 if mode == "conventional" else 0

    model: nn.Module = TorchCrossCorrelation(
        mode=mode,
        max_lag_samples=int(npts_lag) if mode == "v1" else None,
        is_spectral_whitening=is_spectral_whitening_cc,
        whitening_params=whitening_params_cc,
        v1_fft_snap_pow2=v1_fft_snap_pow2,
        v1_fallback=v1_fallback,
    )

    # The bare TorchCrossCorrelation is what the VS path calls into directly
    # (compute_X / compute_Y / combine). Always keep a reference to it.
    xcorr = model.to(device)
    xcorr.eval()

    multi_gpu = device.type == "cuda" and use_gpu and (torch.cuda.device_count() > 1)

    # nn.DataParallel only intercepts forward(). VS mode does not call forward,
    # so wrapping there is a silent no-op (only cuda:0 ever runs). Wrap only
    # for auto_cc, which does call model(data, data) → forward.
    if multi_gpu and auto_cc:
        forward_model: nn.Module = nn.DataParallel(xcorr).to(device)
        logger.info("Auto-CC + multi-GPU: nn.DataParallel across %d devices",
                    torch.cuda.device_count())
    else:
        forward_model = xcorr
        if multi_gpu and not auto_cc:
            logger.warning(
                "Multi-GPU detected but auto_cc=False. VS mode does not "
                "parallelize across nn.DataParallel; only cuda:0 will be "
                "used. For multi-GPU VS-mode runs, launch separate SLURM "
                "tasks pinned to different GPUs via CUDA_VISIBLE_DEVICES."
            )

    if do_compile and (not (multi_gpu and auto_cc)):
        try:
            forward_model = torch.compile(forward_model, backend="inductor", mode=compile_mode)
        except Exception:
            pass

    meta_path = out_dir / basename.replace(".npz", f"_cc_state_{mode}.json")
    completed_src = load_resume_state(meta_path)
    last_out: Optional[Path] = None

    vs_bar = tqdm(src_ch_all, desc=f"VS {basename}", leave=True)

    # ==========================================================
    # 3. CORRELATION ENGINE
    # ==========================================================
    if auto_cc:
        # Symmetric path. forward_model(data, data) routes through forward(),
        # which is what nn.DataParallel / torch.compile wrap. forward()
        # internally calls compute_X(x) + compute_Y(x) + combine.
        out_path = out_dir / f"{basename}_auto_{mode}.npy"
        if check_existing_output(out_path, (nch, cc_out_len)):
            return {"out_path": str(out_path), "io_sec": io_time, "cc_sec": 0.0,
                    "total_time": time.perf_counter() - t_total_start}

        write_runlog(f"Start Auto-CC: {gpu_memory() or ''} | {cpu_memory()}")

        nchunk = int(np.ceil(nch / npair_chunk))
        # On-device accumulator: avoids per-chunk PCIe round-trips on GPU.
        # On CPU this is just a torch CPU tensor — no transfer involved.
        ccall_dev = torch.zeros((nch, cc_out_len), dtype=torch.float32, device=device)

        t_cc_start = time.perf_counter()
        for ichunk in range(nchunk):
            start_idx = npair_chunk * ichunk
            end_idx = min(start_idx + npair_chunk, nch)
            batch_len = end_idx - start_idx

            full_data = data_tensor[start_idx:end_idx, :]
            data_in = full_data.view(batch_len, nseg, npts_seg)

            with torch.inference_mode():
                cc_sum_t = forward_model(data_in, data_in)

                if mode == "conventional":
                    ccall_dev[start_idx:end_idx].add_(cc_sum_t[:, lag_start:lag_end])
                else:
                    ccall_dev[start_idx:end_idx].add_(cc_sum_t)

        ccall_dev.div_(float(flag_mean))
        cc_time += (time.perf_counter() - t_cc_start)

        # Single device→host transfer for the whole file.
        t_io_start = time.perf_counter()
        ccall = ccall_dev.cpu().numpy()
        np.save(out_path, ccall)
        io_time += (time.perf_counter() - t_io_start)
        del ccall_dev
        last_out = out_path

    else:
        # Asymmetric VS path with X_src reuse + on-device ccall accumulator.
        for src_idx in vs_bar:
            src_idx = int(src_idx)
            out_path = out_dir / f"{basename}_cc_{src_idx:03d}_{mode}.npy"

            if check_existing_output(out_path, (nch, cc_out_len)) or (src_idx in completed_src):
                vs_bar.set_postfix_str(f"skip VS={src_idx}")
                last_out = out_path
                continue

            vs_bar.set_postfix_str(f"proc VS={src_idx}")
            write_runlog(f"Start VS {src_idx}: {gpu_memory() or ''} | {cpu_memory()}")

            npair = nch
            nchunk = int(np.ceil(npair / npair_chunk))

            # On-device accumulator. The previous CPU branch built a numpy
            # ccall and copied each chunk via .cpu().numpy() inside the loop,
            # which on GPU forced a synchronous PCIe round-trip per chunk and
            # killed device utilization. Keeping ccall on `device` defers all
            # device→host traffic to a single transfer at the end of this VS.
            ccall_dev = torch.zeros((npair, cc_out_len), dtype=torch.float32, device=device)

            # Source-side spectrum computed ONCE per VS. Shape:
            #   conventional → (1, nseg, nfreq)
            #   v1           → (1, nseg, nblocks, nfreq)
            src_segs = data_tensor[src_idx:src_idx + 1, :].view(1, nseg, npts_seg)
            t_vs0 = time.perf_counter()

            t_cc_start = time.perf_counter()
            with torch.inference_mode():
                X_src = xcorr.compute_X(src_segs)

                for ichunk in range(nchunk):
                    start_idx = npair_chunk * ichunk
                    end_idx = min(start_idx + npair_chunk, npair)
                    batch_len = end_idx - start_idx

                    # Row slices of a contiguous (nch, npts_new) tensor are
                    # contiguous, so .view(...) is safe without .contiguous().
                    recv_segs = data_tensor[start_idx:end_idx, :].view(batch_len, nseg, npts_seg)

                    Y_recv = xcorr.compute_Y(recv_segs)
                    cc_sum_t = xcorr.combine(X_src, Y_recv)

                    if mode == "conventional":
                        ccall_dev[start_idx:end_idx].add_(cc_sum_t[:, lag_start:lag_end])
                    else:
                        ccall_dev[start_idx:end_idx].add_(cc_sum_t)

                    # Drop receiver-side buffers between chunks so peak memory
                    # tracks one chunk's working set, not the cumulative.
                    del recv_segs, Y_recv, cc_sum_t

            ccall_dev.div_(float(flag_mean))
            cc_time += (time.perf_counter() - t_cc_start)

            # Single device→host transfer per VS.
            t_io_start = time.perf_counter()
            ccall = ccall_dev.cpu().numpy()
            np.save(out_path, ccall)
            io_time += (time.perf_counter() - t_io_start)
            del ccall_dev, X_src

            if perf_enabled and log_every_vs:
                write_perf_row(
                    {
                        "file": basename, "mode": mode, "vs_idx": int(src_idx),
                        "nch": int(nch), "npts_seg": int(npts_seg), "nseg": int(nseg),
                        "max_lag_samples": int(npts_lag), "npair_chunk": int(npair_chunk),
                        "device": str(device), "seconds_vs": float(time.perf_counter() - t_vs0),
                    },
                    perf_out_path, add_pid_suffix=True,
                )

            completed_src.add(src_idx)
            save_resume_state(meta_path, completed_src)
            last_out = out_path

    return {
        "out_path": str(last_out) if last_out else None,
        "io_sec": io_time,
        "cc_sec": cc_time,
        "total_time": time.perf_counter() - t_total_start,
    }

def _process_unpack(args: tuple) -> Optional[Dict[str, Any]]:
    return process_single_file(*args)

@timeit
def main(config_path: str | Path) -> None:
    # ----- GPU safety guard -----
    # Hide the GPU from the parent process while it does setup work, so any
    # accidental CUDA call here cannot grab the Exclusive-Process lock before
    # workers spawn. Restored just before mp.Pool so spawned workers inherit
    # the real device list. The try/finally guarantees restoration even if
    # setup raises (otherwise the empty value would leak to subsequent runs).
    _orig_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    try:
        cfg = load_config(config_path)

        data_root = Path(get_cfg(cfg, ["paths", "data_root"], required=True)).expanduser().resolve()
        output_root = Path(get_cfg(cfg, ["paths", "output_root"], required=True)).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        njobs = max(1, int(get_cfg(cfg, ["runtime", "njobs"], 4)))
        mode = str(get_cfg(cfg, ["xcorr", "mode"], "conventional")).lower()

        npz_files = [p for p in data_root.rglob("*.npz") if p.is_file()]
        zarr_dirs = [p for p in data_root.rglob("*.zarr") if p.is_dir()]
        filelist = sorted(npz_files + zarr_dirs)

        logger.info("Found %d valid datasets in %s", len(filelist), data_root)
        if not filelist:
            return

        slurm_cores = os.environ.get("SLURM_CPUS_PER_TASK")
        ncores = int(slurm_cores) if slurm_cores else (os.cpu_count() or 1)
        threads_per_proc = max(1, ncores // njobs)

        fs_raw = float(get_cfg(cfg, ["data", "fs_raw"], required=True))
        decimation = int(get_cfg(cfg, ["preprocess", "decimation"], 1))
        fs_proc = fs_raw / decimation

        max_lag_sec = float(get_cfg(cfg, ["xcorr", "max_lag_sec"], 4.0))
        xcorr_seg_sec = float(get_cfg(cfg, ["xcorr", "xcorr_seg_sec"], 8.0))
        if mode == "v1":
            xcorr_seg_sec = float(get_cfg(cfg, ["xcorr", "xcorr_seg_sec_v1"], xcorr_seg_sec))

        M = int(round(max_lag_sec * fs_proc))
        npts_seg = int(round(xcorr_seg_sec * fs_proc))

        v1_fft_snap_pow2 = bool(get_cfg(cfg, ["xcorr", "v1_fft_snap_pow2"], True))
        v1_fallback      = str(get_cfg(cfg, ["xcorr", "v1_fallback"], "v1_2M"))

        do_compile   = bool(get_cfg(cfg, ["runtime", "torch_compile"], False))
        compile_mode = str(get_cfg(cfg, ["runtime", "compile_mode"], "reduce-overhead"))

        warmup_nseg = 1
        try:
            probe = np.load(str(filelist[0]), mmap_mode="r")
            probe_key  = "data" if "data" in probe else list(probe.keys())[0]
            npts_probe = int(probe[probe_key].shape[-1])
            warmup_nseg = max(1, npts_probe // npts_seg)
            logger.info("warmup_nseg=%d derived from %s (npts=%d, npts_seg=%d)",
                        warmup_nseg, filelist[0].name, npts_probe, npts_seg)
        except Exception as e:
            logger.warning("Could not probe first file for warmup_nseg: %s. Using nseg=1.", e)

        initializer = functools.partial(
            _worker_warmup,
            mode=mode,
            npts_seg=npts_seg,
            max_lag_samples=M,
            v1_fft_snap_pow2=v1_fft_snap_pow2,
            v1_fallback=v1_fallback,
            threads_per_proc=threads_per_proc,
            do_compile=do_compile,
            compile_mode=compile_mode,
            warmup_nseg=warmup_nseg,
        )

        maxtasks = None if do_compile else 20
        logger.info(
            "Pool | njobs=%d | maxtasksperchild=%s | compiled=%s | warmup_nseg=%d",
            njobs, maxtasks, do_compile, warmup_nseg,
        )

    finally:
        # Restore the GPU visibility before mp.Pool is constructed so that
        # workers inherit the real device list at spawn time. Always run,
        # even if an exception was raised above, so re-runs in the same
        # shell don't leak CUDA_VISIBLE_DEVICES="".
        if _orig_cvd is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = _orig_cvd

    with mp.Pool(processes=njobs, initializer=initializer, maxtasksperchild=maxtasks) as pool:
        task_args = [(str(fpath), cfg) for fpath in filelist]
        for result_dict in tqdm(pool.imap_unordered(_process_unpack, task_args, chunksize=1),
                                total=len(filelist), desc="Processing"):
            if result_dict and result_dict.get("out_path"):
                logger.info("Done: %s | Total: %.2fs (I/O: %.2fs, CC: %.2fs)",
                            Path(result_dict["out_path"]).name, result_dict["total_time"],
                            result_dict["io_sec"], result_dict["cc_sec"])

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DAS ambient noise cross-correlation processing pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config file (.yaml/.yml/.json)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug/verbose logging output")
    return parser.parse_args(args=argv)

if __name__ == "__main__":
    # Configure logging only when run as a script, so importing cc.py from
    # other modules / notebooks doesn't clobber their logging setup.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    mp.set_start_method("spawn", force=True)
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    main(args.config)

# Example:
# python -m src.cc --config configs/cc.yaml --verbose
