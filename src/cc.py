"""
:module: src/cc.py
:auth: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: DAS ambient noise interferometry (ANI).
          High-performance engine for Cross-Correlation (NCF generation) 
          and Auto-Correlation (ACF generation for Coda Wave Interferometry).
"""
from __future__ import annotations

import argparse
import functools
import logging
import os
import time
import gc
import math
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

from src.ani import preprocess, TorchCrossCorrelation, whiten_per_segment_torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_WARMED_UP = False


def _set_thread_env(threads: int) -> None:
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
        except Exception as e:
            logger.warning("torch.compile() failed in warmup: %s", e)

    Bwarm  = 32
    nseg_w = max(1, int(warmup_nseg))
    x = torch.zeros((Bwarm, nseg_w, int(npts_seg)), dtype=torch.float32, device=device)
    y = torch.zeros((Bwarm, nseg_w, int(npts_seg)), dtype=torch.float32, device=device)

    with torch.inference_mode():
        _ = model(x, y)
        _ = model(x, y)

    _WARMED_UP = True


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

    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

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
    nch = data_raw.shape[0]
    io_time += (time.perf_counter() - t_io_start)
    
    basename = in_path.name.replace('.zarr', '').replace('.npz', '')

    if nch != nch_expected or npts < min_npts:
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

    lag_start = (npts_seg - 1) - npts_lag if mode == "conventional" else 0
    lag_end = (npts_seg - 1) + npts_lag + 1 if mode == "conventional" else 0

    model: nn.Module = TorchCrossCorrelation(
        mode=mode,
        max_lag_samples=int(npts_lag) if mode == "v1" else None,
        is_spectral_whitening=False if prewhiten else bool(is_spectral_whitening),
        whitening_params=None if prewhiten else (float(fs_proc), float(window_freq_hz), float(f1), float(f2)),
        v1_fft_snap_pow2=v1_fft_snap_pow2,
        v1_fallback=v1_fallback,
    )

    # Optimization 4: Protect V1 chunk size by inflating the fake length sent to auto_np_pair_chunk
    if mode == "v1":
        _fake_nblocks = (npts_seg + model._v1_K - 1) // model._v1_K
        _v1_bytes = nseg * _fake_nblocks * (model._v1_Lfft * 8 + model._v1_nfreq * 16)
        _fake_npts_seg = int(_v1_bytes / 36)
        npair_chunk = auto_np_pair_chunk(
            nch=nch, npts_seg=_fake_npts_seg, device=device,
            frac_mem=frac_mem, min_chunk=min_chunk, max_chunk=max_chunk, nworkers=njobs,
        )
    else:
        npair_chunk = auto_np_pair_chunk(
            nch=nch, npts_seg=npts_seg, device=device,
            frac_mem=frac_mem, min_chunk=min_chunk, max_chunk=max_chunk, nworkers=njobs,
        )

    multi_gpu = device.type == "cuda" and use_gpu and (torch.cuda.device_count() > 1)
    model = nn.DataParallel(model).to(device) if multi_gpu else model.to(device)

    if do_compile and (not multi_gpu):
        try:
            model = torch.compile(model, backend="inductor", mode=compile_mode)
        except Exception:
            pass

    model.eval()

    meta_path = out_dir / basename.replace(".npz", f"_cc_state_{mode}.json")
    completed_src = load_resume_state(meta_path)
    last_out: Optional[Path] = None
    vs_bar = tqdm(src_ch_all, desc=f"VS {basename}", leave=True)

    # ==========================================================
    # 3. CORRELATION ENGINE
    # ==========================================================
    if auto_cc:
        out_path = out_dir / f"{basename}_auto_{mode}.npy"
        if check_existing_output(out_path, (nch, cc_out_len)):
            return {"out_path": str(out_path), "io_sec": io_time, "cc_sec": 0.0, "total_time": time.perf_counter() - t_total_start}
        
        write_runlog(f"Start Auto-CC: {gpu_memory() or ''} | {cpu_memory()}")

        nchunk = int(np.ceil(nch / npair_chunk))
        ccall = np.zeros((nch, cc_out_len), dtype=np.float32)

        t_cc_start = time.perf_counter()
        for ichunk in range(nchunk):
            start_idx = npair_chunk * ichunk
            end_idx = min(start_idx + npair_chunk, nch)
            batch_len = end_idx - start_idx
            
            # Sub-slice is inherently contiguous natively
            data_in = data_tensor[start_idx:end_idx, :].view(batch_len, nseg, npts_seg)

            with torch.inference_mode():
                cc_sum_t = model(data_in, data_in, is_source_spectrum=False)

            if mode == "conventional":
                ccall[start_idx:end_idx, :] += cc_sum_t[:, lag_start:lag_end].cpu().numpy()
            else:
                ccall[start_idx:end_idx, :] += cc_sum_t.cpu().numpy()

        ccall /= float(flag_mean)
        cc_time += (time.perf_counter() - t_cc_start)

        t_io_start = time.perf_counter()
        np.save(out_path, ccall)
        io_time += (time.perf_counter() - t_io_start)
        last_out = out_path

    else:
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
            ccall = np.zeros((npair, cc_out_len), dtype=np.float32)

            t_vs0 = time.perf_counter()
            t_cc_start = time.perf_counter()
            
            # Optimization 1: Encode Source exactly once
            src_trace = data_tensor[src_idx : src_idx + 1, :]
            data1_src = src_trace.view(1, nseg, npts_seg)
            with torch.inference_mode():
                model_base = model.module if multi_gpu else model
                X_src = model_base.encode_source(data1_src)

            for ichunk in range(nchunk):
                start_idx = npair_chunk * ichunk
                end_idx = min(start_idx + npair_chunk, npair)
                batch_len = end_idx - start_idx

                # Expand the cached source spectrum for the chunk
                X_expanded = X_src.expand(batch_len, *X_src.shape[1:])
                data2 = data_tensor[start_idx:end_idx, :].view(batch_len, nseg, npts_seg)

                with torch.inference_mode():
                    cc_sum_t = model(X_expanded, data2, is_source_spectrum=True)

                if mode == "conventional":
                    cc_win = cc_sum_t[:, lag_start:lag_end]
                    ccall[start_idx:end_idx, :] += cc_win.cpu().numpy()
                else:
                    ccall[start_idx:end_idx, :] += cc_sum_t.cpu().numpy()

            ccall /= float(flag_mean)
            cc_time += (time.perf_counter() - t_cc_start)

            t_io_start = time.perf_counter()
            np.save(out_path, ccall)
            io_time += (time.perf_counter() - t_io_start)

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
        "total_time": time.perf_counter() - t_total_start
    }

def _process_unpack(args: tuple) -> Optional[Dict[str, Any]]:
    return process_single_file(*args)

@timeit
def main(config_path: str | Path) -> None:
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
    if not filelist: return

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

    # Optimization 6: Relax maxtasksperchild to prevent massive Python import overhead per-file
    maxtasks = None if do_compile else 20
    logger.info(
        "Pool | njobs=%d | maxtasksperchild=%s | compiled=%s | warmup_nseg=%d",
        njobs, maxtasks, do_compile, warmup_nseg,
    )
    with mp.Pool(processes=njobs, initializer=initializer, maxtasksperchild=maxtasks) as pool:
        task_args = [(str(fpath), cfg) for fpath in filelist]
        for result_dict in tqdm(pool.imap_unordered(_process_unpack, task_args, chunksize=1), total=len(filelist), desc="Processing"):
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
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    main(args.config)

# Example:
# python -m src.cc --config configs/cc.yaml --verbose