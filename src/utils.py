"""
:module: src/utils.py
:auth: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Core utilities for DAS data processing, hardware diagnostics, and HPC safety.

Description:
This module provides the foundational support functions for the High-Performance 
Computing (HPC) Ambient Noise Interferometry (ANI) engine. It is specifically 
designed to handle SLURM environment quirks, manage PyTorch memory budgets, 
and ensure multi-GPU/multiprocessing safety.

Key Features:
    - HPC Hardware Diagnostics: Safely calculates available CPU memory by explicitly 
      reading SLURM cgroup limits (SLURM_MEM_PER_NODE) to prevent node OOM kills.
    - Multiprocessing-Safe GPU Timing: Custom timing decorators (``timeit``) that 
      guard against premature CUDA context initialization. This prevents the parent 
      process from accidentally acquiring the GPU lock under SLURM Exclusive mode 
      before the worker pool spawns.
    - Dynamic Resource Allocation: The ``auto_np_pair_chunk`` function mathematically 
      models the exact tensor byte-footprint of both the conventional and Zhang (2026) 
      v1 algorithms to maximize array chunk sizes without exceeding VRAM/RAM budgets.
    - Data I/O & Checkpointing: Transparent loading of large-scale .npz and .zarr 
      arrays, zero-copy PyTorch tensor conversions, and robust JSON-based auto-resume 
      tracking for interrupted cluster jobs.
    - DSP & Math Tools: 2D FK-filtering, temporal envelopes, and robust power-of-two 
      FFT padding calculators.
"""
from __future__ import annotations

import csv
import functools
import json
import logging
import math
import os
import psutil
import re
import time
import torch
import zarr

import numpy as np

from tqdm import tqdm
from pathlib import Path
from scipy.ndimage import uniform_filter, gaussian_filter
from scipy.fft import fft2, fftshift, ifft2, ifftshift

from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, TypeVar, Tuple, Union, overload, Literal

logger = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike[str], Path]
ArrayLike = Union[np.ndarray, torch.Tensor, list, tuple, zarr.Array]
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

# ==============================================================
# 0. HPC Diagnostics Helpers
# ==============================================================
def _get_cpu_mem_available() -> int:
    """
    Safely calculates available CPU memory, respecting SLURM cgroup allocations.
    psutil.virtual_memory() reports the whole node's memory, which causes OOM
    kills on shared HPC systems if the script overestimates its boundary.
    """
    sys_avail = int(psutil.virtual_memory().available)

    slurm_mem_mb = os.environ.get("SLURM_MEM_PER_NODE")
    if slurm_mem_mb:
        slurm_total_bytes = int(slurm_mem_mb) * 1024 * 1024
        process_used = psutil.Process(os.getpid()).memory_info().rss
        slurm_avail = max(0, slurm_total_bytes - process_used)
        return min(sys_avail, slurm_avail)

    return sys_avail

# ==============================================================
# 1. Load data
# ==============================================================
def load_data(filepath: PathLike, mmap: bool = False) -> tuple[Any, ArrayLike, float, int, float]:
    """
    Load DAS waveform data from either a .npz archive or a .zarr directory.

    :param filepath: Path to the `.npz` file or `.zarr` folder.
    :param mmap: If True, use memory-mapped IO. For Zarr, this is native;
                 for NPZ this is largely a no-op since NPZ is a zip and the
                 underlying arrays are decompressed into memory anyway.

    :return: (data_source, das_array, dt, N, T)
    """
    path = Path(filepath).expanduser()

    logger.info("[load_data] utils at: %s", __file__)
    logger.info("Loading dataset: %s (mmap requested=%s)", path, mmap)

    if path.suffix == '.zarr':
        z_root = zarr.open_group(str(path), mode='r')
        das_array = z_root['data']
        dt = float(das_array.attrs['dt'])
        data_source = z_root
    else:
        # Fallback to legacy NPZ
        data_source = np.load(path, mmap_mode='r' if mmap else None)
        if "data" not in data_source or "dt" not in data_source:
            raise KeyError("NPZ file must contain 'data' and 'dt'.")

        das_array = data_source["data"]
        dt = float(data_source["dt"])

    if das_array.ndim != 2:
        raise ValueError(f"'data' must be 2D (nch × nt); got shape={das_array.shape}")

    n_samples = int(das_array.shape[1])
    duration = n_samples * dt

    logger.info("DAS loaded: shape=%s, dt=%s", das_array.shape, dt)
    return data_source, das_array, dt, n_samples, duration

# ==============================================================
# 2. Tensor / Numpy conversions
# ==============================================================
def convert_to_tensor(
        x: ArrayLike,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        ) -> torch.Tensor:
    """
    Convert input to PyTorch tensor on a specified device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(x, torch.Tensor):
        out = x
        if dtype is not None and out.dtype != dtype:
            out = out.to(dtype=dtype)
        if out.device != device:
            out = out.to(device)
        return out

    if isinstance(x, zarr.Array):
        arr = x[:]
    else:
        arr = np.asarray(x)

    if dtype is None:
        dtype = torch.complex64 if np.iscomplexobj(arr) else torch.float32

    out = torch.as_tensor(arr)
    if out.dtype != dtype:
        out = out.to(dtype=dtype)
    if out.device != device:
        out = out.to(device)
    return out

def convert_to_numpy(x: ArrayLike) -> np.ndarray:
    """
    Convert tensor/array-like to numpy.ndarray on CPU.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, zarr.Array):
        return x[:]
    return np.asarray(x)

# ==============================================================
# 3. Timing + decorators
# ==============================================================
def _can_sync_cuda() -> bool:
    """
    True only if a CUDA context already exists in the current process.

    Calling torch.cuda.synchronize() unconditionally would create a context
    in the calling process. Under SLURM Exclusive Process mode, that's fatal
    in the multiprocessing parent — workers can't then acquire the device.
    torch.cuda.is_initialized() is the correct guard: it returns True only
    after the current process has issued a real CUDA op, never as a side
    effect of merely checking it.
    """
    return torch.cuda.is_available() and torch.cuda.is_initialized()


def runtime(sync_cuda: bool = True) -> float:
    """High-resolution timestamp; CUDA-synced only if this process holds a context."""
    if sync_cuda and _can_sync_cuda():
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            # Exclusive Process / Exclusive Thread mode: lock held elsewhere.
            pass
    return time.perf_counter()


def timeit(func: F) -> F:
    """
    Decorator that measures and logs runtime, CUDA-synced if (and only if) the
    calling process already holds a CUDA context. Safe to apply to functions
    that run in either the parent (no CUDA) or workers (own CUDA context).
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        log = logging.getLogger(func.__module__)

        sync_gpu = _can_sync_cuda()
        if sync_gpu:
            try:
                torch.cuda.synchronize()
            except RuntimeError:
                sync_gpu = False
        t0 = time.perf_counter()

        result = func(*args, **kwargs)

        if sync_gpu:
            try:
                torch.cuda.synchronize()
            except RuntimeError:
                pass

        dt = time.perf_counter() - t0

        msg = f"[{func.__name__}] elapsed = {dt:.3f} s"
        log.info(msg)

        try:
            write_runlog(msg)
        except Exception:
            log.debug("Runlog not available — skipping file logging.")

        return result

    return wrapper

# ==============================================================
# 4. Math helpers
# ==============================================================
def size_mb(tensor: torch.Tensor) -> float:
    return float(tensor.nelement() * tensor.element_size() / (1024 ** 2))

def norm_fro(A: torch.Tensor, Arec: torch.Tensor) -> float:
    return float(torch.linalg.norm(A - Arec, ord="fro") / torch.linalg.norm(A, ord="fro"))

def compute_clip(arr: np.ndarray, pclip: float = 99.0) -> float:
    return float(np.percentile(arr, pclip))

@overload
def nextpow2(x: int) -> int: ...
@overload
def nextpow2(x: float) -> int: ...
@overload
def nextpow2(x: torch.Tensor) -> torch.Tensor: ...

def nextpow2(x: Union[int, float, torch.Tensor]) -> Union[int, torch.Tensor]:
    if isinstance(x, torch.Tensor):
        xt = x.to(dtype=torch.float32)
        xt = torch.clamp(xt, min=1.0)
        return (2.0 ** torch.ceil(torch.log2(xt))).to(dtype=torch.float32)

    xf = float(x)
    if xf <= 1.0:
        return 1
    return int(2 ** int(np.ceil(np.log2(xf))))

# ==============================================================
# 5A. FK filtering (from Haipeng Li)
# ==============================================================
def fk_transform(
    data: np.ndarray,
    dt: float,
    dx: float,
    pad_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the 2D Forward Fourier Transform from the (x, t) domain to the (k, f) domain.

    **DSP Theory:**
    The FK transform (Frequency-Wavenumber) decomposes a wavefield into its constituent 
    plane waves. In DAS data, the horizontal axis (x) represents the fiber distance and 
    the vertical axis (t) represents time. The 2D FFT maps these to spatial frequency 
    (wavenumber, k) and temporal frequency (f).

    :param data: 2D array of DAS data, shape (n_channels, n_samples).
    :param dt: Temporal sampling interval in seconds.
    :param dx: Spatial sampling interval (channel spacing) in meters.
    :param pad_shape: Optional tuple (nx_pad, nt_pad) for zero-padding. Padding improves 
                      spectral resolution and prevents circular convolution artifacts.
    :return: (f_axis, k_axis, fk_spectrum) 
             f_axis: temporal frequency vector (Hz)
             k_axis: spatial wavenumber vector (1/m)
             fk_spectrum: 2D complex-valued Fourier spectrum, centered (shifted).
    """
    if data.ndim != 2:
        raise ValueError(f"'data' must be 2D (nx, nt); got {data.ndim}D")

    shape = pad_shape if pad_shape is not None else data.shape
    nx_out, nt_out = shape

    fk_spectrum = fftshift(fft2(data, s=shape))

    k_axis = fftshift(np.fft.fftfreq(nx_out, dx))
    f_axis = fftshift(np.fft.fftfreq(nt_out, dt))

    return f_axis, k_axis, fk_spectrum


def fk_inverse(
    fk_spectrum: np.ndarray,
    orig_shape: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Transforms a complex FK spectrum back to the space-time (x, t) domain.

    :param fk_spectrum: 2D complex spectrum in (k, f) domain, assumed to be shifted.
    :param orig_shape: Optional tuple (nx, nt) to crop the result back to original size, 
                       undoing any padding applied during the forward transform.
    :return: Real-valued space-time wavefield.
    """
    if fk_spectrum.ndim != 2:
        raise ValueError(f"'fk_spectrum' must be 2D; got {fk_spectrum.ndim}D")

    data = ifft2(ifftshift(fk_spectrum))

    if orig_shape is not None:
        nx, nt = orig_shape
        data = data[:nx, :nt]

    return data.real

def fk_filter(
    data: np.ndarray,
    dt: float,
    dx: float,
    vmin: float,
    vmax: float,
    mode: Literal["eliminate", "extract"] = "eliminate",
    direction: Literal["both", "right", "left"] = "both",
    pad_factor: Tuple[int, int] = (1, 1),
    smooth: Literal["no", "gaussian", "uniform"] = "no",
    sigma: float = 1.0,
    uniform_size: int = 1,
) -> np.ndarray:
    """
    Applies a velocity-based fan filter in the Frequency-Wavenumber (FK) domain.

    **DSP Theory:**
    In the FK domain, coherent seismic waves appear as energy organized along linear 
    trajectories. The slope of these lines represents the apparent phase velocity (v = f/k). 
    FK filtering allows for the separation of signal from noise based on velocity 
    and propagation direction.

    **Applications in DAS:**
    - **Surface Wave Removal:** Eliminating slow-moving Scholte or Rayleigh waves 
      (low v) to reveal faster body waves.
    - **Directional Steering:** Extracting only waves traveling in one direction along 
      the fiber (e.g., separating upgoing vs. downgoing waves in a borehole).
    - **Noise Suppression:** Removing "ringing" interrogator noise which often maps 
      to specific k=0 or f=0 regions.

    :param data: Space-time array (nch, nt).
    :param dt: Temporal sampling (s).
    :param dx: Spatial sampling (m).
    :param vmin: Minimum velocity threshold (m/s).
    :param vmax: Maximum velocity threshold (m/s).
    :param mode: "eliminate" to mute energy within [vmin, vmax]; 
                 "extract" to keep only energy within [vmin, vmax].
    :param direction: "right" (k > 0), "left" (k < 0), or "both".
    :param pad_factor: Multiplier for padding (nx_in*pad_factor[0], nt_in*pad_factor[1]).
    :param smooth: Type of taper to apply to the mask edges to prevent Gibbs ringing.
    :param sigma: Standard deviation for Gaussian smoothing.
    :param uniform_size: Window size for uniform smoothing.
    :return: Filtered space-time wavefield.
    """
    nx_in, nt_in = data.shape
    nx_pad = int(nx_in * pad_factor[0])
    nt_pad = int(nt_in * pad_factor[1])

    freqs, ks, fk_data = fk_transform(data, dt, dx, pad_shape=(nx_pad, nt_pad))

    fk_data = np.flip(fk_data, axis=0)

    f_grid, k_grid = np.meshgrid(freqs, ks, indexing="xy")

    with np.errstate(divide="ignore", invalid="ignore"):
        v_grid = f_grid / k_grid
        v_grid[k_grid == 0] = np.inf

    mask = np.ones_like(fk_data.real)

    mask[(np.abs(v_grid) >= vmin) & (np.abs(v_grid) <= vmax)] = 0.0

    if direction == "right":
        mask[k_grid < 0] = 1.0
    elif direction == "left":
        mask[k_grid > 0] = 1.0

    if smooth == "gaussian":
        mask = gaussian_filter(mask, sigma=sigma)
    elif smooth == "uniform":
        mask = uniform_filter(mask, size=uniform_size)
    elif smooth != "no":
        raise ValueError(f"Invalid smooth mode: {smooth}")

    if mode == "eliminate":
        fk_data *= mask
    elif mode == "extract":
        fk_data *= (1.0 - mask)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    fk_data = np.flip(fk_data, axis=0)

    return fk_inverse(fk_data, orig_shape=(nx_in, nt_in))

# ==============================================================
# 5B. Hilbert transform along time axis (for analytic signal / envelope)
# ==============================================================
def compute_envelope(data: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Calculates the instantaneous amplitude (envelope) via the Hilbert Transform.

    **DSP Theory:**
    The Hilbert transform shifts the phase of a signal by -90 degrees. By combining 
     the original real signal (x) and its Hilbert transform (y) as an analytic signal 
     (z = x + iy), we can calculate the instantaneous amplitude (envelope) as |z|.

    **Applications in DAS:**
    - Tracking the energy of ambient noise over time.
    - Enhancing the arrival of diffuse wave packets where phase is incoherent.
    - Normalization and gain control.

    :param data: Input waveform data.
    :param axis: The axis along which to compute the transform (usually time).
    :return: 2D array of instantaneous amplitudes.
    """
    from scipy.signal import hilbert
    return np.abs(hilbert(data, axis=axis))

# ==============================================================
# 6. Runlog writer
# ==============================================================
_DEFAULT_RUNLOG_PATH = Path("./data/runlog.txt").expanduser().resolve()

def write_runlog(message: str, path: PathLike = _DEFAULT_RUNLOG_PATH) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(message + "\n")

def write_perf_row(
    row: Mapping[str, Any],
    path: PathLike,
    *,
    add_pid_suffix: bool = True,
) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)

    if add_pid_suffix:
        pid = os.getpid()
        p = p.with_name(f"{p.stem}_{pid}{p.suffix}")

    fieldnames = list(row.keys())
    write_header = (not p.exists()) or (p.stat().st_size == 0)

    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(dict(row))

# ==============================================================
# 7. Memory diagnostics
# ==============================================================
def gpu_memory(prefix: str = "") -> Optional[str]:
    if not torch.cuda.is_available():
        return None

    alloc = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)

    return (
        f"{prefix}GPU mem (MB): allocated={alloc:.1f}, "
        f"reserved={reserved:.1f}, max_reserved={max_reserved:.1f}"
    )

def cpu_memory(prefix: str = "") -> str:
    rss = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    return f"{prefix}CPU RSS = {rss:.1f} MB"

# ==============================================================
# 8. Auto batch-size selection
# ==============================================================
def auto_np_pair_chunk(
    nch: int,
    npts_seg: int,
    device: torch.device,
    frac_mem: float = 0.25,
    min_chunk: int = 64,
    max_chunk: int = 4096,
    *,
    mode: str = "conventional",
    nseg: int = 1,
    nblocks: Optional[int] = None,
    Lfft: Optional[int] = None,
    nfreq: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    safety_factor: float = 3.0,
    nworkers: int = 1,
) -> int:
    """
    Heuristic to choose a safe batch size for channel-pair processing.

    Memory model
    ------------
    Conventional (default / fallback):
        bytes_per_pair ≈ (2*npts_seg + (2*npts_seg - 1)) * bps × safety_factor
        — i.e. data1, data2, and a (2N-1)-length CC output.

    v1 (when mode="v1" and nblocks/Lfft/nfreq are provided):
        bytes_per_pair ≈ [
            nseg * nblocks * Lfft * 4 * 2     # x_blocked, y_blocked (real)
          + nseg * nblocks * nfreq * 8 * 2    # X, Y (complex64)
          + nseg * npts_seg * 4               # input segment slab
        ] × safety_factor
        — this is the actual peak working set inside _forward_v1_batched /
        compute_X + compute_Y. 

    :param nch: Number of channels.
    :param npts_seg: Samples per segment.
    :param device: Target device.
    :param frac_mem: Fraction of available memory to budget.
    :param min_chunk: Preferred minimum chunk size (used only if it fits).
    :param max_chunk: Hard cap on chunk size.
    :param mode: "conventional" or "v1". Selects the per-pair byte model.
    :param nseg: Segments per channel. Required for the v1 model.
    :param nblocks: v1 block count. Required for the v1 model.
    :param Lfft: v1 FFT length. Required for the v1 model.
    :param nfreq: v1 spectrum length (Lfft // 2 + 1). Required for v1.
    :param dtype: Data dtype used for tensors.
    :param safety_factor: Multiplier to cover temporaries/workspace.
    :param nworkers: Number of data loading workers (for CPU memory budgeting).
    :return: npair_chunk in [1, min(nch, max_chunk)].
    """
    if nch <= 0:
        return 1
    if npts_seg <= 0:
        return max(1, min(nch, min_chunk))

    bps = 2 if dtype == torch.float16 else 4 if dtype == torch.float32 else 8 if dtype == torch.float64 else 4

    use_v1_model = (
        str(mode).lower() == "v1"
        and nblocks is not None and Lfft is not None and nfreq is not None
    )

    if use_v1_model:
        # Real (x_blocked, y_blocked) + complex (X, Y) + input segments.
        bytes_per_pair = (
            int(nseg) * int(nblocks) * int(Lfft) * bps * 2
            + int(nseg) * int(nblocks) * int(nfreq) * 8 * 2
            + int(nseg) * int(npts_seg) * bps
        )
    else:
        cc_len = 2 * int(npts_seg) - 1
        bytes_per_pair = (2 * int(npts_seg) + cc_len) * bps

    bytes_per_pair = int(math.ceil(bytes_per_pair * float(safety_factor)))
    if bytes_per_pair <= 0:
        return 1

    if device.type == "cuda" and torch.cuda.is_available():
        free_bytes, _ = torch.cuda.mem_get_info()
        avail = int(free_bytes)
    else:
        avail = _get_cpu_mem_available()

    nworkers = max(1, int(nworkers))
    budget = int(avail * float(frac_mem) / nworkers)
    if budget <= 0:
        return 1

    max_pairs_by_mem = max(1, int(budget // bytes_per_pair))
    upper = min(int(nch), int(max_chunk), int(max_pairs_by_mem))
    if upper <= 0:
        return 1
    return int(min_chunk) if min_chunk <= upper else int(upper)

# ==============================================================
# 9. Auto-resume helpers
# ==============================================================
def check_existing_output(out_path: PathLike, expected_shape: tuple[int, int]) -> bool:
    p = Path(out_path).expanduser()
    if not p.exists():
        return False

    try:
        arr = np.load(p)
        if tuple(arr.shape) == tuple(expected_shape):
            return True
        logger.warning("Corrupt output detected at %s (shape=%s), recomputing...", p, arr.shape)
        return False
    except Exception:
        logger.warning("Failed to load %s, recomputing...", p)
        return False

def load_resume_state(meta_path: PathLike) -> set[int]:
    p = Path(meta_path).expanduser()
    if not p.exists():
        return set()

    try:
        with p.open("r", encoding="utf-8") as f:
            state = json.load(f)
        return {int(x) for x in state.get("completed_src", [])}
    except Exception:
        return set()

def save_resume_state(meta_path: PathLike, completed_set: Iterable[int]) -> None:
    p = Path(meta_path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)

    safe_list = [int(x) for x in completed_set]
    with p.open("w", encoding="utf-8") as f:
        json.dump({"completed_src": sorted(safe_list)}, f, indent=2)

# ==============================================================
# 10. Config helpers
# ==============================================================
def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)

    suf = p.suffix.lower()
    if suf in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as e:
            raise ImportError("YAML config requested but PyYAML is not installed. Run: pip install pyyaml") from e
        with p.open("r") as f:
            cfg = yaml.safe_load(f)
            if not isinstance(cfg, dict):
                raise ValueError("Config root must be a mapping/dict.")
            return cfg

    if suf == ".json":
        with p.open("r") as f:
            cfg = json.load(f)
            if not isinstance(cfg, dict):
                raise ValueError("Config root must be a mapping/dict.")
            return cfg

    raise ValueError(f"Unsupported config extension: {suf} (use .yaml/.yml/.json)")

def get_cfg(cfg: Mapping[str, Any], keys: Sequence[str], default: Any = None, *, required: bool = False) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, Mapping) or k not in cur:
            if required:
                raise KeyError(f"Missing config key: {'.'.join(keys)}")
            return default
        cur = cur[k]
    return cur

# ==============================================================
# 11. Filename read helpers
# ==============================================================
def parse_ncf_filename(fname: str) -> Tuple[str, str, str]:
    base = os.path.basename(fname)

    m = re.match(r"(\d{8})_cc_(\d{3})_(\w+)\.npy", base)
    if m is None:
        raise ValueError(f"Filename not recognized: {fname}")

    date, vs, window = m.groups()
    return date, vs, window

def parse_ncf_stack_filename(fname: str) -> Tuple[str, str, str, str]:
    base = os.path.basename(fname)

    m = re.match(r"(.+?)_cc_(\d+)_([^_]+)_(.+)\.np[yz]$", base)

    if m is None:
        raise ValueError(f"Stack filename not recognized: {fname}")

    date, vs, window, mode = m.groups()
    return date, vs, window, mode
