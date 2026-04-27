"""
:module: src/utils.py
:auth: Benz Poobua 
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Utility functions for DAS data processing, timing, GPU/CPU diagnostics, and I/O safety.
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

import numpy as np

from tqdm import tqdm
from pathlib import Path
from scipy.ndimage import uniform_filter, gaussian_filter
from scipy.fft import fft2, fftshift, ifft2, ifftshift

from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, TypeVar, Tuple, Union, overload, Literal


logger = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike[str], Path]
ArrayLike = Union[np.ndarray, torch.Tensor, list, tuple]
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

# ==============================================================
# 1. Load data
# ==============================================================
def load_data(filepath: PathLike, mmap: bool = False) -> tuple[Any, np.ndarray, float, int, float]:
    """
    Load DAS waveform data from a .npz file containing keys 'data' and 'dt'.

    :param filepath: Path to the `.npz` file.
    :param mmap: If True, use memory-mapped IO via np.load(..., mmap_mode='r').

    :return: (data_dict, das_array, dt, N, T)
    """
    path = Path(filepath).expanduser()

    logger.info("[load_data] utils at: %s", __file__)
    logger.info("Loading file: %s (mmap=%s)", path, mmap)

    data_dict = np.load(path, mmap_mode='r' if mmap else None)

    if "data" not in data_dict or "dt" not in data_dict:
        raise KeyError("NPZ file must contain 'data' and 'dt'.")
    
    das_array = data_dict["data"]       # may be np.memmap if mmap=True
    dt = float(data_dict["dt"])         # sampling interval (seconds)

    if das_array.ndim != 2:
        raise ValueError(f"'data' must be 2D (nch × nt); got shape={das_array.shape}")

    n_samples = int(das_array.shape[1])
    duration = n_samples * dt

    logger.info("DAS loaded: shape=%s, dt=%s", das_array.shape, dt)
    return data_dict, das_array, dt, n_samples, duration

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

    :param x: Input array/tensor.
    :param device: Target device. If None, auto-select cuda if available else cpu.
    :param dtype: Optional dtype override. If None, uses float32 or complex64 based on input.

    :return: Tensor on the target device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if isinstance(x, torch.Tensor):
        # Respect dtype override if provided; else keep original
        out = x
        if dtype is not None and out.dtype != dtype:
            out = out.to(dtype=dtype)
        if out.device != device:
            out = out.to(device)
        return out
    
    arr = np.asarray(x)
    if dtype is None:
        dtype = torch.complex64 if np.iscomplexobj(arr) else torch.float32

    # torch.as_tensor is zero-copy for CPU numpy arrays when possible
    out = torch.as_tensor(arr)
    if out.dtype != dtype:
        out = out.to(dtype=dtype)
    if out.device != device:
        out = out.to(device)
    return out

def convert_to_numpy(x: ArrayLike) -> np.ndarray:
    """
    Convert tensor/array-like to numpy.ndarray on CPU.

    :param x: Torch tensor or array-like.
    :return: NumPy array (CPU).
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

# ==============================================================
# 3. Timing + decorators
# ==============================================================
def runtime(sync_cuda: bool = True) -> float:
    """
    Return a high-resolution timestamp. Optionally sync CUDA first.

    :param sync_cuda: If True and CUDA is available, torch.cuda.synchronize() first.
    :return: Current timestamp in seconds.
    """
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()

def timeit(func: F) -> F:
    """
    Decorator to measure and log runtime of a function (CUDA-synced if available).

    :param func: Function to wrap.
    :return: Wrapped function.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        log = logging.getLogger(func.__module__)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        result = func(*args, **kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
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
    """
    Compute tensor memory size in MB.

    :param tensor: Input tensor.
    :return: Size in MB.
    """
    return float(tensor.nelement() * tensor.element_size() / (1024 ** 2))

def norm_fro(A: torch.Tensor, Arec: torch.Tensor) -> float:
    """
    Normalized Frobenius error: ||A - Arec||_F / ||A||_F

    :param A: Original matrix.
    :param Arec: Reconstructed/processed matrix.
    :return: Normalized Frobenius error.
    """
    return float(torch.linalg.norm(A - Arec, ord="fro") / torch.linalg.norm(A, ord="fro"))

def compute_clip(arr: np.ndarray, pclip: float = 99.0) -> float:
    """
    Percentile clipping value for display.

    :param arr: Input array.
    :param pclip: Percentile.
    :return: Clipping value.
    """
    return float(np.percentile(arr, pclip))

@overload
def nextpow2(x: int) -> int: ...
@overload
def nextpow2(x: float) -> int: ...
@overload
def nextpow2(x: torch.Tensor) -> torch.Tensor: ...

def nextpow2(x: Union[int, float, torch.Tensor]) -> Union[int, torch.Tensor]:
    """
    Next power of 2.

    - If x is int/float: returns an int.
    - If x is torch.Tensor: returns a torch.Tensor on the same device.

    :param x: Scalar or tensor.
    :return: Next power of 2.
    """
    if isinstance(x, torch.Tensor):
        xt = x.to(dtype=torch.float32)
        # Guard against non-positive values for log2
        xt = torch.clamp(xt, min=1.0)
        return (2.0 ** torch.ceil(torch.log2(xt))).to(dtype=torch.float32)
    
    # Python scalar path
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
    Compute the Forward 2D Fourier transform (f–k spectrum).

    :param data: 2D data array (nx, nt).
    :param dt: Time sampling interval (s).
    :param dx: Spatial sampling interval (m).
    :param pad_shape: Optional tuple (nx_pad, nt_pad) for FFT padding.
    :return: (freqs [Hz], wavenumbers [cycles/m], fk_spectrum [complex])
    """
    if data.ndim != 2:
        raise ValueError(f"'data' must be 2D (nx, nt); got {data.ndim}D")

    shape = pad_shape if pad_shape is not None else data.shape
    nx_out, nt_out = shape

    # 2D FFT with shift (places DC at center)
    fk_spectrum = fftshift(fft2(data, s=shape))

    # Construct axes
    k_axis = fftshift(np.fft.fftfreq(nx_out, dx))
    f_axis = fftshift(np.fft.fftfreq(nt_out, dt))

    return f_axis, k_axis, fk_spectrum


def fk_inverse(
    fk_spectrum: np.ndarray,
    orig_shape: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Compute the Inverse 2D Fourier transform.

    :param fk_spectrum: 2D f–k spectrum (nk, nf), usually fftshifted.
    :param orig_shape: Optional (nx, nt) to crop the output to.
    :return: Reconstructed time-space data (nx, nt).
    """
    if fk_spectrum.ndim != 2:
        raise ValueError(f"'fk_spectrum' must be 2D; got {fk_spectrum.ndim}D")

    # Inverse FFT (undo shift first)
    data = ifft2(ifftshift(fk_spectrum))

    # Crop to original shape if provided
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
    direction: Literal["both", "right", "left"] = "both",  # Direction control
    pad_factor: Tuple[int, int] = (1, 1),
    smooth: Literal["no", "gaussian", "uniform"] = "no",
    sigma: float = 1.0,
    uniform_size: int = 1,
) -> np.ndarray:
    """
    Apply velocity filtering in the f–k domain with optional directional masking.

    :param data: 2D data array (nx, nt).
    :param dt: Time sampling interval (s).
    :param dx: Spatial sampling interval (m).
    :param vmin: Min absolute velocity to target (m/s).
    :param vmax: Max absolute velocity to target (m/s).
    :param mode: 'eliminate' (remove band) or 'extract' (keep band).
    :param direction: 'both' (symmetric), 'right' (keep +k only), 'left' (keep -k only).
    :param pad_factor: Factors (nx_mul, nt_mul) for FFT padding. Default (1, 1).
    :param smooth: Mask smoothing method: 'no', 'gaussian', or 'uniform'.
    :param sigma: Sigma for gaussian smoothing (if smooth='gaussian').
    :param uniform_size: Kernel size for uniform smoothing (if smooth='uniform').
    :return: Filtered data (nx, nt).
    """
    nx_in, nt_in = data.shape
    nx_pad = int(nx_in * pad_factor[0])
    nt_pad = int(nt_in * pad_factor[1])

    # 1. Forward Transform
    freqs, ks, fk_data = fk_transform(data, dt, dx, pad_shape=(nx_pad, nt_pad))

    # Flip axis 0 (wavenumber) to match specific directional logic
    # (Matches original logic: positive direction L to R handling)
    fk_data = np.flip(fk_data, axis=0)

    # 2. Create Velocity Mask
    # v = f / k. Handle singularities.
    f_grid, k_grid = np.meshgrid(freqs, ks, indexing="xy")
    
    with np.errstate(divide="ignore", invalid="ignore"):
        v_grid = f_grid / k_grid
        # Handle k=0 (infinite velocity)
        v_grid[k_grid == 0] = np.inf
    
    # Base mask: 0 inside the velocity band, 1 outside
    mask = np.ones_like(fk_data.real)

    # Use absolute velocity to keep both Left and Right going waves
    mask[(np.abs(v_grid) >= vmin) & (np.abs(v_grid) <= vmax)] = 0.0

    # --- NEW: Directional Override ---
    if direction == "right":
        mask[k_grid < 0] = 1.0  # Force eliminate all negative wavenumbers
    elif direction == "left":
        mask[k_grid > 0] = 1.0  # Force eliminate all positive wavenumbers

    # 3. Apply Smoothing
    if smooth == "gaussian":
        mask = gaussian_filter(mask, sigma=sigma)
    elif smooth == "uniform":
        mask = uniform_filter(mask, size=uniform_size)
    elif smooth != "no":
        raise ValueError(f"Invalid smooth mode: {smooth}")

    # 4. Apply Mask (Mode Logic)
    if mode == "eliminate":
        # Keep everything OUTSIDE the band (multiply by mask where band=0)
        fk_data *= mask
    elif mode == "extract":
        # Keep everything INSIDE the band (multiply by inverse mask)
        fk_data *= (1.0 - mask)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # 5. Inverse Transform
    # Flip back before inverse
    fk_data = np.flip(fk_data, axis=0)
    
    return fk_inverse(fk_data, orig_shape=(nx_in, nt_in))

# ==============================================================
# 5B. Hilbert transform along time axis (for analytic signal / envelope)
# ==============================================================
def compute_envelope(data: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute the instantaneous amplitude (envelope) via Hilbert transform.
    """
    # Import locally to avoid slow load times if not used
    from scipy.signal import hilbert
    
    # Return absolute value of analytic signal
    return np.abs(hilbert(data, axis=axis))

# ==============================================================
# 6. Runlog writer
# ==============================================================
_DEFAULT_RUNLOG_PATH = Path("./data/runlog.txt").expanduser().resolve()

def write_runlog(message: str, path: PathLike = _DEFAULT_RUNLOG_PATH) -> None:
    """
    Append a message to a runlog text file.

    :param message: Line to append.
    :param path: Runlog file path.
    """
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
    """
    Append one row to a CSV file. Creates file + header if missing.

    :param row: Dict-like row to write.
    :param path: CSV path.
    :param add_pid_suffix: If True, writes per-process CSV to avoid race in multiprocessing.
    """
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
    """
    GPU memory usage summary (CUDA only).

    :param prefix: Optional label prefix.
    :return: Formatted summary string or None if CUDA unavailable.
    """
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
    """
    CPU RAM usage (RSS).

    :param prefix: Optional label prefix.
    :return: Formatted RSS string.
    """
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
    dtype: torch.dtype = torch.float32,
    safety_factor: float = 3.0,
    nworkers: int = 1, 
) -> int:
    """
    Heuristic to choose a safe batch size for channel-pair processing.

    Model (approx):
      - We materialize data1 and data2: 2 * (batch * npts_seg) * bytes_per_sample
      - We materialize CC output:
          conventional: (2*npts_seg-1) per pair
          v1: (2*M+1) per pair  (unknown here, so we conservatively assume conventional)
      - FFT workspace / temporaries: handled via safety_factor

    We do NOT force min_chunk if memory doesn't allow it.

    :param nch: Number of channels.
    :param npts_seg: Samples per segment.
    :param device: Target device.
    :param frac_mem: Fraction of available memory to budget.
    :param min_chunk: Preferred minimum chunk size (used only if fits).
    :param max_chunk: Hard cap on chunk size.
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
    cc_len = 2 * int(npts_seg) - 1  # conservative for conventional
    bytes_per_pair = (2 * npts_seg + cc_len) * bps
    bytes_per_pair = int(math.ceil(bytes_per_pair * float(safety_factor)))
    if bytes_per_pair <= 0:
        return 1

    if device.type == "cuda" and torch.cuda.is_available():
        free_bytes, _ = torch.cuda.mem_get_info()
        avail = int(free_bytes)
    else:
        avail = int(psutil.virtual_memory().available)

    # Divide by workers so total across processes respects frac_mem
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
    """
    Return True if output file exists and has expected shape.

    :param out_path: Output .npy path.
    :param expected_shape: Expected array shape.
    :return: True if valid output exists.
    """
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
    """
    Load JSON resume state (completed virtual sources).

    :param meta_path: Path to JSON state file.
    :return: Set of completed src indices.
    """
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
    """
    Save JSON resume state.

    :param meta_path: Path to JSON state file.
    :param completed_set: Completed source indices.
    """
    p = Path(meta_path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)

    safe_list = [int(x) for x in completed_set]
    with p.open("w", encoding="utf-8") as f:
        json.dump({"completed_src": sorted(safe_list)}, f, indent=2)

# ==============================================================
# 10. Config helpers
# ==============================================================
def load_config(path: str | Path) -> dict[str, Any]:
    """
    Load config from YAML (.yaml/.yml) or JSON (.json).

    YAML requires:
        pip install pyyaml
    """
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
    """
    Nested get: get_cfg(cfg, ["paths","data_root"]).
    """
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
    """
    Parse NCF stacked filename of format:
        YYYYMMDD_cc_XXX_<window>.npy

    Example:
        20210901_cc_080_daily.npy

    :param fname: Path or filename of the NCF stack
    :type fname: str
    :return: (date, vs, window)
    :rtype: Tuple[str, str, str]
    """
    base = os.path.basename(fname)

    m = re.match(r"(\d{8})_cc_(\d{3})_(\w+)\.npy", base)
    if m is None:
        raise ValueError(f"Filename not recognized: {fname}")
    
    date, vs, window = m.groups()
    return date, vs, window

def parse_ncf_stack_filename(fname: str) -> Tuple[str, str, str, str]:
    """
    Parse STACKED NCF filename of format:
        {prefix}_cc_{vs}_{window}_{mode}.npy

    Examples:
        20210928_cc_070_daily_conventional.npy
        20210928_cc_2120_7d_v1.npy
        20210928_cc_3191_30d_v2.npy

    :param fname: Path or filename of the NCF stack
    :type fname: str
    :return: (date, vs, window, mode)
    :rtype: Tuple[str, str, str, str]
    """
    base = os.path.basename(fname)
    
    # 1. (.+?)     -> Captures any date or prefix before '_cc_'
    # 2. (\d+)     -> Captures any number of digits for the VS index (0, 000, 3191)
    # 3. ([^_]+)   -> Captures the window string (daily, 7d, etc.)
    # 4. (.+)      -> Captures the mode string (v1, conventional)
    # 5. \.np[yz]$ -> Allows both .npy and compressed .npz formats
    m = re.match(r"(.+?)_cc_(\d+)_([^_]+)_(.+)\.np[yz]$", base)
    
    if m is None:
        raise ValueError(f"Stack filename not recognized: {fname}")
        
    date, vs, window, mode = m.groups()
    return date, vs, window, mode