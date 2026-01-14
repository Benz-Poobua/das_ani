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
from scipy.ndimage import median_filter, gaussian_filter
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, TypeVar, Tuple, Union, overload


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
# 5. FK transform
# ==============================================================
@torch.no_grad()
def fk_transform(
    data: Union[np.ndarray, torch.Tensor],
    dt: float,
    dx: float,
    fast_len_t: Optional[int] = None,
    fast_len_x: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the f–k (frequency–wavenumber) spectrum using PyTorch FFT.

    :param data: DAS waveform matrix (nch × nt).
    :param dt: Time sampling interval (s).
    :param dx: Spatial sampling interval (m).
    :param fast_len_t: Optional FFT length along time (overrides nextpow2).
    :param fast_len_x: Optional FFT length along channels (overrides nextpow2).
    :param device: Target device. If None, auto-select.

    :return: (freqs [Hz], wavenumbers [cycles/m], fk_spectrum [complex])
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to tensor and ensure contiguous memory
    x = convert_to_tensor(data, device=device).contiguous()

    if x.ndim != 2:
        raise ValueError(f"'data' must be 2D (nch × nt); got shape={tuple(x.shape)}")

    nch, nt = int(x.shape[0]), int(x.shape[1])
    Ft = int(fast_len_t) if fast_len_t is not None else int(nextpow2(nt))
    Fx = int(fast_len_x) if fast_len_x is not None else int(nextpow2(nch))

    # 1) FFT along time axis (lag axis)
    fft_t = torch.fft.rfft(x, n=Ft, dim=1).contiguous()   # (nch, nfreq)
    freqs = torch.fft.rfftfreq(Ft, dt).to(device)         # (nfreq,)

    # 2) FFT along space axis
    fk_spectrum = torch.fft.fft(fft_t, n=Fx, dim=0)       # (nk, nfreq)
    fk_spectrum = fk_spectrum.contiguous()

    wavenumbers = torch.fft.fftfreq(Fx, dx).to(device)   # (nk,)

    return freqs, wavenumbers, fk_spectrum

@torch.no_grad()
def fk_velocity_filter(
    ncf: np.ndarray,
    dt: float,
    dx: float,
    vmin: float = 200.0,
    vmax: float = 2000.0,
    taper_frac: float = 0.10, 
    fast_len_t: int | None = None,
    fast_len_x: int | None = None,
    device: torch.device | None = None,
    ) -> np.ndarray:
    """
    f–k velocity-cone filter for an NCF gather (nch × nt), with smooth taper.

    Keeps energy consistent with velocities in [vmin, vmax] using v = f / |k|,
    and applies cosine tapers to avoid ringing artifacts.
    """
    if ncf.ndim != 2:
        raise ValueError(f"ncf must be 2D (nch × nt). Got {ncf.ndim}D.")

    nch, nt = ncf.shape

    # 1. Forward f–k transform 
    freqs, ks, FK = fk_transform(
        ncf,
        dt=dt,
        dx=dx,
        fast_len_t=fast_len_t,
        fast_len_x=fast_len_x,
        device=device,
    )
    FK = FK.contiguous()

    # Build velocity grid
    # freqs: (nfreq,), ks: (nk,)
    F = freqs[None, :]                 # (1, nfreq)
    K = ks[:, None]                    # (nk, 1)
    V = torch.abs(F / (torch.abs(K) + 1e-12))  # (nk, nfreq)

    # 2. Smooth velocity mask 
    # Define transition zones
    dv = float(taper_frac)
    vmin1, vmin2 = vmin * (1 - dv), vmin * (1 + dv)
    vmax1, vmax2 = vmax * (1 - dv), vmax * (1 + dv)

    w = torch.zeros_like(V)

    # Passband
    w[(V >= vmin2) & (V <= vmax1)] = 1.0

    # Low-velocity transition
    low_zone = (V >= vmin1) & (V < vmin2)
    w[low_zone] = 0.5 * (
        1 - torch.cos(np.pi * (V[low_zone] - vmin1) / (vmin2 - vmin1))
    )

    # High-velocity transition
    high_zone = (V > vmax1) & (V <= vmax2)
    w[high_zone] = 0.5 * (
        1 + torch.cos(np.pi * (V[high_zone] - vmax1) / (vmax2 - vmax1))
    )

    # Remove DC and unstable k≈0 region
    w[:, 0] = 0.0
    w[torch.abs(K[:, 0]) < 1e-9, :] = 0.0

    # Apply tapered mask
    FK_filt = (FK * w).contiguous()

    # 3. Inverse f–k transform 
    ifft_x = torch.fft.ifft(FK_filt, dim=0).contiguous()

    ntime = 2 * (ifft_x.shape[1] - 1)
    x_t = torch.fft.irfft(ifft_x, n=ntime, dim=1).real

    # Crop back to original size
    x_t = x_t[:nch, :nt]

    return x_t.detach().cpu().numpy().astype(np.float32, copy=False)

def fv_filter(
    fv_panel: np.ndarray,
    f_axis: np.ndarray,
    v_axis: np.ndarray,
    fmin: float | None = None,
    fmax: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    normalize: str = "per_f",      # "per_f", "global", or "none"
    denoise: str = "median",       # "median", "gaussian", or "none"
    denoise_size: int = 3,         # for median (odd int)
    denoise_sigma: float = 1.0,    # for gaussian
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter/prepare an f–v dispersion panel for ridge picking.

    Assumes fv_panel shape = (n_vel, n_freq) like your plot_fv_panel.

    :return: (fv_filtered, f_filtered, v_filtered)
    """
    P = np.asarray(fv_panel)
    f = np.asarray(f_axis)
    v = np.asarray(v_axis)

    if P.ndim != 2:
        raise ValueError(f"fv_panel must be 2D; got shape={P.shape}")
    if P.shape != (v.size, f.size):
        raise ValueError(
            f"Shape mismatch: fv_panel={P.shape}, expected (n_vel, n_freq)=({v.size}, {f.size})."
        )

    # --- crop masks ---
    fmask = np.ones_like(f, dtype=bool)
    vmask = np.ones_like(v, dtype=bool)

    if fmin is not None:
        fmask &= (f >= float(fmin))
    if fmax is not None:
        fmask &= (f <= float(fmax))
    if vmin is not None:
        vmask &= (v >= float(vmin))
    if vmax is not None:
        vmask &= (v <= float(vmax))

    f2 = f[fmask]
    v2 = v[vmask]
    P2 = P[np.ix_(vmask, fmask)].copy()

    # --- normalization ---
    normalize = normalize.lower()
    if normalize == "per_f":
        # normalize each frequency column (good for picking)
        colmax = np.max(np.abs(P2), axis=0, keepdims=True)
        P2 = P2 / (colmax + eps)
    elif normalize == "global":
        P2 = P2 / (np.max(np.abs(P2)) + eps)
    elif normalize == "none":
        pass
    else:
        raise ValueError("normalize must be one of: 'per_f', 'global', 'none'")

    # --- denoise ---
    denoise = denoise.lower()
    if denoise == "median":
        # good for speckle / salt-and-pepper
        k = int(denoise_size)
        if k < 1 or k % 2 == 0:
            raise ValueError("denoise_size must be an odd integer >= 1")
        P2 = median_filter(P2, size=(k, k))
    elif denoise == "gaussian":
        P2 = gaussian_filter(P2, sigma=float(denoise_sigma))
    elif denoise == "none":
        pass
    else:
        raise ValueError("denoise must be one of: 'median', 'gaussian', 'none'")

    return P2.astype(np.float32, copy=False), f2, v2

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