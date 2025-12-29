"""
:module: src/utils.py
:auth: Benz Poobua 
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Utility functions for DAS data processing, timing, GPU/CPU diagnostics, and I/O safety.
"""
from __future__ import annotations

import json
import logging
import os
import time
import functools
import psutil
import torch

import numpy as np

from tqdm import tqdm
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, TypeVar, Union, overload

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

    x = convert_to_tensor(data, device=device)
    if x.ndim != 2:
        raise ValueError(f"'data' must be 2D (nch × nt); got shape={tuple(x.shape)}")

    nch, nt = int(x.shape[0]), int(x.shape[1])
    Ft = int(fast_len_t) if fast_len_t is not None else int(nextpow2(nt))
    Fx = int(fast_len_x) if fast_len_x is not None else int(nextpow2(nch))
    
    # 1. FFT along time axis (dim=1)
    fft_t = torch.fft.rfft(data, n=Ft, dim=1)           # (nch, nfreq)
    freqs = torch.fft.rfftfreq(Ft, dt).to(device)       # shape (nfreq,)

    # 2. FFT along space axis (dim=0)
    fk_spectrum = torch.fft.fft(fft_t, n=Fx, dim=0)     # (nk, nfreq)
    wavenumbers = torch.fft.fftfreq(Fx, dx).to(device)  # shape (nk,)

    return freqs, wavenumbers, fk_spectrum

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
    ) -> int:
    """
    Heuristic to choose a safe batch size for channel-pair processing.

    :param nch: Number of channels.
    :param npts_seg: Samples per segment.
    :param device: Target device (cpu/cuda/mps).
    :param frac_mem: Fraction of available memory to budget.
    :param min_chunk: Minimum chunk size.
    :param max_chunk: Maximum chunk size.
    :return: npair_chunk (>=1).
    """
    if nch <= 0:
        return 1
    if npts_seg <= 0:
        return int(min(min_chunk, nch))
    
    # Rough model: two float32 traces + intermediate buffers (tunable constant)
    bytes_per_pair = 64 * int(npts_seg)
    if bytes_per_pair <= 0:
        return int(min(min_chunk, nch))

    if device.type == "cuda" and torch.cuda.is_available():
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        budget = free_bytes * float(frac_mem)
    else:
        budget = psutil.virtual_memory().available * float(frac_mem)

    max_pairs_by_mem = int(budget // bytes_per_pair)

    if max_pairs_by_mem < min_chunk:
        npair_chunk = min_chunk
    else:
        npair_chunk = min(max_pairs_by_mem, max_chunk)

    npair_chunk = min(int(npair_chunk), int(nch))
    return int(max(npair_chunk, 1))

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