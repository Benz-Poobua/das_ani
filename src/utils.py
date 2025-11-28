"""
:module: src/utils.py
:auth: Benz Poobua 
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Utility functions for DAS data processing, timing, GPU/CPU diagnostics.
"""
import os
import sys
import time
import torch
import psutil
import logging
import numpy as np

logger = logging.getLogger(__name__)

# 1. Load data
# ==============================================================
def load_data(filepath, mmap=False):
    """
    Load DAS waveform data from a .npz file.

    :param filepath: path to the `.npz` file; must contain keys 'data' and 'dt'
    :type filepath: str
    :param mmap: If True, use memory-mapped IO (np.load(..., mmap_mode='r')).
                 This is more memory-efficient for large files.
    :type mmap: bool

    :return:
        - **data_dict** (*dict*) -- dictionary loaded from the npz file  
        - **das_array** (*numpy.ndarray*) -- DAS matrix (n_channels × n_samples)  
        - **dt** (*float*) -- sampling interval in seconds  
        - **N** (*int*) -- number of samples  
        - **T** (*float*) -- total duration in seconds  
    """
    logger.info(f'Loading file: {filepath} (mmap={mmap})')

    data_dict = np.load(filepath, mmap_mode='r' if mmap else None)

    if 'data' not in data_dict or 'dt' not in data_dict:
        raise KeyError("NPZ file must contain 'data' and 'dt'.")
    
    das_array = data_dict['data']   # may be a memmap if mmap=True
    dt = float(data_dict['dt'])     # sampling interval (seconds)

    N = das_array.shape[1]          # number of samples
    T = N * dt                      # total duration

    logger.info(f'DAS loaded: shape={das_array.shape}, dt={dt}')
    return data_dict, das_array, dt, N, T

# 2. Tensor/Numpy conversions
# ==============================================================
def convert_to_tensor(x, device=None):
    """
    Convert input to PyTorch tensor on a specified device.

    :param x: Input array or tensor.
    :type x: numpy.ndarray, list, or torch.Tensor
    :param device: Target torch.device ('cpu', 'cuda', 'mps').
    :type device: torch.device or None

    :return: Tensor on the target device.
    :rtype: torch.Tensor
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if isinstance(x, torch.Tensor):
        if x.device != device:
            x = x.to(device)
        return x
    
    x = np.asarray(x)

    if np.iscomplexobj(x):
        return torch.tensor(x, dtype=torch.complex64, device=device)
    
    return torch.tensor(x, dtype=torch.float32, device=device)

def convert_to_numpy(x):
    """
    Convert tensor or array to numpy.ndarray on CPU.

    :param x: Torch tensor or array-like.
    :type x: torch.Tensor or numpy.ndarray

    :return: NumPy array
    :rtype: numpy.ndarray
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

# 3. Timing + decorators
# ==============================================================
def runtime():
    """
    Return synchronized high-resolution timestamp.

    :return: current timestamp in seconds
    :rtype: float
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()

def timeit(func):
    """
    Decorator to measure and log the runtime of any function.

    :param func: Function to wrap.
    :type func: callable

    :return: Wrapped function with timing.
    :rtype: callable

    Usage:
        @timeit
        def my_function(...):
            ...
    """
    def wrapper(*args, **kwargs):
        # Use the logger of the module where the function is defined
        logger = logging.getLogger(func.__module__)

        # Sync GPU before timing 
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        # Execute the wrapped function 
        result = func(*args, **kwargs)

        # Sync GPU before ending timer
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        dt = time.perf_counter() - t0

        msg = f'[{func.__name__}] elapsed = {dt:.3f} s'
        logger.info(msg)

        # Optional runlog write
        try:
            write_runlog(msg)
        except Exception:
            pass

        return result
    
    return wrapper

# 4. Math helper
# ==============================================================
def size(tensor):
    """
    Compute memory size of tensor in MB.

    :param tensor: Tensor for size reporting.
    :type tensor: torch.Tensor

    :return: Memory size in MB.
    :rtype: float
    """
    MB = tensor.nelement() * tensor.element_size() / (1024 ** 2)
    return MB

def norm_fro(A, Arec):
    """
    Compute normalized Frobenius error: ||A - A_rec|| / ||A||.

    :param A: Original matrix.
    :type A: torch.Tensor
    :param Arec: Reconstructed/processed matrix.
    :type Arec: torch.Tensor

    :return: Frobenius error.
    :rtype: float
    """
    return (torch.linalg.norm(A - Arec, ord="fro") /
            torch.linalg.norm(A, ord="fro")).item()

def compute_clip(arr, pclip=99):
    """
    Percentile clipping value for display.

    :param arr: Input amplitude array.
    :type arr: numpy.ndarray
    :param pclip: Percentile (default 99).
    :type pclip: float

    :return: Clipping value.
    :rtype: float
    """
    return float(np.percentile(arr, pclip))

def nextpow2(x):
    """
    Compute next power of 2.

    :param x: Scalar or tensor.
    :type x: int or torch.Tensor

    :return: Next power of 2.
    :rtype: torch.Tensor
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor([x], dtype=torch.float32)
        return 2 ** torch.ceil(torch.log2(x))[0]
   
    x = x.to(torch.float32)
    return 2 ** torch.ceil(torch.log2(x))

# 5. FK transform
# ==============================================================
def fk_transform(data, dt, dx, fast_len_t=None, fast_len_x=None):
    """
    Compute the f–k (frequency–wavenumber) spectrum using PyTorch FFT.
    Automatically pads FFT length to the next power of 2 unless overridden.

    :param data: DAS waveform matrix (nch × nt). Can be numpy array or torch tensor.
    :type data: numpy.ndarray or torch.Tensor
    :param dt: Time sampling interval in seconds.
    :type dt: float
    :param dx: Spatial sampling interval in meters.
    :type dx: float
    :param fast_len_t: Optional FFT length for time axis (overrides nextpow2).
    :type fast_len_t: int or None
    :param fast_len_x: Optional FFT length for space axis (overrides nextpow2).
    :type fast_len_x: int or None

    :return:
        - **freqs** (*torch.Tensor*) -- frequency axis in Hz, shape (nfreq,)
        - **wavenumbers** (*torch.Tensor*) -- wavenumber axis (cycles/m), shape (nk,)
        - **fk_spectrum** (*torch.Tensor*) -- complex f–k spectrum, shape (nk × nfreq)
    :rtype: tuple

    Notes
    -----
    - FFT along time (axis=1) followed by FFT along channels (axis=0).
    - Uses nextpow2() for fast FFTs unless user overrides.
    - Output preserved as torch complex tensor for whitening/filtering in f–k domain.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not isinstance(data, torch.Tensor):
        data = convert_to_tensor(data, device)
    else:
        data = data.to(device)

    nch, nt = data.shape

    Ft = fast_len_t or int(nextpow2(nt))
    Fx = fast_len_x or int(nextpow2(nch))
    
    # 1. FFT along time axis (dim=1)
    # ----------------------------------------------
    fft_t = torch.fft.rfft(data, n=Ft, dim=1)           # -> (nch, nfreq)
    freqs = torch.fft.rfftfreq(Ft, dt).to(device)       # shape (nfreq,)

    # 2. FFT along space axis (dim=0)
    # ----------------------------------------------
    fk_spectrum = torch.fft.fft(fft_t, n=Fx, dim=0)     # -> (nk, nfreq)
    wavenumbers = torch.fft.fftfreq(Fx, dx).to(device)  # shape (nk,)

    return freqs, wavenumbers, fk_spectrum

# 6. Runlog writer
# ==============================================================
_RUNLOG_PATH = os.path.expanduser('./data/runlog.txt')
os.makedirs(os.path.dirname(_RUNLOG_PATH), exist_ok=True)

def write_runlog(message, path=_RUNLOG_PATH):
    """
    Append message to runlog text file.

    :param message: Message to store.
    :type message: str
    :param path: Runlog file path.
    :type path: str

    :return: None
    :rtype: None
    """
    with open(path, 'a') as f:
        f.write(message +'\n')

# 7. Memory diagnostics
# ==============================================================
def gpu_memory(prefix=""):
    """
    GPU memory usage summary.

    :param prefix: Optional prefix label.
    :type prefix: str

    :return: Formatted GPU memory string or None.
    :rtype: str or None
    """
    if not torch.cuda.is_available():
        return None
    
    alloc = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)

    return (f'{prefix}GPU mem (MB): allocated={alloc:.1f}, '
            f'reserved={reserved:.1f}, max_reserved={max_reserved:.1f}')

def cpu_memory(prefix=""):
    """
    CPU RAM usage (RSS).

    :param prefix: Optional prefix.
    :type prefix: str

    :return: Formatted CPU memory string.
    :rtype: str
    """
    rss = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    return f'{prefix}CPU RSS = {rss:.1f} MB'