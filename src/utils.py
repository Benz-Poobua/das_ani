"""
:module: src/utils.py
:auth: Benz Poobua 
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: This module contains scripts that manipulate DAS data.
"""
import os
import sys
import time
import torch
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Device selection
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
logger.info(f'Using device: {device}')

def load_data(filepath):
    """
    Load DAS waveform data from a .npz file.

    :param filepath: path to the `.npz` file; must contain keys 'data' and 'dt'
    :type filepath: str

    :return:
        - **data_dict** (*dict*) -- dictionary loaded from the npz file  
        - **das_array** (*numpy.ndarray*) -- DAS matrix (n_channels × n_samples)  
        - **dt** (*float*) -- sampling interval in seconds  
        - **N** (*int*) -- number of samples  
        - **T** (*float*) -- total duration in seconds  
    """
    logger.info(f'Loading file: {filepath}')

    data_dict = np.load(filepath)
    if 'data' not in data_dict or 'dt' not in data_dict:
        logger.error("NPZ file must contain 'data' and 'dt'.")
        raise KeyError('Missing required keys in NPZ file.')
    
    das_array = data_dict['data']
    dt = float(data_dict['dt'])     # sampling interval (seconds)

    N = das_array.shape[1]          # number of samples
    T = N * dt                      # total duration

    logger.info(f'DAS loaded: shape={das_array.shape}, dt={dt}')
    return data_dict, das_array, dt, N, T

def convert_to_tensor(x, device=None):
    """
    Convert input to a torch.Tensor on a given device.
    If already a tensor, move it to the target device.

    :param x: Input array or tensor.
    :type x: numpy.ndarray, list, or torch.Tensor

    :param device: Target torch.device ('cpu', 'cuda', 'mps').
    :type device: torch.device or None

    :return: DAS matrix as a float32 PyTorch tensor
    :rtype: torch.Tensor

    :return: Tensor on the target device.
    :rtype: torch.Tensor
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Already a torch tensor
    if isinstance(x, torch.Tensor):
        # avoid copying if already correct device and dtype
        if x.device != device:
            x = x.to(device)
        # do not cast complex tensors to float32
        if x.is_complex():
            return x
        return x.to(torch.float32)
    
    # Convert numpy → tensor
    x = np.asarray(x)

    # Complex numpy? preserve complex64
    if np.iscomplexobj(x):
        return torch.tensor(x, dtype=torch.complex64, device=device)
    
    # Otherwise real-valued float32
    return torch.tensor(x, dtype=torch.float32, device=device)

def convert_to_numpy(x):
    """
    Convert tensor or array to numpy.ndarray on CPU.

    :param x: Input tensor or array.
    :type x: torch.Tensor or numpy.ndarray 

    :return: converted NumPy array
    :rtype: numpy.ndarray
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def runtime():
    """
    Return synchronized high-resolution timestamp for benchmarking.

    :return: current timestamp in seconds
    :rtype: float
    """
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return time.perf_counter()

def size(das_torch):
    """
    Compute tensor size in megabytes.

    :param das_torch: DAS tensor
    :type das_torch: torch.Tensor

    :return: tensor size in megabytes
    :rtype: float
    """
    bytes_per_element = das_torch.element_size()
    total_bytes = das_torch.nelement() * bytes_per_element
    MB = total_bytes / (1024**2)

    logger.debug(f'Tensor size = {MB:.3f} MB')
    return MB

def norm_fro(ori_torch, reconstructed_torch):
    """
    Compute normalized Frobenius error: ||A - A_rec|| / ||A||.

    :param ori_torch: original DAS matrix
    :type ori_torch: torch.Tensor
    :param reconstructed_torch: reconstructed / processed DAS matrix
    :type reconstructed_torch: torch.Tensor

    :return: normalized Frobenius reconstruction error
    :rtype: float
    """
    normA = torch.linalg.norm(ori_torch, ord='fro')
    residual = torch.linalg.norm(ori_torch - reconstructed_torch, ord='fro')
    error = (residual / normA).item()

    logger.debug(f'Frobenius error = {error:.6e}')
    return error

def compute_clip(das_array, pclip=99):
    """
    Compute percentile-based clipping value for image display.

    :param das_array: amplitude array
    :type das_array: numpy.ndarray
    :param pclip: percentile for clipping, defaults to 99
    :type pclip: float, optional

    :return: clipping amplitude
    :rtype: float
    """
    clip = float(np.percentile(das_array, pclip))
    logger.debug(f'Clip ({pclip}th percentile) = {clip:.4f}')
    return clip

def nextpow2(x):
    """
    Vectorized next power of 2 for tensor inputs, fully GPU-accelerated.

    :param x: 1D or multi-dim tensor of integers
    :type x: torch.Tensor

    :return: tensor of next powers of 2 (same shape as x)
    :rtype: torch.Tensor
    """
    x = x.to(torch.float32)
    return 2 ** torch.ceil(torch.log2(x))

def fk_transform(data, dt, dx, fast_len_t=None, fast_len_x=None):
    """
    Compute the f–k (frequency–wavenumber) spectrum of DAS data using PyTorch.
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
    logger.info('Computing f-k transform.')

    # Ensure torch tensor
    if not isinstance(data, torch.Tensor):
        logger.debug('Input is not a torch.Tensor; converting via convert_to_tensor().')
        data = convert_to_tensor(data)
    
    data = data.to(device)
    nch, nt = data.shape
    logger.debug(f'Input data shape = (nch={nch}, nt={nt}) on device {device}.')

    # Determine FFT lengths using nextpow2()
    if fast_len_t is None:
        Ft = int(nextpow2(torch.tensor([nt], device=device))[0].item())
    else:
        Ft = fast_len_t

    if fast_len_x is None:
        Fx = int(nextpow2(torch.tensor([nch], device=device))[0].item())
    else:
        Fx = fast_len_x
    
    logger.debug(f'FFT lengths: time-axis Ft={Ft}, space-axis Fx={Fx}')

    # 1. FFT along time axis (dim=1)
    # ----------------------------------------------
    fft_t = torch.fft.rfft(data, n=Ft, dim=1)           # -> (nch, nfreq)
    nfreq = fft_t.shape[1]
    logger.debug(f'Time FFT complete: fft_t shape = {fft_t.shape}')

    # Frequency axis (Hz)
    freqs = torch.fft.rfftfreq(Ft, dt).to(device)       # shape (nfreq,)

    # 2. FFT along space axis (dim=0)
    # ----------------------------------------------
    fk_spectrum = torch.fft.fft(fft_t, n=Fx, dim=0)     # -> (nk, nfreq)
    nk = fk_spectrum.shape[0]
    logger.debug(f'Space FFT complete: fk_spectrum shape = {fk_spectrum.shape}')

    # Wavenumber axis (cycles/m)
    wavenumbers = torch.fft.fftfreq(Fx, dx).to(device)  # shape (nk,)

    logger.info('f-k transform completed successfully.')

    return freqs, wavenumbers, fk_spectrum