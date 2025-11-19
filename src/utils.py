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

# Load .npz DAS file
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

# Convert NumPy → PyTorch
def convert_to_tensor(das_array):
    """
    Convert a NumPy DAS array to a PyTorch tensor on the active device.

    :param das_array: DAS matrix
    :type das_array: numpy.ndarray

    :return: DAS matrix as a float32 PyTorch tensor
    :rtype: torch.Tensor
    """
    logger.debug('Converting NumPy array to torch.Tensor.')
    return torch.from_numpy(das_array.astype(np.float32)).to(device)

# Convert PyTorch → NumPy
def convert_to_numpy(das_torch):
    """
    Convert a PyTorch tensor to a NumPy array on CPU.

    :param das_torch: DAS tensor
    :type das_torch: torch.Tensor

    :return: converted NumPy array
    :rtype: numpy.ndarray
    """
    logger.debug('Converting torch.Tensor to NumPy array.')

    return das_torch.detach().cpu().numpy()

# Runtime measurement helper
def runtime():
    """
    Return synchronized high-resolution timestamp for benchmarking.

    :return: current timestamp in seconds
    :rtype: float
    """
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return time.perf_counter()

# Memory size utility
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

# Normalized Frobenius error
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

# Percentile clip for plotting
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