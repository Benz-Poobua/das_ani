"""
:module: src/mccc.py
:auth: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Multi-Channel Cross-Correlation (MCCC) for DAS array processing.
          Extracts sub-sample precision relative arrival times and dt/t.
"""
from __future__ import annotations

import argparse
import functools
import logging
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from scipy import sparse
from scipy.sparse.linalg import lsmr
from scipy.signal import tukey, resample

# Local imports
from src.utils import bandpass_filter_tukey

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def torch_xcorr_1d_vs_nd(signal_1: torch.Tensor, signal_2: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-correlation of a 1D reference signal against an ND array of signals using FFT.
    
    :param signal_1: Array of signals, shape (nch, npts).
    :type signal_1: torch.Tensor
    :param signal_2: Reference signal repeated for all channels, shape (nch, npts).
    :type signal_2: torch.Tensor
    :return: Cross-correlation result, perfectly centered.
    :rtype: torch.Tensor
    """
    signal_length = signal_1.shape[-1]
    x_cor_sig_length = signal_length * 2 - 1
    
    # Next power of 2 for fast FFT
    fast_length = 2 ** int(np.ceil(np.log2(x_cor_sig_length)))

    fft_1 = torch.fft.rfft(signal_1, n=fast_length, dim=-1)
    fft_2 = torch.fft.rfft(signal_2, n=fast_length, dim=-1)

    # Cross-correlation in frequency domain
    fft_multiplied = torch.conj(fft_1) * fft_2

    # Back to time domain
    prelim_correlation = torch.fft.irfft(fft_multiplied, n=fast_length, dim=-1)

    # Shift and crop to valid lag window
    shift_idx = fast_length // 2
    start_idx = shift_idx - x_cor_sig_length // 2
    end_idx = start_idx + x_cor_sig_length
    
    final_result = torch.roll(prelim_correlation, shift_idx, dims=-1)[:, start_idx:end_idx]
    
    return final_result


def compute_mccc_delays(
    data: torch.Tensor, 
    dt: float, 
    cc_threshold: float = 0.6, 
    damp: float = 0.0, 
    return_all: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Perform Multi-Channel Cross-Correlation (VanDecar & Crosson, 1990).
    
    Cross-correlates all channel pairs and inverts the resulting relative
    time shifts to find the absolute optimized delay for each channel.
    
    :param data: Seismic data array of shape (nch, npts). Must be on the target device.
    :type data: torch.Tensor
    :param dt: Sampling interval in seconds.
    :type dt: float
    :param cc_threshold: Minimum correlation coefficient to include a pair in the inversion.
    :type cc_threshold: float
    :param damp: Damping factor for the least-squares regularization.
    :type damp: float
    :param return_all: If True, returns the optimized delays, the CC matrix, and the raw dt matrix.
    :type return_all: bool
    :return: A tuple containing the optimized relative arrival time for each channel (np.ndarray), 
             and optionally the maximum correlation coefficient matrix and raw time lag matrix.
    :rtype: Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
    """
    nch, npts = data.shape

    # Normalize data so max auto-correlation equals 1
    data_mean = torch.mean(data, dim=1, keepdim=True)
    data_std = torch.std(data, dim=1, keepdim=True)
    data_norm = (data - data_mean) / (data_std + 1e-12)

    # Initialize matrices
    ccmax = torch.zeros((nch, nch), device=data.device)
    dtmax = torch.zeros((nch, nch), device=data.device)

    logger.debug(f"Starting pairwise cross-correlation for {nch} channels...")

    # Compute pairwise cross-correlations
    for i in range(nch):
        signal1 = data_norm
        # Repeat channel 'i' to correlate against all other channels simultaneously
        signal2 = data_norm[i, :].unsqueeze(0).expand(nch, -1)
        
        cc = torch_xcorr_1d_vs_nd(signal1, signal2) / npts
        
        max_vals, max_indices = torch.max(cc, dim=1)
        ccmax[i, :] = max_vals
        dtmax[i, :] = (max_indices - (npts - 1)) * dt

    # Move tensors back to CPU for SciPy sparse inversion
    ccmax_np = ccmax.cpu().numpy()
    dtmax_np = dtmax.cpu().numpy()

    # Setup the Overdetermined Linear System (LSQR Inversion)
    x0, y0 = np.where(ccmax_np > cc_threshold)
    x, y = x0[x0 < y0], y0[x0 < y0]  # Keep only upper triangle to avoid redundant pairs
    
    nrow = len(x) + 1
    ncol = nch

    row = np.tile(np.arange(nrow - 1), 2)
    col = np.concatenate((x, y))
    val = np.concatenate((np.ones(nrow - 1), -np.ones(nrow - 1)))
    
    # Add constraint equation to center the result
    row = np.concatenate((row, (nrow - 1) * np.ones(ncol)))
    col = np.concatenate((col, np.arange(ncol)))
    val = np.concatenate((val, np.ones(ncol)))

    G_cc = sparse.coo_matrix((val, (row, col)), shape=(nrow, ncol))
    dt_obs = dtmax_np[x, y]

    # Regularization (Smoothing)
    D = (np.diag(np.ones(ncol)) - np.diag(np.ones(ncol - 1), k=-1))[1:, :]
    D = sparse.csr_matrix(D) * damp
    
    d = np.concatenate((dt_obs, np.zeros(D.shape[0] + 1)))
    G = sparse.vstack((G_cc, D))

    logger.debug(f"Solving linear system with {nrow} equations for {ncol} channels...")
    
    # Solve linear system
    optimized_delays = lsmr(G, d)[0]

    if return_all:
        return optimized_delays, ccmax_np, dtmax_np
    return optimized_delays, None, None


def run_mccc_windowed(
    data: np.ndarray, 
    pick_time: float, 
    window_half_width: float, 
    dt: float, 
    fs: float, 
    f1: float, 
    f2: float, 
    spatial_smooth_win: int, 
    device: torch.device,
    cc_threshold: float = 0.6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    End-to-end MCCC wrapper: Crops a specific seismic phase, filters, upsamples,
    and computes the fractional velocity change (dt/t) across the array.
    
    :param data: Symmetric NCF data, shape (nch, npts).
    :type data: np.ndarray
    :param pick_time: Target phase arrival time in seconds (center of the window).
    :type pick_time: float
    :param window_half_width: Half-width of the crop window in seconds.
    :type window_half_width: float
    :param dt: Original sampling interval.
    :type dt: float
    :param fs: Original sampling frequency.
    :type fs: float
    :param f1: Bandpass filter lower corner frequency.
    :type f1: float
    :param f2: Bandpass filter upper corner frequency.
    :type f2: float
    :param spatial_smooth_win: Number of channels to average over for spatial smoothing.
    :type spatial_smooth_win: int
    :param device: Compute device ('cpu' or 'cuda').
    :type device: torch.device
    :param cc_threshold: Threshold for MCCC inversion.
    :type cc_threshold: float
    :return: Tuple containing fractional velocity change (dt/t), mean CC per channel, 
             uncertainty estimate for dt/t, and median CC per channel.
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    nch, npts = data.shape
    
    # Define time window limits
    t1 = max(pick_time - window_half_width, 0)
    t2 = pick_time + window_half_width
    
    # Determine upsampling factor to ensure sub-sample precision
    upsample_factor = int(np.ceil(5 / (t2 - t1)))
    idx_t1 = int(np.floor(t1 / dt * upsample_factor))
    idx_t2 = int(np.ceil(t2 / dt * upsample_factor))

    logger.info(f"Running MCCC. Upsampling factor: {upsample_factor}x. Target window: [{t1:.2f}s, {t2:.2f}s]")

    # Fold symmetric NCF (average causal and acausal sides)
    midpt = npts // 2
    data_folded = (np.fliplr(data[:, :midpt + 1]) + data[:, midpt:]) / 2.0
    
    # Filter
    data_filtered = bandpass_filter_tukey(data_folded, fs=fs, f1=f1, f2=f2, alpha=0.05)

    # Upsample along the time axis
    data_up = resample(data_filtered, data_filtered.shape[-1] * upsample_factor, axis=-1)

    # Apply spatial smoothing (Running mean across channels)
    data_smooth = np.zeros_like(data_up)
    for i in range(nch):
        start_ch = max(0, i - spatial_smooth_win // 2)
        end_ch = min(nch, i + spatial_smooth_win // 2 + 1)
        data_smooth[i, :] = np.nanmean(data_up[start_ch:end_ch, :], axis=0)

    # Crop target window and apply Tukey taper
    window_len = idx_t2 - idx_t1
    data_cropped = data_smooth[:, idx_t1:idx_t2] * tukey(window_len, alpha=0.5)

    # Move to PyTorch
    data_tensor = torch.from_numpy(data_cropped).to(device)

    # Run MCCC Inversion
    new_dt = dt / upsample_factor
    m_delays, ccmax, dtmax = compute_mccc_delays(
        data=data_tensor, 
        dt=new_dt, 
        cc_threshold=cc_threshold, 
        damp=0.0, 
        return_all=True
    )

    # Calculate final physical metrics
    dt_t = m_delays / pick_time
    
    cc_median = np.median(ccmax, axis=0)
    
    # Clean up matrices based on threshold for error calculation
    ccmax[ccmax < cc_threshold] = np.nan
    dtmax[ccmax < cc_threshold] = np.nan
    ccmax_mean = np.nanmean(ccmax, axis=0)

    # Error estimation (Variance of the residuals)
    ti_tj = m_delays.reshape(nch, 1) - m_delays.reshape(1, nch)
    dt_t_err = np.sqrt(np.nanmean((dtmax - ti_tj) ** 2, axis=0)) / pick_time

    logger.info("MCCC calculation complete.")

    return dt_t, ccmax_mean, dt_t_err, cc_median

if __name__ == "__main__":
    # Optional CLI setup placeholder
    pass