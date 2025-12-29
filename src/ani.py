"""
:module: src/ani.py
:auth: Benz Poobua 
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: DAS preprocessing (Bensen et al., 2007) + GPU-accelerated cross-correlation.
:reference: Modified from Yan Yang (2022-07-10).
"""
from __future__ import annotations

import logging
import torch
import numpy as np
import scipy.signal as signal

from scipy.signal import butter, convolve, detrend, filtfilt
from torch import nn
from typing import Optional

from src.utils import convert_to_numpy, convert_to_tensor, nextpow2

logger = logging.getLogger(__name__)

# ==============================================================
# 1. Preprocessing utilities (numpy)
# ==============================================================
def bandpass_filter_tukey(
    data: np.ndarray,
    fs: float,
    f1: float,
    f2: float,
    alpha: float = 0.05,
    order: int = 4,
    ) -> np.ndarray:
    """
    Tukey-taper + Butterworth bandpass along time axis.

    :param data: 2D array (nch × nt).
    :param fs: Sampling rate (Hz).
    :param f1: Low-cut (Hz).
    :param f2: High-cut (Hz).
    :param alpha: Tukey window alpha in [0, 1].
    :param order: Butterworth filter order.
    :return: Filtered array float32 (nch × nt).
    """ 
    if data.ndim != 2:
        raise ValueError("bandpass_filter_tukey: data must be 2D (nch × nt).")
    
    nyq = fs / 2.0
    if not (0.0 < f1 < f2 < nyq):
        raise ValueError(f"Invalid f1/f2: require 0 < f1 < f2 < Nyquist={nyq}.")
    
    # Create Tukey window
    nt = int(data.shape[1])
    window = signal.windows.tukey(nt, alpha=float(alpha))

    low = f1 / nyq
    high = f2 / nyq
    b, a = butter(int(order), [low, high], btype="bandpass")

    # Broadcast Tukey window across channels
    tapered = data * window
    filtered = filtfilt(b, a, tapered, axis=1)

    return filtered.astype(np.float32, copy=False)

def running_absolute_mean(trace: np.ndarray, nwin: int) -> np.ndarray:
    """
    Running absolute mean (RAM) normalization for a 1D trace.

    :param trace: 1D array (nt,).
    :param nwin: Window length in samples (>1).
    :return: RAM-normalized trace (nt,).
    """
    if trace.ndim != 1:
        raise ValueError("running_absolute_mean: 'trace' must be 1D.")
    if nwin <= 1:
        return trace.copy()
    
    npts = int(trace.size)
    abs_trace = np.abs(trace)

    # Prepare padded array: length = npts + 2*nwin
    padded = np.empty(npts + 2 * nwin, dtype=trace.dtype)

    # Insert the central region
    padded[nwin:-nwin] = abs_trace

    # Pad front and back with boundary values
    padded[:nwin] = abs_trace[0]
    padded[-nwin:] = abs_trace[-1]

    # Moving average kernel 
    kernel = np.ones(int(nwin), dtype=np.float64) / float(nwin)

    # Convolve and remove padding
    ram = convolve(padded, kernel, mode="same")[nwin:-nwin]

    # Avoid division by zero
    ram = np.where(ram == 0, np.nan, ram)

    return np.nan_to_num(trace / ram, nan=0.0)

def temporal_normalization(data: np.ndarray, fs: float, window_time: float) -> np.ndarray:
    """
    Temporal normalization:

    - window_time == 0 -> one-bit normalization (sign)
    - window_time > 0  -> RAM normalization with window_time seconds

    :param data: 2D array (nch × nt).
    :param fs: Sampling rate (Hz).
    :param window_time: Window duration (s); 0 means one-bit.
    :return: Normalized array (nch × nt).
    """
    if data.ndim != 2:
        raise ValueError("temporal_normalization: data must be 2D (nch × nt).")
    
    # One-Bit Normalization
    if float(window_time) == 0.0:
        logger.info("Applying one-bit normalization.")
        return np.sign(data).astype(np.float32, copy=False)
    
    # Running Absolute Mean (RAM) Normalization
    nwin = int(round(fs * float(window_time)))
    if nwin < 1:
        nwin = 1
    logger.info("Applying RAM normalization: window=%.3fs (%d samples).", window_time, nwin)

    out = data.copy()
    for i in range(out.shape[0]):
        out[i, :] = running_absolute_mean(out[i, :], nwin)

    return out.astype(np.float32, copy=False)

# ==============================================================
# 2. Spectral whitening (torch)
# ==============================================================
def spectral_whitening(
    rfftdata: torch.Tensor,
    df: float,
    window_freq: float,
    f1: float,
    f2: float,
    ) -> torch.Tensor:
    """
    Spectral whitening on complex rFFT data (nch × nfreq).

    Modes:
    - window_freq == 0 : phase-only whitening (amp -> 1)
    - window_freq > 0  : RAM smoothing of amplitude in frequency domain via conv1d

    Applies cosine taper outside [f1, f2].

    :param rfftdata: Complex tensor (nch × nfreq).
    :param df: Frequency bin spacing (Hz).
    :param window_freq: RAM window (Hz). 0 => phase-only.
    :param f1: Low-cut (Hz).
    :param f2: High-cut (Hz).
    :return: Whitened rFFT tensor (nch × nfreq).
    """
    if not isinstance(rfftdata, torch.Tensor):
        raise TypeError("spectral_whitening: rfftdata must be torch.Tensor.")
    if not rfftdata.is_complex():
        raise ValueError("spectral_whitening: rfftdata must be complex.")

    device = rfftdata.device
    nch, nfreq = int(rfftdata.shape[0]), int(rfftdata.shape[1])

    # Compute freq indices
    idxf1 = int(f1 / df) if df > 0 else 0
    idxf2 = int(torch.ceil(torch.tensor(f2 / df, device=device)).item()) if df > 0 else nfreq  

    # Clip to array bounds
    idxf1 = max(0, min(idxf1, nfreq - 1))
    idxf2 = max(0, min(idxf2, nfreq))

    mode = "phase-only" if float(window_freq) == 0.0 else "RAM"
    logger.info("Spectral whitening (%s) | f1=%.3fHz f2=%.3fHz window=%.3fHz", mode, f1, f2, window_freq)

    # 1. Phase-only
    if float(window_freq) == 0.0:
        return torch.exp(1j * torch.angle(rfftdata))
    
    # 2. RAM amplitude smoothing
    nwin = max(int(window_freq / df), 1)

    # Ensure nwin is odd so conv1d output = input length
    if nwin % 2 == 0:
        nwin += 1

    amp = torch.abs(rfftdata)       # (nch, nfreq)
    phase = torch.angle(rfftdata)   # (nch, nfreq)

    # Running mean with 1D convolution (GPU)
    # conv1d expects shape (batch, channels, length)
    amp_3d = amp.unsqueeze(1)       # (nch, 1, nfreq)

    kernel = torch.ones((1, 1, nwin), device=device, dtype=amp.dtype) / float(nwin) 

    # Padding to maintain same length
    pad = nwin // 2

    amp_smooth = torch.nn.functional.conv1d(amp_3d, kernel, padding=pad).squeeze(1)

    # Force shape match (conv padding can create off-by-one depending on backend)
    if amp_smooth.shape[-1] > amp.shape[-1]:
        amp_smooth = amp_smooth[..., : amp.shape[-1]]
    elif amp_smooth.shape[-1] < amp.shape[-1]:
        amp_smooth = torch.nn.functional.pad(amp_smooth, (0, amp.shape[-1] - amp_smooth.shape[-1]), mode="replicate")

    amp_smooth = torch.where(amp_smooth == 0, torch.ones_like(amp_smooth), amp_smooth)

    rfft_out = torch.exp(1j * phase) * (amp / amp_smooth)

    # 3. Cosine Taper outside [f1, f2] 
    if idxf1 > 0:
        taper1 = torch.cos(torch.linspace(torch.pi / 2, torch.pi, idxf1, device=device)) ** 2
        rfft_out[:, :idxf1] *= taper1

    if idxf2 < nfreq:
        taper2 = torch.cos(torch.linspace(torch.pi, torch.pi / 2, nfreq - idxf2, device=device)) ** 2
        rfft_out[:, idxf2:] *= taper2

    return rfft_out

# ==============================================================
# 3. Full preprocessing pipeline
# ==============================================================
def preprocess(
    x: np.ndarray | torch.Tensor,
    fs_raw: float,
    f1: float,
    f2: float,
    decimation: int,
    diff: bool,
    ram_win: float,
    ) -> np.ndarray | torch.Tensor:
    """
    Ambient-noise preprocessing:
      diff (optional) -> detrend -> bandpass (Tukey+Butter) -> decimate ->
      remove median offset -> temporal normalization (one-bit/RAM)

    Preserves input type: numpy in -> numpy out; torch in -> torch out (same device).

    :param x: Input (nch × nt).
    :param fs_raw: Sampling rate (Hz).
    :param f1: Bandpass low-cut (Hz).
    :param f2: Bandpass high-cut (Hz).
    :param decimation: Decimation factor (>=1).
    :param diff: If True, time-derivative (np.gradient * fs_raw).
    :param ram_win: RAM window in seconds; 0 => one-bit.
    :return: Preprocessed data, float32.
    """
    if decimation < 1:
        raise ValueError("preprocess: decimation must be >= 1.")
    
    is_tensor = isinstance(x, torch.Tensor)
    orig_device: Optional[torch.device] = x.device if is_tensor else None

    x_np = convert_to_numpy(x) if is_tensor else np.asarray(x)
    
    if x_np.ndim != 2:
        raise ValueError(f"preprocess: expected 2D (nch × nt); got shape={x_np.shape}")

    logger.info(
        "Preprocess | shape=%s | fs=%.2fHz | band=[%.2f, %.2f]Hz | decim=%d | diff=%s | RAM=%.3fs",
        x_np.shape,
        fs_raw,
        f1,
        f2,
        decimation,
        diff,
        ram_win,
        )

    # 1. Differentiation (optional)
    if diff:
        x_np = np.gradient(x_np, axis=-1) * float(fs_raw)

    # 2. Detrend
    x_np = detrend(x_np, axis=-1)

    # 3. Bandpass filter (Butterworth + Tukey taper)
    x_np = bandpass_filter_tukey(x_np, fs_raw, f1, f2)

    # 4. Decimation
    if decimation > 1:
        x_np = x_np[:, :: int(decimation)]

    fs_proc = float(fs_raw) / float(decimation)

    # 5. Remove channel-wise DC offset (median across channels at each time sample)
    x_np -= np.median(x_np, axis=0)

    # 6. Temporal normalization (one-bit or RAM)
    x_np = temporal_normalization(x_np, fs_proc, float(ram_win))

    x_np = x_np.astype(np.float32, copy=False)

    if is_tensor:
        assert orig_device is not None
        return convert_to_tensor(x_np, device=orig_device)
    
    return x_np

# ==============================================================
# 4. Cross-correlation (torch) + module wrapper
# ==============================================================
@torch.no_grad()
def cross_correlation(
    signal_1: torch.Tensor,
    signal_2: torch.Tensor,
    *,
    is_spectral_whitening: bool = False,
    whitening_params: Optional[tuple[float, float, float, float]] = None,
    ) -> torch.Tensor:
    """
    Multi-channel cross-correlation via FFT (vectorized).

    :param signal_1: Tensor (nch × nt).
    :param signal_2: Tensor (nch × nt), same shape/device as signal_1.
    :param is_spectral_whitening: If True, whiten both spectra before CC.
    :param whitening_params: (fs, window_freq, f1, f2) used by spectral_whitening.
    :return: CC tensor (nch × (2*nt-1)).
    """
    if signal_1.ndim != 2 or signal_2.ndim != 2:
        raise ValueError("cross_correlation: inputs must be 2D (nch × nt).")
    if signal_1.shape != signal_2.shape:
        raise ValueError("cross_correlation: signal_1 and signal_2 must have same shape.")
    if signal_1.device != signal_2.device:
        raise ValueError("cross_correlation: signal_1 and signal_2 must be on the same device.")
    
    nch, npts = int(signal_1.shape[0]), int(signal_1.shape[1])

    # FFT size for full cross-correlation
    x_corr_len = 2 * npts - 1

    # nextpow2() returns int for scalar
    fast_length_raw = nextpow2(x_corr_len)
    fast_length = int(fast_length_raw if isinstance(fast_length_raw, int) else int(fast_length_raw.item()))

    fft_1 = torch.fft.rfft(signal_1, n=fast_length, dim=-1)
    fft_2 = torch.fft.rfft(signal_2, n=fast_length, dim=-1)

    if is_spectral_whitening:
        if whitening_params is None:
            raise ValueError("cross_correlation: whitening_params required when whitening is enabled.")
        fs, window_freq, f1, f2 = whitening_params
        df = float(fs) / float(fast_length)

        fft_1 = spectral_whitening(fft_1, df, float(window_freq), float(f1), float(f2))
        fft_2 = spectral_whitening(fft_2, df, float(window_freq), float(f1), float(f2))

    # Multiply with conjugate for CC spectrum 
    fft_prod = torch.conj(fft_1) * fft_2

    # Invert FFT → cross-correlation in time domain
    cc_full = torch.fft.irfft(fft_prod, n=fast_length, dim=-1)
 
    # Center zero lag
    cc_full = torch.roll(cc_full, shifts=fast_length // 2, dims=-1)

    start = fast_length // 2 - (x_corr_len // 2)
    end = start + x_corr_len

    return cc_full[:, start:end]

class TorchCrossCorrelation(nn.Module):
    """
    Module wrapper around cross_correlation().
    Intended for inference-only usage in the pipeline (use with torch.no_grad()).
    """
    def __init__(
        self,
        *,
        is_spectral_whitening: bool = False,
        whitening_params: Optional[tuple[float, float, float, float]] = None,
        ) -> None:
        super().__init__()

        if is_spectral_whitening and whitening_params is None:
            raise ValueError("TorchCrossCorrelation: whitening_params required when whitening enabled.")
        self.is_spectral_whitening = bool(is_spectral_whitening)
        self.whitening_params = whitening_params

        self.is_spectral_whitening = is_spectral_whitening
        self.whitening_params = whitening_params

        logger.info(
            "TorchCrossCorrelation initialized | whitening=%s | params=%s",
            self.is_spectral_whitening,
            self.whitening_params,
        )

    def forward(self, data1: torch.Tensor, data2: torch.Tensor) -> torch.Tensor:
        return cross_correlation(
            data1,
            data2,
            is_spectral_whitening=self.is_spectral_whitening,
            whitening_params=self.whitening_params,
            )

@torch.no_grad()    
def cross_correlation_full(
    data: torch.Tensor,
    ich1: int,
    ich2: int,
    *,
    is_spectral_whitening: bool = False,
    whitening_params: Optional[tuple[float, float, float, float]] = None,
    ) -> torch.Tensor:
    """
    Pairwise CC between selected channels [ich1:ich2] and all channels, via FFT broadcasting.

    Output shape: (Nsel × Ntotal × (2*nt-1))

    :param data: Tensor (Ntotal × nt).
    :param ich1: Start channel index (inclusive).
    :param ich2: End channel index (exclusive).
    :param is_spectral_whitening: If True, apply whitening.
    :param whitening_params: (fs, window_freq, f1, f2) used by spectral_whitening.
    :return: CC tensor (Nsel × Ntotal × (2*nt-1)).
    """
    if not isinstance(data, torch.Tensor):
        raise TypeError("cross_correlation_full: data must be torch.Tensor.")
    if data.ndim != 2:
        raise ValueError("cross_correlation_full: data must be 2D (Ntotal × nt).")
    if ich1 < 0 or ich2 > int(data.shape[0]) or ich1 >= ich2:
        raise ValueError("cross_correlation_full: invalid channel slice ich1:ich2.")
    if is_spectral_whitening and whitening_params is None:
        raise ValueError("cross_correlation_full: whitening_params required when whitening enabled.")
    
    n_total, npts = int(data.shape[0]), int(data.shape[1])
    n_sel = int(ich2 - ich1)

    # FFT size for full CC
    x_corr_len = 2 * npts - 1
    fast_length_raw = nextpow2(x_corr_len)
    fast_length = int(fast_length_raw if isinstance(fast_length_raw, int) else int(fast_length_raw.item()))

    # Select subset of channels
    sig_sel = data[ich1:ich2, :]                                 # (Nsel × npts)
    sig_all = data                                               # (Ntotal × npts)

    # Forward FFT
    fft_sel = torch.fft.rfft(sig_sel, n=fast_length, dim=-1)     # (Nsel, Nfreq)
    fft_all = torch.fft.rfft(sig_all, n=fast_length, dim=-1)     # (Ntotal, Nfreq)

    # Optional spectral whitening 
    if is_spectral_whitening:
        assert whitening_params is not None
        fs, window_freq, f1, f2 = whitening_params
        df = float(fs) / float(fast_length)

        fft_sel = spectral_whitening(fft_sel, df, float(window_freq), float(f1), float(f2))
        fft_all = spectral_whitening(fft_all, df, float(window_freq), float(f1), float(f2))

    # Broadcasting for pairwise CC:
    # (Nsel, 1, Nfreq) * (1, Ntotal, Nfreq) -> (Nsel, Ntotal, Nfreq)
    fft_prod = torch.conj(fft_sel.unsqueeze(1)) * fft_all.unsqueeze(0)
    cc_full = torch.fft.irfft(fft_prod, n=fast_length, dim=-1)

    cc_full = torch.roll(cc_full, shifts=fast_length // 2, dims=-1)

    start = fast_length // 2 - (x_corr_len // 2)
    end = start + x_corr_len

    out = cc_full[:, :, start:end]

    logger.info(
        "cross_correlation_full output shape=%s (Nsel=%d, Ntotal=%d, CClen=%d)",
        tuple(out.shape),
        n_sel,
        n_total,
        x_corr_len,
        )

    return out