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
import math
import torch
import numpy as np
import scipy.signal as signal

from scipy.signal import butter, convolve, detrend, filtfilt
from torch import nn
from typing import Optional, Literal, Any

from src.utils import convert_to_numpy, convert_to_tensor, nextpow2

try:
    from scipy.special import lambertw  
except Exception:   
    lambertw = None

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
        logger.debug("Applying one-bit normalization.")
        return np.sign(data).astype(np.float32, copy=False)
    
    # Running Absolute Mean (RAM) Normalization
    nwin = int(round(fs * float(window_time)))
    nwin = max(nwin, 1)
    logger.debug("Applying RAM normalization: window=%.3fs (%d samples).", window_time, nwin)

    out = data.copy()
    for i in range(out.shape[0]):
        out[i, :] = running_absolute_mean(out[i, :], nwin)

    return out.astype(np.float32, copy=False)

# ==============================================================
# 2. Spectral whitening (torch) - (cached kernels/tapers)
# ==============================================================

class _WhiteningCache:
    """
    Tiny cache for spectral whitening tensors that are expensive to rebuild:
      - conv1d kernel (depends on dtype/device/nwin)
      - tapers (depends on dtype/device/idxf1/idxf2/nfreq)
    """
    __slots__ = ("kernel", "taper1", "taper2")

    def __init__(self) -> None:
        self.kernel: dict[tuple[Any, ...], torch.Tensor] = {}
        self.taper1: dict[tuple[Any, ...], torch.Tensor] = {}
        self.taper2: dict[tuple[Any, ...], torch.Tensor] = {}

    @staticmethod
    def _key(device: torch.device, dtype: torch.dtype, *rest: Any) -> tuple[Any, ...]:
        # Device is hashable, dtype is hashable
        return (device, dtype, *rest)
    
    def get_kernel(self, device: torch.device, dtype: torch.dtype, nwin: int) -> torch.Tensor:
        key = self._key(device, dtype, nwin)
        k = self.kernel.get(key)
        if k is None:
            # conv1d kernel shape: (out_ch=1, in_ch=1, k)
            k = torch.ones((1, 1, nwin), device=device, dtype=dtype) / float(nwin)
            self.kernel[key] = k
        return k
    
    def get_taper1(self, device: torch.device, dtype: torch.dtype, idxf1: int) -> Optional[torch.Tensor]:
        if idxf1 <= 0:
            return None
        key = self._key(device, dtype, idxf1)
        t = self.taper1.get(key)
        if t is None:
            t = torch.cos(torch.linspace(torch.pi / 2, torch.pi, idxf1, device=device, dtype=dtype)) ** 2
            self.taper1[key] = t
        return t
    
    def get_taper2(self, device: torch.device, dtype: torch.dtype, nfreq: int, idxf2: int) -> Optional[torch.Tensor]:
        if idxf2 >= nfreq:
            return None
        key = self._key(device, dtype, nfreq, idxf2)
        t = self.taper2.get(key)
        if t is None:
            n = nfreq - idxf2
            t = torch.cos(torch.linspace(torch.pi, torch.pi / 2, n, device=device, dtype=dtype)) ** 2
            self.taper2[key] = t
        return t
        
_WHITEN_CACHE = _WhiteningCache()

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
    dtype_amp = rfftdata.real.dtype
    B, nfreq = int(rfftdata.shape[0]), int(rfftdata.shape[1])

    # Compute freq indices
    if df <= 0:
        idxf1, idxf2 = 0, nfreq
    else: 
        idxf1 = int(f1 / df) 
        idxf2 = int(math.ceil(f2 / df))

    # Clip to array bounds
    idxf1 = max(0, min(idxf1, nfreq - 1))
    idxf2 = max(0, min(idxf2, nfreq))

    # Phase-only whitening
    if float(window_freq) == 0.0:
        # exp(i*angle) keeps magnitude 1
        return torch.exp(1j * torch.angle(rfftdata))
    
    # RAM amplitude smoothing
    nwin = max(int(window_freq / df), 1) if df > 0 else 1
    if nwin % 2 == 0:
        nwin += 1

    amp = torch.abs(rfftdata)       # (B, nfreq)
    phase = torch.angle(rfftdata)   # (B, nfreq)

    # Running mean with 1D convolution (GPU)
    # conv1d expects shape (batch, channels, length)
    amp_3d = amp.unsqueeze(1)       # (B, 1, nfreq)

    kernel = _WHITEN_CACHE.get_kernel(device, dtype_amp, nwin)

    # Padding to maintain same length
    pad = nwin // 2

    amp_smooth = torch.nn.functional.conv1d(amp_3d, kernel, padding=pad).squeeze(1)

    # Safety: match length 
    if amp_smooth.shape[-1] != amp.shape[-1]:
        if amp_smooth.shape[-1] > amp.shape[-1]:
            amp_smooth = amp_smooth[..., : amp.shape[-1]]
        else:
            amp_smooth = torch.nn.functional.pad(
                amp_smooth, (0, amp.shape[-1] - amp_smooth.shape[-1]), mode="replicate")

    amp_smooth = torch.where(amp_smooth == 0, torch.ones_like(amp_smooth), amp_smooth)
    rfft_out = torch.exp(1j * phase) * (amp / amp_smooth)

    # 3. Cosine Taper outside [f1, f2] 
    t1 = _WHITEN_CACHE.get_taper1(device, dtype_amp, idxf1)
    if t1 is not None:
        rfft_out[:, :idxf1] *= t1

    t2 = _WHITEN_CACHE.get_taper2(device, dtype_amp, nfreq, idxf2)
    if t2 is not None:
        rfft_out[:, idxf2:] *= t2

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
        x_np.shape, fs_raw, f1, f2, decimation, diff, ram_win)

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
    
    npts = int(signal_1.shape[1])

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
    """
    def __init__(
        self,
        *,
        mode: str = "conventional",  # "conventional" or "v1"
        max_lag_samples: Optional[int] = None,
        is_spectral_whitening: bool = False,
        whitening_params: Optional[tuple[float, float, float, float]] = None,
        v1_fft_snap_pow2: bool = True,
        v1_fallback: str = "v1_2M",
        ) -> None:
        super().__init__()

        self.mode = str(mode).lower()
        self.max_lag_samples = int(max_lag_samples) if max_lag_samples is not None else None

        self.is_spectral_whitening = bool(is_spectral_whitening)
        self.whitening_params = whitening_params

        self.v1_fft_snap_pow2 = bool(v1_fft_snap_pow2)
        self.v1_fallback = str(v1_fallback)

        if self.mode not in {"conventional", "v1"}:
            raise ValueError(f"Unknown mode={self.mode}. Use 'conventional' or 'v1'.")

        if self.v1_fallback not in {"v1_2M", "v1_Mp1"}:
            raise ValueError("v1_fallback must be 'v1_2M' or 'v1_Mp1'.")

        if self.is_spectral_whitening and self.whitening_params is None:
            raise ValueError("TorchCrossCorrelation: whitening_params required when whitening enabled.")

        if self.mode == "v1" and self.max_lag_samples is None:
            raise ValueError("TorchCrossCorrelation(mode='v1'): max_lag_samples is required.")
        
        # Conventional caching 
        self._conv_fast_length: Optional[int] = None
        self._conv_df: Optional[float] = None
        self._conv_xcorr_len: Optional[int] = None

        logger.info(
            "TorchCrossCorrelation initialized | mode=%s | max_lag=%s | whitening=%s",
            self.mode, self.max_lag_samples, self.is_spectral_whitening)
        
    def _conventional_cached(self, data1: torch.Tensor, data2: torch.Tensor) -> torch.Tensor:
        npts = int(data1.shape[1])
        x_corr_len = 2 * npts - 1

        if self._conv_fast_length is None or self._conv_xcorr_len != x_corr_len:
            fast_length_raw = nextpow2(x_corr_len)
            fast_length = int(fast_length_raw if isinstance(fast_length_raw, int) else int(fast_length_raw.item()))
            self._conv_fast_length = fast_length
            self._conv_xcorr_len = x_corr_len

            if self.is_spectral_whitening:
                assert self.whitening_params is not None
                fs = float(self.whitening_params[0])
                self._conv_df = fs / float(fast_length)
            else:
                self._conv_df = None
        
        fast_length = int(self._conv_fast_length)
        fft_1 = torch.fft.rfft(data1, n=fast_length, dim=-1)
        fft_2 = torch.fft.rfft(data2, n=fast_length, dim=-1)

        if self.is_spectral_whitening:
            assert self.whitening_params is not None
            _, window_freq, f1, f2 = self.whitening_params
            df = float(self._conv_df) if self._conv_df is not None else float(self.whitening_params[0]) / float(fast_length)
            fft_1 = spectral_whitening(fft_1, df, float(window_freq), float(f1), float(f2))
            fft_2 = spectral_whitening(fft_2, df, float(window_freq), float(f1), float(f2))

        cc_full = torch.fft.irfft(torch.conj(fft_1) * fft_2, n=fast_length, dim=-1)
        cc_full = torch.roll(cc_full, shifts=fast_length // 2, dims=-1)

        x_corr_len = int(self._conv_xcorr_len)  # type: ignore[arg-type]
        start = fast_length // 2 - (x_corr_len // 2)
        end = start + x_corr_len
        return cc_full[:, start:end]

    def forward(self, data1: torch.Tensor, data2: torch.Tensor) -> torch.Tensor:
        if self.mode == "conventional":
            return self._conventional_cached(data1, data2)

        # v1 path (real-only)
        if data1.is_complex() or data2.is_complex():
            raise ValueError("TorchCrossCorrelation(mode='v1') expects real tensors.")
        assert self.max_lag_samples is not None
        return cross_correlation_v1_real(
            data1,
            data2,
            max_lag_samples=self.max_lag_samples,
            is_spectral_whitening=self.is_spectral_whitening,
            whitening_params=self.whitening_params,
            fft_snap_pow2=self.v1_fft_snap_pow2,
            fallback=self.v1_fallback)

# ==============================================================
# 5. Zhang (2025) Workflow
# ==============================================================
def choose_block_size_v2(
    M: int,
    *,
    fft_snap_pow2: bool = True,
    fallback: Literal["v1_2M", "v1_Mp1"] = "v1_2M",
    ) -> tuple[int, int]:
    """
    Choose block size K for the block-by-block short-lag correlation.
    Returns (K, Lfft) where Lfft = K + 2M (possibly snapped to pow2).

    v2 (from Zhang 2025) formula:
        K* = 2M( -W_{-1}(-1/(4eM)) - 1 )

    Fallback:
        v1_2M  -> K = 2M
        v1_Mp1 -> K = M+1
    """
    if M <= 0:
        raise ValueError("choose_block_size_v2: M must be > 0")
    
    # Compute K* (v2) if Lambert W is available
    K_star: Optional[float] = None
    if lambertw is not None:
        # Argument is negative and close to 0: valid for W_{-1}
        z = -1.0 / (4.0 * math.e * float(M))
        w = lambertw(z, k=-1)

        # Use real part 
        w_real = float(w.real)
        K_star = 2.0 * float(M) * (-w_real - 1.0)

        # Numerical guard
        if not math.isfinite(K_star) or K_star <= 0.0:
            K_star = None

        logger.debug("LambertW available: using v2 block-size formula (M=%d).", M)
    else:
        logger.debug("LambertW not available: using fallback block-size rule (M=%d, fallback=%s).", M, fallback)

    # Fallback if v2 unavailable
    if K_star is None:
        if fallback == "v1_2M":
            K = int(2 * M)
        elif fallback == "v1_Mp1":
            K = int(M + 1)
        else:
            raise ValueError(f"Unknown fallback: {fallback}")
    else:
        K = int(max(1, round(K_star)))

    # Choose FFT length
    L = int(K + 2 * M)
    if fft_snap_pow2:
        Lpow2 = int(nextpow2(L))
        L = Lpow2
        K = int(L - 2*M)

    # Final guards AFTER snap: enforce K>=M+1 by snapping L again if needed
    if K < M + 1:
        K = M + 1
        L = int(K + 2*M)
        if fft_snap_pow2:
            L = int(nextpow2(L))
            K = int(L - 2*M)
    
    return K, L

@torch.no_grad()
def cross_correlation_v1_real(
    signal_1: torch.Tensor,
    signal_2: torch.Tensor,
    *,
    max_lag_samples: int,
    is_spectral_whitening: bool = False,
    whitening_params: Optional[tuple[float, float, float, float]] = None,
    fft_snap_pow2: bool = True,
    fallback: Literal["v1_2M", "v1_Mp1"] = "v1_2M",
) -> torch.Tensor:
    """
    Real-valued block-by-block short-lag correlation.
    Returns only lags in [-M, M], shape = (B, 2M+1).

    signal_1, signal_2: (B, N) real tensors (float32 recommended) on same device.
    """
    if signal_1.ndim != 2 or signal_2.ndim != 2:
        raise ValueError("cross_correlation_v1_real: inputs must be 2D (B × N).")
    if signal_1.shape != signal_2.shape:
        raise ValueError("cross_correlation_v1_real: signal_1 and signal_2 must have same shape.")
    if signal_1.dtype != signal_2.dtype:
        raise ValueError("cross_correlation_v1_real: signal_1 and signal_2 must have same dtype.")
    if signal_1.device != signal_2.device:
        raise ValueError("cross_correlation_v1_real: signals must be on the same device.")
    if max_lag_samples <= 0:
        raise ValueError("cross_correlation_v1_real: max_lag_samples must be > 0.")
    if is_spectral_whitening and whitening_params is None:
        raise ValueError("cross_correlation_v1_real: whitening_params required when whitening enabled.")
    
    # Recommend float32 for speed
    if signal_1.dtype not in (torch.float16, torch.float32, torch.float64):
        raise TypeError(f"Expected real floating dtype; got {signal_1.dtype}")

    M = int(max_lag_samples)
    B, N = int(signal_1.shape[0]), int(signal_1.shape[1])
    device = signal_1.device

    # Choose block size (K) and FFT length (Lfft = K + 2M, possibly snapped)
    K, Lfft = choose_block_size_v2(M, fft_snap_pow2=fft_snap_pow2, fallback=fallback)

    # Number of blocks
    nblocks = int((N + K - 1) // K)  # ceil(N/K) without float

    # Accumulator spectrum
    nfreq = Lfft // 2 + 1
    Rspec = torch.zeros((B, nfreq), dtype=torch.complex64, device=device)

    # Preallocate time-domain buffers (reused each block)
    # x_tilde: [x(lK-M : (l+1)K+M-1)] length K+2M, zero-padded outside [0,N)
    # y_tilde: [0...0, y(lK:(l+1)K-1), 0...0] length K+2M
    x_t = torch.zeros((B, Lfft), device=device, dtype=signal_1.dtype)
    y_t = torch.zeros((B, Lfft), device=device, dtype=signal_2.dtype)

    # Whitening constants
    if is_spectral_whitening:
        fs, window_freq, f1, f2 = whitening_params  # type: ignore[misc]
        df = float(fs) / float(Lfft)
        window_freq = float(window_freq)
        f1 = float(f1)
        f2 = float(f2)

    # Localize for tiny speed gains (Python-level)
    rfft = torch.fft.rfft
    irfft = torch.fft.irfft

    for l in range(nblocks):
        start = l * K
        end = min(start + K, N)
        klen = end - start  # may be < K for last block

        # Reuse buffers
        x_t.zero_()
        y_t.zero_()

        # y_tilde: [0..0, y(start:end), 0..0]
        y_t[:, M:M + klen] = signal_2[:, start:end]

        # x_tilde: x[start-M : start+K+M) placed into x_t[0:K+2M], clipped to [0,N)
        x0 = start - M
        x1 = start + K + M

        # Intersection with [0, N)
        ix0 = max(0, x0)
        ix1 = min(N, x1)

        if ix1 > ix0:
            # Where to place into x_t: offset by (ix0 - x0)
            dst0 = ix0 - x0
            dst1 = dst0 + (ix1 - ix0)
            x_t[:, dst0:dst1] = signal_1[:, ix0:ix1]

        # FFTs
        X = rfft(x_t, n=Lfft, dim=-1)
        Y = rfft(y_t, n=Lfft, dim=-1)

        # Whitening per block (if enabled)
        if is_spectral_whitening:
            X = spectral_whitening(X, df, window_freq, f1, f2)
            Y = spectral_whitening(Y, df, window_freq, f1, f2)

        # Accumulate cross-spectrum conj(X)*Y
        Rspec.addcmul_(X.conj(), Y)
    
    # Inverse FFT → circular correlation length Lfft
    r = irfft(Rspec, n=Lfft, dim=-1)

    # Extract [-M, M]
    out = torch.cat([r[:, Lfft - M:Lfft], r[:, 0:M + 1]], dim=-1)

    return out.to(dtype=torch.float32)