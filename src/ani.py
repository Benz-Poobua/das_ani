"""
:module: src/ani.py
:auth: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: DAS preprocessing (Bensen et al., 2007) + cross-correlation (conventional + Zhang 2026 v1).
:reference: Modified from Yan Yang (2022-07-10).
"""
from __future__ import annotations

import logging
import math
from typing import Any, Optional, Literal

import numpy as np
import scipy.signal as signal
import torch
from scipy.signal import detrend
from scipy.ndimage import uniform_filter1d
from torch import nn

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
    f1: float | None = None,
    f2: float | None = None,
    alpha: float = 0.05,
    order: int = 4,
) -> np.ndarray:
    if data.ndim != 2:
        raise ValueError("bandpass_filter_tukey: data must be 2D (nch × nt).")

    nyq = fs / 2.0
    nt = int(data.shape[1])
    window = signal.windows.tukey(nt, alpha=float(alpha))

    if f1 is not None and f2 is not None:
        btype = "bandpass"
        Wn = [f1 / nyq, f2 / nyq]
    elif f1 is None and f2 is not None:
        btype = "lowpass"
        Wn = f2 / nyq
    elif f1 is not None and f2 is None:
        btype = "highpass"
        Wn = f1 / nyq
    else:
        raise ValueError("Must specify at least one frequency (f1 or f2).")

    sos = signal.butter(int(order), Wn, btype=btype, output='sos')
    tapered = data * window
    filtered = signal.sosfiltfilt(sos, tapered, axis=1)
    return filtered.astype(np.float32, copy=False)

def temporal_normalization(data: np.ndarray, fs: float, window_time: float) -> np.ndarray:
    if data.ndim != 2:
        raise ValueError("temporal_normalization: data must be 2D (nch × nt).")
    if float(window_time) == 0.0:
        return np.sign(data).astype(np.float32, copy=False)

    nwin = max(int(round(fs * float(window_time))), 1)
    ram = uniform_filter1d(np.abs(data), size=nwin, axis=1, mode='nearest')
    ram = np.where(ram == 0, np.nan, ram)
    out = np.nan_to_num(data / ram, nan=0.0)
    return out.astype(np.float32, copy=False)

# ==============================================================
# 2. Spectral whitening (torch)
# ==============================================================
class _WhiteningCache:
    __slots__ = ("kernel", "taper1", "taper2")
    def __init__(self) -> None:
        self.kernel: dict[tuple[Any, ...], torch.Tensor] = {}
        self.taper1: dict[tuple[Any, ...], torch.Tensor] = {}
        self.taper2: dict[tuple[Any, ...], torch.Tensor] = {}

    @staticmethod
    def _key(device: torch.device, dtype: torch.dtype, *rest: Any) -> tuple[Any, ...]:
        return (device, dtype, *rest)

    def get_kernel(self, device: torch.device, dtype: torch.dtype, nwin: int) -> torch.Tensor:
        key = self._key(device, dtype, nwin)
        if key not in self.kernel:
            self.kernel[key] = torch.ones((1, 1, nwin), device=device, dtype=dtype) / float(nwin)
        return self.kernel[key]

    def get_taper1(self, device: torch.device, dtype: torch.dtype, idxf1: int) -> Optional[torch.Tensor]:
        if idxf1 <= 0: return None
        key = self._key(device, dtype, idxf1)
        if key not in self.taper1:
            self.taper1[key] = torch.cos(torch.linspace(torch.pi / 2, torch.pi, idxf1, device=device, dtype=dtype)) ** 2
        return self.taper1[key]

    def get_taper2(self, device: torch.device, dtype: torch.dtype, nfreq: int, idxf2: int) -> Optional[torch.Tensor]:
        if idxf2 >= nfreq: return None
        key = self._key(device, dtype, nfreq, idxf2)
        if key not in self.taper2:
            n = nfreq - idxf2
            self.taper2[key] = torch.cos(torch.linspace(torch.pi, torch.pi / 2, n, device=device, dtype=dtype)) ** 2
        return self.taper2[key]

_WHITEN_CACHE = _WhiteningCache()

def spectral_whitening(rfftdata: torch.Tensor, df: float, window_freq: float, f1: float, f2: float) -> torch.Tensor:
    if rfftdata.ndim < 2:
        raise ValueError("spectral_whitening: rfftdata must be at least 2D.")
    device = rfftdata.device
    dtype_amp = rfftdata.real.dtype
    nfreq = rfftdata.shape[-1]
    idxf1 = max(0, min(int(f1 / df), nfreq))
    idxf2 = max(0, min(int(math.ceil(f2 / df)), nfreq))
    if idxf2 <= idxf1: return torch.zeros_like(rfftdata)

    if window_freq == 0.0:
        amp = torch.abs(rfftdata).clamp_(min=1e-10)
        rfft_out = rfftdata / amp
    else:
        nwin = max(int(window_freq / df), 1); nwin = nwin + 1 if nwin % 2 == 0 else nwin
        amp = torch.abs(rfftdata)
        kernel = _WHITEN_CACHE.get_kernel(device, dtype_amp, nwin)
        amp_smooth = torch.nn.functional.conv1d(amp.unsqueeze(1), kernel, padding=nwin//2).squeeze(1)
        if amp_smooth.shape[-1] > amp.shape[-1]: amp_smooth = amp_smooth[..., :amp.shape[-1]]
        rfft_out = rfftdata / amp_smooth.clamp_(min=1e-10)

    t1 = _WHITEN_CACHE.get_taper1(device, dtype_amp, idxf1)
    if t1 is not None: rfft_out[..., :idxf1] *= t1
    t2 = _WHITEN_CACHE.get_taper2(device, dtype_amp, nfreq, idxf2)
    if t2 is not None: rfft_out[..., idxf2:] *= t2
    return rfft_out


@torch.no_grad()
def whiten_per_segment_torch(
    x: torch.Tensor,
    *,
    fs_proc: float,
    npts_seg: int,
    window_freq_hz: float,
    f1: float,
    f2: float,
) -> torch.Tensor:
    """
    Apply spectral whitening independently to each segment of a multichannel signal.

    Reshapes (nch, npts_new) → (nch*nseg, npts_seg), whitens in the frequency domain,
    then reshapes back to (nch, npts_new).

    :param x: Input tensor, shape (nch, npts_new). Must be float32.
    :param fs_proc: Sampling rate after decimation (Hz).
    :param npts_seg: Samples per segment (must divide npts_new evenly).
    :param window_freq_hz: Smoothing window width for amplitude spectrum (Hz). 0 = full whitening.
    :param f1: Low frequency bound for whitening passband (Hz).
    :param f2: High frequency bound for whitening passband (Hz).
    :return: Whitened tensor, same shape as input (nch, npts_new), float32.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("whiten_per_segment_torch: x must be a torch.Tensor")
    if x.ndim != 2:
        raise ValueError(f"whiten_per_segment_torch: expected 2D (nch, nt); got {tuple(x.shape)}")

    nch, npts_new = int(x.shape[0]), int(x.shape[1])
    if npts_seg <= 0:
        raise ValueError("whiten_per_segment_torch: npts_seg must be > 0")
    if npts_new % npts_seg != 0:
        raise ValueError("whiten_per_segment_torch: npts_new must be divisible by npts_seg")

    nseg = npts_new // npts_seg
    B = nch * nseg
    df = float(fs_proc) / float(npts_seg)

    x = x.to(dtype=torch.float32)
    x2 = x.reshape(B, npts_seg)
    X = torch.fft.rfft(x2, n=npts_seg, dim=-1)
    Xw = spectral_whitening(X, df, float(window_freq_hz), float(f1), float(f2))
    xw = torch.fft.irfft(Xw, n=npts_seg, dim=-1).to(torch.float32)
    return xw.reshape(nch, npts_new)

# ==============================================================
# 3. Full preprocessing pipeline
# ==============================================================
def preprocess(x: np.ndarray | torch.Tensor, fs_raw: float, f1: float, f2: float, 
               decimation: int, diff: bool, ram_win: float) -> np.ndarray | torch.Tensor:
    is_tensor = isinstance(x, torch.Tensor)
    orig_device = x.device if is_tensor else None
    x_np = convert_to_numpy(x) if is_tensor else np.asarray(x)

    if diff: x_np = np.gradient(x_np, axis=-1) * float(fs_raw)
    x_np = detrend(x_np, axis=-1)
    x_np = bandpass_filter_tukey(x_np, fs_raw, f1, f2)
    if decimation > 1: x_np = x_np[:, ::int(decimation)]

    fs_proc = float(fs_raw) / float(decimation)
    x_np -= np.median(x_np, axis=0)
    x_np = temporal_normalization(x_np, fs_proc, float(ram_win))
    
    return convert_to_tensor(x_np, device=orig_device) if is_tensor else x_np.astype(np.float32)

# ==============================================================
# 4. Zhang (2026) helper: choose block size
# ==============================================================
def choose_block_size_v2(M: int, *, fft_snap_pow2: bool = True, 
                        fallback: Literal["v1_2M", "v1_Mp1"] = "v1_2M") -> tuple[int, int]:
    K_star = None
    if lambertw is not None:
        z = -1.0 / (4.0 * math.e * float(M))
        try:
            w = lambertw(z, k=-1)
            K_star = 2.0 * float(M) * (-w.real - 1.0)
        except Exception: pass
    
    K = int(max(1, round(K_star))) if K_star is not None else (int(2*M) if fallback=="v1_2M" else M+1)
    L = int(K + 2 * M)
    if fft_snap_pow2: L = int(nextpow2(L)); K = L - 2*M
    if K < M + 1: K = M + 1; L = int(K + 2*M); L = int(nextpow2(L)) if fft_snap_pow2 else L; K = L - 2*M
    return K, L

# ==============================================================
# 5. Cross-correlation module
# ==============================================================
class TorchCrossCorrelation(nn.Module):
    def __init__(self, *, mode: str = "conventional", max_lag_samples: Optional[int] = None,
                 is_spectral_whitening: bool = False, whitening_params: Optional[tuple] = None,
                 v1_fft_snap_pow2: bool = True, v1_fallback: str = "v1_2M") -> None:
        super().__init__()
        self.mode = mode.lower()
        self.max_lag_samples = max_lag_samples
        self.is_spectral_whitening = is_spectral_whitening
        self.whitening_params = whitening_params
        self.v1_fft_snap_pow2 = v1_fft_snap_pow2
        self.v1_fallback = v1_fallback

        self._conv_L = self._conv_N = self._conv_df = None
        self._v1_M = self._v1_K = self._v1_Lfft = self._v1_nfreq = None
        
        # Legacy buffers (needed for A-B testing/benchmarking)
        self._v1_buf_key = self._v1_x_t = self._v1_y_t = None
        self._rspec_key = self._rspec = None

        if self.mode == "v1":
            self._v1_M = int(max_lag_samples)
            self._v1_K, self._v1_Lfft = choose_block_size_v2(self._v1_M, fft_snap_pow2=v1_fft_snap_pow2, fallback=v1_fallback)
            self._v1_nfreq = self._v1_Lfft // 2 + 1

    def _forward_conventional(self, data1: torch.Tensor, data2: torch.Tensor) -> torch.Tensor:
        N = data1.shape[1]
        if self._conv_L is None or self._conv_N != N:
            self._conv_L = int(nextpow2(2 * N - 1)); self._conv_N = N
            self._conv_df = self.whitening_params[0] / self._conv_L if self.is_spectral_whitening else None
        
        X = torch.fft.rfft(data1, n=self._conv_L, dim=-1)
        Y = torch.fft.rfft(data2, n=self._conv_L, dim=-1)
        if self.is_spectral_whitening:
            X = spectral_whitening(X, self._conv_df, *self.whitening_params[1:])
            Y = spectral_whitening(Y, self._conv_df, *self.whitening_params[1:])
        
        r = torch.fft.irfft(torch.conj(X) * Y, n=self._conv_L, dim=-1)
        return torch.cat([r[:, self._conv_L-(N-1):], r[:, :N]], dim=-1).to(torch.float32)

    def _v1_get_buffers(self, B, Lfft, dtype, device):
        key = (B, Lfft, dtype, device)
        if self._v1_buf_key != key:
            self._v1_buf_key = key
            self._v1_x_t = torch.zeros((B, Lfft), device=device, dtype=dtype)
            self._v1_y_t = torch.zeros((B, Lfft), device=device, dtype=dtype)
        return self._v1_x_t, self._v1_y_t

    def _v1_get_rspec(self, B, nfreq, device):
        key = (B, nfreq, device)
        if self._rspec_key != key:
            self._rspec_key = key
            self._rspec = torch.zeros((B, nfreq), dtype=torch.complex64, device=device)
        else: self._rspec.zero_()
        return self._rspec

    @torch.no_grad()
    def _forward_v1_batched(self, signal_1: torch.Tensor, signal_2: torch.Tensor) -> torch.Tensor:
        if signal_1.ndim != 2 or signal_2.ndim != 2:
            raise ValueError("v1: inputs must be 2D (B × N).")
        if signal_1.shape != signal_2.shape:
            raise ValueError("v1: shape mismatch")
        if signal_1.device != signal_2.device:
            raise ValueError("v1: device mismatch")
        if signal_1.dtype != signal_2.dtype:
            raise ValueError("v1: dtype mismatch")
        if signal_1.is_complex() or signal_2.is_complex():
            raise ValueError("v1: expects real tensors")
        if self._v1_M is None or self._v1_K is None or self._v1_Lfft is None or self._v1_nfreq is None:
            raise RuntimeError("v1: module not initialised (M/K/Lfft/nfreq is None).")

        M, K, Lfft = self._v1_M, self._v1_K, self._v1_Lfft
        B, N = signal_1.shape[0], signal_1.shape[1]
        nblocks = (N + K - 1) // K
        
        x_blocked = torch.zeros((B, nblocks, Lfft), dtype=signal_1.dtype, device=signal_1.device)
        y_blocked = torch.zeros((B, nblocks, Lfft), dtype=signal_1.dtype, device=signal_1.device)

        for blk in range(nblocks):
            start = blk * K
            end = min(start + K, N)
            klen = end - start
            y_blocked[:, blk, M : M + klen] = signal_2[:, start : end]
            x0 = start - M; ix0 = max(0, x0); ix1 = min(N, start + K + M)
            if ix1 > ix0:
                x_blocked[:, blk, (ix0 - x0) : (ix0 - x0) + (ix1 - ix0)] = signal_1[:, ix0 : ix1]

        X = torch.fft.rfft(x_blocked, n=Lfft, dim=-1)
        Y = torch.fft.rfft(y_blocked, n=Lfft, dim=-1)
        del x_blocked, y_blocked

        if self.is_spectral_whitening:
            df = self.whitening_params[0] / Lfft
            Bnb, nfreq = B * nblocks, self._v1_nfreq
            X = spectral_whitening(X.reshape(Bnb, nfreq), df, *self.whitening_params[1:]).reshape(B, nblocks, nfreq)
            Y = spectral_whitening(Y.reshape(Bnb, nfreq), df, *self.whitening_params[1:]).reshape(B, nblocks, nfreq)

        Rspec = (X.conj() * Y).sum(dim=1)
        r = torch.fft.irfft(Rspec, n=Lfft, dim=-1)
        return torch.cat([r[:, Lfft - M : Lfft], r[:, 0 : M + 1]], dim=-1).to(torch.float32)

    @torch.no_grad()
    def _forward_v1_python_fft(self, signal_1, signal_2):
        M, K, Lfft, nfreq = self._v1_M, self._v1_K, self._v1_Lfft, self._v1_nfreq
        B, N = signal_1.shape[0], signal_1.shape[1]
        nblocks = (N + K - 1) // K
        Rspec = self._v1_get_rspec(B, nfreq, signal_1.device)
        x_t, y_t = self._v1_get_buffers(B, Lfft, signal_1.dtype, signal_1.device)
        
        df = self.whitening_params[0] / Lfft if self.is_spectral_whitening else 0
        for l in range(nblocks):
            start = l * K; end = min(start + K, N); klen = end - start
            x_t.zero_(); y_t.zero_()
            y_t.narrow(1, M, klen).copy_(signal_2.narrow(1, start, klen))
            ix0 = max(0, start - M); ix1 = min(N, start + K + M)
            if ix1 > ix0:
                x_t.narrow(1, ix0-(start-M), ix1-ix0).copy_(signal_1.narrow(1, ix0, ix1-ix0))
            X, Y = torch.fft.rfft(x_t, n=Lfft, dim=-1), torch.fft.rfft(y_t, n=Lfft, dim=-1)
            if self.is_spectral_whitening:
                X = spectral_whitening(X, df, *self.whitening_params[1:])
                Y = spectral_whitening(Y, df, *self.whitening_params[1:])
            Rspec.addcmul_(X.conj(), Y)
        r = torch.fft.irfft(Rspec, n=Lfft, dim=-1)
        return torch.cat([r[:, Lfft - M : Lfft], r[:, 0 : M + 1]], dim=-1).to(torch.float32)

    def forward(self, data1: torch.Tensor, data2: torch.Tensor) -> torch.Tensor:
        if self.mode == "conventional": return self._forward_conventional(data1, data2)
        return self._forward_v1_batched(data1, data2)