"""
:module: src/ani.py
:auth: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: DAS preprocessing (Bensen et al., 2007) + cross-correlation (conventional + Zhang 2025 v1).
:reference: Modified from Yan Yang (2022-07-10).
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Optional, Literal

import numpy as np
import scipy.signal as signal
import torch
from scipy.signal import butter, convolve, detrend, filtfilt
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
    f1: float,
    f2: float,
    alpha: float = 0.05,
    order: int = 4,
) -> np.ndarray:
    if data.ndim != 2:
        raise ValueError("bandpass_filter_tukey: data must be 2D (nch × nt).")

    nyq = fs / 2.0
    if not (0.0 < f1 < f2 < nyq):
        raise ValueError(f"Invalid f1/f2: require 0 < f1 < f2 < Nyquist={nyq}.")

    nt = int(data.shape[1])
    window = signal.windows.tukey(nt, alpha=float(alpha))

    low = f1 / nyq
    high = f2 / nyq
    b, a = butter(int(order), [low, high], btype="bandpass")

    tapered = data * window
    filtered = filtfilt(b, a, tapered, axis=1)

    return filtered.astype(np.float32, copy=False)


def running_absolute_mean(trace: np.ndarray, nwin: int) -> np.ndarray:
    if trace.ndim != 1:
        raise ValueError("running_absolute_mean: 'trace' must be 1D.")
    if nwin <= 1:
        return trace.copy()

    npts = int(trace.size)
    abs_trace = np.abs(trace)

    padded = np.empty(npts + 2 * nwin, dtype=trace.dtype)
    padded[nwin:-nwin] = abs_trace
    padded[:nwin] = abs_trace[0]
    padded[-nwin:] = abs_trace[-1]

    kernel = np.ones(int(nwin), dtype=np.float64) / float(nwin)
    ram = convolve(padded, kernel, mode="same")[nwin:-nwin]
    ram = np.where(ram == 0, np.nan, ram)

    return np.nan_to_num(trace / ram, nan=0.0)


def temporal_normalization(data: np.ndarray, fs: float, window_time: float) -> np.ndarray:
    if data.ndim != 2:
        raise ValueError("temporal_normalization: data must be 2D (nch × nt).")

    if float(window_time) == 0.0:
        return np.sign(data).astype(np.float32, copy=False)

    nwin = int(round(fs * float(window_time)))
    nwin = max(nwin, 1)

    out = data.copy()
    for i in range(out.shape[0]):
        out[i, :] = running_absolute_mean(out[i, :], nwin)

    return out.astype(np.float32, copy=False)


# ==============================================================
# 2. Spectral whitening (torch) - cached kernels/tapers
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
        k = self.kernel.get(key)
        if k is None:
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
    if not isinstance(rfftdata, torch.Tensor):
        raise TypeError("spectral_whitening: rfftdata must be torch.Tensor.")
    if not rfftdata.is_complex():
        raise ValueError("spectral_whitening: rfftdata must be complex.")

    device = rfftdata.device
    dtype_amp = rfftdata.real.dtype
    nfreq = int(rfftdata.shape[1])

    if df <= 0:
        idxf1, idxf2 = 0, nfreq
    else:
        idxf1 = int(f1 / df)
        idxf2 = int(math.ceil(f2 / df))

    idxf1 = max(0, min(idxf1, nfreq))
    idxf2 = max(0, min(idxf2, nfreq))
    if idxf2 <= idxf1:
        return torch.zeros_like(rfftdata)

    phase = torch.angle(rfftdata)

    df = float(df)
    window_freq = float(window_freq)

    if window_freq == 0.0:
        rfft_out = torch.exp(1j * phase)
    else:
        nwin = max(int(window_freq / df), 1) if df > 0 else 1
        if nwin % 2 == 0:
            nwin += 1

        amp = torch.abs(rfftdata)
        amp_3d = amp.unsqueeze(1)

        kernel = _WHITEN_CACHE.get_kernel(device, dtype_amp, nwin)
        pad = nwin // 2
        amp_smooth = torch.nn.functional.conv1d(amp_3d, kernel, padding=pad).squeeze(1)

        if amp_smooth.shape[-1] != amp.shape[-1]:
            if amp_smooth.shape[-1] > amp.shape[-1]:
                amp_smooth = amp_smooth[..., : amp.shape[-1]]
            else:
                amp_smooth = torch.nn.functional.pad(
                    amp_smooth, (0, amp.shape[-1] - amp_smooth.shape[-1]), mode="replicate"
                )

        amp_smooth = torch.where(amp_smooth == 0, torch.ones_like(amp_smooth), amp_smooth)
        rfft_out = torch.exp(1j * phase) * (amp / amp_smooth)

    t1 = _WHITEN_CACHE.get_taper1(device, dtype_amp, idxf1)
    if t1 is not None and 0 < idxf1 < nfreq:
        rfft_out[:, :idxf1] *= t1

    t2 = _WHITEN_CACHE.get_taper2(device, dtype_amp, nfreq, idxf2)
    if t2 is not None and 0 <= idxf2 < nfreq:
        rfft_out[:, idxf2:] *= t2

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
def preprocess(
    x: np.ndarray | torch.Tensor,
    fs_raw: float,
    f1: float,
    f2: float,
    decimation: int,
    diff: bool,
    ram_win: float,
) -> np.ndarray | torch.Tensor:
    if decimation < 1:
        raise ValueError("preprocess: decimation must be >= 1.")

    is_tensor = isinstance(x, torch.Tensor)
    orig_device: Optional[torch.device] = x.device if is_tensor else None

    x_np = convert_to_numpy(x) if is_tensor else np.asarray(x)
    if x_np.ndim != 2:
        raise ValueError(f"preprocess: expected 2D (nch × nt); got shape={x_np.shape}")

    if diff:
        x_np = np.gradient(x_np, axis=-1) * float(fs_raw)

    x_np = detrend(x_np, axis=-1)
    x_np = bandpass_filter_tukey(x_np, fs_raw, f1, f2)

    if decimation > 1:
        x_np = x_np[:, :: int(decimation)]

    fs_proc = float(fs_raw) / float(decimation)

    x_np -= np.median(x_np, axis=0)
    x_np = temporal_normalization(x_np, fs_proc, float(ram_win))
    x_np = x_np.astype(np.float32, copy=False)

    if is_tensor:
        assert orig_device is not None
        return convert_to_tensor(x_np, device=orig_device)

    return x_np


# ==============================================================
# 4. Zhang (2025) helper: choose block size
# ==============================================================
def choose_block_size_v2(
    M: int,
    *,
    fft_snap_pow2: bool = True,
    fallback: Literal["v1_2M", "v1_Mp1"] = "v1_2M",
) -> tuple[int, int]:
    if M <= 0:
        raise ValueError("choose_block_size_v2: M must be > 0")

    K_star: Optional[float] = None
    if lambertw is not None:
        z = -1.0 / (4.0 * math.e * float(M))
        try:
            w = lambertw(z, k=-1)
            w_real = float(w.real)
            K_star = 2.0 * float(M) * (-w_real - 1.0)
            if not math.isfinite(K_star) or K_star <= 0.0:
                K_star = None
        except Exception:
            K_star = None

    if K_star is None:
        K = int(2 * M) if fallback == "v1_2M" else int(M + 1)
    else:
        K = int(max(1, round(K_star)))

    L = int(K + 2 * M)
    if fft_snap_pow2:
        L = int(nextpow2(L))
        K = int(L - 2 * M)

    if K < M + 1:
        K = M + 1
        L = int(K + 2 * M)
        if fft_snap_pow2:
            L = int(nextpow2(L))
            K = int(L - 2 * M)

    return K, L


# ==============================================================
# 5. Cross-correlation module
# ==============================================================
class TorchCrossCorrelation(nn.Module):
    """
    mode="conventional": full-lag (2N-1), standard lag ordering
    mode="v1": short-lag (2M+1) using Zhang block-FFT
    """

    def __init__(
        self,
        *,
        mode: str = "conventional",
        max_lag_samples: Optional[int] = None,
        is_spectral_whitening: bool = False,
        whitening_params: Optional[tuple[float, float, float, float]] = None,
        v1_fft_snap_pow2: bool = True,
        v1_fallback: Literal["v1_2M", "v1_Mp1"] = "v1_2M",
    ) -> None:
        super().__init__()

        self.mode = str(mode).lower()
        if self.mode not in {"conventional", "v1"}:
            raise ValueError(f"Unknown mode={self.mode}. Use 'conventional' or 'v1'.")

        self.max_lag_samples = int(max_lag_samples) if max_lag_samples is not None else None
        if self.mode == "v1" and self.max_lag_samples is None:
            raise ValueError("TorchCrossCorrelation(mode='v1'): max_lag_samples is required.")

        self.is_spectral_whitening = bool(is_spectral_whitening)
        self.whitening_params = whitening_params
        if self.is_spectral_whitening and self.whitening_params is None:
            raise ValueError("TorchCrossCorrelation: whitening_params required when whitening enabled.")

        self.v1_fft_snap_pow2 = bool(v1_fft_snap_pow2)
        self.v1_fallback = v1_fallback

        # conventional cache
        self._conv_L: Optional[int] = None
        self._conv_N: Optional[int] = None
        self._conv_df: Optional[float] = None

        # v1 cache (params + buffers)
        self._v1_M: Optional[int] = None
        self._v1_K: Optional[int] = None
        self._v1_Lfft: Optional[int] = None
        self._v1_nfreq: Optional[int] = None

        self._v1_buf_key: Optional[tuple[int, int, torch.dtype, torch.device]] = None
        self._v1_x_t: Optional[torch.Tensor] = None
        self._v1_y_t: Optional[torch.Tensor] = None

        self._rspec_key: Optional[tuple[int, int, torch.device]] = None
        self._rspec: Optional[torch.Tensor] = None

        if self.mode == "v1":
            self._v1_M = int(self.max_lag_samples)
            self._v1_K, self._v1_Lfft = choose_block_size_v2(
                self._v1_M, fft_snap_pow2=self.v1_fft_snap_pow2, fallback=self.v1_fallback
            )
            self._v1_nfreq = int(self._v1_Lfft // 2 + 1)

        logger.info(
            "TorchCrossCorrelation init | mode=%s | max_lag=%s | whitening=%s | use_cpp=%s",
            self.mode,
            self.max_lag_samples,
            self.is_spectral_whitening,
        )

    def _forward_conventional(self, data1: torch.Tensor, data2: torch.Tensor) -> torch.Tensor:
        if data1.ndim != 2 or data2.ndim != 2:
            raise ValueError("conventional: inputs must be 2D (B × N).")
        if data1.shape != data2.shape:
            raise ValueError("conventional: shape mismatch")
        if data1.device != data2.device:
            raise ValueError("conventional: device mismatch")

        N = int(data1.shape[1])

        if self._conv_L is None or self._conv_N != N:
            L_lin = 2 * N - 1
            L = int(nextpow2(L_lin))
            self._conv_L = L
            self._conv_N = N
            if self.is_spectral_whitening:
                assert self.whitening_params is not None
                fs = float(self.whitening_params[0])
                self._conv_df = fs / float(L)
            else:
                self._conv_df = None

        L = int(self._conv_L)
        X = torch.fft.rfft(data1, n=L, dim=-1)
        Y = torch.fft.rfft(data2, n=L, dim=-1)

        if self.is_spectral_whitening:
            assert self.whitening_params is not None
            _, window_freq, f1, f2 = self.whitening_params
            df = float(self._conv_df) if self._conv_df is not None else float(self.whitening_params[0]) / float(L)
            X = spectral_whitening(X, df, float(window_freq), float(f1), float(f2))
            Y = spectral_whitening(Y, df, float(window_freq), float(f1), float(f2))

        r = torch.fft.irfft(torch.conj(X) * Y, n=L, dim=-1)
        cc_full = torch.cat([r[:, L - (N - 1) : L], r[:, :N]], dim=-1)
        return cc_full.to(dtype=torch.float32)

    def _v1_get_buffers(
        self,
        B: int,
        Lfft: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (int(B), int(Lfft), dtype, device)
        if self._v1_buf_key != key or self._v1_x_t is None or self._v1_y_t is None:
            self._v1_buf_key = key
            self._v1_x_t = torch.zeros((B, Lfft), device=device, dtype=dtype)
            self._v1_y_t = torch.zeros((B, Lfft), device=device, dtype=dtype)
        return self._v1_x_t, self._v1_y_t

    def _v1_get_rspec(self, B: int, nfreq: int, device: torch.device) -> torch.Tensor:
        key = (int(B), int(nfreq), device)
        if self._rspec_key != key or self._rspec is None:
            self._rspec_key = key
            self._rspec = torch.zeros((B, nfreq), dtype=torch.complex64, device=device)
        else:
            self._rspec.zero_()
        return self._rspec

    @torch.no_grad()
    def _forward_v1_python_fft(self, signal_1: torch.Tensor, signal_2: torch.Tensor) -> torch.Tensor:
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

        assert self._v1_M is not None and self._v1_K is not None and self._v1_Lfft is not None and self._v1_nfreq is not None

        M = int(self._v1_M)
        K = int(self._v1_K)
        Lfft = int(self._v1_Lfft)

        B, N = int(signal_1.shape[0]), int(signal_1.shape[1])
        nblocks = (N + K - 1) // K
        nfreq = int(self._v1_nfreq)

        Rspec = self._v1_get_rspec(B, nfreq, signal_1.device)
        x_t, y_t = self._v1_get_buffers(B, Lfft, signal_1.dtype, signal_1.device)

        if self.is_spectral_whitening:
            assert self.whitening_params is not None
            fs, window_freq, f1, f2 = self.whitening_params
            df = float(fs) / float(Lfft)
            window_freq = float(window_freq)
            f1 = float(f1)
            f2 = float(f2)

        rfft = torch.fft.rfft
        irfft = torch.fft.irfft

        for l in range(nblocks):
            start = l * K
            end = min(start + K, N)
            klen = end - start

            # Full zero is simple and reliable for Pure Python
            x_t.zero_()
            y_t.zero_()

            # Narrow+copy_ is usually faster than advanced indexing
            y_dst = y_t.narrow(1, M, klen)
            y_src = signal_2.narrow(1, start, klen)
            y_dst.copy_(y_src)

            x0 = start - M
            x1 = start + K + M
            ix0 = max(0, x0)
            ix1 = min(N, x1)
            if ix1 > ix0:
                length = ix1 - ix0
                dst0 = ix0 - x0
                x_dst = x_t.narrow(1, dst0, length)
                x_src = signal_1.narrow(1, ix0, length)
                x_dst.copy_(x_src)

            X = rfft(x_t, n=Lfft, dim=-1)
            Y = rfft(y_t, n=Lfft, dim=-1)

            if self.is_spectral_whitening:
                X = spectral_whitening(X, df, window_freq, f1, f2)
                Y = spectral_whitening(Y, df, window_freq, f1, f2)

            Rspec.addcmul_(X.conj(), Y)

        r = irfft(Rspec, n=Lfft, dim=-1)
        out = torch.cat([r[:, Lfft - M : Lfft], r[:, 0 : M + 1]], dim=-1)
        return out.to(dtype=torch.float32)

    def forward(self, data1: torch.Tensor, data2: torch.Tensor) -> torch.Tensor:
        if self.mode == "conventional":
            return self._forward_conventional(data1, data2)
            
        return self._forward_v1_python_fft(data1, data2)