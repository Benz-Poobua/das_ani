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
import warnings
from typing import Any, Optional, Literal

import numpy as np
import scipy.signal as signal
import torch
from scipy.signal import decimate, detrend
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
    """
    Vectorized temporal normalization using a running absolute mean (RAM).
    """
    if data.ndim != 2:
        raise ValueError("temporal_normalization: data must be 2D (nch × nt).")
    if float(window_time) == 0.0:
        return np.sign(data).astype(np.float32, copy=False)

    nwin = max(int(round(fs * float(window_time))), 1)
    ram  = uniform_filter1d(np.abs(data), size=nwin, axis=1, mode='nearest')

    out = np.zeros(data.shape, dtype=np.float32)
    np.divide(data, ram, out=out, where=ram > 0)
    return out

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
        nwin = max(int(window_freq / df), 1)
        if nwin % 2 == 0:
            nwin += 1
        amp = torch.abs(rfftdata)

        orig_shape = amp.shape
        amp_flat = amp.reshape(-1, 1, nfreq)

        kernel = _WHITEN_CACHE.get_kernel(device, dtype_amp, nwin)
        amp_smooth = torch.nn.functional.conv1d(amp_flat, kernel, padding=nwin // 2).squeeze(1)

        if amp_smooth.shape[-1] > nfreq:
            amp_smooth = amp_smooth[..., :nfreq]

        rfft_out = rfftdata / amp_smooth.reshape(orig_shape).clamp_(min=1e-10)

    t1 = _WHITEN_CACHE.get_taper1(device, dtype_amp, idxf1)
    if t1 is not None: rfft_out[..., :idxf1] *= t1
    t2 = _WHITEN_CACHE.get_taper2(device, dtype_amp, nfreq, idxf2)
    if t2 is not None: rfft_out[..., idxf2:] *= t2
    return rfft_out


@torch.inference_mode()
def whiten_per_segment_torch(
    x: torch.Tensor,
    *,
    fs_proc: float,
    npts_seg: int,
    window_freq_hz: float,
    f1: float,
    f2: float,
    chunk_nch: int = 64,
) -> torch.Tensor:
    """
    Apply spectral whitening independently to each segment of a multichannel signal.
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
    if chunk_nch < 1:
        raise ValueError("whiten_per_segment_torch: chunk_nch must be >= 1")

    nseg = npts_new // npts_seg
    df   = float(fs_proc) / float(npts_seg)
    x    = x.to(dtype=torch.float32)
    out  = torch.empty_like(x)

    for ch0 in range(0, nch, chunk_nch):
        ch1 = min(ch0 + chunk_nch, nch)
        B   = (ch1 - ch0) * nseg

        x2  = x[ch0:ch1, :].reshape(B, npts_seg)
        X   = torch.fft.rfft(x2, n=npts_seg, dim=-1)
        Xw  = spectral_whitening(X, df, float(window_freq_hz), float(f1), float(f2))
        xw  = torch.fft.irfft(Xw, n=npts_seg, dim=-1).to(torch.float32)

        out[ch0:ch1, :] = xw.reshape(ch1 - ch0, npts_new)

    return out

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
    """Full DAS preprocessing pipeline."""
    if decimation < 1:
        raise ValueError(f"preprocess: decimation must be >= 1, got {decimation}.")

    is_tensor  = isinstance(x, torch.Tensor)
    orig_device = x.device if is_tensor else None
    x_np = convert_to_numpy(x) if is_tensor else np.asarray(x)

    if x_np.ndim != 2:
        raise ValueError(f"preprocess: expected 2D (nch × nt); got shape {x_np.shape}.")

    if diff:
        x_np = np.diff(x_np, prepend=x_np[:, :1], axis=-1) * float(fs_raw)

    x_np = detrend(x_np, axis=-1)
    x_np = bandpass_filter_tukey(x_np, fs_raw, f1, f2)

    if decimation > 1:
        nyq_decimated = float(fs_raw) / (2.0 * float(decimation))
        if float(f2) >= nyq_decimated:
            raise ValueError(
                f"preprocess: f2={f2} Hz >= decimated Nyquist ({nyq_decimated:.2f} Hz) "
                f"at fs_raw={fs_raw} Hz, decimation={decimation}. "
                "Reduce decimation or lower f2 to avoid aliasing."
            )
        if float(f2) > 0.8 * nyq_decimated:
            warnings.warn(
                f"preprocess: f2={f2} Hz is within 20% of the decimated Nyquist. "
                "The anti-alias filter transition band may attenuate signal near f2.",
                UserWarning, stacklevel=2,
            )
        x_np = decimate(x_np, q=int(decimation), axis=-1, zero_phase=True)

    fs_proc  = float(fs_raw) / float(decimation)
    x_np    -= np.median(x_np, axis=0)
    x_np     = temporal_normalization(x_np, fs_proc, float(ram_win))

    return convert_to_tensor(x_np, device=orig_device) if is_tensor else x_np.astype(np.float32)

# ==============================================================
# 4. Zhang (2026) helper: choose block size
# ==============================================================
def choose_block_size_v2(
    M: int, *, fft_snap_pow2: bool = True, fallback: Literal["v1_2M", "v1_Mp1"] = "v1_2M",
) -> tuple[int, int]:
    if M <= 0:
        raise ValueError(f"choose_block_size_v2: M must be > 0, got {M}.")

    K_star = None
    if lambertw is not None:
        z = -1.0 / (4.0 * math.e * float(M))
        try:
            w = lambertw(z, k=-1)
            K_star = 2.0 * float(M) * (-w.real - 1.0)
            if not math.isfinite(K_star) or K_star <= 0.0:
                K_star = None
        except Exception:
            pass

    if K_star is not None:
        K = int(max(1, round(K_star)))
    else:
        K = int(2 * M) if fallback == "v1_2M" else int(M + 1)

    L = int(K + 2 * M)
    if fft_snap_pow2:
        L = int(nextpow2(L))
        K = L - 2 * M
    if K < M + 1:
        K = M + 1
        L = int(K + 2 * M)
        if fft_snap_pow2:
            L = int(nextpow2(L))
        K = L - 2 * M
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
        
        # Buffer caches
        self._v1_x_blocked_key = self._v1_x_blocked = None
        self._v1_y_blocked_key = self._v1_y_blocked = None

        if self.mode == "v1":
            self._v1_M = int(max_lag_samples)
            self._v1_K, self._v1_Lfft = choose_block_size_v2(self._v1_M, fft_snap_pow2=v1_fft_snap_pow2, fallback=v1_fallback)
            self._v1_nfreq = self._v1_Lfft // 2 + 1

    @torch.inference_mode()
    def _conv_encode(self, data: torch.Tensor) -> torch.Tensor:
        B, nseg, N = data.shape
        if self._conv_L is None or self._conv_N != N:
            self._conv_L  = int(nextpow2(2 * N - 1))
            self._conv_N  = N
            self._conv_df = self.whitening_params[0] / self._conv_L if self.is_spectral_whitening else None

        X = torch.fft.rfft(data, n=self._conv_L, dim=-1)
        if self.is_spectral_whitening:
            X = spectral_whitening(X, self._conv_df, *self.whitening_params[1:])
        return X

    @torch.inference_mode()
    def _conv_correlate(self, X: torch.Tensor, Y: torch.Tensor, N: int) -> torch.Tensor:
        Rspec = (X.conj() * Y).sum(dim=1)
        conv_L = self._conv_L 
        r = torch.fft.irfft(Rspec, n=conv_L, dim=-1)
        return torch.cat([r[:, conv_L - (N - 1):], r[:, :N]], dim=-1).to(torch.float32)

    def _v1_get_blocked(self, Bn, nblocks, Lfft, dtype, device, is_source: bool):
        key = (Bn, nblocks, Lfft, dtype, device)
        if is_source:
            if getattr(self, "_v1_x_blocked_key", None) != key:
                self._v1_x_blocked_key = key
                self._v1_x_blocked = torch.empty((Bn, nblocks, Lfft), dtype=dtype, device=device)
            self._v1_x_blocked.zero_()
            return self._v1_x_blocked
        else:
            if getattr(self, "_v1_y_blocked_key", None) != key:
                self._v1_y_blocked_key = key
                self._v1_y_blocked = torch.empty((Bn, nblocks, Lfft), dtype=dtype, device=device)
            self._v1_y_blocked.zero_()
            return self._v1_y_blocked

    @torch.inference_mode()
    def _v1_encode(self, data: torch.Tensor, is_source: bool) -> torch.Tensor:
        B, nseg, N = data.shape
        Bn = B * nseg
        M, K, Lfft, nfreq = self._v1_M, self._v1_K, self._v1_Lfft, self._v1_nfreq
        nblocks = (N + K - 1) // K
        
        # Optimization 2: Cache blocked arrays
        blocked = self._v1_get_blocked(Bn, nblocks, Lfft, data.dtype, data.device, is_source)
        s = data.reshape(Bn, N)
        
        if is_source:
            for blk in range(nblocks):
                start = blk * K
                x0 = start - M
                ix0 = max(0, x0)
                ix1 = min(N, start + K + M)
                if ix1 > ix0:
                    dst0 = ix0 - x0
                    length = ix1 - ix0
                    blocked[:, blk, dst0 : dst0 + length] = s[:, ix0 : ix1]
        else:
            for blk in range(nblocks):
                start = blk * K
                end = min(start + K, N)
                klen = end - start
                blocked[:, blk, M : M + klen] = s[:, start : end]
                
        Spec = torch.fft.rfft(blocked, n=Lfft, dim=-1)
        if self.is_spectral_whitening:
            df = float(self.whitening_params[0]) / float(Lfft)
            Spec = spectral_whitening(Spec, df, *self.whitening_params[1:])
            
        # Returns (B, nseg, nblocks, nfreq) so DataParallel can seamlessly scatter batch dims
        return Spec.reshape(B, nseg, nblocks, nfreq)

    @torch.inference_mode()
    def _v1_correlate(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        B, nseg, nblocks, nfreq = X.shape
        X = X.reshape(B * nseg, nblocks, nfreq)
        Y = Y.reshape(B * nseg, nblocks, nfreq)
        
        Rspec = (X.conj() * Y).sum(dim=1)
        Rspec = Rspec.reshape(B, nseg, nfreq).sum(dim=1)
        
        r = torch.fft.irfft(Rspec, n=self._v1_Lfft, dim=-1)
        M = self._v1_M
        Lfft = self._v1_Lfft
        return torch.cat([r[:, Lfft - M : Lfft], r[:, 0 : M + 1]], dim=-1).to(torch.float32)

    @torch.inference_mode()
    def encode_source(self, data: torch.Tensor) -> torch.Tensor:
        """Optimization 1: Encode the virtual source exactly once before the receiver chunk loop."""
        if self.mode == "conventional":
            return self._conv_encode(data)
        return self._v1_encode(data, is_source=True)

    def forward(self, data1: torch.Tensor, data2: torch.Tensor, is_source_spectrum: bool = False) -> torch.Tensor:
        """
        Asymmetric forward pass. 
        If is_source_spectrum=True, data1 is already encoded and expanded to match data2 batch length.
        """
        if self.mode == "conventional":
            N = data2.shape[-1]
            X = data1 if is_source_spectrum else self._conv_encode(data1)
            Y = self._conv_encode(data2)
            return self._conv_correlate(X, Y, N)
        else:
            X = data1 if is_source_spectrum else self._v1_encode(data1, is_source=True)
            Y = self._v1_encode(data2, is_source=False)
            return self._v1_correlate(X, Y)

    @torch.inference_mode()
    def _forward_v1_python_fft(self, signal_1, signal_2):
        """
        LEGACY METHOD (Internal / Benchmarking use only).
        WARNING: Returns raw accumulated windows without dividing by `flag_mean`.
        Caller must manually normalize.
        """
        pass # Intentionally stubbed as you are strictly using the batched method now.