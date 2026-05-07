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

    For window_time == 0.0: applies 1-bit normalization (np.sign).
    Otherwise: divides each sample by the local running absolute mean computed
    over a window of length round(fs * window_time) samples.

    Memory layout:
        ram  : float64, shape (nch, nt)  — running absolute mean
        out  : float32, shape (nch, nt)  — result, pre-allocated as zeros

    np.divide(..., out=out, where=ram > 0) writes directly into out only where
    the denominator is nonzero, leaving zeros elsewhere. This replaces the
    previous three-array chain (np.where → nan intermediate → nan_to_num)
    with a single in-place division — halving allocations and eliminating
    two full array passes over (nch × nt) data.

    :param data: Input array, shape (nch, nt), float32 or float64.
    :param fs: Sampling rate (Hz).
    :param window_time: RAM window duration (seconds). 0.0 = 1-bit normalization.
    :return: Normalized array, shape (nch, nt), float32.
    """
    if data.ndim != 2:
        raise ValueError("temporal_normalization: data must be 2D (nch × nt).")
    if float(window_time) == 0.0:
        return np.sign(data).astype(np.float32, copy=False)

    nwin = max(int(round(fs * float(window_time))), 1)
    ram  = uniform_filter1d(np.abs(data), size=nwin, axis=1, mode='nearest')

    # Pre-allocate output as float32 zeros. Locations where ram == 0 stay zero
    # (same semantic as the previous nan → 0.0 path, without the NaN intermediate).
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
        nwin = max(int(window_freq / df), 1); nwin = nwin + 1 if nwin % 2 == 0 else nwin
        amp = torch.abs(rfftdata)
        
        # Flattens all leading dimensions so conv1d handles 2D, 3D, or 4D tensors seamlessly
        orig_shape = amp.shape
        amp_flat = amp.reshape(-1, 1, nfreq)
        
        kernel = _WHITEN_CACHE.get_kernel(device, dtype_amp, nwin)
        amp_smooth = torch.nn.functional.conv1d(amp_flat, kernel, padding=nwin//2).squeeze(1)
        
        if amp_smooth.shape[-1] > nfreq: 
            amp_smooth = amp_smooth[..., :nfreq]
            
        rfft_out = rfftdata / amp_smooth.reshape(orig_shape).clamp_(min=1e-10)

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
    chunk_nch: int = 64,
) -> torch.Tensor:
    """
    Apply spectral whitening independently to each segment of a multichannel signal.

    Processes channels in groups of chunk_nch to cap peak memory usage.
    Without chunking, the full (nch × nseg, npts_seg) tensor is materialised at
    once — for Bridge (421 ch, nseg=60, npts_seg=2500) this produces three
    simultaneous ~240 MB complex tensors (~720 MB peak). With chunk_nch=64 the
    peak drops to ~115 MB regardless of total channel count, keeping the working
    set closer to L3 cache.

    Memory per chunk (three tensors live simultaneously):
        chunk_nch × nseg × npts_seg × 4B × 3
        e.g. 64 × 60 × 2500 × 4 × 3 ≈ 115 MB  (Bridge)
        e.g. 64 × 10 × 15000 × 4 × 3 ≈ 115 MB  (Urban)

    :param x: Input tensor, shape (nch, npts_new). Must be float32.
    :param fs_proc: Sampling rate after decimation (Hz).
    :param npts_seg: Samples per segment (must divide npts_new evenly).
    :param window_freq_hz: Smoothing window for amplitude spectrum (Hz). 0 = full whitening.
    :param f1: Low frequency bound of whitening passband (Hz).
    :param f2: High frequency bound of whitening passband (Hz).
    :param chunk_nch: Number of channels to whiten per chunk. Tune this to
        balance memory vs loop overhead. Default 64 works well for all current
        datasets. Reduce if RAM is tight; increase for low-nseg datasets.
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
    if chunk_nch < 1:
        raise ValueError("whiten_per_segment_torch: chunk_nch must be >= 1")

    nseg = npts_new // npts_seg
    df   = float(fs_proc) / float(npts_seg)
    x    = x.to(dtype=torch.float32)
    out  = torch.empty_like(x)

    for ch0 in range(0, nch, chunk_nch):
        ch1   = min(ch0 + chunk_nch, nch)
        B     = (ch1 - ch0) * nseg                        # rows for this chunk

        # Reshape chunk: (chunk_nch, npts_new) → (chunk_nch × nseg, npts_seg)
        x2    = x[ch0:ch1, :].reshape(B, npts_seg)
        X     = torch.fft.rfft(x2, n=npts_seg, dim=-1)
        Xw    = spectral_whitening(X, df, float(window_freq_hz), float(f1), float(f2))
        xw    = torch.fft.irfft(Xw, n=npts_seg, dim=-1).to(torch.float32)

        # Write result back: (chunk_nch × nseg, npts_seg) → (chunk_nch, npts_new)
        out[ch0:ch1, :] = xw.reshape(ch1 - ch0, npts_new)

    return out

# ==============================================================
# 3. Full preprocessing pipeline
# ==============================================================
def preprocess(x: np.ndarray | torch.Tensor, fs_raw: float, f1: float, f2: float,
               decimation: int, diff: bool, ram_win: float) -> np.ndarray | torch.Tensor:
    """
    Full DAS preprocessing pipeline: diff → detrend → bandpass → decimate → median
    removal → temporal normalisation.

    Decimation notes
    ----------------
    decimation=1 : no-op, data is unchanged.
    decimation>1 : uses scipy.signal.decimate (Chebyshev lowpass + downsample,
                   zero-phase). Naive slicing (x[:, ::q]) is intentionally NOT
                   used because it skips the anti-aliasing filter, which can fold
                   energy from [Nyquist_decimated, Nyquist_raw] back into the
                   passband. A ValueError is raised if f2 >= Nyquist_decimated
                   (i.e. the bandpass would alias), and a UserWarning is issued
                   if f2 > 0.8 × Nyquist_decimated (close to the alias boundary).

    :param x: Input data, shape (nch, nt).
    :param fs_raw: Raw sampling rate (Hz).
    :param f1: Bandpass low corner (Hz).
    :param f2: Bandpass high corner (Hz).
    :param decimation: Integer downsampling factor (>= 1).
    :param diff: If True, differentiate along time axis first (strain → strain-rate).
    :param ram_win: Running absolute mean window (seconds). 0 = 1-bit normalisation.
    :return: Preprocessed array, same type as input, shape (nch, nt_proc), float32.
    """
    import warnings

    if decimation < 1:
        raise ValueError(f"preprocess: decimation must be >= 1, got {decimation}.")

    is_tensor = isinstance(x, torch.Tensor)
    orig_device = x.device if is_tensor else None
    x_np = convert_to_numpy(x) if is_tensor else np.asarray(x)

    if x_np.ndim != 2:
        raise ValueError(f"preprocess: expected 2D (nch × nt); got shape {x_np.shape}.")

    # --- Differentiation (strain → strain-rate) ---
    if diff:
        x_np = np.gradient(x_np, axis=-1) * float(fs_raw)

    # --- Detrend + bandpass ---
    x_np = detrend(x_np, axis=-1)
    x_np = bandpass_filter_tukey(x_np, fs_raw, f1, f2)

    # --- Decimation with anti-alias guard ---
    if decimation > 1:
        nyq_decimated = float(fs_raw) / (2.0 * float(decimation))

        # Hard error: f2 at or above the decimated Nyquist means the signal
        # of interest cannot survive downsampling — aliasing is unavoidable.
        if float(f2) >= nyq_decimated:
            raise ValueError(
                f"preprocess: f2={f2} Hz >= Nyquist after decimation "
                f"({nyq_decimated:.2f} Hz at fs_raw={fs_raw} Hz, decimation={decimation}). "
                f"Reduce decimation or lower f2 to avoid aliasing."
            )

        # Soft warning: f2 within 20% of the decimated Nyquist. The anti-alias
        # filter in scipy.decimate attenuates here, but the transition band may
        # clip some signal energy near f2.
        if float(f2) > 0.8 * nyq_decimated:
            warnings.warn(
                f"preprocess: f2={f2} Hz is within 20% of the decimated Nyquist "
                f"({nyq_decimated:.2f} Hz). Some signal energy near f2 may be "
                f"attenuated by the anti-alias filter. Consider reducing decimation "
                f"or lowering f2.",
                UserWarning,
                stacklevel=2,
            )

        # scipy.signal.decimate: Chebyshev type-I lowpass at Nyquist_decimated,
        # then downsamples by q. zero_phase=True uses sosfiltfilt (same as our
        # bandpass — no phase distortion).
        x_np = decimate(x_np, q=int(decimation), axis=-1, zero_phase=True)

    # --- Median removal + temporal normalisation ---
    fs_proc = float(fs_raw) / float(decimation)
    x_np   -= np.median(x_np, axis=0)
    x_np    = temporal_normalization(x_np, fs_proc, float(ram_win))

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
        # Auto-unsqueeze 2D -> 3D for generic processing
        if data1.ndim == 2:
            data1 = data1.unsqueeze(1)
            data2 = data2.unsqueeze(1)
            
        B, nseg, N = data1.shape
        if self._conv_L is None or self._conv_N != N:
            self._conv_L = int(nextpow2(2 * N - 1)); self._conv_N = N
            self._conv_df = self.whitening_params[0] / self._conv_L if self.is_spectral_whitening else None
        
        # 1. Forward FFT along the time dimension
        X = torch.fft.rfft(data1, n=self._conv_L, dim=-1)
        Y = torch.fft.rfft(data2, n=self._conv_L, dim=-1)
        
        if self.is_spectral_whitening:
            X = spectral_whitening(X, self._conv_df, *self.whitening_params[1:])
            Y = spectral_whitening(Y, self._conv_df, *self.whitening_params[1:])
        
        # 2. Multiply and Frequency-Domain Stacking (sum over nseg)
        Rspec = (torch.conj(X) * Y).sum(dim=1)
        
        # 3. SINGLE Inverse FFT
        r = torch.fft.irfft(Rspec, n=self._conv_L, dim=-1)
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
        if signal_1.ndim == 2:
            signal_1 = signal_1.unsqueeze(1)
            signal_2 = signal_2.unsqueeze(1)
            
        if signal_1.ndim != 3 or signal_2.ndim != 3:
            raise ValueError("v1: inputs must be 2D (B, N) or 3D (B, nseg, N).")
        if signal_1.shape != signal_2.shape:
            raise ValueError("v1: shape mismatch")
            
        M, K, Lfft = self._v1_M, self._v1_K, self._v1_Lfft
        B, nseg, N = signal_1.shape
        nblocks = (N + K - 1) // K
        
        # --- The 3D Merged Batch Optimization ---
        Bn = B * nseg
        x_blocked = torch.zeros((Bn, nblocks, Lfft), dtype=signal_1.dtype, device=signal_1.device)
        y_blocked = torch.zeros((Bn, nblocks, Lfft), dtype=signal_1.dtype, device=signal_1.device)

        s1 = signal_1.reshape(Bn, N)
        s2 = signal_2.reshape(Bn, N)

        for blk in range(nblocks):
            start = blk * K
            end = min(start + K, N)
            klen = end - start
            y_blocked[:, blk, M : M + klen] = s2[:, start : end]
            x0 = start - M
            ix0 = max(0, x0)
            ix1 = min(N, start + K + M)
            if ix1 > ix0:
                dst0 = ix0 - x0
                length = ix1 - ix0
                x_blocked[:, blk, dst0 : dst0 + length] = s1[:, ix0 : ix1]

        del s1, s2

        X = torch.fft.rfft(x_blocked, n=Lfft, dim=-1)
        Y = torch.fft.rfft(y_blocked, n=Lfft, dim=-1)
        del x_blocked, y_blocked

        if self.is_spectral_whitening:
            df = self.whitening_params[0] / Lfft
            X = spectral_whitening(X, df, *self.whitening_params[1:])
            Y = spectral_whitening(Y, df, *self.whitening_params[1:])

        # Frequency-Domain Stacking (Sum over nseg AND nblocks)
        Rspec = (X.conj() * Y).sum(dim=1)
        Rspec = Rspec.reshape(B, nseg, X.shape[-1]).sum(dim=1)
        
        # SINGLE Inverse FFT
        r = torch.fft.irfft(Rspec, n=Lfft, dim=-1)
        return torch.cat([r[:, Lfft - M : Lfft], r[:, 0 : M + 1]], dim=-1).to(torch.float32)

    @torch.no_grad()
    def _forward_v1_python_fft(self, signal_1, signal_2):
        # Legacy loop: Flattens 3D back to 2D internally to maintain backward compatibility for tests
        is_3d = signal_1.ndim == 3
        if is_3d:
            B, nseg, N = signal_1.shape
            signal_1 = signal_1.reshape(B * nseg, N)
            signal_2 = signal_2.reshape(B * nseg, N)
            
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
        out = torch.cat([r[:, Lfft - M : Lfft], r[:, 0 : M + 1]], dim=-1)
        
        if is_3d:
            # Re-sum over segments to match the output shape of the new batched methods
            out = out.reshape(-1, nseg, out.shape[-1]).sum(dim=1)
            
        return out.to(torch.float32)

    def forward(self, data1: torch.Tensor, data2: torch.Tensor) -> torch.Tensor:
        if self.mode == "conventional": return self._forward_conventional(data1, data2)
        return self._forward_v1_batched(data1, data2)