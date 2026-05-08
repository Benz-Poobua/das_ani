"""
:module: src/ani.py
:auth: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: DAS preprocessing (Bensen et al., 2007) + cross-correlation (conventional + Zhang 2026 v1).
:reference: Modified from Yan Yang (2022-07-10).

Revision notes (this version):
    - TorchCrossCorrelation refactored into compute_X / compute_Y / combine
      so VS-mode callers can compute the source spectrum once per virtual
      source and reuse it across receiver chunks. The legacy
      forward(data1, data2) entry point is preserved.
    - preprocess: np.diff multiply is now in-place (one fewer full-array alloc).
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

    For ``window_time == 0.0``: 1-bit normalization (``np.sign``).
    Otherwise: divides each sample by the local RAM over a window of
    ``round(fs × window_time)`` samples.

    :param data: Input array, shape (nch, nt).
    :param fs: Sampling rate (Hz).
    :param window_time: RAM window length (seconds). 0.0 = 1-bit normalisation.
    :return: Normalised array, shape (nch, nt), float32.
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

        # Flatten all leading dimensions into a single batch dim so conv1d
        # handles any input rank (2D, 3D, 4D, …) without separate code paths.
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

    :param x: Input tensor, shape (nch, npts_new). Must be float32.
    :param fs_proc: Sampling rate after decimation (Hz).
    :param npts_seg: Samples per segment (must divide npts_new evenly).
    :param window_freq_hz: Smoothing window for amplitude spectrum (Hz). 0 = full whitening.
    :param f1: Low frequency bound of whitening passband (Hz).
    :param f2: High frequency bound of whitening passband (Hz).
    :param chunk_nch: Channels processed per chunk.
    :return: Whitened tensor, same shape as input (nch, npts_new), float32.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("whiten_per_segment_torch: x must be a torch.Tensor")
    if x.ndim != 2:
        raise ValueError(
            f"whiten_per_segment_torch: expected 2D (nch, nt); got {tuple(x.shape)}"
        )
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

        # (chunk_nch, npts_new) → (chunk_nch × nseg, npts_seg)
        x2  = x[ch0:ch1, :].reshape(B, npts_seg)
        X   = torch.fft.rfft(x2, n=npts_seg, dim=-1)
        Xw  = spectral_whitening(X, df, float(window_freq_hz), float(f1), float(f2))
        xw  = torch.fft.irfft(Xw, n=npts_seg, dim=-1).to(torch.float32)

        # (chunk_nch × nseg, npts_seg) → (chunk_nch, npts_new)
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
    """
    Full DAS preprocessing pipeline.

    Order of operations:
        1. Differentiation (optional) — strain → strain-rate via ``np.diff``.
        2. Detrend — remove linear trend along time axis.
        3. Bandpass + Tukey taper — ``bandpass_filter_tukey``.
        4. Decimation (optional) — anti-aliased downsampling via
           ``scipy.signal.decimate`` (Chebyshev lowpass + downsample, zero-phase).
        5. Median removal — subtract per-sample median across channels.
        6. Temporal normalisation — ``temporal_normalization``.
    """
    if decimation < 1:
        raise ValueError(f"preprocess: decimation must be >= 1, got {decimation}.")

    is_tensor  = isinstance(x, torch.Tensor)
    orig_device = x.device if is_tensor else None
    x_np = convert_to_numpy(x) if is_tensor else np.asarray(x)

    if x_np.ndim != 2:
        raise ValueError(f"preprocess: expected 2D (nch × nt); got shape {x_np.shape}.")

    # 1. Differentiation
    if diff:
        # Reverted to np.gradient for 100% mathematical parity with legacy baselines.
        # Uses 2nd-order central differences: (x[i+1] - x[i-1]) / 2
        x_np = np.gradient(x_np, axis=-1) * float(fs_raw)

    # 2. Detrend + 3. Bandpass
    x_np = detrend(x_np, axis=-1)
    x_np = bandpass_filter_tukey(x_np, fs_raw, f1, f2)

    # 4. Decimation with anti-alias guard
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
                f"preprocess: f2={f2} Hz is within 20% of the decimated Nyquist "
                f"({nyq_decimated:.2f} Hz). The anti-alias filter transition band "
                "may attenuate signal near f2. Consider reducing decimation or f2.",
                UserWarning,
                stacklevel=2,
            )

        x_np = decimate(x_np, q=int(decimation), axis=-1, zero_phase=True)

    # 5. Median removal + 6. Temporal normalisation
    fs_proc  = float(fs_raw) / float(decimation)
    x_np    -= np.median(x_np, axis=0)
    x_np     = temporal_normalization(x_np, fs_proc, float(ram_win))

    return convert_to_tensor(x_np, device=orig_device) if is_tensor else x_np.astype(np.float32, copy=False)

# ==============================================================
# 4. Zhang (2026) helper: choose block size
# ==============================================================
def choose_block_size_v2(
    M: int,
    *,
    fft_snap_pow2: bool = True,
    fallback: Literal["v1_2M", "v1_Mp1"] = "v1_2M",
) -> tuple[int, int]:
    """
    Choose optimal block size K and FFT length L for the Zhang v1 algorithm.
    """
    if M <= 0:
        raise ValueError(f"choose_block_size_v2: M must be > 0, got {M}.")

    K_star = None
    if lambertw is not None:
        z = -1.0 / (4.0 * math.e * float(M))
        try:
            w      = lambertw(z, k=-1)
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
    """
    Frequency-domain cross-correlation engine.

    Public API
    ----------
    forward(data1, data2)
        Symmetric path; equivalent to ``combine(compute_X(data1), compute_Y(data2))``.
        Used by auto-CC and any caller that has both signals on hand.

    compute_X(x), compute_Y(y), combine(X, Y)
        Asymmetric path used by VS-mode pipelines: compute the source-side
        spectrum once per virtual source and broadcast over receiver chunks.

        For mode='conventional', ``compute_X`` and ``compute_Y`` are identical
        (they're both an rfft + optional whitening). The split exists so the
        v1 path — where x and y require different padding patterns — can
        share the same call surface.

    Shapes
    ------
    Inputs to compute_X / compute_Y / forward:    (B, nseg, N) real.
    compute_X / compute_Y output:
        conventional → (B, nseg, nfreq) complex
        v1           → (B, nseg, nblocks, nfreq) complex
    combine output:                                (B_recv, 2M+1 or 2N-1) real (float32).
        Broadcasts a B_X=1 source spectrum over a B_Y=batch_len receiver
        spectrum, so the source's rfft is computed once per VS instead of
        ``batch_len × nchunk`` times.
    """

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

        # Conventional: lazy-init on first call.
        self._conv_L = self._conv_N = self._conv_df = None

        # v1: K/L are M-only and known at __init__; nblocks depends on input N.
        self._v1_M = self._v1_K = self._v1_Lfft = self._v1_nfreq = None

        # Legacy buffers (kept for _forward_v1_python_fft only).
        self._v1_buf_key = self._v1_x_t = self._v1_y_t = None
        self._rspec_key = self._rspec = None

        if self.mode == "v1":
            self._v1_M = int(max_lag_samples)
            self._v1_K, self._v1_Lfft = choose_block_size_v2(
                self._v1_M, fft_snap_pow2=v1_fft_snap_pow2, fallback=v1_fallback
            )
            self._v1_nfreq = self._v1_Lfft // 2 + 1

    # ----- Lazy init helpers -----
    def _conv_init(self, N: int) -> None:
        if self._conv_L is None or self._conv_N != N:
            self._conv_L = int(nextpow2(2 * N - 1))
            self._conv_N = N
            self._conv_df = (
                self.whitening_params[0] / self._conv_L
                if self.is_spectral_whitening else None
            )

    # ----- v1 block builders -----
    @staticmethod
    def _v1_nblocks(N: int, K: int) -> int:
        return (N + K - 1) // K

    def _v1_build_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Source-side block layout (full overlap context).

        Each block holds [pre-context M][center K][post-context M], zero-padded
        for first/last blocks where context falls outside the signal.
        """
        M, K, Lfft = self._v1_M, self._v1_K, self._v1_Lfft
        B, nseg, N = x.shape
        Bn = B * nseg
        nblocks = self._v1_nblocks(N, K)

        x_blocked = torch.zeros((Bn, nblocks, Lfft), dtype=x.dtype, device=x.device)
        s = x.reshape(Bn, N)
        for blk in range(nblocks):
            start = blk * K
            x0    = start - M
            ix0   = max(0, x0)
            ix1   = min(N, start + K + M)
            if ix1 > ix0:
                dst0   = ix0 - x0
                length = ix1 - ix0
                x_blocked[:, blk, dst0:dst0 + length] = s[:, ix0:ix1]
        return x_blocked

    def _v1_build_y(self, y: torch.Tensor) -> torch.Tensor:
        """
        Receiver-side block layout (centered, M-zero borders).

        Each block holds [M zeros][center K samples of y][M zeros + tail zeros],
        where the trailing zeros come from the implicit zero-padding to length
        Lfft = K + 2M (and the K-window is truncated for the final block).
        """
        M, K, Lfft = self._v1_M, self._v1_K, self._v1_Lfft
        B, nseg, N = y.shape
        Bn = B * nseg
        nblocks = self._v1_nblocks(N, K)

        y_blocked = torch.zeros((Bn, nblocks, Lfft), dtype=y.dtype, device=y.device)
        s = y.reshape(Bn, N)
        for blk in range(nblocks):
            start = blk * K
            end   = min(start + K, N)
            klen  = end - start
            y_blocked[:, blk, M:M + klen] = s[:, start:end]
        return y_blocked

    # ----- Spectrum computation: conventional -----
    @torch.inference_mode()
    def _compute_conv_spec(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"conventional: expected 3D (B, nseg, N); got {tuple(x.shape)}.")
        if x.is_complex():
            raise ValueError("conventional: expects real tensors, got complex input.")
        self._conv_init(int(x.shape[-1]))

        X = torch.fft.rfft(x, n=self._conv_L, dim=-1)
        if self.is_spectral_whitening:
            if self.whitening_params is None:
                raise RuntimeError("conventional: whitening_params required when whitening is enabled.")
            X = spectral_whitening(X, self._conv_df, *self.whitening_params[1:])
        return X

    # ----- Spectrum computation: v1 -----
    def _v1_check_init(self) -> None:
        if (self._v1_M is None or self._v1_K is None
                or self._v1_Lfft is None or self._v1_nfreq is None):
            raise RuntimeError("v1: module not initialised (M/K/Lfft/nfreq is None).")

    @torch.inference_mode()
    def _compute_v1_X_spec(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"v1: expected 3D (B, nseg, N); got {tuple(x.shape)}.")
        if x.is_complex():
            raise ValueError("v1: expects real tensors, got complex input.")
        self._v1_check_init()

        Lfft = self._v1_Lfft
        x_blocked = self._v1_build_x(x)            # (B*nseg, nblocks, Lfft)
        X = torch.fft.rfft(x_blocked, n=Lfft, dim=-1)  # (B*nseg, nblocks, nfreq)
        del x_blocked

        if self.is_spectral_whitening:
            if self.whitening_params is None:
                raise RuntimeError("v1: whitening_params required when whitening is enabled.")
            df = float(self.whitening_params[0]) / float(Lfft)
            X = spectral_whitening(X, df, *self.whitening_params[1:])

        # Reshape to (B, nseg, nblocks, nfreq) so combine() can broadcast cleanly.
        return X.view(x.shape[0], x.shape[1], -1, self._v1_nfreq)

    @torch.inference_mode()
    def _compute_v1_Y_spec(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim != 3:
            raise ValueError(f"v1: expected 3D (B, nseg, N); got {tuple(y.shape)}.")
        if y.is_complex():
            raise ValueError("v1: expects real tensors, got complex input.")
        self._v1_check_init()

        Lfft = self._v1_Lfft
        y_blocked = self._v1_build_y(y)
        Y = torch.fft.rfft(y_blocked, n=Lfft, dim=-1)
        del y_blocked

        if self.is_spectral_whitening:
            if self.whitening_params is None:
                raise RuntimeError("v1: whitening_params required when whitening is enabled.")
            df = float(self.whitening_params[0]) / float(Lfft)
            Y = spectral_whitening(Y, df, *self.whitening_params[1:])

        return Y.view(y.shape[0], y.shape[1], -1, self._v1_nfreq)

    # ----- Public spectrum API -----
    @torch.inference_mode()
    def compute_X(self, x: torch.Tensor) -> torch.Tensor:
        """Source-side spectrum. See class docstring for shape contract."""
        if self.mode == "conventional":
            return self._compute_conv_spec(x)
        return self._compute_v1_X_spec(x)

    @torch.inference_mode()
    def compute_Y(self, y: torch.Tensor) -> torch.Tensor:
        """Receiver-side spectrum. See class docstring for shape contract."""
        if self.mode == "conventional":
            return self._compute_conv_spec(y)
        return self._compute_v1_Y_spec(y)

    # ----- Combine -----
    @torch.inference_mode()
    def combine(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Combine source and receiver spectra into the cross-correlation output.

        For VS mode, X has B_X=1 and Y has B_Y=batch_len; broadcasting handles
        the rest. For symmetric forward(), B_X==B_Y.
        """
        if self.mode == "conventional":
            # X: (B_X, nseg, nfreq), Y: (B_Y, nseg, nfreq)
            Rspec = (X.conj() * Y).sum(dim=1)              # (B_Y, nfreq)
            conv_L = self._conv_L
            N = self._conv_N
            r = torch.fft.irfft(Rspec, n=conv_L, dim=-1)
            return torch.cat([r[:, conv_L - (N - 1):], r[:, :N]], dim=-1).to(torch.float32)

        # v1
        # X: (B_X, nseg, nblocks, nfreq), Y: (B_Y, nseg, nblocks, nfreq)
        Rspec = (X.conj() * Y).sum(dim=(1, 2))             # (B_Y, nfreq)
        Lfft = self._v1_Lfft
        M    = self._v1_M
        r = torch.fft.irfft(Rspec, n=Lfft, dim=-1)
        return torch.cat([r[:, Lfft - M:Lfft], r[:, 0:M + 1]], dim=-1).to(torch.float32)

    # ----- Backward-compat wrappers -----
    @torch.inference_mode()
    def _forward_conventional(self, data1: torch.Tensor, data2: torch.Tensor) -> torch.Tensor:
        return self.combine(self._compute_conv_spec(data1),
                            self._compute_conv_spec(data2))

    @torch.inference_mode()
    def _forward_v1_batched(self, signal_1: torch.Tensor, signal_2: torch.Tensor) -> torch.Tensor:
        # Strict validation kept for parity with the previous interface.
        if signal_1.ndim != 3 or signal_2.ndim != 3:
            raise ValueError(
                "v1: expected 3D (B, nseg, N); "
                f"got signal_1={tuple(signal_1.shape)}, signal_2={tuple(signal_2.shape)}."
            )
        if signal_1.shape != signal_2.shape:
            raise ValueError(f"v1: shape mismatch {tuple(signal_1.shape)} vs {tuple(signal_2.shape)}")
        if signal_1.device != signal_2.device:
            raise ValueError(f"v1: device mismatch {signal_1.device} vs {signal_2.device}")
        if signal_1.dtype != signal_2.dtype:
            raise ValueError(f"v1: dtype mismatch {signal_1.dtype} vs {signal_2.dtype}")
        if signal_1.is_complex() or signal_2.is_complex():
            raise ValueError("v1: expects real tensors, got complex input")

        return self.combine(self._compute_v1_X_spec(signal_1),
                            self._compute_v1_Y_spec(signal_2))

    # ----- Legacy single-buffer path (benchmarking only) -----
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
        else:
            self._rspec.zero_()
        return self._rspec

    @torch.inference_mode()
    def _forward_v1_python_fft(self, signal_1, signal_2):
        """
        LEGACY METHOD (Internal / Benchmarking use only).

        WARNING: Returns the raw accumulation of cross-correlation windows
        without dividing by ``flag_mean`` (nseg). For comparison against the
        production batched pipeline, divide the result by nseg manually.
        """
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
        for blk in range(nblocks):
            start = blk * K
            end   = min(start + K, N)
            klen  = end - start
            x_t.zero_(); y_t.zero_()
            y_t.narrow(1, M, klen).copy_(signal_2.narrow(1, start, klen))
            ix0 = max(0, start - M); ix1 = min(N, start + K + M)
            if ix1 > ix0:
                x_t.narrow(1, ix0 - (start - M), ix1 - ix0).copy_(signal_1.narrow(1, ix0, ix1 - ix0))
            X, Y = torch.fft.rfft(x_t, n=Lfft, dim=-1), torch.fft.rfft(y_t, n=Lfft, dim=-1)
            if self.is_spectral_whitening:
                X = spectral_whitening(X, df, *self.whitening_params[1:])
                Y = spectral_whitening(Y, df, *self.whitening_params[1:])
            Rspec.addcmul_(X.conj(), Y)
        r = torch.fft.irfft(Rspec, n=Lfft, dim=-1)
        out = torch.cat([r[:, Lfft - M:Lfft], r[:, 0:M + 1]], dim=-1)

        if is_3d:
            out = out.reshape(-1, nseg, out.shape[-1]).sum(dim=1)
        return out.to(torch.float32)

    # ----- Standard nn.Module entry point -----
    def forward(self, data1: torch.Tensor, data2: torch.Tensor) -> torch.Tensor:
        return self.combine(self.compute_X(data1), self.compute_Y(data2))
