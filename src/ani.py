"""
:module: src/ani.py
:auth: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: High-performance DAS preprocessing and ambient noise cross-correlation engine.
:reference: Preprocessing heavily adapted from Bensen et al. (2007).
            Original baseline modified from Yan Yang (2022-07-10).
            Advanced cross-correlation implements Zhang (2026) v1.

Description:
This module contains the core mathematical and digital signal processing (DSP)
pipeline for Ambient Noise Interferometry (ANI). It pairs memory-efficient NumPy
preprocessing with a hardware-aware PyTorch backend, designed specifically to
maximize GPU utilization and minimize memory overhead on HPC clusters.

Architecture (selectable preprocessing backend, config key `preprocess.mode`):
    Standalone raw-data utilities (called once at ingest, NOT in preprocess):
        - decimate_raw            : anti-aliased downsampling of raw DAS data
        - differentiate_cpu       : strain -> strain-rate via NumPy gradient
        - differentiate_torch     : strain -> strain-rate via PyTorch gradient

    Per-window preprocessing chain (preprocess()), identical stage order in
    every backend: detrend -> bandpass -> median removal -> temporal norm.

        pure_numpy : all stages on CPU via scipy/numpy. Legacy, bit-exact
                     benchmark used by the published baseline.
        hybrid     : CPU detrend + bandpass (scipy sosfiltfilt; mathematically
                     canonical) -> single host-to-device transfer -> torch
                     median removal + temporal normalization on the target
                     device. Numerical-fidelity choice for CPU or GPU runs.
        pure_torch : all stages in torch on the target device, including the
                     rFFT-domain |H(f)|^2 bandpass. Fastest on GPU; the
                     bandpass approximates sosfiltfilt (see
                     bandpass_filter_tukey_torch fidelity note).

    Spectral whitening is performed downstream as part of the
    cross-correlation kernel (rFFT domain).

Key Features:
    - Asymmetric Cross-Correlation: The `TorchCrossCorrelation` engine is split into
      `compute_X`, `compute_Y`, and `combine`. This allows the FFT of a Virtual
      Source (VS) to be computed exactly once and broadcast across thousands of
      receiver chunks, eliminating redundant computations.
    - Convolutional Spectral Whitening: Flattens frequency arrays and uses
      `torch.nn.functional.conv1d` to apply smoothing windows, allowing the GPU's
      highly optimized deep learning cores to accelerate DSP math.
    - Zero-Copy & In-Place Memory: Enforces strict in-place operations (e.g.,
      `np.divide` via `out=`) during temporal normalization and differentiation
      to prevent massive RAM allocation spikes.
    - Strict Inference Mode: Utilizes `@torch.inference_mode()` across all PyTorch
      functions to strip gradient metadata, minimizing dispatch overhead and unlocking
      maximum execution speed in the C++/CUDA backends.
"""
from __future__ import annotations

import logging
import math
import warnings
from typing import Any, Optional, Literal

import numpy as np
import scipy.signal as signal
import torch
import torch.nn.functional as F
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
# 1. CPU preprocessing utilities (numpy / scipy)
# ==============================================================

def bandpass_filter_tukey(
    data: np.ndarray,
    fs: float,
    f1: float | None = None,
    f2: float | None = None,
    alpha: float = 0.05,
    order: int = 4,
) -> np.ndarray:
    """
    Applies a time-domain Tukey window taper followed by a zero-phase Butterworth filter.

    **DSP Theory & Application:**
    1. **Tapering (Tukey Window):** DAS and seismic recording segments often have abrupt
       start and end amplitudes. Filtering or Fourier transforming un-tapered data causes
       "spectral leakage" (Gibbs phenomenon). The Tukey window (tapered cosine) smoothly
       ramps the edges of the trace to zero while preserving the original amplitude in
       the center.
    2. **Filtering (Butterworth):** A Butterworth filter is chosen for its maximally flat
       passband. To maintain exact phase relationships---which is mathematically critical
       for travel-time measurements in cross-correlation---the filter is applied forward
       and backward using ``sosfiltfilt`` (zero-phase filtering).
    3. **Numerical Stability:** The filter is designed using Second-Order Sections (SOS)
       rather than standard polynomial forms (b, a) to prevent numerical instability and
       precision loss at higher filter orders.

    :param data: 2D NumPy array of continuous DAS data, shape (nchannels, nsamples).
    :param fs: Sampling frequency of the input data in Hz.
    :param f1: Lower frequency bound (Hz). If None, operates as a lowpass filter.
    :param f2: Upper frequency bound (Hz). If None, operates as a highpass filter.
    :param alpha: Shape parameter of the Tukey window. Represents the fraction of the
                  window inside the cosine tapered region. A value of 0.05 means a
                  2.5% taper on each end of the trace.
    :param order: The order of the Butterworth filter. Higher orders yield sharper
                  rolloffs but can cause ringing in the time domain.
    :return: Tapered and filtered 2D array, cast to float32.
    """
    if data.ndim != 2:
        raise ValueError("bandpass_filter_tukey: data must be 2D (nch x nt).")

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
    CPU/numpy temporal normalization (Running Absolute Mean or 1-bit).

    **DSP Theory & Application:**
    Ambient Noise Interferometry (ANI) relies on extracting the Earth's Green's function
    from continuous, diffuse background noise. Local transients (e.g., passing vehicles,
    earthquakes, or optical glitches in the DAS interrogator) can completely dominate the
    cross-correlation result. Temporal normalization equalizes the amplitude of the
    signal so that loud discrete events do not overpower the continuous ambient
    wavefield.

    **Methods:**
    - **Running Absolute Mean (RAM) (window_time > 0):** Moving average of |signal|
      over a specified time window; the original signal is divided by this envelope.
      Implemented via ``scipy.ndimage.uniform_filter1d`` for C-level vectorization.
    - **1-Bit Normalization (window_time = 0.0):** Retains only the sign of the
      waveform; destroys absolute amplitude information but perfectly equalizes the
      trace.

    :param data: Input 2D NumPy array, shape (nchannels, nsamples).
    :param fs: Sampling rate of the input data in Hz.
    :param window_time: Length of the running absolute mean window in seconds.
                        Set to 0.0 to trigger 1-bit normalization.
    :return: Temporally normalized 2D array, shape (nch, nt), float32.
    """
    if data.ndim != 2:
        raise ValueError("temporal_normalization: data must be 2D (nch x nt).")
    if float(window_time) == 0.0:
        return np.sign(data).astype(np.float32, copy=False)

    nwin = max(int(round(fs * float(window_time))), 1)
    ram  = uniform_filter1d(np.abs(data), size=nwin, axis=1, mode='nearest')

    out = np.zeros(data.shape, dtype=np.float32)
    np.divide(data, ram, out=out, where=ram > 0)
    return out


# ==============================================================
# 2. Standalone raw-data utilities
#    These are NOT part of preprocess(). They are called once at
#    ingest, before per-window preprocessing begins.
# ==============================================================

def decimate_raw(
    x: np.ndarray,
    fs_raw: float,
    q: int,
    f2_target: float | None = None,
) -> np.ndarray:
    """
    Anti-aliased decimation of raw DAS data via Chebyshev lowpass + downsample.

    Use this once at ingest if the raw sampling rate is unnecessarily high
    relative to your target frequency band. After decimation, the effective
    sampling rate is ``fs_raw / q`` and ``preprocess()`` should be called
    with ``fs = fs_raw / q``.

    :param x: 2D array (nch, nt) of raw DAS samples.
    :param fs_raw: Raw sampling frequency (Hz).
    :param q: Decimation factor (integer >= 1). Pass 1 to skip decimation.
    :param f2_target: If provided, validate that the new Nyquist
                      ``(fs_raw / 2q)`` comfortably exceeds the target
                      upper passband; raises ValueError if it does not.
    :return: Decimated 2D array (nch, nt // q), float32.
    """
    if q < 1:
        raise ValueError(f"decimate_raw: q must be >= 1, got {q}.")
    if q == 1:
        return np.asarray(x, dtype=np.float32)

    nyq_decimated = float(fs_raw) / (2.0 * float(q))
    if f2_target is not None:
        if float(f2_target) >= nyq_decimated:
            raise ValueError(
                f"decimate_raw: f2_target={f2_target} Hz >= decimated Nyquist "
                f"({nyq_decimated:.2f} Hz) at fs_raw={fs_raw} Hz, q={q}. "
                "Reduce q or lower f2_target to avoid aliasing."
            )
        if float(f2_target) > 0.8 * nyq_decimated:
            warnings.warn(
                f"decimate_raw: f2_target={f2_target} Hz is within 20% of the "
                f"decimated Nyquist ({nyq_decimated:.2f} Hz). The anti-alias filter "
                "transition band may attenuate signal near f2_target.",
                UserWarning,
                stacklevel=2,
            )

    return decimate(np.asarray(x), q=int(q), axis=-1, zero_phase=True).astype(
        np.float32, copy=False
    )


def differentiate_cpu(x: np.ndarray, fs: float) -> np.ndarray:
    """
    Convert strain to strain-rate (or velocity to acceleration) on CPU via
    ``np.gradient``, which uses second-order central differences.

    Apply this once at ingest if your interrogator reports strain rather
    than strain-rate and your downstream analysis assumes strain-rate.

    :param x: 2D array (nch, nt) of strain samples.
    :param fs: Sampling frequency (Hz).
    :return: 2D array (nch, nt) of strain-rate samples, float32.
    """
    return (np.gradient(np.asarray(x), axis=-1) * float(fs)).astype(np.float32, copy=False)


def differentiate_torch(x: torch.Tensor, fs: float) -> torch.Tensor:
    """
    Convert strain to strain-rate on GPU via ``torch.gradient``.

    Mathematically identical to ``differentiate_cpu``: PyTorch's
    ``torch.gradient`` uses the same second-order central differences as
    NumPy's ``np.gradient``.

    :param x: 2D tensor (nch, nt) of strain samples on any device.
    :param fs: Sampling frequency (Hz).
    :return: 2D tensor (nch, nt) of strain-rate samples.
    """
    return torch.gradient(x, spacing=1.0 / float(fs), dim=-1)[0]


# ==============================================================
# 3. GPU preprocessing utilities (torch)
# ==============================================================

@torch.inference_mode()
def detrend_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Remove the best-fit linear trend along the time axis (last axis) for
    every channel, via a least-squares fit of two basis functions
    (slope + intercept). Mathematically identical to
    ``scipy.signal.detrend(x, axis=-1)``.

    Vectorized across all channels using a single ``torch.linalg.lstsq``
    call, which dispatches to LAPACK (CPU) or cuSOLVER (CUDA).

    :param x: 2D tensor (nch, nt).
    :return: Detrended tensor of the same shape and device.
    """
    if x.ndim != 2:
        raise ValueError(f"detrend_torch: expected 2D (nch x nt); got {tuple(x.shape)}.")
    nt = x.shape[1]
    t = torch.linspace(-1.0, 1.0, nt, dtype=x.dtype, device=x.device)
    A = torch.stack([t, torch.ones_like(t)], dim=-1)            # (nt, 2)
    sol = torch.linalg.lstsq(A, x.T, driver='gels').solution    # (2, nch)
    trend = torch.matmul(A, sol).T                              # (nch, nt)
    return x - trend


@torch.inference_mode()
def remove_median_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Subtract the per-sample median across channels (axis 0).

    **Important:** uses ``torch.quantile(x, 0.5, dim=0)`` rather than
    ``torch.median``. The two differ when the channel count is even:
    NumPy ``np.median`` and ``torch.quantile(q=0.5)`` average the two
    middle values, while ``torch.median`` returns the lower of the two.
    The quantile-based version matches NumPy parity to single-precision
    epsilon, which is the convention used by the all-CPU preprocessing
    path and by the published baseline.

    :param x: 2D tensor (nch, nt) of DAS samples.
    :return: Median-subtracted tensor of the same shape and device.
    """
    if x.ndim != 2:
        raise ValueError(f"remove_median_torch: expected 2D (nch x nt); got {tuple(x.shape)}.")
    median_trace = torch.quantile(x, 0.5, dim=0, keepdim=True)
    return x - median_trace


@torch.inference_mode()
def temporal_normalization_torch(
    x: torch.Tensor,
    fs: float,
    window_time: float,
) -> torch.Tensor:
    """
    Running-absolute-mean (RAM) or 1-bit temporal normalization in PyTorch.

    Dynamically routes to the optimal algorithm based on the active device:

    - **CUDA path**: grouped 1D convolution (``F.conv1d``), which dispatches
      to cuDNN and exploits massive channel-level parallelism on GPU.
    - **CPU/MPS path**: O(N) cumulative-sum trick (no convolution), since
      grouped conv1d is poorly optimized outside CUDA.

    For pure CPU pipelines (no torch involvement), prefer the numpy
    ``temporal_normalization`` function in this module --- scipy's
    ``uniform_filter1d`` is heavily optimized C code that beats the PyTorch
    CPU path comfortably. The torch CPU/MPS path here exists for use cases
    where the data is already a torch tensor (e.g. local Mac testing or a
    fully GPU-resident pipeline that runs on a CPU fallback).

    :param x: 2D tensor (nch, nt).
    :param fs: Sampling frequency in Hz.
    :param window_time: RAM window length in seconds; set to 0.0 for
                        1-bit normalization.
    :return: Temporally normalized tensor of the same shape and device.
    """
    if x.ndim != 2:
        raise ValueError(f"temporal_normalization_torch: expected 2D (nch x nt); got {tuple(x.shape)}.")
    if float(window_time) == 0.0:
        return torch.sign(x)

    nwin = max(int(round(fs * float(window_time))), 1)
    if nwin == 1:
        return torch.sign(x)

    x_abs = torch.abs(x)
    pad_left = nwin // 2
    pad_right = nwin - 1 - pad_left

    if x.device.type == 'cuda':
        # ---- CUDA: grouped conv1d (cuDNN) ----
        x_padded = F.pad(x_abs.unsqueeze(0), pad=(pad_left, pad_right), mode='replicate')
        kernel = torch.ones((x.shape[0], 1, nwin), dtype=x.dtype, device=x.device) / float(nwin)
        ram = F.conv1d(x_padded, kernel, groups=x.shape[0]).squeeze(0)
    else:
        # ---- CPU / MPS: O(N) cumulative sum ----
        x_padded = F.pad(
            x_abs.unsqueeze(0), pad=(pad_left, pad_right), mode='replicate'
        ).squeeze(0)
        cs = torch.empty(
            (x.shape[0], x_padded.shape[1] + 1), dtype=x.dtype, device=x.device
        )
        cs[:, 0] = 0.0
        torch.cumsum(x_padded, dim=1, out=cs[:, 1:])
        ram = (cs[:, nwin:] - cs[:, :-nwin]) / float(nwin)

    out = torch.zeros_like(x)
    mask = ram > 0
    out[mask] = x[mask] / ram[mask]
    return out


# --------------------------------------------------------------
# 3b. GPU bandpass: Tukey taper + zero-phase Butterworth, applied
#     as the analytic squared magnitude response |H(f)|^2 in the
#     rFFT domain. Used by preprocess(mode="pure_torch").
# --------------------------------------------------------------
class _BandpassCache:
    """
    Cache of constant tensors for :func:`bandpass_filter_tukey_torch`:
    Tukey windows and squared Butterworth magnitude masks, keyed by
    (shape, filter params, device, dtype). Mirrors the role of
    :class:`_WhiteningCache` for spectral whitening -- both exist to avoid
    re-allocating constant tensors inside the per-window processing loop.
    """
    __slots__ = ("window", "mask")
    def __init__(self):
        self.window: dict = {}
        self.mask: dict = {}

    def get_window(self, nt, alpha, device, dtype):
        key = (nt, float(alpha), device, dtype)
        w = self.window.get(key)
        if w is None:
            w = torch.as_tensor(signal.windows.tukey(nt, alpha=float(alpha)),
                                device=device, dtype=dtype)
            self.window[key] = w
        return w

    def get_mask(self, nt, fs, f1, f2, order, device, dtype):
        key = (nt, float(fs), f1, f2, int(order), device, dtype)
        m = self.mask.get(key)
        if m is None:
            nyq = fs / 2.0
            if f1 is not None and f2 is not None:
                btype, Wn = "bandpass", [f1 / nyq, f2 / nyq]
            elif f1 is None:
                btype, Wn = "lowpass", f2 / nyq
            else:
                btype, Wn = "highpass", f1 / nyq
            sos = signal.butter(int(order), Wn, btype=btype, output="sos")
            freqs = np.fft.rfftfreq(nt, d=1.0 / fs)
            _, H = signal.sosfreqz(sos, worN=freqs, fs=fs)
            Hmag2 = np.abs(H).astype(np.float64) ** 2      # filtfilt => |H|^2, zero phase
            m = torch.as_tensor(Hmag2, device=device, dtype=dtype)
            self.mask[key] = m
        return m

_BP_CACHE = _BandpassCache()


@torch.inference_mode()
def bandpass_filter_tukey_torch(x, fs, f1=None, f2=None, alpha=0.05, order=4):
    """
    Torch analogue of :func:`bandpass_filter_tukey`: Tukey taper followed by
    a zero-phase Butterworth applied as its analytic squared magnitude
    response ``|H(f)|^2`` in the rFFT domain. Forward-backward filtering
    (``sosfiltfilt``) has exactly zero phase and squared magnitude, so the
    mask is the analytic frequency response of the canonical CPU filter.

    **Fidelity note:** ``sosfiltfilt`` operates in the time domain with
    reflective edge padding, whereas this mask multiplication corresponds to
    *circular* convolution over the window. With the Tukey taper forcing the
    trace edges to zero, the two agree closely inside the passband but are
    not bit-identical. Use ``preprocess(mode="hybrid")`` when exact scipy
    parity matters; quantify the difference for your band with the
    preprocess experiment in ``python -m src.eval`` or
    ``tests/test_preprocess.py``.

    :param x: 2D tensor (nch, nt) on any device.
    :param fs: Sampling frequency (Hz).
    :param f1: Lower corner (Hz). None -> lowpass.
    :param f2: Upper corner (Hz). None -> highpass.
    :param alpha: Tukey window shape parameter (fraction tapered).
    :param order: Butterworth order (per direction, as in the CPU filter).
    :return: Filtered tensor, same shape/device/dtype as input.
    """
    if x.ndim != 2:
        raise ValueError("bandpass_filter_tukey_torch: data must be 2D (nch x nt).")
    if f1 is None and f2 is None:
        raise ValueError("Must specify at least one frequency (f1 or f2).")
    nt = x.shape[1]
    window = _BP_CACHE.get_window(nt, alpha, x.device, x.dtype)
    mask   = _BP_CACHE.get_mask(nt, fs, f1, f2, order, x.device, x.dtype)
    X = torch.fft.rfft(x * window, n=nt, dim=-1)
    X = X * mask                                   # broadcast (nfreq,) over channels
    return torch.fft.irfft(X, n=nt, dim=-1).to(x.dtype)


# ==============================================================
# 4. Spectral whitening (torch)
# ==============================================================
class _WhiteningCache:
    """
    A specialized cache for PyTorch tensors used during spectral whitening.

    **Purpose:**
    To maximize GPU performance, we must avoid re-allocating and re-initializing
    taper and convolution kernel tensors during every iteration of the processing
    loop. This class stores pre-computed constant tensors (kernels and cosine tapers)
    indexed by device, dtype, and shape.

    **Attributes:**
    - kernel: Stores 1D boxcar kernels for amplitude spectrum smoothing.
    - taper1: Stores cosine-squared tapers for the low-frequency roll-off.
    - taper2: Stores cosine-squared tapers for the high-frequency roll-off.
    """
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
            self.taper1[key] = torch.cos(
                torch.linspace(torch.pi / 2, torch.pi, idxf1, device=device, dtype=dtype)
            ) ** 2
        return self.taper1[key]

    def get_taper2(self, device: torch.device, dtype: torch.dtype, nfreq: int, idxf2: int) -> Optional[torch.Tensor]:
        if idxf2 >= nfreq: return None
        key = self._key(device, dtype, nfreq, idxf2)
        if key not in self.taper2:
            n = nfreq - idxf2
            self.taper2[key] = torch.cos(
                torch.linspace(torch.pi, torch.pi / 2, n, device=device, dtype=dtype)
            ) ** 2
        return self.taper2[key]

_WHITEN_CACHE = _WhiteningCache()


def spectral_whitening(rfftdata: torch.Tensor, df: float, window_freq: float, f1: float, f2: float) -> torch.Tensor:
    """
    Performs frequency-domain normalization (whitening) on a complex RFFT spectrum.

    **DSP Theory & Application:**
    Spectral whitening is used to flatten the ambient noise spectrum, ensuring
    that all frequencies within the passband contribute equally to the
    cross-correlation. This effectively sharpens the correlation peak and
    attenuates monochromatic noise (e.g., electronic hum or machinery).

    **Implementation Details:**
    - **Smoothing:** If ``window_freq > 0``, the amplitude spectrum is smoothed
      using a moving average (via high-speed 1D convolution) before division.
      This prevents the "over-whitening" of low-energy bins that may contain
      only numerical noise.
    - **Vectorization:** The function flattens all leading dimensions into a
      single batch, allowing it to process any input rank (e.g., multichannel
      segments) in a single GPU kernel launch.
    - **Band-limiting:** Frequencies outside [f1, f2] are zeroed out with
      cosine-squared tapers applied at the edges.

    :param rfftdata: Complex-valued Fourier spectrum (B, nfreq).
    :param df: Frequency resolution (Hz/sample).
    :param window_freq: Width of the smoothing window (Hz). 0.0 = full whitening.
    :param f1: Lower limit of the whitening passband (Hz).
    :param f2: Upper limit of the whitening passband (Hz).
    :return: Whitened complex spectrum of the same shape.
    """
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
    High-level wrapper to apply spectral whitening to time-domain segments in
    PyTorch.

    **Logic Flow:**
    1. Reshapes continuous channel data into discrete time segments (B, npts_seg).
    2. Computes the Forward Real FFT (RFFT) for each segment.
    3. Calls ``spectral_whitening`` to normalize the frequency content.
    4. Computes the Inverse Real FFT (IRFFT) to return to the time domain.
    5. Reconstructs the original continuous channel shape.

    **Performance Note:**
    Uses a ``chunk_nch`` loop to control peak VRAM usage. This is critical when
    processing thousands of DAS channels simultaneously on a GPU.

    :param x: 2D input tensor (nch, npts_continuous).
    :param fs_proc: Sampling rate (Hz).
    :param npts_seg: Length of the segments used for FFT (samples).
    :param window_freq_hz: Smoothing window for whitening (Hz).
    :param f1: Low-cut frequency (Hz).
    :param f2: High-cut frequency (Hz).
    :param chunk_nch: Number of channels to process per batch to manage memory.
    :return: Time-domain whitened tensor.
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

        x2  = x[ch0:ch1, :].reshape(B, npts_seg)
        X   = torch.fft.rfft(x2, n=npts_seg, dim=-1)
        Xw  = spectral_whitening(X, df, float(window_freq_hz), float(f1), float(f2))
        xw  = torch.fft.irfft(Xw, n=npts_seg, dim=-1).to(torch.float32)

        out[ch0:ch1, :] = xw.reshape(ch1 - ch0, npts_new)

    return out


# ==============================================================
# 5. Full preprocessing pipeline (selectable backend)
# ==============================================================

#: Valid values for the ``preprocess.mode`` config key / ``mode`` argument.
PREPROCESS_MODES = ("hybrid", "pure_torch", "pure_numpy")


def resolve_preprocess_mode(mode: str | None, use_gpu: bool) -> str:
    """
    Resolve and validate the preprocessing backend.

    :param mode: One of ``"hybrid"``, ``"pure_torch"``, ``"pure_numpy"``, or
                 ``None``. When ``None`` the legacy (pre-``preprocess.mode``)
                 behaviour is preserved: ``use_gpu=False`` maps to
                 ``"pure_numpy"`` and ``use_gpu=True`` maps to ``"hybrid"``
                 (the Option A architecture).
    :param use_gpu: Runtime GPU toggle (``runtime.use_gpu`` in the config).
    :return: Validated lower-case mode string.
    :raises ValueError: If ``mode`` is not one of the supported backends.
    """
    if mode is None:
        return "hybrid" if use_gpu else "pure_numpy"
    m = str(mode).strip().lower()
    if m not in PREPROCESS_MODES:
        raise ValueError(
            f"preprocess mode must be one of {PREPROCESS_MODES}; got {mode!r}."
        )
    return m


def _preprocess_pure_numpy(
    x_np: np.ndarray, fs: float, f1: float, f2: float, ram_win: float
) -> np.ndarray:
    """All-CPU reference chain: detrend -> bandpass -> median -> temporal."""
    x_np = detrend(x_np, axis=-1)
    x_np = bandpass_filter_tukey(x_np, fs, f1, f2)
    x_np -= np.median(x_np, axis=0)
    x_np = temporal_normalization(x_np, fs, float(ram_win))
    return x_np.astype(np.float32, copy=False)


def preprocess(
    x: np.ndarray | torch.Tensor,
    fs: float,
    f1: float,
    f2: float,
    ram_win: float,
    use_gpu: bool = False,
    device: torch.device | None = None,
    mode: str | None = None,
) -> np.ndarray | torch.Tensor:
    """
    Per-window ANI preprocessing chain with a selectable backend.

    Order of operations (Bensen et al. 2007 convention), identical in all
    backends:

        1. Detrend  -- remove the linear trend along the time axis
        2. Bandpass -- Tukey taper + zero-phase Butterworth
        3. Median   -- subtract the per-sample median across channels
        4. Temporal -- running-absolute-mean (RAM) or 1-bit normalization

    **Backends** (config key ``preprocess.mode``):

    - ``pure_numpy`` (legacy, CPU, benchmark): all four stages on CPU via
      scipy / numpy. This is the bit-exact reference used by the published
      baseline and by the fidelity tests/eval. Returns ``np.ndarray``.

    - ``hybrid`` (numerical fidelity on CPU **or** GPU): detrend +
      bandpass run on CPU with the same canonical scipy routines as
      ``pure_numpy`` (``scipy.signal.detrend`` + ``sosfiltfilt``), preserving
      the mathematically canonical zero-phase Butterworth response. A single
      host-to-device transfer then moves the window to ``device``, where
      median removal + temporal normalization complete in torch. Returns a
      ``torch.Tensor`` left resident on ``device`` so downstream whitening /
      cross-correlation continue without further transfers.

    - ``pure_torch`` (GPU-tailored): all four stages in torch on ``device``,
      including the rFFT-domain ``|H(f)|^2`` bandpass
      (:func:`bandpass_filter_tukey_torch`). This removes the CPU filtering
      bottleneck and the extra host round-trip, but the bandpass is a
      frequency-domain approximation of ``sosfiltfilt`` (see that function's
      fidelity note). Quantify the difference for your band with
      ``python -m src.eval`` (preprocess experiment). Returns a
      ``torch.Tensor`` on ``device``.

    Spectral whitening is performed downstream as part of the
    cross-correlation kernel (via :func:`whiten_per_segment_torch` or inside
    :class:`TorchCrossCorrelation`).

    :param x: 2D array or tensor (nch, nt).
    :param fs: Processing sampling rate (Hz). If you decimated at ingest,
               pass ``fs_raw / q`` here.
    :param f1: Lower bandpass corner (Hz).
    :param f2: Upper bandpass corner (Hz).
    :param ram_win: Running-absolute-mean window in seconds (0.0 = 1-bit).
    :param use_gpu: Selects the default torch device and, when ``mode`` is
                    ``None``, the legacy mode mapping
                    (``False -> pure_numpy``, ``True -> hybrid``).
    :param device: Target torch device for the torch stages. Defaults to
                   ``cuda`` if ``use_gpu`` and CUDA is available, otherwise
                   ``cpu`` (useful for local testing).
    :param mode: ``"hybrid"`` | ``"pure_torch"`` | ``"pure_numpy"`` | None.
    :return: ``np.ndarray`` float32 (``pure_numpy``) or ``torch.Tensor``
             float32 on ``device`` (``hybrid`` / ``pure_torch``).
    """
    mode = resolve_preprocess_mode(mode, use_gpu)
    is_tensor_in = isinstance(x, torch.Tensor)

    if mode == "pure_numpy":
        x_np = convert_to_numpy(x) if is_tensor_in else np.asarray(x)
        if x_np.ndim != 2:
            raise ValueError(f"preprocess: expected 2D (nch x nt); got {x_np.shape}.")
        return _preprocess_pure_numpy(x_np, fs, f1, f2, ram_win)

    if device is None:
        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

    if mode == "pure_torch":
        x_t = (x.to(device=device, dtype=torch.float32) if is_tensor_in
               else torch.as_tensor(np.ascontiguousarray(x), device=device,
                                    dtype=torch.float32))
        if x_t.ndim != 2:
            raise ValueError(f"preprocess: expected 2D (nch x nt); got {tuple(x_t.shape)}.")
        x_t = detrend_torch(x_t)
        x_t = bandpass_filter_tukey_torch(x_t, fs, f1, f2)
        x_t = remove_median_torch(x_t)
        return temporal_normalization_torch(x_t, fs, float(ram_win))

    # ---- mode == "hybrid" ----
    x_np = convert_to_numpy(x) if is_tensor_in else np.asarray(x)
    if x_np.ndim != 2:
        raise ValueError(f"preprocess: expected 2D (nch x nt); got {x_np.shape}.")
    x_np = detrend(x_np, axis=-1)
    x_np = bandpass_filter_tukey(x_np, fs, f1, f2)
    # Single host -> device transfer; the tensor stays on `device` through
    # the remaining torch stages and out of this function.
    x_t = torch.as_tensor(np.ascontiguousarray(x_np), device=device, dtype=torch.float32)
    x_t = remove_median_torch(x_t)
    return temporal_normalization_torch(x_t, fs, float(ram_win))

# ==============================================================
# 6. Zhang (2026) helper: choose block size
# ==============================================================
def choose_block_size_v1(
    M: int,
    *,
    fft_snap_pow2: bool = True,
    fallback: Literal["v1_2M", "v1_Mp1"] = "v1_2M",
) -> tuple[int, int]:
    """
    Calculates the theoretically optimal block size (K) and FFT length (L) for
    the Zhang v1 Cross-Correlation algorithm.

    **DSP Theory & Optimization:**
    The Zhang v1 algorithm utilizes an block-by-block approach to compute long
    cross-correlations. The computational cost is a function of the FFT length $L$
    relative to the "useful" samples $K$ produced per block.

    If $K$ is too small, the algorithm performs redundant FFTs. If $K$ is too large,
    the $\\mathcal{O}(L \\log L)$ cost of the FFT itself becomes inefficient.

    The theoretical optimum for $K$ satisfies the transcendental equation involving
    the Lambert W function (specifically the $W_{-1}$ branch). This function
    minimizes the number of operations per output sample for a given maximum
    lag $M$.

    **Algorithm Logic:**
    1. **Optimal Search:** Uses the Lambert W function to find $K^*$ that minimizes
       the cost function based on the requested lag $M$.
    2. **Power-of-Two Snapping:** If ``fft_snap_pow2`` is True, it rounds $L$ up
       to the nearest $2^n$ and adjusts $K$ accordingly. This is critical for
       maximum performance in libraries like FFTW, cuFFT, or PyTorch's MKL backend.
    3. **Constraints:** Enforces $K \\ge M + 1$ to ensure that every block contains
       at least enough context to compute a full lag window.

    :param M: Maximum lag in samples. The total correlation length is $2M + 1$.
    :param fft_snap_pow2: If True, forces the total FFT length $L$ to be a power
                           of two for optimal hardware performance.
    :param fallback: The heuristic to use if ``scipy.special.lambertw`` is
                     unavailable. "v1_2M" ($K=2M$) is generally more efficient
                     for modern CPUs.
    :return: A tuple of (K, L)
             K: Samples consumed per input block.
             L: Total length of the FFT ($K + 2M$).
    """
    if M <= 0:
        raise ValueError(f"choose_block_size_v1: M must be > 0, got {M}.")

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
# 7. Cross-correlation module
# ==============================================================
class TorchCrossCorrelation(nn.Module):
    """
    High-performance frequency-domain engine for seismic cross-correlation.

    **Architecture & Design:**
    This module implements both the standard "conventional" cross-correlation and
    the optimized Zhang (2026) v1 algorithm. It is refactored into a three-stage
    asymmetric pipeline (Compute X -> Compute Y -> Combine) to maximize throughput
    on HPC clusters.

    **The Asymmetric Advantage (VS Mode):**
    In Virtual Source (VS) interferometry, a single source channel is typically
    correlated against thousands of receiver channels. By splitting the engine,
    the source-side spectrum ($X$) is computed and cached exactly once. The
    receiver-side spectra ($Y$) are processed in batches, and PyTorch's
    broadcasting mechanism efficiently replicates the source spectrum across
    the batch during the multiplication stage. This eliminates $N-1$ redundant
    FFT operations per virtual source.

    **Algorithmic Modes:**
    1. **Conventional:** Performs full-length correlation ($2N-1$ samples).
       Ideal for short segments or when the full lag range is required.
    2. **Zhang v1:** Optimized for calculating a specific maximum lag $M$
       where $M \\ll N$. Uses an block-by-block approach to minimize
       computational complexity and memory footprint. Output length is $2M+1$.

    **Cross-correlation convention:**
        R_{xy}(m) = sum_n x(n) y(n+m)
    where ``data1 = x`` (virtual source; gets auxiliary context windows in v1)
    and ``data2 = y`` (receiver; gets zero-padded flanks in v1). Positive lag
    m > 0 corresponds to wave propagation from data1 to data2. Reversing the
    arguments time-reverses the output: R_{yx}(m) = R_{xy}(-m).

    **Public API:**
    - ``forward(data1, data2)``: Symmetric path; calculates $C_{12}$.
    - ``compute_X(x)``: Computes the source-side complex spectrum.
    - ``compute_Y(y)``: Computes the receiver-side complex spectrum.
    - ``combine(X, Y)``: Multiplies spectra, sums across segments/blocks,
      and performs the inverse FFT to produce the time-domain NCF.
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
            self._v1_K, self._v1_Lfft = choose_block_size_v1(
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
        Constructs the block-wise layout for the Zhang v1 algorithm.

        **DSP Theory:**
        To compute a correlation up to lag $M$ using FFTs of length $L$, the
        input must be partitioned such that the circular convolution artifacts
        are discarded.

        - **Source-side (build_x):** Each block of size $K$ is padded with $M$
        samples of context from both the preceding and following data
        (total length $L = K + 2M$). This ensures the source "sees" enough
        of the receiver's signal to compute all lags.
        - **Receiver-side (build_y):** The $K$ samples are centered within
        the $L$-length buffer, padded with $M$ zeros on both sides.

        :param x/y: Input tensor of shape (Batch, nseg, N).
        :return: Blocked tensor of shape (Batch*nseg, nblocks, Lfft).
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
        """
        Transforms time-domain segments into the frequency domain for correlation.

        **Implementation Details:**
        - Automatically handles FFT length $L$ selection (Power-of-two snapping).
        - Applies optional spectral whitening to the complex RFFT coefficients.
        - Operates under ``torch.inference_mode()`` for maximum C++/CUDA speed.

        **Shapes:**
        - Input: $(B, nseg, N)$ real.
        - Output (Conventional): $(B, nseg, nfreq)$ complex.
        - Output (v1): $(B, nseg, nblocks, nfreq)$ complex.

        :param x: Real-valued input waveform tensor.
        :return: Complex-valued frequency-domain spectrum.
        """
        if self.mode == "conventional":
            return self._compute_conv_spec(x)
        return self._compute_v1_X_spec(x)

    @torch.inference_mode()
    def compute_Y(self, y: torch.Tensor) -> torch.Tensor:
        """Receiver-side spectrum."""
        if self.mode == "conventional":
            return self._compute_conv_spec(y)
        return self._compute_v1_Y_spec(y)

    # ----- Combine -----
    @torch.inference_mode()
    def combine(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Aggregates source and receiver spectra into the final time-domain NCF.

        :param X: Source-side complex spectrum.
        :param Y: Receiver-side complex spectrum.
        :return: Real-valued time-domain NCF, shape $(B, 2M+1)$ or $(B, 2N-1)$.
        """
        if self.mode == "conventional":
            Rspec = (X.conj() * Y).sum(dim=1)
            conv_L = self._conv_L
            N = self._conv_N
            r = torch.fft.irfft(Rspec, n=conv_L, dim=-1)
            return torch.cat([r[:, conv_L - (N - 1):], r[:, :N]], dim=-1).to(torch.float32)

        # v1
        Rspec = (X.conj() * Y).sum(dim=(1, 2))
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
        LEGACY REFERENCE METHOD.

        Computes the Zhang v1 algorithm using a sequential loop over blocks.
        This is significantly slower than the batched/vectorized production
        methods but serves as the "Gold Standard" for mathematical verification
        of the block-by-block logic. Use only for benchmarking and unit testing.
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
        """
        Compute the cross-correlation R_{xy}(m) = sum_n x(n) y(n+m).

        Convention:
            data1 (x): virtual source channel; gets auxiliary context windows in v1.
            data2 (y): receiver channel; gets zero-padded flanks in v1.

        Positive lag m > 0 corresponds to wave propagation from data1 to data2.
        Reversing the arguments time-reverses the output (R_{yx}(m) = R_{xy}(-m)).
        """
        return self.combine(self.compute_X(data1), self.compute_Y(data2))
