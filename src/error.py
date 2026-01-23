"""
:module: src/error.py
:auth: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Error evaluation utilities for conventional vs v1 modes.
"""
from __future__ import annotations

import logging
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from typing import Any, Optional, Tuple, Union, Dict

ArrayLike = np.ndarray
PathLike = Union[str, Path]

logger = logging.getLogger(__name__)

# =====================================================
# I/O helpers
# =====================================================
def load_npy(path: PathLike, *, mmap: bool = False) -> np.ndarray:
    """
    Load a ``.npy`` file as a NumPy array (optionally as a memmap).

    :param path: Path to the ``.npy`` file.
    :type path: PathLike
    :param mmap: If True, load with ``mmap_mode="r"`` (read-only memmap).
    :type mmap: bool

    :return: Loaded array (in-memory ndarray or memmap).
    :rtype: np.ndarray
    """
    p = Path(path).expanduser().resolve()
    return np.load(p, mmap_mode="r" if mmap else None)

def as_array(x: Union[ArrayLike, PathLike], *, mmap: bool = False) -> np.ndarray:
    """
    Convert input into a NumPy array.

    If ``x`` is path-like, the file is loaded via :func:`load_npy`.
    If ``x`` is already an ndarray, it is returned as-is.

    :param x: Array or path to a ``.npy`` file.
    :type x: Union[np.ndarray, PathLike]
    :param mmap: Only used if ``x`` is a path; loads with memmap when True.
    :type mmap: bool

    :return: NumPy array (possibly a memmap if ``mmap=True``).
    :rtype: np.ndarray
    """
    if isinstance(x, (str, Path)):
        return load_npy(x, mmap=mmap)
    if isinstance(x, np.ndarray):
        return x
    raise TypeError(f"Expected np.ndarray or path-like; got {type(x)}")

# =====================================================
# 1. Relative Frobenius Norm
# =====================================================
def rel_frobenius(V: np.ndarray, R: np.ndarray, *, eps: float = 1e-15) -> float:
    """
    Compute relative Frobenius norm error between two matrices.

    Defined as:
        ``||V - R||_F / (||R||_F + eps)``

    :param V: Test matrix (e.g., v1 result), shape (n_traces, n_samples).
    :type V: np.ndarray
    :param R: Reference matrix (e.g., conventional result), same shape as ``V``.
    :type R: np.ndarray
    :param eps: Small constant to avoid division by zero.
    :type eps: float

    :return: Relative Frobenius norm error.
    :rtype: float
    """
    if V.shape != R.shape:
        raise ValueError(f"rel_frobenius: shape mismatch V{V.shape} vs R{R.shape}")

    diff = V - R
    fro_R = float(np.linalg.norm(R, ord="fro"))
    fro_diff = float(np.linalg.norm(diff, ord="fro"))
    return fro_diff / (fro_R + float(eps))

# =====================================================
# 2. Maximum Absolute Error
# =====================================================
def max_abs_error(V: np.ndarray, R: np.ndarray) -> float:
    """
    Compute maximum absolute element-wise error.

    Defined as:
        ``max(|V - R|)``

    :param V: Test matrix.
    :type V: np.ndarray
    :param R: Reference matrix, same shape as ``V``.
    :type R: np.ndarray

    :return: Maximum absolute error.
    :rtype: float
    """
    if V.shape != R.shape:
        raise ValueError(f"max_abs_error: shape mismatch V{V.shape} vs R{R.shape}")
    return float(np.max(np.abs(V - R)))

# =====================================================
# 3. Cosine Similarity (per trace)
# =====================================================
def cosine_similarity_per_trace(
    V: np.ndarray,
    R: np.ndarray,
    *,
    axis: int = 1,
    eps: float = 1e-15,
) -> np.ndarray:
    """
    Compute cosine similarity between corresponding traces.

    For each trace:
        ``cos(theta) = (v · r) / (||v|| ||r||)``

    If both traces are (near) zero vectors, cosine similarity is defined as 1.0.

    :param V: Test matrix, shape (n_traces, n_samples).
    :type V: np.ndarray
    :param R: Reference matrix, same shape as ``V``.
    :type R: np.ndarray
    :param axis: Axis along which each trace is defined (default: 1).
    :type axis: int
    :param eps: Small stabilizer to avoid division by zero.
    :type eps: float

    :return: Cosine similarity per trace, shape (n_traces,).
    :rtype: np.ndarray
    """
    if V.shape != R.shape:
        raise ValueError(f"cosine_similarity_per_trace: shape mismatch V{V.shape} vs R{R.shape}")

    dot = np.sum(V * R, axis=axis)
    nr = np.linalg.norm(R, axis=axis)
    nv = np.linalg.norm(V, axis=axis)

    cos = dot / (nr * nv + eps)

    # If both are zero vectors, define similarity as 1
    zero_mask = (nr < eps) & (nv < eps)
    cos = np.asarray(cos, dtype=np.float64)
    cos[zero_mask] = 1.0
    return cos

# =====================================================
# 4. Spectral comparison (band error + leakage ratios)
# =====================================================
@dataclass(frozen=True)
class SpectralSummary:
    """Spectral comparison summary."""
    rel_spec_err_band: float
    leak_ratio_ref: float
    leak_ratio_test: float


def spectral_compare(
    V: np.ndarray,
    R: np.ndarray,
    *,
    dt: float,
    f1: float,
    f2: float,
    eps: float = 1e-15,
) -> SpectralSummary:
    """
    Compare spectra of two NCF matrices in the frequency domain.

    Metrics:
    - Band-limited relative spectral error within [f1, f2]
    - Leakage ratio for each input: (out-of-band energy) / (in-band energy)

    :param V: Test NCF matrix, shape (n_traces, n_samples).
    :type V: np.ndarray
    :param R: Reference NCF matrix, same shape as ``V``.
    :type R: np.ndarray
    :param dt: Sampling interval of the NCF time axis (seconds).
    :type dt: float
    :param f1: Low cutoff for the in-band region (Hz).
    :type f1: float
    :param f2: High cutoff for the in-band region (Hz).
    :type f2: float
    :param eps: Small stabilizer to avoid division by zero.
    :type eps: float

    :return: Spectral comparison summary.
    :rtype: SpectralSummary
    """
    if V.shape != R.shape:
        raise ValueError(f"spectral_compare: shape mismatch V{V.shape} vs R{R.shape}")
    if dt <= 0:
        raise ValueError(f"spectral_compare: dt must be > 0; got {dt}")
    if not (0 <= f1 < f2):
        raise ValueError(f"spectral_compare: require 0 <= f1 < f2; got f1={f1}, f2={f2}")

    n_tr, n_samp = R.shape
    freq = np.fft.rfftfreq(n_samp, d=float(dt))

    AR = np.fft.rfft(R, axis=1)
    AV = np.fft.rfft(V, axis=1)

    band = (freq >= f1) & (freq <= f2)
    oob = ~band

    num = np.linalg.norm((AV - AR)[:, band], ord="fro")
    den = np.linalg.norm(AR[:, band], ord="fro") + eps
    rel_spec_err_band = float(num / den)

    leak_R = float(np.linalg.norm(AR[:, oob], ord="fro") / (np.linalg.norm(AR[:, band], ord="fro") + eps))
    leak_V = float(np.linalg.norm(AV[:, oob], ord="fro") / (np.linalg.norm(AV[:, band], ord="fro") + eps))

    return SpectralSummary(rel_spec_err_band=rel_spec_err_band, leak_ratio_ref=leak_R, leak_ratio_test=leak_V)

# =====================================================
# 5. Structural Similarity Index (SSIM) for dispersion panels
# =====================================================
def normalize_dispersion(
    D: np.ndarray,
    *,
    p: float = 99.5,
    eps: float = 1e-15,
) -> np.ndarray:
    """
    Robustly normalize a dispersion panel using percentile clipping.

    The scale is computed as ``percentile(|D|, p)`` and the output is:
        ``clip(D, -scale, +scale) / scale``

    This produces values typically in [-1, 1] (exactly within [-1, 1] after clipping).

    :param D: Dispersion panel (e.g., f–v image), shape (n_freq, n_vel) or similar.
    :type D: np.ndarray
    :param p: Percentile for ``|D|`` clipping (default: 99.5).
    :type p: float
    :param eps: Stabilizer if the percentile scale is near zero.
    :type eps: float

    :return: Normalized dispersion panel.
    :rtype: np.ndarray
    """
    scale = float(np.percentile(np.abs(D), p))
    scale = max(scale, eps)
    return np.clip(D, -scale, scale) / scale


def ssim_index(
    D_ref: np.ndarray,
    D_test: np.ndarray,
    *,
    p: float = 99.5,
    gaussian_weights: bool = True,
    sigma: float = 1.5,
) -> float:
    """
    Compute SSIM between two dispersion images.

    Both inputs are robustly normalized (percentile clipping) before SSIM,
    so SSIM is computed on comparable intensity scales.

    :param D_ref: Reference dispersion image (f–v panel).
    :type D_ref: np.ndarray
    :param D_test: Test dispersion image, same shape as ``D_ref``.
    :type D_test: np.ndarray
    :param p: Percentile used for normalization (default: 99.5).
    :type p: float
    :param gaussian_weights: If True, apply Gaussian weighting in SSIM.
    :type gaussian_weights: bool
    :param sigma: Gaussian sigma used when ``gaussian_weights=True``.
    :type sigma: float

    :return: SSIM value (clipped to [-1, 1]).
    :rtype: float
    """
    if D_ref.shape != D_test.shape:
        raise ValueError(f"ssim_index: shape mismatch ref{D_ref.shape} vs test{D_test.shape}")

    D_ref_n = normalize_dispersion(D_ref, p=p)
    D_test_n = normalize_dispersion(D_test, p=p)

    val = ssim(
        D_ref_n,
        D_test_n,
        data_range=2.0,
        gaussian_weights=gaussian_weights,
        sigma=float(sigma),
        use_sample_covariance=False,
    )
    return float(np.clip(val, -1.0, 1.0))

# =====================================================
# 6. Phase-velocity curve comparison
# =====================================================
@dataclass(frozen=True)
class PickError:
    """Phase-velocity pick comparison summary."""
    n_freq: int
    median_abs_dc: float
    rms_abs_dc: float
    p95_abs_dc: float
    max_abs_dc: float
    median_rel: float
    p95_rel: float


def pick_diff(
    P_ref: np.ndarray,
    P_test: np.ndarray,
    *,
    eps: float = 1e-15,
) -> PickError:
    """
    Compare two phase-velocity pick curves c(f).

    Differences are defined as:
        ``Δc(f) = c_test(f) - c_ref(f)``

    Reported metrics include absolute statistics on |Δc| and relative statistics:
        ``|Δc| / max(|c_ref|, eps)``

    :param P_ref: Reference phase-velocity curve, shape (n_freq,).
    :type P_ref: np.ndarray
    :param P_test: Test phase-velocity curve, same shape as ``P_ref``.
    :type P_test: np.ndarray
    :param eps: Stabilizer for relative error when ``c_ref`` is near zero.
    :type eps: float

    :return: Pick error summary statistics.
    :rtype: PickError
    """
    P_ref = np.asarray(P_ref, dtype=np.float64)
    P_test = np.asarray(P_test, dtype=np.float64)

    if P_ref.shape != P_test.shape:
        raise ValueError(f"pick_diff: shape mismatch ref{P_ref.shape} vs test{P_test.shape}")

    mask = np.isfinite(P_ref) & np.isfinite(P_test)
    n = int(np.sum(mask))
    if n == 0:
        return PickError(
            n_freq=0,
            median_abs_dc=float("nan"),
            rms_abs_dc=float("nan"),
            p95_abs_dc=float("nan"),
            max_abs_dc=float("nan"),
            median_rel=float("nan"),
            p95_rel=float("nan"),
        )

    dc = P_test[mask] - P_ref[mask]
    abs_dc = np.abs(dc)

    denom = np.maximum(np.abs(P_ref[mask]), eps)
    rel = abs_dc / denom

    return PickError(
        n_freq=n,
        median_abs_dc=float(np.median(abs_dc)),
        rms_abs_dc=float(np.sqrt(np.mean(dc**2))),
        p95_abs_dc=float(np.percentile(abs_dc, 95)),
        max_abs_dc=float(np.max(abs_dc)),
        median_rel=float(np.median(rel)),
        p95_rel=float(np.percentile(rel, 95)),
    )