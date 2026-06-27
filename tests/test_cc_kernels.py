"""
Correctness of the cross-correlation kernels in src/ani.py.

Reference convention (module docstring of TorchCrossCorrelation):
    R_xy(m) = sum_n x(n) y(n+m)
which equals scipy.signal.correlate(y, x, mode="full") indexed at m + N - 1.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.signal import correlate as scipy_correlate

from src.ani import TorchCrossCorrelation, choose_block_size_v1, spectral_whitening

N = 400      # samples per segment
M = 37       # max lag (samples) for v1
ATOL_FRAC = 1e-5  # tolerance as a fraction of max |reference|


def _ref_xcorr_full(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """R_xy(m) for m in [-(N-1), N-1] via scipy."""
    return scipy_correlate(y, x, mode="full", method="direct")


def _rand_pair(rng, n=N):
    x = rng.standard_normal(n).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)
    return x, y


# ---------------------------------------------------------------------------
# Conventional mode
# ---------------------------------------------------------------------------
def test_conventional_matches_direct_correlation(rng):
    x, y = _rand_pair(rng)
    model = TorchCrossCorrelation(mode="conventional")
    out = model(torch.from_numpy(x)[None, None, :],
                torch.from_numpy(y)[None, None, :]).numpy()[0]
    ref = _ref_xcorr_full(x.astype(np.float64), y.astype(np.float64))
    assert out.shape == (2 * N - 1,)
    tol = ATOL_FRAC * np.max(np.abs(ref))
    assert np.allclose(out, ref, atol=tol)


def test_conventional_zero_lag_is_center(rng):
    x, _ = _rand_pair(rng)
    model = TorchCrossCorrelation(mode="conventional")
    out = model(torch.from_numpy(x)[None, None, :],
                torch.from_numpy(x)[None, None, :]).numpy()[0]
    # Autocorrelation peaks at zero lag = center sample N-1.
    assert int(np.argmax(out)) == N - 1


def test_conventional_argument_reversal_flips_lags(rng):
    x, y = _rand_pair(rng)
    model = TorchCrossCorrelation(mode="conventional")
    r_xy = model(torch.from_numpy(x)[None, None, :],
                 torch.from_numpy(y)[None, None, :]).numpy()[0]
    r_yx = model(torch.from_numpy(y)[None, None, :],
                 torch.from_numpy(x)[None, None, :]).numpy()[0]
    tol = ATOL_FRAC * np.max(np.abs(r_xy))
    assert np.allclose(r_yx, r_xy[::-1], atol=tol)


def test_conventional_sums_over_segments(rng):
    nseg = 3
    xs = rng.standard_normal((nseg, N)).astype(np.float32)
    ys = rng.standard_normal((nseg, N)).astype(np.float32)
    model = TorchCrossCorrelation(mode="conventional")
    out = model(torch.from_numpy(xs)[None, :, :],
                torch.from_numpy(ys)[None, :, :]).numpy()[0]
    ref = sum(_ref_xcorr_full(xs[s].astype(np.float64), ys[s].astype(np.float64))
              for s in range(nseg))
    tol = ATOL_FRAC * np.max(np.abs(ref))
    assert np.allclose(out, ref, atol=tol)


# ---------------------------------------------------------------------------
# v1 (block-wise short-lag) mode
# ---------------------------------------------------------------------------
def test_v1_matches_conventional_within_lag_window(rng):
    x, y = _rand_pair(rng)
    conv = TorchCrossCorrelation(mode="conventional")
    v1 = TorchCrossCorrelation(mode="v1", max_lag_samples=M)

    full = conv(torch.from_numpy(x)[None, None, :],
                torch.from_numpy(y)[None, None, :]).numpy()[0]
    short = v1(torch.from_numpy(x)[None, None, :],
               torch.from_numpy(y)[None, None, :]).numpy()[0]

    assert short.shape == (2 * M + 1,)
    ref = full[(N - 1) - M:(N - 1) + M + 1]
    tol = ATOL_FRAC * np.max(np.abs(ref))
    assert np.allclose(short, ref, atol=tol), (
        "v1 must reproduce the conventional correlator inside |m| <= M "
        f"(max abs diff {np.max(np.abs(short - ref)):.3e})"
    )


def test_v1_batched_matches_legacy_python_fft(rng):
    """The vectorized production path vs the sequential 'gold standard'."""
    B, nseg = 3, 2
    xs = torch.from_numpy(rng.standard_normal((B, nseg, N)).astype(np.float32))
    ys = torch.from_numpy(rng.standard_normal((B, nseg, N)).astype(np.float32))
    v1 = TorchCrossCorrelation(mode="v1", max_lag_samples=M)

    fast = v1._forward_v1_batched(xs, ys).numpy()
    slow = v1._forward_v1_python_fft(xs.clone(), ys.clone()).numpy()
    tol = ATOL_FRAC * np.max(np.abs(slow))
    assert np.allclose(fast, slow, atol=tol)


def test_asymmetric_api_matches_forward(rng):
    """compute_X / compute_Y / combine (VS path) == forward (symmetric path)."""
    for mode, kwargs in (("conventional", {}), ("v1", {"max_lag_samples": M})):
        model = TorchCrossCorrelation(mode=mode, **kwargs)
        x = torch.from_numpy(rng.standard_normal((2, 2, N)).astype(np.float32))
        y = torch.from_numpy(rng.standard_normal((2, 2, N)).astype(np.float32))
        a = model(x, y).numpy()
        b = model.combine(model.compute_X(x), model.compute_Y(y)).numpy()
        assert np.allclose(a, b, atol=1e-6 * max(1.0, np.max(np.abs(a))))


def test_v1_whitening_path_is_finite(rng):
    fs = 100.0
    model = TorchCrossCorrelation(
        mode="v1", max_lag_samples=M,
        is_spectral_whitening=True,
        whitening_params=(fs, 0.0, 2.0, 10.0),
    )
    x = torch.from_numpy(rng.standard_normal((2, 1, N)).astype(np.float32))
    y = torch.from_numpy(rng.standard_normal((2, 1, N)).astype(np.float32))
    out = model(x, y).numpy()
    assert out.shape == (2, 2 * M + 1)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Block sizing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m", [10, 100, 500, 4000])
def test_choose_block_size_constraints(m):
    K, L = choose_block_size_v1(m, fft_snap_pow2=True)
    assert K >= m + 1, "every block must contain a full lag window"
    assert L == K + 2 * m
    assert L & (L - 1) == 0, "snapped FFT length must be a power of two"

    K2, L2 = choose_block_size_v1(m, fft_snap_pow2=False)
    assert L2 == K2 + 2 * m
    assert K2 >= m + 1


def test_choose_block_size_tracks_lambert_optimum():
    """K* = 2M(-W_-1(-1/(4eM)) - 1); unsnapped K should be its round-off."""
    lambertw = pytest.importorskip("scipy.special").lambertw
    m = 500
    z = -1.0 / (4.0 * math.e * m)
    k_star = 2.0 * m * (-lambertw(z, k=-1).real - 1.0)
    K, _ = choose_block_size_v1(m, fft_snap_pow2=False)
    assert abs(K - k_star) <= 1.0


def test_choose_block_size_rejects_nonpositive():
    with pytest.raises(ValueError):
        choose_block_size_v1(0)


# ---------------------------------------------------------------------------
# Spectral whitening
# ---------------------------------------------------------------------------
def test_spectral_whitening_flattens_passband(rng):
    fs, n = 100.0, 1024
    t = np.arange(n) / fs
    sig = (np.sin(2 * np.pi * 5 * t) + 0.05 * np.sin(2 * np.pi * 15 * t))
    X = torch.fft.rfft(torch.from_numpy(sig.astype(np.float32))[None, :], dim=-1)
    df = fs / n
    Xw = spectral_whitening(X, df, window_freq=0.0, f1=2.0, f2=20.0)
    amp = torch.abs(Xw).numpy()[0]
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    band = (freqs > 3.0) & (freqs < 19.0)
    # Full whitening => |X| ~ 1 inside the passband regardless of input power.
    assert np.allclose(amp[band], 1.0, atol=1e-3)
    # Outside the passband (beyond the cosine tapers) energy is zeroed.
    assert np.all(amp[freqs < 1.0] <= 1.0 + 1e-3)
