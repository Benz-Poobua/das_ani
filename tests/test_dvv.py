"""Tests for src/dvv.py (stretching dv/v, ND bandpass, aggregation)."""
from __future__ import annotations

import numpy as np
import pytest

from src.dvv import (
    _bandpass_nd,
    aggregate_dvv_results,
    compute_dvv,
    compute_dvv_single_pair,
)

FS = 100.0
DT = 1.0 / FS


def _coda_trace(nt: int = 2000) -> np.ndarray:
    """Gaussian-windowed multi-tone 'coda' centered in the trace (zero edges)."""
    t = np.arange(nt) * DT
    t0 = t[nt // 2]
    env = np.exp(-((t - t0) ** 2) / (2 * 2.0 ** 2))
    return env * (np.sin(2 * np.pi * 4.0 * t) + 0.5 * np.sin(2 * np.pi * 6.5 * t + 0.7))


# ---------------------------------------------------------------------------
# ND bandpass
# ---------------------------------------------------------------------------
def test_bandpass_nd_preserves_shape(rng):
    x = rng.standard_normal((2, 3, 4, 256))
    out = _bandpass_nd(x, DT, 2.0, 10.0)
    assert out.shape == x.shape


def test_bandpass_nd_attenuates_out_of_band():
    t = np.arange(4096) * DT
    inband = np.sin(2 * np.pi * 5.0 * t)
    outband = np.sin(2 * np.pi * 30.0 * t)
    y_in = _bandpass_nd(inband, DT, 2.0, 10.0)
    y_out = _bandpass_nd(outband, DT, 2.0, 10.0)
    assert np.std(y_in) > 0.5 * np.std(inband)
    assert np.std(y_out) < 0.01 * np.std(outband)


def test_bandpass_nd_validates_corners(rng):
    x = rng.standard_normal(128)
    with pytest.raises(ValueError):
        _bandpass_nd(x, DT, -1.0, 10.0)
    with pytest.raises(ValueError):
        _bandpass_nd(x, DT, 10.0, 2.0)
    with pytest.raises(ValueError):
        _bandpass_nd(x, DT, 2.0, 60.0)  # above Nyquist


# ---------------------------------------------------------------------------
# Stretching dv/v
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("eps_true", [-0.02, 0.0, 0.012])
def test_stretching_recovers_known_factor(eps_true):
    """
    Monitor m(t) = ref(t * (1 + eps_true)). The scan stretches the reference
    by candidate eps and correlates against the monitor, so the best
    candidate equals eps_true and the function returns eps_true * 100
    (raw stretching factor in percent; dv/v = -eps).
    """
    nt = 2000
    t = np.arange(nt) * DT
    ref = _coda_trace(nt)
    ref_i = np.interp(t * (1 + eps_true), t, ref, left=0.0, right=0.0)
    monitors = np.vstack([ref_i, ref_i])

    dv_range, n_steps = 0.03, 121  # grid step = 0.05%
    eps_pct, cc = compute_dvv_single_pair(
        ref, monitors, t, window=(t[0], t[-1]),
        dv_range=dv_range, n_steps=n_steps,
    )
    grid_step_pct = 100.0 * 2 * dv_range / (n_steps - 1)
    assert np.all(cc > 0.95)
    assert np.allclose(eps_pct, eps_true * 100.0, atol=grid_step_pct)


def test_stretching_empty_window_returns_nan():
    nt = 100
    t = np.arange(nt) * DT
    ref = _coda_trace(nt)
    eps, cc = compute_dvv_single_pair(ref, ref[None, :], t, window=(99.0, 100.0))
    assert np.all(np.isnan(eps))
    assert np.all(cc == 0)


def test_compute_dvv_shapes_and_qc(rng):
    n_hours, n_src, n_rec, nt = 3, 2, 2, 1024
    base = _coda_trace(nt)
    data = np.broadcast_to(base, (n_hours, n_src, n_rec, nt)).copy()
    data += 0.01 * rng.standard_normal(data.shape)

    res = compute_dvv(data, DT, freq_bands=[(2.0, 10.0)], window=None, n_steps=21)
    band = res["2.0-10.0"]
    for key in ("dvv_raw", "dvv_qc", "cc"):
        assert band[key].shape == (n_hours, n_src, n_rec)
    # identical traces -> high correlation, near-zero stretch (within one
    # grid step of the eps scan: 100 * 2*0.05/20 = 0.5%)
    assert np.nanmean(band["cc"]) > 0.9
    assert np.nanmax(np.abs(band["dvv_raw"])) <= 1.0


def test_aggregate_dvv_results_masks_low_cc():
    grid = np.ones((2, 1, 2))
    cc = np.array([[[0.9, 0.2]], [[0.9, 0.9]]])
    raw = {"band": {"dvv_raw": grid, "dvv_qc": grid, "cc": cc}}
    out = aggregate_dvv_results(raw, cc_threshold=0.6, dvv_limit=5.0)["band"]
    assert out.loc[0, "n_pairs"] == 1   # low-cc pair dropped in hour 0
    assert out.loc[1, "n_pairs"] == 2
