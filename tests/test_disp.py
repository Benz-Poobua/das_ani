"""Tests for the dispersion-based dv/v + depth-mapping helpers in src/disp.py."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from src.disp import (
    _extract_datetime,
    approximate_depth_profile,
    compute_dvv_from_dispersion,
)


# ---------------------------------------------------------------------------
# dv/v between dispersion curves
# ---------------------------------------------------------------------------
def test_dvv_from_dispersion_uniform_change():
    f = np.linspace(2.0, 8.0, 25)
    v_ref = 400.0 - 20.0 * f          # dispersive reference
    v_cur = v_ref * 1.02              # +2% everywhere
    freqs, dvv = compute_dvv_from_dispersion(f, v_ref, f, v_cur)
    assert np.allclose(freqs, f)
    assert np.allclose(dvv, 0.02, atol=1e-12)


def test_dvv_from_dispersion_identical_is_zero():
    f = np.linspace(1.0, 10.0, 19)
    v = 300.0 + 5.0 * f
    _, dvv = compute_dvv_from_dispersion(f, v, f, v)
    assert np.allclose(dvv, 0.0, atol=1e-14)


def test_dvv_from_dispersion_masks_non_overlap():
    """Current curve covers a narrower band; outside it -> dropped, not NaN."""
    f_ref = np.linspace(2.0, 8.0, 25)
    v_ref = np.full_like(f_ref, 350.0)
    f_cur = np.linspace(4.0, 6.0, 9)
    v_cur = np.full_like(f_cur, 360.0)
    freqs, dvv = compute_dvv_from_dispersion(f_ref, v_ref, f_cur, v_cur)
    assert freqs.min() >= 4.0 and freqs.max() <= 6.0
    assert np.all(np.isfinite(dvv))
    assert np.allclose(dvv, 10.0 / 350.0, atol=1e-12)


def test_dvv_from_dispersion_interpolates_between_picks():
    f_ref = np.array([2.0, 4.0, 6.0])
    v_ref = np.array([300.0, 300.0, 300.0])
    f_cur = np.array([1.0, 7.0])
    v_cur = np.array([330.0, 330.0])  # constant +10% across the band
    freqs, dvv = compute_dvv_from_dispersion(f_ref, v_ref, f_cur, v_cur)
    assert np.allclose(dvv, 0.1, atol=1e-12)
    assert freqs.size == 3


# ---------------------------------------------------------------------------
# Depth mapping
# ---------------------------------------------------------------------------
def test_depth_profile_rule_of_thumb():
    freqs = np.array([1.0, 2.0, 5.0])
    vels = np.array([300.0, 300.0, 300.0])
    depths = approximate_depth_profile(freqs, vels, factor=1.0 / 3.0)
    # depth = (v / f) / 3
    assert np.allclose(depths, np.array([100.0, 50.0, 20.0]), rtol=1e-3)


def test_depth_profile_zero_frequency_guard():
    depths = approximate_depth_profile(np.array([0.0, 1.0]), np.array([300.0, 300.0]))
    assert np.all(np.isfinite(depths))
    assert depths[0] > depths[1]  # f -> 0 maps to (very) deep, not inf/NaN


# ---------------------------------------------------------------------------
# Filename timestamp parsing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,expected", [
    ("20250722_cc_080_7d_v1.npy", datetime(2025, 7, 22)),
    ("20250722_025000_bridge_cc_080_1h_v1.npy", datetime(2025, 7, 22, 2, 50)),
])
def test_extract_datetime(name, expected):
    assert _extract_datetime(name) == expected


def test_extract_datetime_rejects_garbage():
    with pytest.raises(ValueError):
        _extract_datetime("no_timestamp_here.npy")
