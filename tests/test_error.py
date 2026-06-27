"""Unit tests for the fidelity metrics in src/error.py (numpy-only)."""
from __future__ import annotations

import numpy as np
import pytest

from src.error import (
    cosine_similarity_per_trace,
    max_abs_error,
    pick_diff,
    rel_frobenius,
    spectral_compare,
)


def test_rel_frobenius_identical_is_zero(rng):
    a = rng.standard_normal((5, 20))
    assert rel_frobenius(a, a) == pytest.approx(0.0, abs=1e-12)


def test_rel_frobenius_known_value():
    R = np.eye(3)
    V = np.eye(3) * 1.1
    # ||V - R||_F / ||R||_F = (0.1*sqrt(3)) / sqrt(3) = 0.1
    assert rel_frobenius(V, R) == pytest.approx(0.1, rel=1e-9)


def test_rel_frobenius_shape_mismatch():
    with pytest.raises(ValueError):
        rel_frobenius(np.zeros((2, 2)), np.zeros((3, 2)))


def test_max_abs_error_known():
    a = np.array([[0.0, 1.0], [2.0, 3.0]])
    b = np.array([[0.0, 1.5], [2.0, 2.0]])
    assert max_abs_error(a, b) == pytest.approx(1.0)


def test_cosine_similarity_cases(rng):
    a = rng.standard_normal((3, 50))
    assert np.allclose(cosine_similarity_per_trace(a, a), 1.0, atol=1e-9)

    v = np.array([[1.0, 0.0], [0.0, 2.0]])
    r = np.array([[0.0, 1.0], [0.0, 2.0]])
    cos = cosine_similarity_per_trace(v, r)
    assert cos[0] == pytest.approx(0.0, abs=1e-9)   # orthogonal
    assert cos[1] == pytest.approx(1.0, rel=1e-9)   # parallel

    z = np.zeros((1, 4))
    assert cosine_similarity_per_trace(z, z)[0] == 1.0  # both dead -> defined as 1


def test_spectral_compare_identical(rng):
    a = rng.standard_normal((4, 256))
    s = spectral_compare(a, a, dt=0.01, f1=2.0, f2=20.0)
    assert s.rel_spec_err_band == pytest.approx(0.0, abs=1e-12)
    assert s.leak_ratio_ref == pytest.approx(s.leak_ratio_test)


def test_spectral_compare_validation(rng):
    a = rng.standard_normal((2, 64))
    with pytest.raises(ValueError):
        spectral_compare(a, a, dt=-1.0, f1=1.0, f2=2.0)
    with pytest.raises(ValueError):
        spectral_compare(a, a, dt=0.01, f1=5.0, f2=2.0)


def test_pick_diff_stats():
    ref = np.array([100.0, 200.0, 300.0, np.nan])
    test = np.array([101.0, 198.0, 300.0, 400.0])
    pe = pick_diff(ref, test)
    assert pe.n_freq == 3                      # NaNs excluded
    assert pe.max_abs_dc == pytest.approx(2.0)
    assert pe.median_abs_dc == pytest.approx(1.0)


def test_pick_diff_all_nan():
    pe = pick_diff(np.array([np.nan]), np.array([np.nan]))
    assert pe.n_freq == 0
    assert np.isnan(pe.median_abs_dc)
