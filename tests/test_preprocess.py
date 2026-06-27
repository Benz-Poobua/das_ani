"""
Numerical parity of the three preprocessing backends in src/ani.py.

Reference chain ("ground truth"): mode="pure_numpy" (scipy detrend +
sosfiltfilt bandpass + np.median + scipy RAM/1-bit).

Expectations:
- hybrid     : identical to pure_numpy within float32 round-off (the scipy
               stages are literally shared; the torch median / temporal-norm
               stages are numpy-parity by construction).
- pure_torch : same stage order, but the bandpass is the analytic |H(f)|^2
               rFFT mask instead of time-domain sosfiltfilt. Differences are
               small but non-zero; the bounds asserted here are deliberately
               loose upper bounds. Tighten them after calibrating on your
               data with `python -m src.eval` (preprocess experiment).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.signal import detrend as scipy_detrend

from src.ani import (
    PREPROCESS_MODES,
    bandpass_filter_tukey,
    bandpass_filter_tukey_torch,
    detrend_torch,
    preprocess,
    remove_median_torch,
    resolve_preprocess_mode,
    temporal_normalization,
    temporal_normalization_torch,
)
from src.error import rel_frobenius, cosine_similarity_per_trace

FS = 100.0
F1, F2 = 2.0, 10.0
CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# Mode resolution
# ---------------------------------------------------------------------------
def test_resolve_preprocess_mode_legacy_mapping():
    assert resolve_preprocess_mode(None, use_gpu=False) == "pure_numpy"
    assert resolve_preprocess_mode(None, use_gpu=True) == "hybrid"


@pytest.mark.parametrize("mode", PREPROCESS_MODES)
def test_resolve_preprocess_mode_passthrough(mode):
    assert resolve_preprocess_mode(mode, use_gpu=False) == mode
    assert resolve_preprocess_mode(mode.upper(), use_gpu=True) == mode  # case-insensitive


def test_resolve_preprocess_mode_rejects_unknown():
    with pytest.raises(ValueError, match="preprocess mode"):
        resolve_preprocess_mode("hybird", use_gpu=False)  # typo must fail fast


# ---------------------------------------------------------------------------
# Stage-level parity (torch vs numpy/scipy)
# ---------------------------------------------------------------------------
def test_detrend_torch_matches_scipy(das_panel):
    ref = scipy_detrend(das_panel, axis=-1)
    out = detrend_torch(torch.from_numpy(das_panel)).numpy()
    scale = float(np.max(np.abs(ref))) or 1.0
    assert np.allclose(out, ref, atol=1e-8 * scale, rtol=1e-6)


@pytest.mark.parametrize("nch", [11, 12])  # odd AND even channel counts
def test_remove_median_torch_matches_numpy(rng, nch):
    x = rng.standard_normal((nch, 257))
    ref = x - np.median(x, axis=0)
    out = remove_median_torch(torch.from_numpy(x)).numpy()
    # torch.quantile(0.5) averages the two middle values for even nch,
    # exactly like np.median (torch.median would NOT).
    assert np.allclose(out, ref, atol=1e-12)


@pytest.mark.parametrize("window_time", [0.0, 0.25, 0.5])
def test_temporal_normalization_torch_matches_scipy(das_panel, window_time):
    x32 = das_panel.astype(np.float32)
    ref = temporal_normalization(x32, FS, window_time)
    out = temporal_normalization_torch(torch.from_numpy(x32), FS, window_time).numpy()
    # The torch CPU path computes the running mean via a float32 cumulative
    # sum, whose round-off grows with trace length; tolerance reflects that.
    assert np.allclose(out, ref, atol=5e-3), (
        f"RAM window {window_time}s: torch path deviates from scipy "
        f"(max abs diff {np.max(np.abs(out - ref)):.3e})"
    )


def test_bandpass_torch_close_to_scipy(das_panel):
    """|H(f)|^2 rFFT mask vs sosfiltfilt -- loose bound, see module docstring."""
    x = scipy_detrend(das_panel, axis=-1)  # remove trend so edges are tame
    ref = bandpass_filter_tukey(x, FS, F1, F2)
    out = bandpass_filter_tukey_torch(
        torch.from_numpy(x).to(torch.float32), FS, F1, F2
    ).numpy()
    assert rel_frobenius(out, ref.astype(np.float32)) < 5e-2
    cos = cosine_similarity_per_trace(out.astype(np.float64), ref.astype(np.float64))
    assert float(np.min(cos)) > 0.99


# ---------------------------------------------------------------------------
# Full-chain parity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ram_win", [0.0, 0.5])
def test_hybrid_matches_pure_numpy(das_panel, ram_win):
    ref = preprocess(das_panel, FS, F1, F2, ram_win, mode="pure_numpy")
    out_t = preprocess(das_panel, FS, F1, F2, ram_win, device=CPU, mode="hybrid")
    assert isinstance(ref, np.ndarray)
    assert isinstance(out_t, torch.Tensor)
    out = out_t.numpy()
    assert out.shape == ref.shape
    if ram_win == 0.0:
        # 1-bit output: signs may only disagree where the input is ~0.
        agree = np.mean(out == ref)
        assert agree > 0.999
    else:
        # Sub-1e-4 in practice; the residual is the float32 cumsum round-off
        # of the torch CPU running-mean (the scipy stages are shared).
        assert rel_frobenius(out, ref) < 1e-4


def test_pure_torch_close_to_pure_numpy(das_panel):
    """Loose upper bound on the GPU-tailored chain (RAM normalization)."""
    ram_win = 0.5
    ref = preprocess(das_panel, FS, F1, F2, ram_win, mode="pure_numpy")
    out = preprocess(das_panel, FS, F1, F2, ram_win, device=CPU, mode="pure_torch").numpy()
    assert out.shape == ref.shape
    assert np.all(np.isfinite(out))
    assert rel_frobenius(out, ref) < 0.1
    cos = cosine_similarity_per_trace(out.astype(np.float64), ref.astype(np.float64))
    assert float(np.mean(cos)) > 0.99


def test_preprocess_return_types_and_devices(das_panel):
    out_np = preprocess(das_panel, FS, F1, F2, 0.0, mode="pure_numpy")
    assert isinstance(out_np, np.ndarray) and out_np.dtype == np.float32

    for mode in ("hybrid", "pure_torch"):
        out = preprocess(das_panel, FS, F1, F2, 0.0, device=CPU, mode=mode)
        assert isinstance(out, torch.Tensor)
        assert out.dtype == torch.float32
        assert out.device.type == "cpu"


def test_preprocess_accepts_tensor_input(das_panel):
    x_t = torch.from_numpy(das_panel)
    a = preprocess(das_panel, FS, F1, F2, 0.5, device=CPU, mode="hybrid").numpy()
    b = preprocess(x_t, FS, F1, F2, 0.5, device=CPU, mode="hybrid").numpy()
    assert np.allclose(a, b, atol=1e-6)


def test_preprocess_rejects_non_2d():
    with pytest.raises(ValueError, match="expected 2D"):
        preprocess(np.zeros(100), FS, F1, F2, 0.0, mode="pure_numpy")
    with pytest.raises(ValueError, match="expected 2D"):
        preprocess(np.zeros((2, 3, 4)), FS, F1, F2, 0.0, device=CPU, mode="pure_torch")
