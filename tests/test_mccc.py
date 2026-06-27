"""Tests for src/mccc.py (multi-channel cross-correlation)."""
from __future__ import annotations

import numpy as np
import torch

from src.mccc import compute_mccc_delays, torch_xcorr_1d_vs_nd

DT = 0.01


def _ricker(nt: int, fs: float, t0: float, f0: float = 8.0) -> np.ndarray:
    t = np.arange(nt) / fs - t0
    a = (np.pi * f0 * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def test_xcorr_zero_lag_at_center(rng):
    x = rng.standard_normal((4, 256)).astype(np.float32)
    xt = torch.from_numpy(x)
    cc = torch_xcorr_1d_vs_nd(xt, xt)
    npts = x.shape[-1]
    assert cc.shape == (4, 2 * npts - 1)
    # auto-correlation of every channel peaks at zero lag = index npts-1
    assert torch.all(torch.argmax(cc, dim=1) == npts - 1)


def test_xcorr_detects_known_shift():
    nt, fs = 512, 1.0 / DT
    base = _ricker(nt, fs, t0=2.0)
    shift = 7  # samples; y delayed w.r.t. x
    delayed = np.roll(base, shift)
    x = torch.from_numpy(np.vstack([base, base]).astype(np.float32))
    y = torch.from_numpy(np.vstack([delayed, delayed]).astype(np.float32))
    cc = torch_xcorr_1d_vs_nd(x, y)
    lag = int(torch.argmax(cc[0])) - (nt - 1)
    assert lag == shift


def test_mccc_recovers_relative_delays(rng):
    """
    Channels are time-shifted copies of one wavelet; MCCC must recover the
    relative delays (up to the zero-mean gauge fixed by the constraint row).
    """
    fs = 1.0 / DT
    nt = 512
    shifts = np.array([0, 2, -3, 5, 1])  # samples
    nch = len(shifts)
    base = _ricker(nt, fs, t0=2.5)
    data = np.vstack([np.roll(base, s) for s in shifts]).astype(np.float32)
    data += 0.001 * rng.standard_normal(data.shape).astype(np.float32)

    delays, ccmax, dtmax = compute_mccc_delays(
        torch.from_numpy(data), DT, cc_threshold=0.3, return_all=True,
    )

    assert delays.shape == (nch,)
    assert ccmax.shape == (nch, nch)
    # Gauge: constraint row centers the solution near zero mean.
    assert abs(np.mean(delays)) < DT

    true_rel = (shifts - shifts.mean()) * DT
    assert np.allclose(delays, true_rel, atol=0.6 * DT), (
        f"recovered {delays}, expected {true_rel}"
    )

    # Pairwise lags measured directly should match the imposed shifts.
    # torch_xcorr_1d_vs_nd(s1, s2) peaks at the lag by which s2 is delayed
    # relative to s1, so dtmax[i, j] = (shift_i - shift_j) * dt.
    for i in range(nch):
        for j in range(nch):
            expected = (shifts[i] - shifts[j]) * DT
            assert abs(dtmax[i, j] - expected) <= DT + 1e-9
