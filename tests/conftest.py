"""
Shared fixtures for the DAS-ANI test suite.

Run from the repo root with:  pytest  (or: make test)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make `import src.<module>` work regardless of how pytest was invoked.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(20260610)


def make_das_noise(
    rng: np.random.Generator,
    nch: int = 12,
    nt: int = 4000,
    fs: float = 100.0,
    *,
    trend: bool = True,
) -> np.ndarray:
    """
    Synthetic DAS-like panel: band-limited noise + a coherent component
    shared across channels (so median removal has something to remove)
    + optional per-channel linear trend (so detrend has something to remove).
    """
    t = np.arange(nt) / fs
    common = np.sin(2 * np.pi * 3.0 * t) + 0.5 * np.sin(2 * np.pi * 7.0 * t + 0.3)
    x = 0.7 * rng.standard_normal((nch, nt)) + common[None, :]
    if trend:
        slopes = rng.uniform(-2.0, 2.0, size=(nch, 1))
        offsets = rng.uniform(-5.0, 5.0, size=(nch, 1))
        x = x + slopes * t[None, :] + offsets
    return x.astype(np.float64)


@pytest.fixture()
def das_panel(rng: np.random.Generator) -> np.ndarray:
    return make_das_noise(rng)


def write_test_npz(
    path: Path,
    rng: np.random.Generator,
    *,
    nch: int = 8,
    nt: int = 3000,
    fs: float = 50.0,
    dx: float = 8.0,
) -> Path:
    """Write a spec-compliant input archive (data/dt/dx/start_sample/end_sample)."""
    data = make_das_noise(rng, nch=nch, nt=nt, fs=fs).astype(np.float32)
    np.savez(
        path,
        data=data,
        dt=1.0 / fs,
        dx=dx,
        start_sample=0,
        end_sample=nt,
    )
    return path
