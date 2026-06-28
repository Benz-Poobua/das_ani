"""Unit + small integration tests for src/stack.py."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from src.stack import (
    _floor_to_base,
    _streaming_mean,
    _time_fmt_for,
    base_stack_ncf,
    parse_date_vs,
    parse_date_vs_method,
)


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,exp_dt,exp_vs,exp_method", [
    ("20210901_000000_cc_080.npy", datetime(2021, 9, 1), 80, None),
    ("20211110_150000_cc_080_conventional.npy", datetime(2021, 11, 10, 15), 80, "conventional"),
    ("20211110_150000_auto_conventional.npy", datetime(2021, 11, 10, 15), -1, "conventional"),
    ("20210901_cc_080_30d_v1_2M.npy", datetime(2021, 9, 1), 80, "v1_2M"),
    ("20250722_025000_bridge_cc_012_v1.npy", datetime(2025, 7, 22, 2, 50), 12, "v1"),
])
def test_parse_date_vs_method(name, exp_dt, exp_vs, exp_method):
    dt_obj, vs, method = parse_date_vs_method(name)
    assert dt_obj == exp_dt
    assert vs == exp_vs
    assert method == exp_method


def test_parse_date_vs_rejects_garbage():
    with pytest.raises(ValueError):
        parse_date_vs("not_a_real_file.npy")


@pytest.mark.parametrize("label,fmt", [
    ("1d", "%Y%m%d"), ("30d", "%Y%m%d"), ("1h", "%Y%m%d_%H%M%S"), ("daily", "%Y%m%d"),
])
def test_time_fmt_for(label, fmt):
    assert _time_fmt_for(label) == fmt


# ---------------------------------------------------------------------------
# Streaming mean
# ---------------------------------------------------------------------------
def test_streaming_mean(tmp_path, rng):
    arrays = [rng.standard_normal((4, 16)).astype(np.float32) for _ in range(5)]
    files = []
    for i, a in enumerate(arrays):
        p = tmp_path / f"a{i}.npy"
        np.save(p, a)
        files.append(p)

    stack, n = _streaming_mean(files)
    assert n == 5
    assert stack.dtype == np.float32
    assert np.allclose(stack, np.mean(arrays, axis=0), atol=1e-6)


def test_streaming_mean_skips_mismatched_shapes(tmp_path, rng):
    p1 = tmp_path / "good.npy"
    p2 = tmp_path / "bad.npy"
    np.save(p1, np.ones((2, 4), dtype=np.float32))
    np.save(p2, np.ones((3, 4), dtype=np.float32))
    stack, n = _streaming_mean([p1, p2])
    assert n == 1
    assert stack.shape == (2, 4)


def test_streaming_mean_empty():
    stack, n = _streaming_mean([])
    assert stack is None and n == 0


# ---------------------------------------------------------------------------
# Base stacking (integration on a tmp tree)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ts,label,expected", [
    # All intraday timestamps of a day floor to local midnight for "1d".
    (datetime(2021, 9, 1, 0, 0, 0), "1d", datetime(2021, 9, 1)),
    (datetime(2021, 9, 1, 5, 50, 0), "1d", datetime(2021, 9, 1)),
    (datetime(2021, 9, 1, 23, 59, 59), "1d", datetime(2021, 9, 1)),
    # "1h" floors to the top of the hour.
    (datetime(2021, 9, 1, 3, 40, 0), "1h", datetime(2021, 9, 1, 3)),
    # Multi-step bins are measured from the epoch.
    (datetime(2021, 9, 2, 12, 0, 0), "2d", datetime(2021, 9, 2)),
])
def test_floor_to_base(ts, label, expected):
    assert _floor_to_base(ts, label) == expected


def test_base_stack_ncf_averages_all_intraday_windows(tmp_path, rng):
    """A '1d' base stack must average ALL intraday windows of a day into one
    daily NCF (regression test for the timestamp-grouping bug that kept only a
    single window per day)."""
    raw = tmp_path / "raw"
    out = tmp_path / "stacks"
    raw.mkdir()

    # Two intraday slices of the same (day, vs, method) must average into ONE
    # 1d base; a second VS gets its own 1d base.
    a = rng.standard_normal((3, 9)).astype(np.float32)
    b = rng.standard_normal((3, 9)).astype(np.float32)
    np.save(raw / "20250722_010000_urban_cc_000_v1.npy", a)
    np.save(raw / "20250722_020000_urban_cc_000_v1.npy", b)
    np.save(raw / "20250722_010000_urban_cc_004_v1.npy", a * 2)

    base_stack_ncf(raw, out, "1d", overwrite=True, njobs=1)

    produced = sorted(p.name for p in Path(out).glob("*.npy"))
    # Exactly one daily base per (day, vs), named by date only.
    assert produced == [
        "20250722_cc_000_1d_v1.npy",
        "20250722_cc_004_1d_v1.npy",
    ]

    # The cc_000 daily base is the mean of its two intraday windows -- the
    # behaviour that was broken before the fix.
    cc000 = np.load(Path(out) / "20250722_cc_000_1d_v1.npy")
    np.testing.assert_allclose(cc000, (a + b) / 2.0, atol=1e-6)

    # cc_004 has a single window, so its base equals that window.
    cc004 = np.load(Path(out) / "20250722_cc_004_1d_v1.npy")
    np.testing.assert_allclose(cc004, a * 2.0, atol=1e-6)

    for p in Path(out).glob("*.npy"):
        arr = np.load(p)
        assert arr.shape == (3, 9)
        assert arr.dtype == np.float32
