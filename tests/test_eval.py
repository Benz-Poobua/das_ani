"""Tests for src/eval.py helpers and the preprocessing-backend experiment."""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from src.eval import (
    BenchmarkRunner,
    _set_nested,
    fidelity_metrics,
    run_preprocess_test,
)
from tests.conftest import write_test_npz

FS = 50.0
NCH = 8


def test_set_nested_creates_paths():
    cfg = {}
    _set_nested(cfg, "xcorr.mode", "v1")
    _set_nested(cfg, "preprocess.mode", "hybrid")
    assert cfg == {"xcorr": {"mode": "v1"}, "preprocess": {"mode": "hybrid"}}


def test_fidelity_metrics_identical(rng):
    a = rng.standard_normal((4, 64))
    fid = fidelity_metrics(a, a)
    assert fid["rel_fro"] == pytest.approx(0.0, abs=1e-12)
    assert fid["max_abs"] == pytest.approx(0.0, abs=1e-12)
    assert fid["cos_mean"] == pytest.approx(1.0)
    assert fid["cos_p05"] == pytest.approx(1.0)


@pytest.fixture()
def bench_setup(tmp_path, rng):
    """Tiny deployment + config file for BenchmarkRunner-based experiments."""
    data_root = tmp_path / "raw"
    out_dir = tmp_path / "bench"
    data_root.mkdir()
    fpath = write_test_npz(
        data_root / "20250722_025000_test.npz", rng, nch=NCH, nt=3000, fs=FS,
    )
    cfg = {
        "paths": {"data_root": str(data_root), "output_root": str(tmp_path / "ncf")},
        "runtime": {"njobs": 1, "use_gpu": False},
        "data": {"fs_raw": FS, "first_chan": 0, "last_chan": NCH - 1,
                 "dx": 8.0, "src_stride": 4, "min_length_sec": 10.0},
        "ingest": {"decimation": 1, "diff": False},
        "preprocess": {"mode": "pure_numpy", "f1": 2.0, "f2": 10.0, "ram_win_sec": 0.5},
        "xcorr": {"mode": "v1", "auto_cc": False, "is_spectral_whitening": True,
                  "window_freq_hz": 0.0, "max_lag_sec": 2.0,
                  "xcorr_seg_sec": 20.0, "xcorr_seg_sec_v1": 20.0,
                  "v1_fft_snap_pow2": True, "v1_fallback": "v1_2M"},
        "perf": {"enabled": False},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path, [fpath], out_dir


def test_run_preprocess_test_rows_and_fidelity(bench_setup):
    cfg_path, files, out_dir = bench_setup
    runner = BenchmarkRunner(cfg_path, files, out_dir)

    rows = run_preprocess_test(
        runner, ["pure_numpy", "hybrid", "pure_torch"], repeats=2, use_gpu=False,
    )

    assert len(rows) == 3  # one row per (file, mode)
    by_mode = {r.mode: r for r in rows}
    assert set(by_mode) == {"pure_numpy", "hybrid", "pure_torch"}

    for r in rows:
        assert r.experiment == "preprocess"
        assert r.note == files[0].name
        assert r.wall_sec >= 0.0
        assert np.isfinite(r.rel_fro)

    # pure_numpy compared against itself must be exact.
    assert by_mode["pure_numpy"].rel_fro == pytest.approx(0.0, abs=1e-12)
    # hybrid matches the benchmark to float32 round-off (cumsum-bounded).
    assert by_mode["hybrid"].rel_fro < 1e-4
    # pure_torch is an approximation, but a controlled one.
    assert by_mode["pure_torch"].rel_fro < 0.1
    assert by_mode["pure_torch"].cos_mean > 0.99


def test_run_preprocess_test_rejects_unknown_mode(bench_setup):
    cfg_path, files, out_dir = bench_setup
    runner = BenchmarkRunner(cfg_path, files, out_dir)
    with pytest.raises(ValueError, match="Unknown preprocess mode"):
        run_preprocess_test(runner, ["numpy_pure"], repeats=1)
