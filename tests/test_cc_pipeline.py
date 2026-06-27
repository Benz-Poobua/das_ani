"""
End-to-end integration tests for src/cc.py::process_single_file on a tiny
synthetic deployment. Exercises:
  - the config wiring (incl. the new preprocess.mode key),
  - VSG output naming / shape / dtype,
  - conventional-vs-v1 NCF fidelity through the full pipeline,
  - pure_numpy vs hybrid preprocessing through the full pipeline.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from src.cc import process_single_file
from src.error import rel_frobenius
from tests.conftest import write_test_npz

FS = 50.0
NCH = 8
NT = 3000          # 60 s at 50 Hz
SEG_SEC = 20.0     # -> nseg = 3
MAX_LAG_SEC = 2.0  # -> M = 100 samples
M = int(round(MAX_LAG_SEC * FS))


def _base_cfg(data_root: Path, output_root: Path) -> dict:
    return {
        "paths": {"data_root": str(data_root), "output_root": str(output_root)},
        "runtime": {
            "njobs": 1, "use_gpu": False, "mmap": True,
            "frac_mem": 0.1, "min_chunk": 4, "max_chunk": 64,
            "torch_compile": False,
        },
        "data": {
            "fs_raw": FS, "first_chan": 0, "last_chan": NCH - 1,
            "dx": 8.0, "src_stride": 4, "min_length_sec": 10.0,
        },
        "ingest": {"decimation": 1, "diff": False},
        "preprocess": {"mode": "pure_numpy", "f1": 2.0, "f2": 10.0,
                       "ram_win_sec": 0.0, "whiten_chunk_nch": 8},
        "xcorr": {
            "mode": "conventional", "auto_cc": False,
            "is_spectral_whitening": True, "window_freq_hz": 0.0,
            "max_lag_sec": MAX_LAG_SEC,
            "xcorr_seg_sec": SEG_SEC, "xcorr_seg_sec_v1": SEG_SEC,
            "v1_fft_snap_pow2": True, "v1_fallback": "v1_2M",
        },
        "perf": {"enabled": False},
    }


@pytest.fixture()
def deployment(tmp_path, rng, monkeypatch):
    """One spec-compliant input file + isolated cwd (for the runlog)."""
    monkeypatch.chdir(tmp_path)
    data_root = tmp_path / "raw"
    data_root.mkdir()
    fpath = write_test_npz(
        data_root / "20250722_025000_test.npz", rng, nch=NCH, nt=NT, fs=FS,
    )
    return data_root, fpath


def _run(cfg: dict, fpath: Path) -> dict:
    out = process_single_file(str(fpath), cfg)
    assert out is not None and out.get("out_path")
    return out


def test_vsg_outputs_naming_shape_dtype(deployment, tmp_path):
    data_root, fpath = deployment
    cfg = _base_cfg(data_root, tmp_path / "ncf_conv")
    _run(cfg, fpath)

    out_dir = tmp_path / "ncf_conv"
    produced = sorted(p.name for p in out_dir.glob("*.npy"))
    # src_stride=4 over channels 0..7 -> virtual sources 000 and 004.
    assert produced == [
        "20250722_025000_test_cc_000_conventional.npy",
        "20250722_025000_test_cc_004_conventional.npy",
    ]
    for p in out_dir.glob("*.npy"):
        vsg = np.load(p)
        assert vsg.shape == (NCH, 2 * M + 1)
        assert vsg.dtype == np.float32
        assert np.all(np.isfinite(vsg))

    # Resume state must list both completed virtual sources.
    state = out_dir / "20250722_025000_test_cc_state_conventional.json"

    assert state.exists()


def test_v1_matches_conventional_end_to_end(deployment, tmp_path):
    data_root, fpath = deployment

    cfg_conv = _base_cfg(data_root, tmp_path / "ncf_conv")
    _run(cfg_conv, fpath)

    cfg_v1 = copy.deepcopy(_base_cfg(data_root, tmp_path / "ncf_v1"))
    cfg_v1["xcorr"]["mode"] = "v1"
    _run(cfg_v1, fpath)

    for vs in ("000", "004"):
        a = np.load(tmp_path / "ncf_conv" / f"20250722_025000_test_cc_{vs}_conventional.npy")
        b = np.load(tmp_path / "ncf_v1" / f"20250722_025000_test_cc_{vs}_v1.npy")
        assert a.shape == b.shape
        assert rel_frobenius(b, a) < 1e-4, (
            f"VS {vs}: v1 deviates from conventional through the full pipeline"
        )


def test_hybrid_preprocess_matches_pure_numpy_end_to_end(deployment, tmp_path):
    data_root, fpath = deployment

    cfg_np = _base_cfg(data_root, tmp_path / "ncf_np")
    cfg_np["preprocess"]["ram_win_sec"] = 0.5  # RAM norm: smooth comparison
    _run(cfg_np, fpath)

    cfg_hy = copy.deepcopy(_base_cfg(data_root, tmp_path / "ncf_hy"))
    cfg_hy["preprocess"]["mode"] = "hybrid"
    cfg_hy["preprocess"]["ram_win_sec"] = 0.5
    _run(cfg_hy, fpath)

    for vs in ("000", "004"):
        a = np.load(tmp_path / "ncf_np" / f"20250722_025000_test_cc_{vs}_conventional.npy")
        b = np.load(tmp_path / "ncf_hy" / f"20250722_025000_test_cc_{vs}_conventional.npy")
        # Spectral whitening can amplify tiny preprocessing round-off in
        # low-amplitude bins; bound is loose but far below physical signal.
        assert rel_frobenius(b, a) < 1e-3


def test_invalid_preprocess_mode_fails_fast(deployment, tmp_path):
    data_root, fpath = deployment
    cfg = _base_cfg(data_root, tmp_path / "ncf_bad")
    cfg["preprocess"]["mode"] = "torch_pure"  # typo
    with pytest.raises(ValueError, match="preprocess mode"):
        process_single_file(str(fpath), cfg)


def test_auto_cc_output(deployment, tmp_path):
    data_root, fpath = deployment
    cfg = _base_cfg(data_root, tmp_path / "ncf_auto")
    cfg["xcorr"]["auto_cc"] = True
    _run(cfg, fpath)
    acf = np.load(tmp_path / "ncf_auto" / "20250722_025000_test_auto_conventional.npy")
    assert acf.shape == (NCH, 2 * M + 1)
    # Zero lag of an autocorrelation is the energy peak.
    assert np.all(np.argmax(acf, axis=1) == M)
