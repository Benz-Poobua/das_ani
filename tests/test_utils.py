"""Unit tests for src/utils.py (I/O, config, filename parsing, math helpers)."""
from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from src.utils import (
    check_existing_output,
    get_cfg,
    load_config,
    load_data,
    load_resume_state,
    nextpow2,
    normalize_traces,
    parse_ncf_filename,
    parse_ncf_stack_filename,
    save_resume_state,
)
from tests.conftest import write_test_npz


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("x,expected", [
    (1, 1), (2, 2), (3, 4), (5, 8), (1024, 1024), (1025, 2048), (0.5, 1),
])
def test_nextpow2(x, expected):
    assert nextpow2(x) == expected


def test_normalize_traces_unit_peak(rng):
    x = rng.standard_normal((4, 100)) * np.array([[1.0], [10.0], [0.1], [5.0]])
    out = normalize_traces(x)
    assert np.allclose(np.max(np.abs(out), axis=-1), 1.0)


def test_normalize_traces_dead_trace_stays_zero(rng):
    x = rng.standard_normal((3, 50))
    x[1, :] = 0.0
    out = normalize_traces(x)
    assert np.all(out[1] == 0.0)
    assert np.all(np.isfinite(out))


def test_normalize_traces_nd(rng):
    x = rng.standard_normal((2, 3, 64))
    out = normalize_traces(x)
    assert out.shape == x.shape
    assert np.allclose(np.max(np.abs(out), axis=-1), 1.0)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def test_get_cfg_nested_default_required():
    cfg = {"a": {"b": {"c": 7}}}
    assert get_cfg(cfg, ["a", "b", "c"]) == 7
    assert get_cfg(cfg, ["a", "x"], default=3) == 3
    with pytest.raises(KeyError):
        get_cfg(cfg, ["a", "x"], required=True)


def test_load_config_yaml_and_json(tmp_path):
    yaml_p = tmp_path / "cfg.yaml"
    yaml_p.write_text("runtime:\n  njobs: 2\npreprocess:\n  mode: hybrid\n")
    cfg = load_config(yaml_p)
    assert cfg["runtime"]["njobs"] == 2
    assert cfg["preprocess"]["mode"] == "hybrid"

    json_p = tmp_path / "cfg.json"
    json_p.write_text(json.dumps({"runtime": {"njobs": 4}}))
    assert load_config(json_p)["runtime"]["njobs"] == 4


def test_load_config_rejects_unknown_extension(tmp_path):
    p = tmp_path / "cfg.toml"
    p.write_text("x = 1")
    with pytest.raises(ValueError):
        load_config(p)


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------
def test_parse_ncf_filename():
    date, vs, window = parse_ncf_filename("20250722_cc_080_v1.npy")
    assert (date, vs, window) == ("20250722", "080", "v1")
    with pytest.raises(ValueError):
        parse_ncf_filename("garbage.npy")


def test_parse_ncf_stack_filename():
    date, vs, window, mode = parse_ncf_stack_filename(
        "20250722_025000_bridge_cc_080_7d_v1.npy"
    )
    assert vs == "080"
    assert window == "7d"
    assert mode == "v1"


# ---------------------------------------------------------------------------
# Data I/O (.npz file specification)
# ---------------------------------------------------------------------------
def test_load_data_npz_roundtrip(tmp_path, rng):
    p = write_test_npz(tmp_path / "20250722_025000_test.npz", rng, nch=4, nt=500, fs=50.0)
    _, das, dt, n, duration = load_data(p)
    assert das.shape == (4, 500)
    assert dt == pytest.approx(0.02)
    assert n == 500
    assert duration == pytest.approx(10.0)


def test_load_data_requires_data_and_dt(tmp_path, rng):
    p = tmp_path / "bad.npz"
    np.savez(p, data=rng.standard_normal((2, 10)))  # no dt
    with pytest.raises(KeyError):
        load_data(p)


def test_load_data_warns_on_missing_spec_keys(tmp_path, rng, caplog):
    p = tmp_path / "legacy.npz"
    np.savez(p, data=rng.standard_normal((2, 10)).astype(np.float32), dt=0.01)
    with caplog.at_level(logging.WARNING, logger="src.utils"):
        load_data(p)
    assert any("start_sample" in r.message for r in caplog.records), (
        "legacy archives without dx/start_sample/end_sample must trigger a warning"
    )


def test_load_data_rejects_non_2d(tmp_path):
    p = tmp_path / "bad2.npz"
    np.savez(p, data=np.zeros(10, dtype=np.float32), dt=0.01)
    with pytest.raises(ValueError):
        load_data(p)


# ---------------------------------------------------------------------------
# Auto-resume helpers
# ---------------------------------------------------------------------------
def test_check_existing_output(tmp_path):
    p = tmp_path / "out.npy"
    assert not check_existing_output(p, (3, 5))
    np.save(p, np.zeros((3, 5), dtype=np.float32))
    assert check_existing_output(p, (3, 5))
    assert not check_existing_output(p, (3, 6))  # wrong shape -> recompute


def test_resume_state_roundtrip(tmp_path):
    meta = tmp_path / "state.json"
    assert load_resume_state(meta) == set()
    save_resume_state(meta, {3, 1, 2})
    assert load_resume_state(meta) == {1, 2, 3}
