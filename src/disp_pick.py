"""
:module: src/disp_pick.py
:author: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Compute dispersion images (f–v) and pick dispersion curves
          from NCF stacks (daily, 7d, 15d, 30d) per virtual source.

Workflow (recommended):
1) Load stacked symmetric NCF (-T ... 0 ... +T)
2) Optional: zero-lag mute/taper (helps FK/FV stability)
3) Optional: FK velocity filter on symmetric NCF
4) Fold / mode select -> one-sided gather (t >= 0)
5) F–V transform (phase shift) + optional picking
6) Save outputs + meta.json + runlog + perf CSV

Adds:
- runlog messages
- perf CSV row per processed file
- richer meta.json with runtime + parameters
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from src.utils import (
    timeit,
    load_config,
    get_cfg,
    fk_velocity_filter,
    write_runlog,
    write_perf_row,
)
from src.disp import compute_dispersion_from_ncf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_VS_RE = re.compile(r"_cc_(\d+)")
_KNOWN_CC_METHODS = {"v1", "conventional"}


# =====================================================
# Helpers
# =====================================================
def extract_cc_method(stem: str) -> Optional[str]:
    last = stem.split("_")[-1].lower()
    return last if last in _KNOWN_CC_METHODS else None


def extract_vs_index(path_or_name: str) -> int:
    m = _VS_RE.search(path_or_name)
    if not m:
        raise ValueError(f"Cannot parse VS index from: {path_or_name}")
    return int(m.group(1))


def _pick_device(device_str: str) -> str:
    s = (device_str or "auto").lower()
    if s not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"runtime.device must be one of auto/cpu/cuda/mps, got: {device_str}")
    return s


def _resolve_torch_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _get_disp_mode(cfg: Dict[str, Any]) -> str:
    mode = get_cfg(cfg, ["dispersion", "mode"], None)
    if mode is None:
        mode = cfg.get("disp_mode", "both")  # backward compat
    mode = str(mode).lower()
    if mode not in {"both", "causal", "acausal"}:
        raise ValueError(f"Unknown dispersion mode: {mode}")
    return mode


def _resolve_outdir(results_root: Path, stack_window: str, base: str, vs_subdir: bool) -> Path:
    if not vs_subdir:
        return results_root / stack_window
    vs_idx = extract_vs_index(base)
    return results_root / stack_window / f"VS_{vs_idx:03d}"


def _sentinel_path(outdir: Path, base: str, *, picking_enabled: bool, save_pick: bool) -> Path:
    # Prefer pick file as sentinel if picking is enabled AND we save picks.
    if picking_enabled and save_pick:
        return outdir / f"{base}_pick.npy"
    return outdir / f"{base}_meta.json"


def _dump_meta(meta_path: Path, meta: Dict[str, Any]) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def _cfg_slice(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Keep meta.json readable: store only relevant cfg branches."""
    out: Dict[str, Any] = {}
    for k in ("io", "geometry", "fk", "fv", "runtime", "picking", "save", "dispersion", "runlog", "perf"):
        if k in cfg:
            out[k] = cfg[k]
    if "disp_mode" in cfg and "dispersion" not in out:
        out["disp_mode"] = cfg["disp_mode"]
    return out


def _apply_zero_lag_taper(
    ncf: np.ndarray,
    *,
    mode: str = "mute",   # "mute" or "taper"
    half_width: int = 2,  # in samples around zero lag (symmetric)
) -> np.ndarray:
    """
    Reduce the huge broadband zero-lag spike that can smear FK/FV.

    Assumes ncf is symmetric with zero lag at mid index.
    """
    if ncf.ndim != 2:
        return ncf

    nch, nt = ncf.shape
    mid = nt // 2
    hw = int(max(0, half_width))

    out = ncf.copy()  # safe
    if mode == "mute":
        out[:, mid - hw : mid + hw + 1] = 0.0
        return out

    if mode == "taper":
        # cosine taper down to 0 at mid, symmetric
        idx = np.arange(-hw, hw + 1)
        if idx.size == 0:
            return out
        w = 0.5 * (1.0 + np.cos(np.pi * idx / max(hw, 1)))  # 1 at edges, 0 at mid
        w = w.astype(np.float32)
        out[:, mid - hw : mid + hw + 1] *= w[None, :]
        return out

    raise ValueError(f"zero_lag.mode must be mute or taper; got {mode}")


def _fold_symmetric_to_one_sided(ncf: np.ndarray, disp_mode: str) -> np.ndarray:
    """
    Convert symmetric (-T...0...+T) -> one-sided (0...+T) depending on mode.
    Output includes zero-lag at index 0.
    """
    nch, nt_full = ncf.shape
    mid = nt_full // 2

    if disp_mode == "both":
        acausal_flipped = np.flip(ncf[:, :mid], axis=1)
        causal = ncf[:, mid + 1 :]

        n_len = min(acausal_flipped.shape[1], causal.shape[1])
        acausal_flipped = acausal_flipped[:, :n_len]
        causal = causal[:, :n_len]

        stacked_side = (causal + acausal_flipped) / 2.0
        zero_lag = ncf[:, mid : mid + 1]
        return np.concatenate([zero_lag, stacked_side], axis=1)

    if disp_mode == "causal":
        return ncf[:, mid:]  # [zero | +]

    if disp_mode == "acausal":
        zero_lag = ncf[:, mid : mid + 1]
        acausal_flipped = np.flip(ncf[:, :mid], axis=1)
        return np.concatenate([zero_lag, acausal_flipped], axis=1)

    raise ValueError(f"Unknown dispersion mode: {disp_mode}")


# =====================================================
# Worker
# =====================================================
def process_one_ncf(ncf_path: str, cfg: Dict[str, Any]) -> Optional[str]:
    t0 = time.perf_counter()

    ncf_p = Path(ncf_path)
    base = ncf_p.stem
    cc_method = extract_cc_method(base)

    # ---- cfg: io ----
    results_root = Path(get_cfg(cfg, ["io", "results_root"], "results/dispersion")).expanduser()
    stack_window = str(get_cfg(cfg, ["io", "stack_window"], "daily"))
    vs_subdir = bool(get_cfg(cfg, ["io", "vs_subdir"], True))

    # ---- cfg: geometry ----
    fs = float(get_cfg(cfg, ["geometry", "fs"], 250.0))
    dx = float(get_cfg(cfg, ["geometry", "dx"], 8.16))

    # ---- cfg: mode ----
    disp_mode = _get_disp_mode(cfg)

    # ---- cfg: runtime ----
    device_str = _pick_device(str(get_cfg(cfg, ["runtime", "device"], "auto")))
    device_obj = _resolve_torch_device(device_str)
    batch_size_v = get_cfg(cfg, ["runtime", "batch_size_v"], None)
    empty_cache = bool(get_cfg(cfg, ["runtime", "empty_cache"], True))

    # ---- cfg: fv ----
    fv_kwargs: Dict[str, Any] = {
        "vmin": float(get_cfg(cfg, ["fv", "vmin"], 200.0)),
        "vmax": float(get_cfg(cfg, ["fv", "vmax"], 4000.0)),
        "dv": float(get_cfg(cfg, ["fv", "dv"], 10.0)),
        "fmin": float(get_cfg(cfg, ["fv", "fmin"], 0.1)),
        "fmax": float(get_cfg(cfg, ["fv", "fmax"], 50.0)),
        "normalize": bool(get_cfg(cfg, ["fv", "normalize"], True)),
        "empty_cache": empty_cache,
        "device": device_obj,
    }
    if batch_size_v is not None:
        fv_kwargs["batch_size_v"] = int(batch_size_v)

    # ---- cfg: picking ----
    picking_enabled = bool(get_cfg(cfg, ["picking", "enabled"], True))
    pick_kwargs: Optional[Dict[str, Any]] = None
    if picking_enabled:
        f_ref_set = get_cfg(cfg, ["picking", "f_ref_set"], None)
        vmax_set = get_cfg(cfg, ["picking", "vmax_set"], None)
        pick_kwargs = {
            "step": int(get_cfg(cfg, ["picking", "step"], 5)),
            "f_ref_set": None if f_ref_set is None else list(f_ref_set),
            "vmax_set": None if vmax_set is None else list(vmax_set),
        }

    # ---- cfg: fk ----
    fk_cfg = get_cfg(cfg, ["fk"], {})
    do_fk = bool(fk_cfg.get("enabled", False))
    fk_vmin = float(fk_cfg.get("vmin", 150.0))
    fk_vmax = float(fk_cfg.get("vmax", 2500.0))
    fk_taper_frac = float(fk_cfg.get("taper_frac", 0.10))

    # ---- cfg: zero-lag ----
    zl_cfg = get_cfg(cfg, ["zero_lag"], {})
    zl_enabled = bool(zl_cfg.get("enabled", False))
    zl_mode = str(zl_cfg.get("mode", "mute")).lower()
    zl_half_width = int(zl_cfg.get("half_width", 2))

    # ---- cfg: save ----
    overwrite = bool(get_cfg(cfg, ["save", "overwrite"], False))
    save_panel = bool(get_cfg(cfg, ["save", "save_panel"], True))
    save_axes = bool(get_cfg(cfg, ["save", "save_axes"], True))
    save_pick = bool(get_cfg(cfg, ["save", "save_pick"], True))
    save_meta = bool(get_cfg(cfg, ["save", "save_meta"], True))

    # ---- cfg: runlog/perf ----
    runlog_enabled = bool(get_cfg(cfg, ["runlog", "enabled"], True))
    perf_enabled = bool(get_cfg(cfg, ["perf", "enabled"], False))
    perf_out_path = str(get_cfg(cfg, ["perf", "out_path"], "data/runlogs/perf_disp.csv"))

    outdir = _resolve_outdir(results_root, stack_window, base, vs_subdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sentinel = _sentinel_path(outdir, base, picking_enabled=picking_enabled, save_pick=save_pick)
    if sentinel.exists() and not overwrite:
        logger.info("[SKIP] %s already done -> %s", base, sentinel)
        return None

    if runlog_enabled:
        write_runlog(
            f"[disp_pick] start base={base} stack={stack_window} mode={disp_mode} "
            f"fk={do_fk} zl={zl_enabled} device={device_obj} cc_method={cc_method or 'NA'}"
        )

    # -------------------------
    # 1) Load symmetric NCF
    # -------------------------
    if not ncf_p.exists():
        raise FileNotFoundError(f"NCF file not found: {ncf_p}")

    ncf = np.load(ncf_p)
    if ncf.ndim != 2:
        raise ValueError(f"NCF must be 2D (nrec, nlag); got shape={ncf.shape}")

    nch, nt_full = ncf.shape

    # -------------------------
    # 2) Optional zero-lag mute/taper (on symmetric)
    # -------------------------
    ncf_zl = ncf
    if zl_enabled:
        ncf_zl = _apply_zero_lag_taper(ncf, mode=zl_mode, half_width=zl_half_width)

    # -------------------------
    # 3) Optional FK filter (on symmetric)
    # -------------------------
    ncf_fk = ncf_zl
    if do_fk:
        ncf_fk = fk_velocity_filter(
            ncf_fk,
            dt=1.0 / fs,
            dx=dx,
            vmin=fk_vmin,
            vmax=fk_vmax,
            taper_frac=fk_taper_frac,
            device=device_obj,
        )

    # -------------------------
    # 4) Fold / mode select -> one-sided
    # -------------------------
    ncf_proc = _fold_symmetric_to_one_sided(ncf_fk, disp_mode)

    # -------------------------
    # 5) FV + picking
    # -------------------------
    fv_panel, f_axis, v_axis, picks = compute_dispersion_from_ncf(
        ncf=ncf_proc,
        fs=fs,
        dx=dx,
        fv_kwargs=fv_kwargs,
        pick_kwargs=pick_kwargs if picking_enabled else None,
    )

    # -------------------------
    # 6) Save outputs
    # -------------------------
    if save_panel:
        np.save(outdir / f"{base}_fv_panel.npy", fv_panel.detach().cpu().numpy())
    if save_axes:
        np.save(outdir / f"{base}_f_axis.npy", f_axis.detach().cpu().numpy())
        np.save(outdir / f"{base}_v_axis.npy", v_axis.detach().cpu().numpy())
    if save_pick and (picks is not None):
        np.save(outdir / f"{base}_pick.npy", picks)

    # -------------------------
    # 7) meta + runlog + perf
    # -------------------------
    t1 = time.perf_counter()
    wall_sec = float(t1 - t0)

    if save_meta:
        meta = {
            "ncf_path": str(ncf_p.resolve()),
            "outdir": str(outdir.resolve()),
            "base": base,
            "cc_method": cc_method,
            "stack_window": stack_window,
            "geometry": {"fs": float(fs), "dx": float(dx)},
            "workflow": {
                "zero_lag": {"enabled": zl_enabled, "mode": zl_mode, "half_width": zl_half_width},
                "fk_on_symmetric": bool(do_fk),
                "fold_mode": disp_mode,
            },
            "shapes": {
                "orig": [int(nch), int(nt_full)],
                "proc_one_sided": [int(ncf_proc.shape[0]), int(ncf_proc.shape[1])],
            },
            "fk": {
                "enabled": do_fk,
                "vmin": fk_vmin if do_fk else None,
                "vmax": fk_vmax if do_fk else None,
                "taper_frac": fk_taper_frac if do_fk else None,
            },
            "runtime": {
                "device": str(device_obj),
                "batch_size_v": fv_kwargs.get("batch_size_v", None),
                "torch_version": torch.__version__,
                "platform": platform.platform(),
                "python": platform.python_version(),
            },
            "fv_kwargs": {k: repr(v) for k, v in fv_kwargs.items()},
            "picking": pick_kwargs if picking_enabled else None,
            "timing": {"wall_sec": wall_sec},
            "cfg": _cfg_slice(cfg),
        }
        _dump_meta(outdir / f"{base}_meta.json", meta)

    if runlog_enabled:
        write_runlog(
            f"[disp_pick] done base={base} wall_sec={wall_sec:.3f} "
            f"fk={do_fk} zl={zl_enabled} mode={disp_mode} device={device_obj}"
        )

    if perf_enabled:
        write_perf_row(
            {
                "file": ncf_p.name,
                "base": base,
                "cc_method": cc_method or "NA",
                "stack_window": stack_window,
                "disp_mode": disp_mode,
                "zero_lag": int(zl_enabled),
                "fk_enabled": int(do_fk),
                "fk_vmin": float(fk_vmin) if do_fk else float("nan"),
                "fk_vmax": float(fk_vmax) if do_fk else float("nan"),
                "nch": int(nch),
                "nt_full": int(nt_full),
                "nt_proc": int(ncf_proc.shape[1]),
                "device": str(device_obj),
                "batch_size_v": int(fv_kwargs["batch_size_v"]) if "batch_size_v" in fv_kwargs else -1,
                "seconds": wall_sec,
            },
            perf_out_path,
            add_pid_suffix=True,
        )

    return str(sentinel)


# =====================================================
# Main
# =====================================================
@timeit
def main(cfg: Dict[str, Any]) -> None:
    ncf_root = Path(get_cfg(cfg, ["io", "ncf_root"], "data/ncf_stacks")).expanduser()
    results_root = Path(get_cfg(cfg, ["io", "results_root"], "results/dispersion")).expanduser()
    njobs = int(get_cfg(cfg, ["runtime", "njobs"], 4))

    stack_windows = get_cfg(cfg, ["io", "stack_windows"], None)
    if stack_windows is None:
        stack_windows = [str(get_cfg(cfg, ["io", "stack_window"], "daily"))]
    if isinstance(stack_windows, str):
        stack_windows = [stack_windows]
    stack_windows = [str(s) for s in stack_windows]

    results_root.mkdir(parents=True, exist_ok=True)

    logger.info("NCF root:     %s", ncf_root)
    logger.info("Results root: %s", results_root)
    logger.info("Runtime:      njobs=%d", njobs)
    logger.info("Windows:      %s", stack_windows)
    logger.info("FK:           enabled=%s", bool(get_cfg(cfg, ["fk", "enabled"], False)))
    logger.info("Mode:         %s", _get_disp_mode(cfg))

    for stack_window in stack_windows:
        cfg.setdefault("io", {})
        cfg["io"]["stack_window"] = stack_window

        in_dir = ncf_root / stack_window
        if not in_dir.exists():
            logger.warning("[SKIP] Input directory missing: %s", in_dir)
            continue

        filelist = sorted(in_dir.glob("*.npy"))
        if not filelist:
            logger.warning("[%s] No NCF files found.", stack_window)
            continue

        logger.info("[%s] Processing %d files...", stack_window, len(filelist))

        with ProcessPoolExecutor(max_workers=njobs) as ex:
            futures = [ex.submit(process_one_ncf, str(p), cfg) for p in filelist]

            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Disp [{stack_window}]"):
                try:
                    fut.result()
                except Exception:
                    logger.exception("[%s] Worker Error", stack_window)


# =====================================================
# CLI
# =====================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dispersion Imaging & Picking Pipeline")
    p.add_argument("--config", type=str, default="configs/disp.yaml", help="Path to YAML config")

    p.add_argument("--stack_windows", type=str, nargs="+", help="Override io.stack_windows")
    p.add_argument("--njobs", type=int, help="Override runtime.njobs")

    p.add_argument("--mode", type=str, choices=["both", "causal", "acausal"], help="Override dispersion folding mode")
    p.add_argument("--fk", action="store_true", help="Force enable FK filtering")
    p.add_argument("--no-fk", action="store_true", help="Force disable FK filtering")

    p.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return p.parse_args()


def _override_cfg(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.stack_windows is not None:
        cfg.setdefault("io", {})
        cfg["io"]["stack_windows"] = list(args.stack_windows)
        cfg["io"]["stack_window"] = str(args.stack_windows[0])

    if args.njobs is not None:
        cfg.setdefault("runtime", {})
        cfg["runtime"]["njobs"] = int(args.njobs)

    if args.mode is not None:
        cfg.setdefault("dispersion", {})
        cfg["dispersion"]["mode"] = str(args.mode)

    if args.fk:
        cfg.setdefault("fk", {})
        cfg["fk"]["enabled"] = True
    elif args.no_fk:
        cfg.setdefault("fk", {})
        cfg["fk"]["enabled"] = False

    return cfg


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_config(cfg_path)
    cfg = _override_cfg(cfg, args)

    main(cfg)