"""
:module: src/disp_pick.py
:author: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Compute dispersion images (f–v) and pick dispersion curves
          from NCF stacks (daily, 7d, 15d, 30d) per virtual source.

Workflow:
---------
Input folder:
    <io.ncf_root>/<io.stack_window>/
        20210901_cc_000_daily.npy
        ...

Output folder (if io.vs_subdir=true):
    <io.results_root>/<io.stack_window>/VS_000/
        20210901_cc_000_daily_fv_panel.npy
        20210901_cc_000_daily_f_axis.npy
        20210901_cc_000_daily_v_axis.npy
        20210901_cc_000_daily_pick.npy
        20210901_cc_000_daily_meta.json

If io.vs_subdir=false:
    <io.results_root>/<io.stack_window>/
        <base>_fv_panel.npy, ...
"""
from __future__ import annotations

import argparse
import json
import logging
import re

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, Optional

from src.utils import timeit, load_config, get_cfg
from src.disp import compute_dispersion_from_ncf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_VS_RE = re.compile(r"_cc_(\d+)")
# =====================================================
# Helpers
# =====================================================
def extract_vs_index(path_or_name: str) -> int:
    """Extract virtual source index from e.g. 20210901_cc_000_daily.npy -> 0."""
    m = _VS_RE.search(path_or_name)
    if not m:
        raise ValueError(f"Cannot parse VS index from: {path_or_name}")
    return int(m.group(1))

def _pick_device(device_str: str) -> str:
    """Return string for cfg sanity; compute_dispersion_from_ncf uses fv_kwargs device object."""
    s = (device_str or "auto").lower()
    if s not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"runtime.device must be one of auto/cpu/cuda/mps, got: {device_str}")
    return s

def _resolve_outdir(results_root: Path, stack_window: str, base: str, vs_subdir: bool) -> Path:
    if not vs_subdir:
        return results_root / stack_window
    vs_idx = extract_vs_index(base)
    return results_root / stack_window / f"VS_{vs_idx:03d}"

def _sentinel_path(outdir: Path, base: str, picking_enabled: bool) -> Path:
    # If picking is enabled and save_pick=True => pick is the best sentinel.
    # If picking disabled => meta.json is the sentinel.
    return outdir / (f"{base}_pick.npy" if picking_enabled else f"{base}_meta.json")

def _dump_meta(meta_path: Path, meta: Dict[str, Any]) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

# =====================================================
# Worker
# =====================================================
def process_one_ncf(ncf_path: str, cfg: Dict[str, Any]) -> Optional[str]:
    """
    Process ONE stacked NCF file:
      - load NCF
      - compute fv-panel
      - optional pick
      - save outputs
    Returns sentinel path string if done, else None if skipped.
    """
    ncf_p = Path(ncf_path)
    base = ncf_p.stem

    # ---- cfg: io ----
    ncf_root = Path(get_cfg(cfg, ["io", "ncf_root"], "data/ncf_stacks")).expanduser()
    results_root = Path(get_cfg(cfg, ["io", "results_root"], "results/dispersion")).expanduser()
    stack_window = str(get_cfg(cfg, ["io", "stack_window"], "daily"))
    vs_subdir = bool(get_cfg(cfg, ["io", "vs_subdir"], True))

    # ---- cfg: geometry ----
    fs = float(get_cfg(cfg, ["geometry", "fs"], 250.0))
    dx = float(get_cfg(cfg, ["geometry", "dx"], 8.16))

    # ---- cfg: runtime ----
    device_str = _pick_device(str(get_cfg(cfg, ["runtime", "device"], "auto")))
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
        }
    if batch_size_v is not None:
        fv_kwargs["batch_size_v"] = int(batch_size_v)

    # Device object for disp.dispersion_curve
    import torch  # local import keeps worker import light

    if device_str == "auto":
        if torch.cuda.is_available():
            fv_kwargs["device"] = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            fv_kwargs["device"] = torch.device("mps")
        else:
            fv_kwargs["device"] = torch.device("cpu")
    else:
        fv_kwargs["device"] = torch.device(device_str)

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

    # ---- cfg: save ----
    overwrite = bool(get_cfg(cfg, ["save", "overwrite"], False))
    save_panel = bool(get_cfg(cfg, ["save", "save_panel"], True))
    save_axes = bool(get_cfg(cfg, ["save", "save_axes"], True))
    save_pick = bool(get_cfg(cfg, ["save", "save_pick"], True))
    save_meta = bool(get_cfg(cfg, ["save", "save_meta"], True))

    # Output directory + sentinel
    outdir = _resolve_outdir(results_root, stack_window, base, vs_subdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sentinel = _sentinel_path(outdir, base, picking_enabled and save_pick)
    if sentinel.exists() and not overwrite:
        logger.info(f"[SKIP] {base} already done → {sentinel}")
        return None
    
    # Load NCF
    if not ncf_p.exists():
        raise FileNotFoundError(f"NCF file not found: {ncf_p}")
    ncf = __import__("numpy").load(ncf_p)

    # Compute
    fv_panel, f_axis, v_axis, picks = compute_dispersion_from_ncf(
        ncf=ncf,
        fs=fs,
        dx=dx,
        fv_kwargs=fv_kwargs,
        pick_kwargs=pick_kwargs if picking_enabled else None,
    )

    # Save arrays
    import numpy as np

    if save_panel:
        np.save(outdir / f"{base}_fv_panel.npy", fv_panel.detach().cpu().numpy())
    if save_axes:
        np.save(outdir / f"{base}_f_axis.npy", f_axis.detach().cpu().numpy())
        np.save(outdir / f"{base}_v_axis.npy", v_axis.detach().cpu().numpy())
    if save_pick and (picks is not None):
        np.save(outdir / f"{base}_pick.npy", picks)
    
    # Metadata
    if save_meta:
        meta = {
            "ncf_path": str(ncf_p.resolve()),
            "outdir": str(outdir.resolve()),
            "stack_window": stack_window,
            "vs_subdir": bool(vs_subdir),
            "geometry": {"fs": float(fs), "dx": float(dx)},
            "shapes": {
                "ncf": list(np.asarray(ncf).shape),
                "fv_panel": list(fv_panel.shape),
                "f_axis": list(f_axis.shape),
                "v_axis": list(v_axis.shape),
                "picks": None if picks is None else list(np.asarray(picks).shape),
            },
            "fv_kwargs": {k: repr(v) for k, v in fv_kwargs.items()},
            "picking": {
                "enabled": bool(picking_enabled),
                "pick_kwargs": None if pick_kwargs is None else pick_kwargs,
            },
            "save": {
                "overwrite": bool(overwrite),
                "save_panel": bool(save_panel),
                "save_axes": bool(save_axes),
                "save_pick": bool(save_pick),
                "save_meta": bool(save_meta),
            },
        }
        _dump_meta(outdir / f"{base}_meta.json", meta)

    return str(sentinel)

# =====================================================
# Main
# =====================================================
@timeit 
def main(cfg: Dict[str, Any]) -> None:
    """
    Run dispersion workflow for one stack window folder, using cfg.
    """
    ncf_root = Path(get_cfg(cfg, ["io", "ncf_root"], "data/ncf_stacks")).expanduser()
    results_root = Path(get_cfg(cfg, ["io", "results_root"], "results/dispersion")).expanduser()
    njobs = int(get_cfg(cfg, ["runtime", "njobs"], 4))
    vs_subdir = bool(get_cfg(cfg, ["io", "vs_subdir"], True))

    # List of stack windows (default: ["daily"])
    stack_windows = get_cfg(cfg, ["io", "stack_windows"], None)
    if stack_windows is None:
        # Backward-compatible: if user still has io.stack_window
        stack_windows = [str(get_cfg(cfg, ["io", "stack_window"], "daily"))]
    if isinstance(stack_windows, str):
        stack_windows = [stack_windows]
    stack_windows = [str(s) for s in stack_windows]

    results_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"NCF root:     {ncf_root}")
    logger.info(f"Results root: {results_root}")
    logger.info(f"Runtime:      njobs={njobs}")
    logger.info(f"Layout:       vs_subdir={vs_subdir}")
    logger.info(f"Windows:      {stack_windows}")

    # Process each window sequentially; inside each, parallelize over files
    for stack_window in stack_windows:
        # Set current window into cfg so worker can read it
        cfg.setdefault("io", {})
        cfg["io"]["stack_window"] = stack_window

        in_dir = ncf_root / stack_window
        if not in_dir.exists():
            logger.warning(f"[SKIP] Input directory not found: {in_dir}")
            continue

        filelist = sorted(in_dir.glob("*.npy"))
        logger.info(f"[{stack_window}] Input: {in_dir} | files={len(filelist)}")
        logger.info(f"[{stack_window}] Output: {results_root / stack_window}")

        if not filelist:
            logger.warning(f"[{stack_window}] No NCF files found. Skipping.")
            continue

        # Multiprocessing: cfg must be picklable/JSON-ish (dict of primitives/lists)
        with ProcessPoolExecutor(max_workers=njobs) as ex:
            futures = [ex.submit(process_one_ncf, str(p), cfg) for p in filelist]

            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Dispersion [{stack_window}]",
            ):
                try:
                    outp = fut.result()
                    if outp:
                        logger.info(f"[{stack_window}] Done → {outp}")
                except Exception as e:
                    logger.error(f"[{stack_window}] Worker error: {e}")

# =====================================================
# CLI
# =====================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dispersion imaging + picking from stacked NCF files"
    )
    p.add_argument(
        "--config",
        type=str,
        default="disp.yaml",
        help="Path to YAML config file (default: disp.yaml)",
    )
    p.add_argument(
        "--stack_windows",
        type=str,
        nargs="+",
        choices=["daily", "7d", "15d", "30d"],
        default=None,
        help="Override io.stack_windows in config (e.g., --stack_windows daily 7d 15d)",
    )
    p.add_argument(
        "--njobs",
        type=int,
        default=None,
        help="Override runtime.njobs in config",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return p.parse_args()

def _override_cfg(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Apply CLI overrides without editing YAML.
    """
    if args.stack_windows is not None:
        cfg.setdefault("io", {})
        cfg["io"]["stack_windows"] = list(args.stack_windows)
        # Keep io.stack_window consistent (use first) for any code that still reads it
        cfg["io"]["stack_window"] = str(args.stack_windows[0])

    if args.njobs is not None:
        cfg.setdefault("runtime", {})
        cfg["runtime"]["njobs"] = int(args.njobs)

    return cfg

if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    cfg_path = Path(args.config).expanduser().resolve()
    logger.info(f"Loading config: {cfg_path}")

    cfg = load_config(cfg_path)
    cfg = _override_cfg(cfg, args)

    main(cfg)

# Example
# python -m src.disp_pick --config configs/disp.yaml
# python -m src.disp_pick --config configs/disp.yaml --stack_window 30d --njobs 12 --verbose