"""
:module: src/eval_modes.py
:auth: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Evaluate conventional vs v1 outputs (runtime + correctness).

What this script does:
1) Scan NCF output_root for pairs:
     <file>.npz_cc_<vs>_conventional.npy
     <file>.npz_cc_<vs>_v1.npy
2) Compute correctness metrics (conv vs v1) per pair -> eval_ncf.csv
3) (Optional) Compare dispersion panels + picks if disp_config provided -> eval_disp.csv, eval_pick.csv
4) Summarize perf_cc logs (supports pid-suffixed CSVs) -> runtime_summary.csv
5) Make nicer plots (seaborn) -> outdir/plots/*.png

Notes:
- Ranked runtime plots avoid putting 100+ filenames on axes.
- Speedup row reports speedup for mean/median/p10/p90/total (not just median).
- Parallel eval supported via --njobs (uses ProcessPoolExecutor; safe on macOS spawn).
"""

from __future__ import annotations

import argparse
import logging
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# nice plots
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.error import (
    cosine_similarity_per_trace,
    load_npy,
    max_abs_error,
    pick_diff,
    rel_frobenius,
    spectral_compare,
    ssim_index,
)
from src.utils import get_cfg, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------
# Filename parsing / matching
# ----------------------------
_NCF_RE = re.compile(r"^(?P<filebase>[^.]+)(?:\.npz)?_cc_(?P<vs>\d{3})_(?P<mode>conventional|v1)\.npy$")

_DATE8_RE = re.compile(r"(?P<date>\d{8})")

def _extract_date8(s: str) -> Optional[str]:
    """
    Extract first YYYYMMDD occurrence from a string; returns None if not found.
    """
    m = _DATE8_RE.search(str(s))
    return m.group("date") if m else None

def _scan_ncf_pairs(ncf_root: Path) -> List[Tuple[Path, Path, str, int]]:
    """
    Find NCF pairs in ncf_root:
      <file>.npz_cc_<vs>_conventional.npy
      <file>.npz_cc_<vs>_v1.npy

    Returns list of (conv_path, v1_path, filebase, vs_idx)
    """
    conv: Dict[Tuple[str, int], Path] = {}
    v1: Dict[Tuple[str, int], Path] = {}

    for p in ncf_root.rglob("*.npy"):
        m = _NCF_RE.match(p.name)
        if not m:
            continue
        filebase = m.group("filebase")
        vs = int(m.group("vs"))
        mode = m.group("mode")
        key = (filebase, vs)
        if mode == "conventional":
            conv[key] = p
        else:
            v1[key] = p

    pairs: List[Tuple[Path, Path, str, int]] = []
    for key, pconv in conv.items():
        if key in v1:
            pv1 = v1[key]
            filebase, vs = key
            pairs.append((pconv, pv1, filebase, int(vs)))

    pairs.sort(key=lambda t: (t[2], t[3]))
    return pairs


def _disp_paths(
    disp_root: Path,
    *,
    filebase: str,
    vs_idx: int,
    window: str,
    mode: str,
) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """
    Builds specific dispersion paths using the filebase to avoid incorrect pairing.
    """
    vs_dir = disp_root / f"VS_{vs_idx:03d}"
    if not vs_dir.exists():
        return None, None, None
    
    # Replace wildcard '*' with the specific filebase
    name_panel = f"{filebase}_cc_{vs_idx:03d}_{window}_{mode}_fv_panel.npy"
    name_pick = f"{filebase}_cc_{vs_idx:03d}_{window}_{mode}_pick.npy"

    panel = vs_dir / name_panel
    pick = vs_dir / name_pick

    return (
        panel if panel.exists() else None,
        pick if pick.exists() else None,
        vs_dir
    )


# ----------------------------
# Perf CSV helpers
# ----------------------------
def _read_perf_csv_anydelim(perf_csv: Path) -> Optional[List[Dict[str, Any]]]:
    """
    Read perf_cc*.csv (supports tab or comma).
    Returns list of rows as dict with numeric conversion where possible.
    """
    if not perf_csv.exists():
        logger.warning("Perf CSV not found: %s", perf_csv)
        return None

    text = perf_csv.read_text().strip().splitlines()
    if not text:
        return None

    delim = "\t" if ("\t" in text[0]) else ","
    header = text[0].split(delim)
    rows: List[Dict[str, Any]] = []

    for line in text[1:]:
        parts = line.split(delim)
        if len(parts) != len(header):
            continue
        d: Dict[str, Any] = dict(zip(header, parts))

        # Convert numeric fields when possible
        for k in ("vs_idx", "nch", "npts_seg", "nseg", "max_lag_samples", "npair_chunk"):
            if k in d:
                try:
                    d[k] = int(float(d[k]))
                except Exception:
                    pass
        if "seconds_vs" in d:
            try:
                d["seconds_vs"] = float(d["seconds_vs"])
            except Exception:
                pass

        # normalize mode
        if "mode" in d:
            d["mode"] = str(d["mode"]).lower()

        rows.append(d)

    return rows


def load_perf_rows_glob(perf_path: Path) -> Optional[List[Dict[str, Any]]]:
    """
    Loads perf rows from:
      - perf_path if exists
      - else: glob for pid-suffixed files in same directory: <stem>*<suffix>
    """
    p = perf_path.expanduser().resolve()
    if p.exists() and p.is_file():
        return _read_perf_csv_anydelim(p)

    parent = p.parent
    stem = p.stem
    suffix = p.suffix
    candidates = sorted(parent.glob(f"{stem}*{suffix}"))

    if not candidates:
        logger.warning("No perf CSV candidates found for %s in %s", p.name, parent)
        return None

    all_rows: List[Dict[str, Any]] = []
    logger.info("Perf CSV not found exactly; loading %d candidates via glob.", len(candidates))
    for c in candidates:
        rr = _read_perf_csv_anydelim(c)
        if rr:
            all_rows.extend(rr)
    return all_rows if all_rows else None


# ----------------------------
# Evaluation core
# ----------------------------
def eval_ncf_pair(
    conv_path: Path,
    v1_path: Path,
    *,
    dt: float,
    f1: float,
    f2: float,
    eps: float = 1e-15,
    mmap: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate one NCF pair (conv vs v1).

    :param conv_path: conventional NCF .npy
    :param v1_path: v1 NCF .npy
    :param dt: sampling interval for NCF lag axis
    :param f1: spectral band low (Hz)
    :param f2: spectral band high (Hz)
    :param eps: stabilizer
    :param mmap: whether to use memmap for loading
    """
    R = load_npy(conv_path, mmap=mmap)
    V = load_npy(v1_path, mmap=mmap)

    rel_f = rel_frobenius(V, R, eps=eps)
    max_a = max_abs_error(V, R)

    cos = cosine_similarity_per_trace(V, R, eps=eps)
    cos_mean = float(np.mean(cos))
    cos_p05 = float(np.percentile(cos, 5))

    spec = spectral_compare(V, R, dt=dt, f1=f1, f2=f2, eps=eps)

    # per-trace relative L2
    num = np.linalg.norm(V - R, axis=1)
    den = np.linalg.norm(R, axis=1) + eps
    rel_l2 = num / den

    return {
        "rel_fro": float(rel_f),
        "max_abs": float(max_a),
        "mean_rel_l2_per_trace": float(np.mean(rel_l2)),
        "median_rel_l2_per_trace": float(np.median(rel_l2)),
        "p95_rel_l2_per_trace": float(np.percentile(rel_l2, 95)),
        "mean_cos_sim_per_trace": cos_mean,
        "p05_cos_sim_per_trace": cos_p05,
        "rel_spec_err_band": float(spec.rel_spec_err_band),
        "leak_ratio_ref": float(spec.leak_ratio_ref),
        "leak_ratio_test": float(spec.leak_ratio_test),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """
    Write list-of-dict rows to CSV.
    """
    if not rows:
        logger.warning("No rows to write: %s", path)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())

    lines = [",".join(keys)]
    for r in rows:
        vals = []
        for k in keys:
            v = r.get(k, "")
            if isinstance(v, float):
                vals.append(f"{v:.12g}")
            else:
                vals.append(str(v))
        lines.append(",".join(vals))

    path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote %d rows -> %s", len(rows), path)


def runtime_summary(rows: List[Dict[str, Any]], *, drop_first_vs: bool = True, njobs_cc: int = 1) -> List[Dict[str, Any]]:
    """
    Summarize perf rows into per-mode statistics.
    If drop_first_vs=True, drop the minimum vs_idx per file+mode (warm-up).

    Also appends a speedup row "speedup(conv/v1)" where each numeric column is conv/v1.
    """
    if not rows:
        return []

    # Group by (file, mode)
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        if "file" not in r or "mode" not in r or "seconds_vs" not in r:
            continue
        key = (str(r["file"]), str(r["mode"]))
        grouped.setdefault(key, []).append(r)

    cleaned: List[Dict[str, Any]] = []
    for _, rr in grouped.items():
        rr2 = sorted(rr, key=lambda x: int(x.get("vs_idx", 0)))
        if drop_first_vs and rr2:
            rr2 = rr2[1:]
        cleaned.extend(rr2)

    by_mode: Dict[str, List[float]] = {}
    for r in cleaned:
        mode = str(r["mode"])
        t = float(r["seconds_vs"])
        by_mode.setdefault(mode, []).append(t)

    out: List[Dict[str, Any]] = []
    for mode, ts in sorted(by_mode.items()):
        a = np.asarray(ts, dtype=np.float64)
        out.append(
            {
                "mode": mode,
                "n_vs": int(a.size),
                "mean_seconds_vs": float(np.mean(a)),
                "median_seconds_vs": float(np.median(a)),
                "p10_seconds_vs": float(np.percentile(a, 10)),
                "p90_seconds_vs": float(np.percentile(a, 90)),
                "total_seconds_all_vs": float(np.sum(a)),
                "cc_njobs": int(njobs_cc),
            }
        )

    modes = {d["mode"]: d for d in out}

    def safe_div(a: float, b: float) -> float:
        return float(a) / float(b) if float(b) > 0 else float("nan")

    if "conventional" in modes and "v1" in modes:
        conv = modes["conventional"]
        v1 = modes["v1"]
        out.append(
            {
                "mode": "speedup(conv/v1)",
                "n_vs": "",
                "mean_seconds_vs": safe_div(conv["mean_seconds_vs"], v1["mean_seconds_vs"]),
                "median_seconds_vs": safe_div(conv["median_seconds_vs"], v1["median_seconds_vs"]),
                "p10_seconds_vs": safe_div(conv["p10_seconds_vs"], v1["p10_seconds_vs"]),
                "p90_seconds_vs": safe_div(conv["p90_seconds_vs"], v1["p90_seconds_vs"]),
                "total_seconds_all_vs": safe_div(conv["total_seconds_all_vs"], v1["total_seconds_all_vs"]),
                "cc_njobs": int(njobs_cc),
            }
        )

    return out

# ----------------------------
# Plotting (seaborn)
# ----------------------------
def _set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.95)


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_runtime_distribution(
    perf_rows: List[Dict[str, Any]],
    out_png: Path,
    *,
    title: str = "",
    logy: bool = False,
) -> None:
    df = pd.DataFrame(perf_rows).copy()
    if df.empty or "mode" not in df or "seconds_vs" not in df:
        return

    df["mode"] = df["mode"].astype(str).str.lower()
    df["seconds_vs"] = pd.to_numeric(df["seconds_vs"], errors="coerce")
    df = df.dropna(subset=["seconds_vs", "mode"])

    plt.figure(figsize=(10, 5))
    ax = sns.violinplot(data=df, x="mode", y="seconds_vs", inner=None, cut=0)
    sns.boxplot(data=df, x="mode", y="seconds_vs", width=0.25, showfliers=False, ax=ax)
    sns.stripplot(data=df, x="mode", y="seconds_vs", size=2, alpha=0.35, jitter=0.25, ax=ax)

    ax.set_xlabel("")
    ax.set_ylabel("seconds_vs (per virtual source)")
    ax.set_title(title or "Per-VS runtime (distribution)")
    if logy:
        ax.set_yscale("log")

    sns.despine()
    _savefig(out_png)


def plot_runtime_total_per_file(
    perf_rows: List[Dict[str, Any]],
    out_png: Path,
    *,
    title: str = "",
    top_k: int = 50,
    annotate_top: int = 10,
) -> None:
    """
    Total runtime per file (sum over VS), ranked by total runtime.
    X-axis uses rank (1..top_k), NOT filenames.
    """
    df = pd.DataFrame(perf_rows).copy()
    if df.empty or not {"file", "mode", "seconds_vs"}.issubset(df.columns):
        return

    df["mode"] = df["mode"].astype(str).str.lower()
    df["seconds_vs"] = pd.to_numeric(df["seconds_vs"], errors="coerce")
    df = df.dropna(subset=["seconds_vs", "mode", "file"])

    df_tot = (
        df.groupby(["file", "mode"], as_index=False)["seconds_vs"]
        .sum()
        .rename(columns={"seconds_vs": "total_seconds"})
    )

    # rank files by conventional total if available; otherwise by overall total
    if (df_tot["mode"] == "conventional").any():
        ref = df_tot[df_tot["mode"] == "conventional"][["file", "total_seconds"]].copy()
        ref = ref.sort_values("total_seconds", ascending=False).reset_index(drop=True)
    else:
        ref = (
            df_tot.groupby("file", as_index=False)["total_seconds"]
            .sum()
            .sort_values("total_seconds", ascending=False)
            .reset_index(drop=True)
        )

    ref["rank"] = np.arange(1, len(ref) + 1)
    if top_k is not None and top_k > 0:
        ref = ref.head(int(top_k))

    df_plot = df_tot.merge(ref[["file", "rank"]], on="file", how="inner")

    plt.figure(figsize=(12, 5))
    ax = sns.barplot(
        data=df_plot,
        x="rank",
        y="total_seconds",
        hue="mode",
        dodge=True,
        errorbar=None,
    )

    ax.set_xlabel("file rank (1 = slowest by total runtime)")
    ax.set_ylabel("total seconds (sum over VS)")
    ax.set_title(title or f"Total runtime per file (top {len(ref)} slowest)")
    ax.legend(title="mode", frameon=True, loc="best")

    if annotate_top and annotate_top > 0:
        top_files = ref.head(int(annotate_top))
        for _, rr in top_files.iterrows():
            rank = int(rr["rank"])
            fname = str(rr["file"])

            sub = df_plot[df_plot["rank"] == rank]
            if sub.empty:
                continue

            if (sub["mode"] == "conventional").any():
                y = float(sub[sub["mode"] == "conventional"]["total_seconds"].max())
            else:
                y = float(sub["total_seconds"].max())

            ax.text(rank - 1, y, fname, rotation=90, va="bottom", ha="center", fontsize=8)

    sns.despine()
    _savefig(out_png)


def plot_runtime_cumulative(
    perf_rows: List[Dict[str, Any]],
    out_png: Path,
    *,
    title: str = "",
) -> None:
    """
    Cumulative runtime over files ranked by conventional total runtime (slowest -> fastest).
    X-axis is rank, not filename.
    """
    df = pd.DataFrame(perf_rows).copy()
    if df.empty or not {"file", "mode", "seconds_vs"}.issubset(df.columns):
        return

    df["mode"] = df["mode"].astype(str).str.lower()
    df["seconds_vs"] = pd.to_numeric(df["seconds_vs"], errors="coerce")
    df = df.dropna(subset=["seconds_vs", "mode", "file"])

    df_tot = (
        df.groupby(["file", "mode"], as_index=False)["seconds_vs"]
        .sum()
        .rename(columns={"seconds_vs": "total_seconds"})
    )

    # rank files by conventional if available
    if (df_tot["mode"] == "conventional").any():
        ref = df_tot[df_tot["mode"] == "conventional"][["file", "total_seconds"]].copy()
        ref = ref.sort_values("total_seconds", ascending=False).reset_index(drop=True)
    else:
        ref = (
            df_tot.groupby("file", as_index=False)["total_seconds"]
            .sum()
            .sort_values("total_seconds", ascending=False)
            .reset_index(drop=True)
        )
    ref["rank"] = np.arange(1, len(ref) + 1)

    df_plot = df_tot.merge(ref[["file", "rank"]], on="file", how="inner")
    df_plot = df_plot.sort_values(["mode", "rank"]).reset_index(drop=True)

    df_plot["cum_seconds"] = df_plot.groupby("mode")["total_seconds"].cumsum()

    plt.figure(figsize=(10, 5))
    ax = sns.lineplot(data=df_plot, x="rank", y="cum_seconds", hue="mode", marker="o")

    ax.set_xlabel("file rank (1 = slowest)")
    ax.set_ylabel("cumulative seconds (sum over VS)")
    ax.set_title(title or "Cumulative runtime vs ranked files (slowest→fastest)")
    ax.legend(title="mode", frameon=True, loc="best")

    sns.despine()
    _savefig(out_png)


def plot_correctness_metrics(
    ncf_rows: List[Dict[str, Any]],
    out_png: Path,
    *,
    title: str = "",
) -> None:
    df = pd.DataFrame(ncf_rows).copy()
    if df.empty:
        return

    metric_cols = [
        "rel_fro",
        "max_abs",
        "median_rel_l2_per_trace",
        "p95_rel_l2_per_trace",
        "mean_cos_sim_per_trace",
        "rel_spec_err_band",
    ]
    keep = ["file", "vs_idx"] + [c for c in metric_cols if c in df.columns]
    df = df[keep].copy()

    for c in keep:
        if c in ("file",):
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["vs_idx"])
    df = df.sort_values(["file", "vs_idx"]).reset_index(drop=True)
    df["pair_idx"] = np.arange(1, len(df) + 1)

    long = df.melt(
        id_vars=["pair_idx"],
        value_vars=[c for c in metric_cols if c in df.columns],
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])

    if long.empty:
        return

    plt.figure(figsize=(12, 5))
    ax = sns.lineplot(data=long, x="pair_idx", y="value", hue="metric", alpha=0.9)

    ax.set_xlabel("pair index (sorted by file, vs)")
    ax.set_ylabel("value")
    ax.set_title(title or "Correctness metrics (v1 vs conventional)")
    ax.legend(title="metric", frameon=True, loc="best")

    sns.despine()
    _savefig(out_png)


def plot_disp_ssim(
    disp_rows: List[Dict[str, Any]],
    out_png: Path,
    *,
    title: str = "",
) -> None:
    df = pd.DataFrame(disp_rows).copy()
    if df.empty or "ssim" not in df.columns:
        return

    df["ssim"] = pd.to_numeric(df["ssim"], errors="coerce")
    df = df.dropna(subset=["ssim"])

    # Safety clip (SSIM ∈ [0, 1])
    df["ssim"] = df["ssim"].clip(0.0, 1.0)

    plt.figure(figsize=(9, 5))
    ax = sns.histplot(
        data=df,
        x="ssim",
        bins=15,             
        kde=False,
        fill=True,
        element="bars",
        color="red",
        edgecolor="black",
        linewidth=1.2,         
        alpha=1.0,           
        shrink=0.9,
    )

    ax.set_title(title or "Dispersion panel SSIM (v1 vs conventional)")
    ax.set_xlabel("SSIM")
    ax.set_ylabel("count")

    ax.set_xlim(0.0, 1.1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0", "1"])

    sns.despine()
    _savefig(out_png)


# ----------------------------
# Parallel worker
# ----------------------------
def _eval_one_pair_worker(args: tuple) -> tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    (
        conv_path_s,
        v1_path_s,
        filebase,
        vs_idx,
        dt,
        f1,
        f2,
        mmap,
        disp_results_root_s,
        disp_window,
        njobs_cc,
    ) = args

    conv_path = Path(conv_path_s)
    v1_path = Path(v1_path_s)

    # --- NCF metrics ---
    ncf_row: Dict[str, Any] = {
        "file": str(filebase) + ".npz",
        "vs_idx": int(vs_idx),
        "conv_path": str(conv_path),
        "v1_path": str(v1_path),
        "cc_njobs": int(njobs_cc),
    }
    try:
        metrics = eval_ncf_pair(conv_path, v1_path, dt=float(dt), f1=float(f1), f2=float(f2), mmap=bool(mmap))
        ncf_row.update(metrics)
    except Exception as e:
        ncf_row["error"] = str(e)

    disp_row: Optional[Dict[str, Any]] = None
    pick_row: Optional[Dict[str, Any]] = None

    # --- Dispersion (corrected pairing) ---
    if disp_results_root_s is not None:
        disp_root = Path(disp_results_root_s)

        # Etracting YYYYMMDD robustly; fallback to original string
        date8 = _extract_date8(filebase)
        disp_filebase = date8 if date8 is not None else filebase

        try:
            panel_conv, pick_conv, _ = _disp_paths(
                disp_root, 
                filebase=disp_filebase, # Use the date-only version here
                vs_idx=int(vs_idx), 
                window=str(disp_window), 
                mode="conventional"
            )
            panel_v1, pick_v1, _ = _disp_paths(
                disp_root, 
                filebase=disp_filebase, # Use the date-only version here
                vs_idx=int(vs_idx), 
                window=str(disp_window), 
                mode="v1"
            )

            if panel_conv and panel_v1:
                try:
                    D_ref = load_npy(panel_conv, mmap=bool(mmap))
                    D_v1 = load_npy(panel_v1, mmap=bool(mmap))
                    val = ssim_index(D_ref, D_v1)
                    disp_row = {
                        "file": filebase,
                        "vs_idx": int(vs_idx),
                        "window": str(disp_window),
                        "panel_conv": str(panel_conv.name),
                        "panel_v1": str(panel_v1.name),
                        "ssim": float(val),
                        "cc_njobs": int(njobs_cc),
                    }
                except Exception as e:
                    disp_row = {"file": filebase, "vs_idx": int(vs_idx), "error": f"SSIM error: {e}"}

            if pick_conv and pick_v1:
                try:
                    P_ref = load_npy(pick_conv, mmap=bool(mmap))
                    P_tst = load_npy(pick_v1, mmap=bool(mmap))
                    pe = pick_diff(P_ref, P_tst)
                    d = asdict(pe)
                    d.update({
                        "file": filebase,
                        "vs_idx": int(vs_idx),
                        "window": str(disp_window),
                        "cc_njobs": int(njobs_cc),
                    })
                    pick_row = d
                except Exception as e:
                    pick_row = {"file": filebase, "vs_idx": int(vs_idx), "error": f"Pick error: {e}"}
        except Exception as e:
            disp_row = {"file": filebase, "vs_idx": int(vs_idx), "error": str(e)}

    return ncf_row, disp_row, pick_row

# ----------------------------
# Main
# ----------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate conventional vs v1 (runtime + errors).")
    p.add_argument("--cc_config", type=str, required=True, help="Path to cc.yaml")
    p.add_argument("--disp_config", type=str, required=False, default=None, help="Path to disp.yaml (optional)")
    p.add_argument("--outdir", type=str, default="./data/runlogs/eval_modes", help="Output directory for reports")
    p.add_argument("--mmap", action="store_true", help="Use memmap to load .npy files")
    p.add_argument("--drop_first_vs", action="store_true", help="Drop first VS per file+mode in runtime summary")
    p.add_argument("--plots", action="store_true", help="Generate seaborn plots")
    p.add_argument("--logy_runtime", action="store_true", help="Log-scale y-axis for per-VS runtime distribution")
    p.add_argument("--max_files_bar", type=int, default=50, help="Top-K slowest files to show in ranked bar plot")
    p.add_argument("--title", type=str, default="", help="Optional plot title prefix")
    p.add_argument("--njobs", type=int, default=1, help="Parallel workers for pair eval (ProcessPool)")
    return p.parse_args(args=argv)


def main(
    cc_config: str | Path,
    disp_config: Optional[str | Path],
    outdir: str | Path,
    *,
    mmap: bool,
    drop_first_vs: bool,
    make_plots: bool,
    logy_runtime: bool,
    max_files_bar: int,
    title: str,
    njobs: int,
) -> None:
    cc_cfg = load_config(cc_config)
    disp_cfg = load_config(disp_config) if disp_config else {}

    # --- CC runtime parallelism (source of truth) ---
    njobs_cc = int(get_cfg(cc_cfg, ["runtime", "njobs"], 1))

    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- paths from cc.yaml ----
    ncf_root = Path(get_cfg(cc_cfg, ["paths", "output_root"], required=True)).expanduser().resolve()

    # ---- signal params from cc.yaml ----
    fs_raw = float(get_cfg(cc_cfg, ["data", "fs_raw"], required=True))
    decimation = int(get_cfg(cc_cfg, ["preprocess", "decimation"], 1))
    fs_proc = fs_raw / float(decimation)

    # NCF lag sampling interval
    dt = 1.0 / float(fs_proc)

    f1 = float(get_cfg(cc_cfg, ["preprocess", "f1"], 1.0))
    f2 = float(get_cfg(cc_cfg, ["preprocess", "f2"], 10.0))

    # ---- dispersion root from disp.yaml (optional) ----
    disp_results_root: Optional[Path] = None
    disp_window = "daily"
    if disp_cfg:
        root = get_cfg(disp_cfg, ["io", "results_root"], required=False) or ""
        if str(root).strip():
            disp_window = str(get_cfg(disp_cfg, ["io", "stack_window"], "daily"))
            # Always set the intended root; existence can be checked per-file later
            disp_results_root = Path(root).expanduser().resolve() / disp_window

    logger.info("NCF root: %s", ncf_root)
    if disp_results_root:
        logger.info("Disp root: %s", disp_results_root)

    # ---- NCF pair scan ----
    pairs = _scan_ncf_pairs(ncf_root)
    logger.info("Found %d NCF pairs (conv,v1).", len(pairs))
    if not pairs:
        logger.warning("No NCF pairs found in %s", ncf_root)
        return

    # ---- Evaluate pairs (parallel optional) ----
    nj = max(1, int(njobs))
    tasks = [
        (
            str(conv_path),
            str(v1_path),
            str(filebase).replace(".npz", ""),
            int(vs_idx),
            float(dt),
            float(f1),
            float(f2),
            bool(mmap),
            str(disp_results_root) if disp_results_root is not None else None,
            str(disp_window),
            int(njobs_cc)
        )
        for (conv_path, v1_path, filebase, vs_idx) in pairs
    ]

    ncf_rows: List[Dict[str, Any]] = []
    disp_rows: List[Dict[str, Any]] = []
    pick_rows: List[Dict[str, Any]] = []

    if nj == 1:
        for t in tasks:
            ncf_row, disp_row, pick_row = _eval_one_pair_worker(t)
            ncf_rows.append(ncf_row)
            if disp_row:
                disp_rows.append(disp_row)
            if pick_row:
                pick_rows.append(pick_row)
    else:
        logger.info("Parallel eval: njobs=%d over %d pairs", nj, len(tasks))
        with ProcessPoolExecutor(max_workers=nj) as ex:
            futs = [ex.submit(_eval_one_pair_worker, t) for t in tasks]
            for fut in as_completed(futs):
                ncf_row, disp_row, pick_row = fut.result()
                ncf_rows.append(ncf_row)
                if disp_row:
                    disp_rows.append(disp_row)
                if pick_row:
                    pick_rows.append(pick_row)

    # deterministic ordering for diffs
    ncf_rows.sort(key=lambda r: (str(r.get("file", "")), int(r.get("vs_idx", -1))))

    disp_rows.sort(key=lambda r: (str(r.get("file", "")), int(r.get("vs_idx", -1))))
    pick_rows.sort(key=lambda r: (str(r.get("file", "")), int(r.get("vs_idx", -1))))

    # ---- add run metadata columns to all outputs ----
    run_meta = {
        "njobs": int(njobs),
        "mmap": bool(mmap),
        "drop_first_vs": bool(drop_first_vs),
    }

    for r in ncf_rows:
        r.update(run_meta)
    for r in disp_rows:
        r.update(run_meta)
    for r in pick_rows:
        r.update(run_meta)
    
    # ---- Write reports ----
    write_csv(outdir / "eval_ncf.csv", ncf_rows)
    if disp_rows:
        write_csv(outdir / "eval_disp.csv", disp_rows)
    if pick_rows:
        write_csv(outdir / "eval_pick.csv", pick_rows)

    # ---- Runtime summary from perf_cc.csv (supports pid-suffixed) ----
    perf_csv = Path(get_cfg(cc_cfg, ["perf", "out_path"], "./data/runlogs/perf_cc.csv")).expanduser().resolve()
    perf_rows = load_perf_rows_glob(perf_csv)

    if perf_rows:
        summ = runtime_summary(perf_rows, drop_first_vs=drop_first_vs, njobs_cc=njobs_cc)
        for r in summ:
            r["njobs"] = int(njobs)
            r["drop_first_vs"] = bool(drop_first_vs)

        write_csv(outdir / "runtime_summary.csv", summ)
    else:
        logger.warning("No perf rows loaded (perf_cc*.csv missing or empty).")

    # ---- Plots ----
    if make_plots:
        _set_plot_style()
        plots_dir = outdir / "plots"
        
        cc_tag = f"cc_njobs={njobs_cc}"
        prefix = (title.strip() + " | ") if title.strip() else ""
        prefix = prefix + cc_tag + " | "

        if perf_rows:
            plot_runtime_distribution(
                perf_rows,
                plots_dir / "runtime_seconds_vs_dist.png",
                title=prefix + "Per-VS runtime",
                logy=bool(logy_runtime),
            )
            # plot_runtime_total_per_file(
            #     perf_rows,
            #     plots_dir / "runtime_total_per_file_ranked.png",
            #     title=prefix + "Total per file (ranked)",
            #     top_k=int(max_files_bar),
            #     annotate_top=min(10, int(max_files_bar)),
            # )
            plot_runtime_cumulative(
                perf_rows,
                plots_dir / "runtime_cumulative_ranked.png",
                title=prefix + "Cumulative runtime (ranked)",
            )

        plot_correctness_metrics(
            ncf_rows,
            plots_dir / "correctness_metrics_ranked.png",
            title=prefix + "Correctness (ranked pairs)",
        )

        if disp_rows:
            plot_disp_ssim(
                disp_rows,
                plots_dir / "disp_ssim_hist.png",
                title=prefix + "Dispersion SSIM",
            )

        logger.info("Saved plots to: %s", plots_dir)

    logger.info("Done. Reports in: %s", outdir)


if __name__ == "__main__":
    args = parse_args()
    main(
        cc_config=args.cc_config,
        disp_config=args.disp_config,
        outdir=args.outdir,
        mmap=bool(args.mmap),
        drop_first_vs=bool(args.drop_first_vs),
        make_plots=bool(args.plots),
        logy_runtime=bool(args.logy_runtime),
        max_files_bar=int(args.max_files_bar),
        title=str(args.title),
        njobs=int(args.njobs),
    )

# Example:
# python -m src.eval_modes \
#   --cc_config configs/cc.yaml \
#   --disp_config configs/disp.yaml \
#   --outdir data/runlogs/eval_modes \
#   --mmap \
#   --drop_first_vs \
#   --plots \
#   --logy_runtime \
#   --max_files_bar 50 \
#   --title "GPU test" \
#   --njobs 8