"""
:module: src/eval_modes.py
:auth: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Evaluate conventional vs v1 outputs (runtime + correctness).

What this script does:
1) Scan NCF output_root for pairs:
     <filebase>_cc_<vs>_conventional.npy
     <filebase>_cc_<vs>_v1.npy
2) Compute correctness metrics (conv vs v1) per pair -> eval_ncf.csv
3) (Optional) Evaluate DISP outputs by scanning disp results_root (independent of NCF scan)
     - pairs *_fv_panel.npy (conv vs v1) -> eval_disp.csv
     - pairs *_pick.npy (conv vs v1) -> eval_pick.csv
4) Summarize perf_cc logs (supports pid-suffixed CSVs) -> runtime_summary.csv
5) Make plots (seaborn) -> outdir/plots/*.png

Important runtime note:
- If perf rows only contain per-VS timing (seconds_vs), then "total_seconds_all_vs"
  is NOT wall time; it's "sum of per-VS seconds". This is useful but *not* makespan.
- If perf rows include a per-file wall time column (e.g., wall_sec, seconds_file, wall_time),
  this script will additionally produce true "per-file wall-time" plots + summary.
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
_NCF_RE = re.compile(
    r"^(?P<filebase>.+?)_cc_(?P<vs>\d{3})_(?P<mode>conventional|v1)\.npy$"
)

# Disp outputs from disp_pick.py (recommended):
#   <base>_fv_panel.npy, <base>_pick.npy, <base>_meta.json
# where <base> ends with ..._conventional or ..._v1
_DISP_PANEL_RE = re.compile(r"^(?P<base>.+?)_fv_panel\.npy$")
_DISP_PICK_RE = re.compile(r"^(?P<base>.+?)_pick\.npy$")

# method token (last underscore-separated)
_KNOWN_CC_METHODS = {"v1", "conventional"}


def _extract_cc_method_from_base(base: str) -> Optional[str]:
    last = str(base).split("_")[-1].lower()
    return last if last in _KNOWN_CC_METHODS else None


def _base_without_method(base: str) -> str:
    """
    If base ends with _v1 or _conventional, strip it for pairing.
    """
    parts = str(base).split("_")
    if parts and parts[-1].lower() in _KNOWN_CC_METHODS:
        return "_".join(parts[:-1])
    return str(base)


def _scan_ncf_pairs(ncf_root: Path) -> List[Tuple[Path, Path, str, int]]:
    """
    Find NCF pairs in ncf_root:
      <filebase>_cc_<vs>_conventional.npy
      <filebase>_cc_<vs>_v1.npy

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


def _scan_disp_pairs(disp_root: Path) -> Tuple[List[Tuple[Path, Path, str]], List[Tuple[Path, Path, str]]]:
    """
    Scan dispersion results_root/<stack_window>/ (may include VS subdirs) for:
      - fv_panel pairs
      - pick pairs

    Pairing rule:
      base_key := base_without_method(<base>)
      method   := last token in <base> (v1/conventional)

    Returns:
      (panel_pairs, pick_pairs)
      where each item is (conv_path, v1_path, base_key)
    """
    # map: base_key -> {"conventional": path, "v1": path}
    panels: Dict[str, Dict[str, Path]] = {}
    picks: Dict[str, Dict[str, Path]] = {}

    for p in disp_root.rglob("*.npy"):
        name = p.name

        m1 = _DISP_PANEL_RE.match(name)
        if m1:
            base = m1.group("base")
            method = _extract_cc_method_from_base(base)
            if method in _KNOWN_CC_METHODS:
                key = _base_without_method(base)
                panels.setdefault(key, {})[method] = p
            continue

        m2 = _DISP_PICK_RE.match(name)
        if m2:
            base = m2.group("base")
            method = _extract_cc_method_from_base(base)
            if method in _KNOWN_CC_METHODS:
                key = _base_without_method(base)
                picks.setdefault(key, {})[method] = p
            continue

    panel_pairs: List[Tuple[Path, Path, str]] = []
    for key, mm in panels.items():
        if "conventional" in mm and "v1" in mm:
            panel_pairs.append((mm["conventional"], mm["v1"], key))

    pick_pairs: List[Tuple[Path, Path, str]] = []
    for key, mm in picks.items():
        if "conventional" in mm and "v1" in mm:
            pick_pairs.append((mm["conventional"], mm["v1"], key))

    panel_pairs.sort(key=lambda t: t[2])
    pick_pairs.sort(key=lambda t: t[2])

    return panel_pairs, pick_pairs


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

        for k in ("seconds_vs", "seconds", "wall_sec", "wall_time", "seconds_file"):
            if k in d:
                try:
                    d[k] = float(d[k])
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


def _infer_wall_time_col(df: pd.DataFrame) -> Optional[str]:
    """
    Try to detect a per-file wall-time column in perf rows.
    Preference order:
      wall_sec, wall_time, seconds_file, seconds
    """
    for c in ("wall_sec", "wall_time", "seconds_file"):
        if c in df.columns:
            return c
    # "seconds" is common in other perf logs; accept if it looks like per-file (not per-vs)
    if "seconds" in df.columns and "vs_idx" not in df.columns:
        return "seconds"
    return None


# ----------------------------
# Evaluation core (NCF)
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
    """
    R = load_npy(conv_path, mmap=mmap)
    V = load_npy(v1_path, mmap=mmap)

    rel_f = rel_frobenius(V, R, eps=eps)
    max_a = max_abs_error(V, R)

    cos = cosine_similarity_per_trace(V, R, eps=eps)
    cos_mean = float(np.mean(cos))
    cos_p05 = float(np.percentile(cos, 5))
    cos_neg_frac = float(np.mean(cos < 0.0))  # sign-flip / anti-phase detector

    spec = spectral_compare(V, R, dt=dt, f1=f1, f2=f2, eps=eps)

    # per-trace relative L2
    num = np.linalg.norm(V - R, axis=1)
    den = np.linalg.norm(R, axis=1) + eps
    rel_l2 = num / den

    # lag-energy curve mismatch (sensitive to subtle artifacts)
    # E(τ) = sum_ch ncf(ch, τ)^2
    eR = np.sum(R * R, axis=0)
    eV = np.sum(V * V, axis=0)
    e_num = np.linalg.norm(eV - eR)
    e_den = np.linalg.norm(eR) + eps
    rel_energy_curve = float(e_num / e_den)

    return {
        "rel_fro": float(rel_f),
        "max_abs": float(max_a),
        "mean_rel_l2_per_trace": float(np.mean(rel_l2)),
        "median_rel_l2_per_trace": float(np.median(rel_l2)),
        "p95_rel_l2_per_trace": float(np.percentile(rel_l2, 95)),
        "mean_cos_sim_per_trace": cos_mean,
        "p05_cos_sim_per_trace": cos_p05,
        "neg_cos_frac_per_trace": cos_neg_frac,
        "rel_spec_err_band": float(spec.rel_spec_err_band),
        "leak_ratio_ref": float(spec.leak_ratio_ref),
        "leak_ratio_test": float(spec.leak_ratio_test),
        "rel_energy_curve": rel_energy_curve,
    }


# ----------------------------
# Evaluation core (DISP)
# ----------------------------
def eval_disp_panel_pair(
    conv_panel: Path,
    v1_panel: Path,
    *,
    mmap: bool = False,
) -> Dict[str, Any]:
    D_ref = load_npy(conv_panel, mmap=mmap)
    D_v1 = load_npy(v1_panel, mmap=mmap)
    val = ssim_index(D_ref, D_v1)
    return {"ssim": float(val)}


def eval_pick_pair(
    conv_pick: Path,
    v1_pick: Path,
    *,
    mmap: bool = False,
) -> Dict[str, Any]:
    P_ref = load_npy(conv_pick, mmap=mmap)
    P_tst = load_npy(v1_pick, mmap=mmap)
    pe = pick_diff(P_ref, P_tst)
    return asdict(pe)


# ----------------------------
# CSV writer
# ----------------------------
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


# ----------------------------
# Runtime summary
# ----------------------------
def runtime_summary_vs(
    rows: List[Dict[str, Any]],
    *,
    drop_first_vs: bool = True,
    njobs_cc: int = 1,
) -> List[Dict[str, Any]]:
    """
    Summarize per-VS timings (seconds_vs) into per-mode stats.
    This is NOT wall time if you used multiprocessing.
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
                "metric_type": "sum_of_vs_seconds",
                "n_vs": int(a.size),
                "mean": float(np.mean(a)),
                "median": float(np.median(a)),
                "p10": float(np.percentile(a, 10)),
                "p90": float(np.percentile(a, 90)),
                "total": float(np.sum(a)),
                "cc_njobs": int(njobs_cc),
                "note": "NOT wall time under multiprocessing; sums per-VS seconds.",
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
                "metric_type": "sum_of_vs_seconds",
                "n_vs": "",
                "mean": safe_div(conv["mean"], v1["mean"]),
                "median": safe_div(conv["median"], v1["median"]),
                "p10": safe_div(conv["p10"], v1["p10"]),
                "p90": safe_div(conv["p90"], v1["p90"]),
                "total": safe_div(conv["total"], v1["total"]),
                "cc_njobs": int(njobs_cc),
                "note": "Speedup in sum-of-VS seconds (not makespan).",
            }
        )

    return out


def runtime_summary_wall(
    perf_rows: List[Dict[str, Any]],
    *,
    njobs_cc: int,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    If perf logs contain a per-file wall-time column, summarize it.
    Returns (rows, wall_col_used).
    """
    if not perf_rows:
        return [], None

    df = pd.DataFrame(perf_rows).copy()
    if df.empty or "mode" not in df.columns:
        return [], None

    wall_col = _infer_wall_time_col(df)
    if wall_col is None or wall_col not in df.columns:
        return [], None

    # require file id
    if "file" not in df.columns:
        return [], None

    df["mode"] = df["mode"].astype(str).str.lower()
    df[wall_col] = pd.to_numeric(df[wall_col], errors="coerce")
    df = df.dropna(subset=[wall_col, "file", "mode"])

    # per file summary: if multiple rows per file exist, take max as "makespan"
    # (common when many rows are written during processing)
    df_file = (
        df.groupby(["file", "mode"], as_index=False)[wall_col]
        .max()
        .rename(columns={wall_col: "wall_sec_file"})
    )

    out: List[Dict[str, Any]] = []
    for mode, sub in df_file.groupby("mode"):
        a = sub["wall_sec_file"].to_numpy(dtype=np.float64)
        out.append(
            {
                "mode": str(mode),
                "metric_type": "wall_time_per_file",
                "n_files": int(a.size),
                "mean": float(np.mean(a)),
                "median": float(np.median(a)),
                "p10": float(np.percentile(a, 10)),
                "p90": float(np.percentile(a, 90)),
                "total": float(np.sum(a)),
                "cc_njobs": int(njobs_cc),
                "note": f"Derived from perf column '{wall_col}' (per-file makespan).",
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
                "metric_type": "wall_time_per_file",
                "n_files": "",
                "mean": safe_div(conv["mean"], v1["mean"]),
                "median": safe_div(conv["median"], v1["median"]),
                "p10": safe_div(conv["p10"], v1["p10"]),
                "p90": safe_div(conv["p90"], v1["p90"]),
                "total": safe_div(conv["total"], v1["total"]),
                "cc_njobs": int(njobs_cc),
                "note": "Speedup in per-file wall time (makespan).",
            }
        )

    return out, wall_col


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


def plot_runtime_distribution_vs(
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


def plot_runtime_cumulative_ranked_vs(
    perf_rows: List[Dict[str, Any]],
    out_png: Path,
    *,
    title: str = "",
) -> None:
    """
    Cumulative sum of per-file totals where per-file total = sum(seconds_vs over VS).
    NOTE: this is *not wall time* under parallel processing.
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
        .rename(columns={"seconds_vs": "total_seconds_vs_sum"})
    )

    # rank files by conventional if available
    if (df_tot["mode"] == "conventional").any():
        ref = df_tot[df_tot["mode"] == "conventional"][["file", "total_seconds_vs_sum"]].copy()
        ref = ref.sort_values("total_seconds_vs_sum", ascending=False).reset_index(drop=True)
    else:
        ref = (
            df_tot.groupby("file", as_index=False)["total_seconds_vs_sum"]
            .sum()
            .sort_values("total_seconds_vs_sum", ascending=False)
            .reset_index(drop=True)
        )
    ref["rank"] = np.arange(1, len(ref) + 1)

    df_plot = df_tot.merge(ref[["file", "rank"]], on="file", how="inner")
    df_plot = df_plot.sort_values(["mode", "rank"]).reset_index(drop=True)
    df_plot["cum_seconds"] = df_plot.groupby("mode")["total_seconds_vs_sum"].cumsum()

    plt.figure(figsize=(10, 5))
    ax = sns.lineplot(data=df_plot, x="rank", y="cum_seconds", hue="mode", marker="o")

    ax.set_xlabel("file rank (1 = slowest)")
    ax.set_ylabel("cumulative seconds (sum of per-VS seconds)")
    ax.set_title(title or "Cumulative runtime vs ranked files (sum of per-VS seconds)")
    ax.legend(title="mode", frameon=True, loc="best")

    sns.despine()
    _savefig(out_png)


def plot_runtime_wall_per_file_ranked(
    perf_rows: List[Dict[str, Any]],
    out_png: Path,
    *,
    title: str = "",
) -> None:
    """
    True makespan plot if perf rows contain a per-file wall-time column.
    """
    df = pd.DataFrame(perf_rows).copy()
    if df.empty or not {"file", "mode"}.issubset(df.columns):
        return

    wall_col = _infer_wall_time_col(df)
    if wall_col is None or wall_col not in df.columns:
        return

    df["mode"] = df["mode"].astype(str).str.lower()
    df[wall_col] = pd.to_numeric(df[wall_col], errors="coerce")
    df = df.dropna(subset=["file", "mode", wall_col])

    df_file = (
        df.groupby(["file", "mode"], as_index=False)[wall_col]
        .max()
        .rename(columns={wall_col: "wall_sec_file"})
    )

    # rank by conventional wall per-file if available
    if (df_file["mode"] == "conventional").any():
        ref = df_file[df_file["mode"] == "conventional"][["file", "wall_sec_file"]].copy()
        ref = ref.sort_values("wall_sec_file", ascending=False).reset_index(drop=True)
    else:
        ref = (
            df_file.groupby("file", as_index=False)["wall_sec_file"]
            .sum()
            .sort_values("wall_sec_file", ascending=False)
            .reset_index(drop=True)
        )
    ref["rank"] = np.arange(1, len(ref) + 1)

    df_plot = df_file.merge(ref[["file", "rank"]], on="file", how="inner")
    df_plot = df_plot.sort_values(["rank", "mode"])

    plt.figure(figsize=(12, 5))
    ax = sns.lineplot(data=df_plot, x="rank", y="wall_sec_file", hue="mode", marker="o")
    ax.set_xlabel("file rank (1 = slowest)")
    ax.set_ylabel("wall seconds per file (makespan)")
    ax.set_title(title or f"Per-file wall time vs ranked files (from '{wall_col}')")
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
        "neg_cos_frac_per_trace",
        "rel_spec_err_band",
        "rel_energy_curve",
    ]
    keep = ["file", "vs_idx"] + [c for c in metric_cols if c in df.columns]
    df = df[keep].copy()

    for c in keep:
        if c == "file":
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
# Parallel workers
# ----------------------------
def _eval_one_ncf_pair_worker(args: tuple) -> Dict[str, Any]:
    (conv_path_s, v1_path_s, filebase, vs_idx, dt, f1, f2, mmap, njobs_cc) = args
    conv_path = Path(conv_path_s)
    v1_path = Path(v1_path_s)

    row: Dict[str, Any] = {
        "file": str(filebase),
        "vs_idx": int(vs_idx),
        "conv_path": str(conv_path),
        "v1_path": str(v1_path),
        "cc_njobs": int(njobs_cc),
    }
    try:
        metrics = eval_ncf_pair(conv_path, v1_path, dt=float(dt), f1=float(f1), f2=float(f2), mmap=bool(mmap))
        row.update(metrics)
    except Exception as e:
        row["error"] = str(e)
    return row


def _eval_one_disp_panel_worker(args: tuple) -> Dict[str, Any]:
    (conv_s, v1_s, base_key, mmap, window) = args
    conv_p = Path(conv_s)
    v1_p = Path(v1_s)
    row = {"base_key": str(base_key), "window": str(window), "panel_conv": str(conv_p), "panel_v1": str(v1_p)}
    try:
        row.update(eval_disp_panel_pair(conv_p, v1_p, mmap=bool(mmap)))
    except Exception as e:
        row["error"] = str(e)
    return row


def _eval_one_pick_worker(args: tuple) -> Dict[str, Any]:
    (conv_s, v1_s, base_key, mmap, window) = args
    conv_p = Path(conv_s)
    v1_p = Path(v1_s)
    row = {"base_key": str(base_key), "window": str(window), "pick_conv": str(conv_p), "pick_v1": str(v1_p)}
    try:
        row.update(eval_pick_pair(conv_p, v1_p, mmap=bool(mmap)))
    except Exception as e:
        row["error"] = str(e)
    return row


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
    p.add_argument("--title", type=str, default="", help="Optional plot title prefix")
    p.add_argument("--njobs", type=int, default=1, help="Parallel workers for eval (ProcessPool)")
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
    title: str,
    njobs: int,
) -> None:
    cc_cfg = load_config(cc_config)
    disp_cfg = load_config(disp_config) if disp_config else {}

    njobs_cc = int(get_cfg(cc_cfg, ["runtime", "njobs"], 1))

    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- NCF root from cc.yaml ----
    ncf_root = Path(get_cfg(cc_cfg, ["paths", "output_root"], required=True)).expanduser().resolve()

    # ---- signal params from cc.yaml ----
    fs_raw = float(get_cfg(cc_cfg, ["data", "fs_raw"], required=True))
    decimation = int(get_cfg(cc_cfg, ["preprocess", "decimation"], 1))
    fs_proc = fs_raw / float(decimation)
    dt = 1.0 / float(fs_proc)

    f1 = float(get_cfg(cc_cfg, ["preprocess", "f1"], 1.0))
    f2 = float(get_cfg(cc_cfg, ["preprocess", "f2"], 10.0))

    logger.info("NCF root: %s", ncf_root)

    # ---- scan NCF pairs ----
    pairs = _scan_ncf_pairs(ncf_root)
    logger.info("Found %d NCF pairs (conv,v1).", len(pairs))
    if not pairs:
        logger.warning("No NCF pairs found in %s", ncf_root)
        return

    # ---- evaluate NCF pairs ----
    nj = max(1, int(njobs))
    tasks = [
        (str(pconv), str(pv1), str(filebase), int(vs), float(dt), float(f1), float(f2), bool(mmap), int(njobs_cc))
        for (pconv, pv1, filebase, vs) in pairs
    ]

    ncf_rows: List[Dict[str, Any]] = []
    if nj == 1:
        for t in tasks:
            ncf_rows.append(_eval_one_ncf_pair_worker(t))
    else:
        logger.info("Parallel NCF eval: njobs=%d over %d pairs", nj, len(tasks))
        with ProcessPoolExecutor(max_workers=nj) as ex:
            futs = [ex.submit(_eval_one_ncf_pair_worker, t) for t in tasks]
            for fut in as_completed(futs):
                ncf_rows.append(fut.result())

    ncf_rows.sort(key=lambda r: (str(r.get("file", "")), int(r.get("vs_idx", -1))))
    write_csv(outdir / "eval_ncf.csv", ncf_rows)

    # ---- optional dispersion eval (scan-based, independent of NCF scan) ----
    disp_rows: List[Dict[str, Any]] = []
    pick_rows: List[Dict[str, Any]] = []

    if disp_cfg:
        disp_results_root = get_cfg(disp_cfg, ["io", "results_root"], required=False)
        disp_window = str(get_cfg(disp_cfg, ["io", "stack_window"], "daily"))
        if disp_results_root:
            disp_root = Path(disp_results_root).expanduser().resolve() / disp_window
            logger.info("Disp root: %s", disp_root)

            if disp_root.exists():
                panel_pairs, pick_pairs = _scan_disp_pairs(disp_root)
                logger.info("Found %d DISP panel pairs, %d pick pairs.", len(panel_pairs), len(pick_pairs))

                # panels
                panel_tasks = [(str(a), str(b), key, bool(mmap), disp_window) for (a, b, key) in panel_pairs]
                if nj == 1:
                    for t in panel_tasks:
                        disp_rows.append(_eval_one_disp_panel_worker(t))
                else:
                    with ProcessPoolExecutor(max_workers=nj) as ex:
                        futs = [ex.submit(_eval_one_disp_panel_worker, t) for t in panel_tasks]
                        for fut in as_completed(futs):
                            disp_rows.append(fut.result())

                # picks
                pick_tasks = [(str(a), str(b), key, bool(mmap), disp_window) for (a, b, key) in pick_pairs]
                if nj == 1:
                    for t in pick_tasks:
                        pick_rows.append(_eval_one_pick_worker(t))
                else:
                    with ProcessPoolExecutor(max_workers=nj) as ex:
                        futs = [ex.submit(_eval_one_pick_worker, t) for t in pick_tasks]
                        for fut in as_completed(futs):
                            pick_rows.append(fut.result())

                disp_rows.sort(key=lambda r: str(r.get("base_key", "")))
                pick_rows.sort(key=lambda r: str(r.get("base_key", "")))

                if disp_rows:
                    write_csv(outdir / "eval_disp.csv", disp_rows)
                if pick_rows:
                    write_csv(outdir / "eval_pick.csv", pick_rows)
            else:
                logger.warning("Disp root does not exist: %s", disp_root)

    # ---- runtime summary from perf_cc*.csv ----
    perf_csv = Path(get_cfg(cc_cfg, ["perf", "out_path"], "./data/runlogs/perf_cc.csv")).expanduser().resolve()
    perf_rows = load_perf_rows_glob(perf_csv)

    runtime_rows: List[Dict[str, Any]] = []
    wall_used: Optional[str] = None

    if perf_rows:
        # 1) sum-of-VS seconds summary
        runtime_rows.extend(runtime_summary_vs(perf_rows, drop_first_vs=drop_first_vs, njobs_cc=njobs_cc))

        # 2) wall-time summary if available
        wall_rows, wall_used = runtime_summary_wall(perf_rows, njobs_cc=njobs_cc)
        if wall_rows:
            runtime_rows.extend(wall_rows)

        write_csv(outdir / "runtime_summary.csv", runtime_rows)
    else:
        logger.warning("No perf rows loaded (perf_cc*.csv missing or empty).")

    # ---- plots ----
    if make_plots:
        _set_plot_style()
        plots_dir = outdir / "plots"

        cc_tag = f"cc_njobs={njobs_cc}"
        prefix = (title.strip() + " | ") if title.strip() else ""
        prefix = prefix + cc_tag + " | "

        # runtime plots
        if perf_rows:
            plot_runtime_distribution_vs(
                perf_rows,
                plots_dir / "runtime_seconds_vs_dist.png",
                title=prefix + "Per-VS runtime",
                logy=bool(logy_runtime),
            )
            plot_runtime_cumulative_ranked_vs(
                perf_rows,
                plots_dir / "runtime_cumulative_ranked.png",
                title=prefix + "Cumulative runtime (sum of per-VS seconds)",
            )
            # only if we have a wall-time column
            if wall_used is not None:
                plot_runtime_wall_per_file_ranked(
                    perf_rows,
                    plots_dir / "runtime_wall_per_file_ranked.png",
                    title=prefix + "Per-file wall time (makespan)",
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
#   --title "CPU test" \
#   --njobs 8