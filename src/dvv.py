"""
:module: src/dvv.py
:author: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: dv/v monitoring tools for DAS ambient-noise interferometry.

Provides three families of methods built on top of utils.normalize_traces and
a private ND-bandpass helper:

    1. **Stretching dv/v** — grid-search over a stretching factor that
       maximizes the Pearson correlation between a stretched reference trace
       and each monitor trace inside a chosen time window. Robust at low
       SNR, naturally suited to coda-wave interferometry.

       - :func:`compute_dvv_single_pair`  (per-pair, vectorized over hours)
       - :func:`compute_dvv`              (orchestrator across freq bands and
                                          spatial pairs)
       - :func:`aggregate_dvv_results`    (QC + per-hour mean/std summary)

    2. **Cross-correlation dv/v** — peak-shift estimation between a reference
       and each monitor, with optional FFT-based resampling for sub-sample
       precision.

       - :func:`compute_dvv_xcorr`

    3. **Attenuation (Q-factor)** — global least-squares fit of the spectral
       ratio across offsets and frequencies, ``ln(A0/Aj) = C - π·(x0 - xj)·f /
       (Q·v)``. Used as a deployment-level QC and as input to amplitude
       corrections in some dv/v workflows.

       - :func:`compute_attenuation`

Sign convention. The standard CWI relation is ``dt/t = ε = -dv/v``: positive
dv/v means velocity *increased* (waves arrive earlier in the monitor),
negative means velocity *decreased* (waves arrive later).

  - The stretching scan (:func:`compute_dvv_single_pair` / :func:`compute_dvv`)
    returns the **raw best-fit stretching factor ε in percent** (legacy
    convention); convert to dv/v with ``dvv = -ε``.
  - :func:`compute_dvv_xcorr` returns dv/v directly (already sign-converted).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import butter, correlate, resample, sosfiltfilt
from tqdm.auto import tqdm

from src.utils import normalize_traces

logger = logging.getLogger(__name__)


# =====================================================
# Internal: ND zero-phase Butterworth bandpass
# =====================================================
def _bandpass_nd(
    data: np.ndarray,
    dt: float,
    fmin: float,
    fmax: float,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass over the last (time) axis.

    Operates on N-dimensional arrays by reshaping the leading dims into a
    single batch dim, running ``scipy.signal.sosfiltfilt`` once, and
    reshaping back. ``sosfiltfilt`` is the canonical scipy implementation
    for forward-backward IIR filtering with proper edge handling, and is
    numerically equivalent to running ``sosfilt`` twice with reversal.

    This helper is intentionally private to ``dvv.py`` because the project's
    public preprocessing pipeline (``ani.bandpass_filter_tukey``) applies an
    additional Tukey window, which is appropriate for cross-correlation
    preprocessing but not for dv/v frequency-band gating.

    :param data: Input data, shape ``(..., nt)``. Last axis is time.
    :param dt: Sampling interval (seconds).
    :param fmin: Low corner frequency (Hz).
    :param fmax: High corner frequency (Hz).
    :param order: Butterworth filter order (per direction). Default 4.
    :return: Filtered data, same shape and dtype as input.
    """
    original_shape = data.shape
    nt = original_shape[-1]
    if data.ndim == 1:
        data_2d = data.reshape(1, nt)
    elif data.ndim > 2:
        data_2d = data.reshape(-1, nt)
    else:
        data_2d = data

    fs = 1.0 / float(dt)
    nyq = 0.5 * fs
    low = float(fmin) / nyq
    high = float(fmax) / nyq

    if not (0.0 < low < 1.0):
        raise ValueError(f"fmin={fmin} Hz is outside (0, Nyquist={nyq} Hz).")
    if not (0.0 < high < 1.0):
        raise ValueError(f"fmax={fmax} Hz is outside (0, Nyquist={nyq} Hz).")
    if low >= high:
        raise ValueError(f"fmin must be < fmax; got {fmin} >= {fmax}.")

    sos = butter(int(order), [low, high], btype="band", output="sos")
    out_2d = sosfiltfilt(sos, data_2d, axis=-1)
    return out_2d.reshape(original_shape).astype(data.dtype, copy=False)


# =====================================================
# 1. Stretching dv/v
# =====================================================
def compute_dvv_single_pair(
    ref_trace: np.ndarray,
    curr_traces_matrix: np.ndarray,
    t: np.ndarray,
    window: Tuple[float, float],
    dv_range: float = 0.05,
    n_steps: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stretching dv/v for a single (reference, monitors) pair, vectorized over
    monitor traces.

    Workflow:
        1. Restrict to the time window ``[window[0], window[1]]``.
        2. Build a grid of ``n_steps`` stretching factors ``ε ∈ [-dv_range,
           +dv_range]`` and, for each, interpolate the reference onto the
           stretched time axis ``t·(1 + ε)``.
        3. For every monitor trace, compute Pearson correlation against each
           stretched reference and pick the ε that maximizes it.
        4. Return the best-fit ε (in percent) and the corresponding
           correlation. Note the CWI convention ``dv/v = -ε``; the conversion
           is left to the caller so the raw scan output stays sign-neutral.

    :param ref_trace: Reference trace (typically the mean stack), shape ``(nt,)``.
    :param curr_traces_matrix: Monitor traces, shape ``(n_traces, nt)``.
    :param t: Time vector matching the trace samples, shape ``(nt,)``.
    :param window: ``(t_min, t_max)`` analysis window in seconds (e.g. coda).
    :param dv_range: Maximum scanned ε in each direction (e.g. 0.05 → ±5%).
    :param n_steps: Number of grid points for the ε scan.
    :return: ``(eps_pct, max_cc)`` where ``eps_pct`` is the best-fit
             stretching factor in percent (shape ``(n_traces,)``; convert to
             dv/v via ``-eps_pct``) and ``max_cc`` is the corresponding best
             correlation per trace.
    """
    win_idx = np.where((t >= window[0]) & (t <= window[1]))[0]

    if len(win_idx) == 0:
        n_traces = curr_traces_matrix.shape[0]
        return np.full(n_traces, np.nan), np.zeros(n_traces)

    t_win = t[win_idx]
    ref_win = ref_trace[win_idx]
    curr_win = curr_traces_matrix[:, win_idx]

    epsilons = np.linspace(-dv_range, dv_range, n_steps)

    # Build stretched-reference matrix, shape (n_steps, n_samples_in_window).
    ref_interp = interp1d(
        t_win, ref_win, kind="linear",
        bounds_error=False, fill_value=0.0,
    )
    ref_stretched_matrix = np.array([ref_interp(t_win * (1.0 + eps)) for eps in epsilons])

    # Pearson correlation between every (monitor, stretched-ref) pair.
    numerator = np.dot(curr_win, ref_stretched_matrix.T)               # (n_traces, n_steps)
    norm_curr = np.sqrt(np.sum(curr_win ** 2, axis=1))                 # (n_traces,)
    norm_ref = np.sqrt(np.sum(ref_stretched_matrix ** 2, axis=1))      # (n_steps,)
    denominator = np.outer(norm_curr, norm_ref)                        # (n_traces, n_steps)

    with np.errstate(divide="ignore", invalid="ignore"):
        corr_matrix = numerator / denominator
        corr_matrix[denominator == 0] = 0.0

    best_idx = np.argmax(corr_matrix, axis=1)
    max_cc = np.max(corr_matrix, axis=1)
    best_eps = epsilons[best_idx]

    # Raw stretching factor in percent (legacy convention). Downstream
    # interpretation: dv/v[%] = -1 * (returned value).
    return best_eps * 100.0, max_cc


def compute_dvv(
    data: np.ndarray,
    dt: float,
    freq_bands: List[Tuple[float, float]],
    dv_range: float = 0.05,
    cc_threshold: float = 0.6,
    window: Optional[Tuple[float, float]] = None,
    n_steps: int = 50,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Per-band, per-pair stretching dv/v across the full DAS array.

    For each frequency band:
        1. Bandpass-filter the entire 4D array.
        2. Build a per-pair reference as the mean over hours.
        3. Run :func:`compute_dvv_single_pair` for every ``(src, rec)`` pair.

    Does **not** spatially aggregate — returns full ``(n_hours, n_src, n_rec)``
    grids of the stretching factor (percent; ``dv/v = -ε``) and correlation.
    Aggregation lives in :func:`aggregate_dvv_results`.

    :param data: 4D dataset, shape ``(n_hours, n_src, n_rec, nt)``.
    :param dt: Sampling interval (seconds).
    :param freq_bands: List of ``(fmin, fmax)`` tuples (Hz).
    :param dv_range: Max ε scanned by the stretching grid (e.g. 0.05 → ±5%).
    :param cc_threshold: Below this correlation, the QC mask flags the dv/v
                         value as NaN in ``dvv_qc``. Raw values are kept in
                         ``dvv_raw``.
    :param window: Optional ``(t_min, t_max)`` analysis window (seconds).
                   Default ``None`` uses the full trace length.
    :param n_steps: Number of grid points in each pair's ε scan.
    :return: ``{band_key: {'dvv_raw': ..., 'dvv_qc': ..., 'cc': ...}}``,
             each grid of shape ``(n_hours, n_src, n_rec)``.
    """
    if data.ndim != 4:
        raise ValueError(f"data must be 4D (n_hours, n_src, n_rec, nt); got {data.shape}.")

    n_hours, n_src, n_rec, nt = data.shape
    t_axis = np.arange(nt) * dt
    win = (float(t_axis[0]), float(t_axis[-1])) if window is None else (float(window[0]), float(window[1]))

    results: Dict[str, Dict[str, np.ndarray]] = {}

    for fmin, fmax in freq_bands:
        band_key = f"{fmin}-{fmax}"
        logger.info("compute_dvv: bandpass %s Hz", band_key)

        data_filt = _bandpass_nd(data, dt, fmin, fmax)

        # Reference stack: mean over hours per (src, rec).
        ref_stack = np.mean(data_filt, axis=0)

        band_dvv = np.full((n_hours, n_src, n_rec), np.nan)
        band_cc = np.full((n_hours, n_src, n_rec), np.nan)

        for s in tqdm(range(n_src), desc=f"Band {band_key} Hz, src"):
            for r in range(n_rec):
                ref_trace = ref_stack[s, r, :]

                # Skip dead reference traces.
                if np.max(np.abs(ref_trace)) == 0:
                    continue

                curr_traces = data_filt[:, s, r, :]
                dvv_vals, cc_vals = compute_dvv_single_pair(
                    ref_trace, curr_traces, t_axis, win,
                    dv_range=dv_range, n_steps=n_steps,
                )
                band_dvv[:, s, r] = dvv_vals
                band_cc[:, s, r] = cc_vals

        results[band_key] = {
            "dvv_raw": band_dvv,
            "dvv_qc": np.where(band_cc < cc_threshold, np.nan, band_dvv),
            "cc": band_cc,
        }

    return results


def aggregate_dvv_results(
    raw_results: Dict[str, Dict[str, np.ndarray]],
    cc_threshold: float = 0.6,
    dvv_limit: float = 5.0,
) -> Dict[str, pd.DataFrame]:
    """
    QC + per-hour aggregation of raw dv/v grids into a tidy time series.

    For each band, applies a combined QC mask (correlation ≥ ``cc_threshold``
    AND ``|dv/v|`` ≤ ``dvv_limit``) per ``(src, rec)`` pair, then computes
    per-hour mean / std / count over the surviving pairs.

    :param raw_results: Output of :func:`compute_dvv`.
    :param cc_threshold: Minimum correlation to include a pair.
    :param dvv_limit: Maximum allowed ``|dv/v|`` in percent. Pairs outside
                      this are dropped.
    :return: ``{band_key: DataFrame}`` with columns
             ``['dvv_mean', 'dvv_std', 'n_pairs']``, indexed by hour.
    """
    final_stats: Dict[str, pd.DataFrame] = {}

    for band_name, data in raw_results.items():
        logger.info("aggregate_dvv_results: aggregating band %s", band_name)

        dvv_grid = data["dvv_raw"]
        cc_grid = data["cc"]
        n_hours = dvv_grid.shape[0]

        ts_mean = np.full(n_hours, np.nan)
        ts_std = np.full(n_hours, np.nan)
        ts_count = np.zeros(n_hours, dtype=int)

        for h in range(n_hours):
            dvv_h = dvv_grid[h, :, :]
            cc_h = cc_grid[h, :, :]

            mask_cc = cc_h >= cc_threshold
            mask_val = np.abs(dvv_h) <= dvv_limit
            valid = mask_cc & mask_val

            valid_dvv = dvv_h[valid]
            if valid_dvv.size > 0:
                ts_mean[h] = np.mean(valid_dvv)
                ts_std[h] = np.std(valid_dvv)
                ts_count[h] = int(valid_dvv.size)

        final_stats[band_name] = pd.DataFrame(
            {"dvv_mean": ts_mean, "dvv_std": ts_std, "n_pairs": ts_count}
        )

    return final_stats


# =====================================================
# 2. Cross-correlation dv/v
# =====================================================
def compute_dvv_xcorr(
    data: np.ndarray,
    dt: float,
    offsets: np.ndarray,
    freq_bands: List[Tuple[float, float]],
    v_ref: float = 250.0,
    dt_resample: float = 0.001,
    cc_threshold: float = 0.6,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Per-band, per-pair dv/v from cross-correlation peak time shifts.

    Workflow:
        1. FFT-resample the entire 4D dataset to ``dt_resample`` for
           sub-sample lag resolution.
        2. For each band: bandpass → trace-wise normalize (uses
           :func:`src.utils.normalize_traces`) → compute reference as mean
           over hours.
        3. For each ``(src, rec, hour)``: cross-correlate the monitor against
           the reference, find the peak, and convert the integer-sample
           shift into ``dv/v = -dt_shift / t_ref`` where
           ``t_ref = |offset| / v_ref`` is the theoretical travel time.

    Sign convention (matches the standard CWI definition ``dt/t = -dv/v``):
    a delayed monitor (later peak) yields ``shift_samples < 0`` (peak shifts
    left of the correlation center) → ``dt_shift > 0`` → ``dvv < 0``
    (velocity *decreased*).

    :param data: 4D dataset, shape ``(n_hours, n_src, n_rec, nt)``.
    :param dt: Original sampling interval (seconds).
    :param offsets: Source-receiver geometry, shape ``(n_src, n_rec)``,
                    in meters. Sign carries direction; ``|offsets|`` is used
                    for the travel-time normalization.
    :param freq_bands: List of ``(fmin, fmax)`` tuples (Hz).
    :param v_ref: Reference apparent velocity (m/s) used to convert offset
                  to travel time. Default 250.
    :param dt_resample: Target finer sampling interval (seconds). If
                        smaller than ``dt``, the data is FFT-resampled.
                        Default 0.001 → 1 ms.
    :param cc_threshold: Pairs below this correlation are NaN'd in
                         ``dvv_qc`` (raw values kept in ``dvv_raw``).
    :return: ``{band_key: {'dvv_raw': ..., 'dvv_qc': ..., 'cc': ...}}``,
             each shape ``(n_hours, n_src, n_rec)``.
    """
    if data.ndim != 4:
        raise ValueError(f"data must be 4D (n_hours, n_src, n_rec, nt); got {data.shape}.")
    if offsets.shape[0] != data.shape[1] or offsets.shape[1] != data.shape[2]:
        raise ValueError(
            f"offsets shape {offsets.shape} must be (n_src={data.shape[1]}, "
            f"n_rec={data.shape[2]})."
        )

    n_hours, n_src, n_rec, nt = data.shape

    # Global FFT resampling for sub-sample lag resolution.
    if dt_resample is not None and dt_resample < dt:
        nt_new = int(nt * (dt / dt_resample))
        logger.info("compute_dvv_xcorr: resampling %d -> %d (dt %.4fs -> %.4fs)",
                    nt, nt_new, dt, dt_resample)
        data = resample(data, nt_new, axis=-1)
        dt = float(dt_resample)

    results: Dict[str, Dict[str, np.ndarray]] = {}

    for fmin, fmax in freq_bands:
        band_key = f"{fmin}-{fmax}"
        logger.info("compute_dvv_xcorr: bandpass %s Hz", band_key)

        data_filt = _bandpass_nd(data, dt, fmin, fmax)
        data_filt = normalize_traces(data_filt)

        ref_stack = np.mean(data_filt, axis=0)

        band_dvv = np.full((n_hours, n_src, n_rec), np.nan)
        band_cc = np.full((n_hours, n_src, n_rec), np.nan)

        for s in tqdm(range(n_src), desc=f"Band {band_key} Hz, src"):
            for r in range(n_rec):
                dist = abs(offsets[s, r])
                if dist < 1.0:
                    continue
                t_ref = dist / v_ref

                ref_trace = ref_stack[s, r, :]
                if np.max(np.abs(ref_trace)) == 0:
                    continue

                curr_traces = data_filt[:, s, r, :]

                for h in range(n_hours):
                    curr_tr = curr_traces[h]
                    if np.max(np.abs(curr_tr)) == 0:
                        continue

                    cc_vec = correlate(ref_trace, curr_tr, mode="same")
                    idx_max = int(np.argmax(cc_vec))
                    max_cc = float(cc_vec[idx_max])

                    # Center of correlate(*, mode='same') is len // 2 (matches
                    # the standard scipy convention).
                    center_idx = len(cc_vec) // 2
                    shift_samples = idx_max - center_idx
                    dt_shift = -shift_samples * dt
                    dvv_val = -dt_shift / t_ref if t_ref > 0 else np.nan

                    band_dvv[h, s, r] = dvv_val
                    band_cc[h, s, r] = max_cc

        dvv_qc = band_dvv.copy()
        dvv_qc[band_cc < cc_threshold] = np.nan

        results[band_key] = {
            "dvv_raw": band_dvv,
            "cc": band_cc,
            "dvv_qc": dvv_qc,
        }

    return results


# =====================================================
# 3. Attenuation / Q-factor estimation
# =====================================================
def compute_attenuation(
    data: np.ndarray,
    dt: float,
    offset: np.ndarray,
    v_phase: float = 250.0,
    fmin: float = 5.0,
    fmax: float = 10.0,
    offset_min: float = 4.0,
    offset_max: float = 100.0,
    smooth_width: int = 5,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """
    Q-factor estimation via global least-squares spectral-ratio fit, separately
    for positive and negative offsets.

    Equation:
        ``ln(A0/Aj) = C - π·(x0 - xj)·f / (Q·v_phase)``

    Stacks all (offset, frequency) observations into a single linear system
    and solves ``[1, X] · [C, 1/Q]ᵀ = Y``. Returns ``Q`` (and the regression
    diagnostics) for each side.

    :param data: Seismic gather, shape ``(nrec, nt)``.
    :param dt: Sampling interval (seconds).
    :param offset: Signed receiver offsets, shape ``(nrec,)``. Positive and
                   negative are processed independently.
    :param v_phase: Phase velocity (m/s) used to convert the regression slope
                    to ``Q``.
    :param fmin: Lower edge of the frequency band used in the regression (Hz).
    :param fmax: Upper edge of the frequency band (Hz).
    :param offset_min: Minimum absolute offset to include (meters).
    :param offset_max: Maximum absolute offset to include (meters).
    :param smooth_width: Spectral smoothing kernel length (samples).
    :return: ``(results_pos, results_neg)`` dictionaries containing
             ``Q``, ``R2``, regression vectors ``X``, ``Y``, ``Y_pred``,
             and the per-side ``data``, ``offset``, ``spectra_smooth``,
             ``freqs``. Missing sides have ``Q = NaN``.
    """
    offset_min = abs(offset_min)
    offset_max = abs(offset_max)

    nt = data.shape[-1]
    freqs_full = np.fft.rfftfreq(nt, d=dt)
    f_mask = (freqs_full >= fmin) & (freqs_full <= fmax)
    freqs = freqs_full[f_mask]

    results_pos: Dict[str, object] = {"freqs": freqs}
    results_neg: Dict[str, object] = {"freqs": freqs}

    def _process_direction(sg: np.ndarray, off: np.ndarray):
        """Spectral-ratio least squares on one signed-offset half-gather."""
        nrec = sg.shape[0]
        spectra = np.abs(np.fft.rfft(sg, axis=1))

        kernel = np.ones(smooth_width) / smooth_width
        spectra_smooth = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode="same"),
            axis=1, arr=spectra,
        )

        # Reference trace: closest to the source.
        A0 = spectra_smooth[0, f_mask]
        x0 = off[0]

        Y_list = []
        X_list = []
        for j in range(nrec):
            Aj = spectra_smooth[j, f_mask]
            xj = off[j]
            X_val = -1.0 * (np.pi * (x0 - xj) * freqs) / v_phase
            Y_val = np.log((A0 + 1e-15) / (Aj + 1e-15))
            X_list.append(X_val)
            Y_list.append(Y_val)

        Y = np.concatenate(Y_list)
        X = np.concatenate(X_list)

        # Design matrix [1, X] for the linear system Y = C + (1/Q) · X.
        A_mat = np.vstack([np.ones_like(X), X]).T
        (C_hat, inv_Q_hat), _, _, _ = np.linalg.lstsq(A_mat, Y, rcond=None)

        Q_est = 1.0 / inv_Q_hat if inv_Q_hat > 0 else np.nan
        Y_pred = C_hat + inv_Q_hat * X
        ss_res = np.sum((Y - Y_pred) ** 2)
        ss_tot = np.sum((Y - np.mean(Y)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        return Q_est, r2, X, Y, Y_pred, spectra_smooth[:, f_mask]

    # Positive offsets.
    pos_idx = np.where((offset >= offset_min) & (offset <= offset_max))[0]
    if len(pos_idx) > 5:
        data_pos = data[pos_idx]
        offset_pos = offset[pos_idx]
        Q_est, r2, X, Y, Y_pred, sp_smooth = _process_direction(data_pos, offset_pos)
        results_pos.update({
            "Q": Q_est, "R2": r2,
            "X": X, "Y": Y, "Y_pred": Y_pred,
            "data": data_pos, "offset": offset_pos,
            "spectra_smooth": sp_smooth,
        })
    else:
        results_pos["Q"] = np.nan

    # Negative offsets — flip so the closest-to-source trace is at index 0.
    neg_idx = np.where((offset <= -offset_min) & (offset >= -offset_max))[0]
    if len(neg_idx) > 5:
        data_neg = np.flipud(data[neg_idx, :])
        offset_neg = np.abs(np.flip(offset[neg_idx]))
        Q_est, r2, X, Y, Y_pred, sp_smooth = _process_direction(data_neg, offset_neg)
        results_neg.update({
            "Q": Q_est, "R2": r2,
            "X": X, "Y": Y, "Y_pred": Y_pred,
            "data": data_neg, "offset": offset_neg,
            "spectra_smooth": sp_smooth,
        })
    else:
        results_neg["Q"] = np.nan

    return results_pos, results_neg
