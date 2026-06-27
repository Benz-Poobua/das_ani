"""
:module: src/disp.py
:author: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Dispersion imaging (f–v transform), dispersion curve picking,
          dispersion-based dv/v + depth mapping, and a config-driven
          time-lapse dispersion wrapper for DAS ambient noise interferometry.
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from src.utils import (
    convert_to_tensor,
    convert_to_numpy,
    get_cfg,
    nextpow2,
    timeit,
    parse_ncf_stack_filename,
    fk_filter,
)

logger = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, torch.Tensor]

# =====================================================
# 1. Dispersion image (f-v panel) via phase-shift method
# =====================================================
@timeit
@torch.no_grad()
def dispersion_curve(
    data: ArrayLike,
    offset: ArrayLike,
    t: ArrayLike,
    *,
    vmin: float = 200.0,
    vmax: float = 4000.0,
    dv: float = 10.0,
    fmin: float = 0.1,
    fmax: float = 50.0,
    normalize: bool = True,
    device: Optional[torch.device] = None,
    batch_size_v: Optional[int] = None,
    empty_cache: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the f-v (frequency-velocity) dispersion image using the phase-shift method.

    :param data: Gather (nrec, nt).
    :param offset: Offsets (nrec,).
    :param t: Time axis (nt,).
    :param vmin: Minimum phase velocity (m/s).
    :param vmax: Maximum phase velocity (m/s).
    :param dv: Velocity sampling interval (m/s).
    :param fmin: Minimum frequency (Hz).
    :param fmax: Maximum frequency (Hz).
    :param normalize: Normalize each frequency slice by its max.
    :param device: Torch device. If None, uses GPU if available.
    :param batch_size_v: Velocities per batch. If None, heuristic.
    :param empty_cache: If True and CUDA, empty cache between batches.

    :return: (fv_panel [nv,nf], f_axis [nf], v_axis [nv])
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # Convert inputs
    data_t = convert_to_tensor(data, device=device)
    off_t = convert_to_tensor(offset, device=device)
    t_t = convert_to_tensor(t, device=device)

    if data_t.ndim != 2:
        raise ValueError("'data' must be 2D (nrec, nt).")
    if off_t.ndim != 1 or off_t.numel() != data_t.shape[0]:
        raise ValueError("'offset' must be 1D with length nrec.")
    if t_t.ndim != 1 or t_t.numel() != data_t.shape[1]:
        raise ValueError("'t' must be 1D with length nt.")
    
    nrec, nt = int(data_t.shape[0]), int(data_t.shape[1])

    if nt < 2:
        raise ValueError("Time axis must have at least 2 samples.")

    dt = float((t_t[1] - t_t[0]).item())
    if dt <= 0:
        raise ValueError(f"Invalid dt={dt}. Check your time vector.")

    # Velocity axis
    if dv <= 0:
        raise ValueError("dv must be > 0.")
    if vmax <= vmin:
        raise ValueError("vmax must be > vmin.")
    v_axis = torch.arange(vmin, vmax + dv, dv, device=device, dtype=torch.float32)
    nv = int(v_axis.numel())

    # FFT / frequency axis
    nfft = int(nextpow2(torch.tensor(nt, device=device)).item())
    f_full = torch.fft.rfftfreq(nfft, dt).to(device)
    freq_mask = (f_full >= float(fmin)) & (f_full <= float(fmax))
    f_axis = f_full[freq_mask]
    nf = int(f_axis.numel())
    if nf < 2:
        raise ValueError(
            f"Too few frequencies after masking: nf={nf}. "
            f"Try adjusting fmin/fmax or check dt/nt."
            )

    logger.info(
        f"[dispersion_curve] device={device} | nrec={nrec} nt={nt} dt={dt:.6f} | "
        f"nv={nv} (v=[{vmin},{vmax}] dv={dv}) | nf={nf} (f=[{fmin},{fmax}])"
        )

    # FFT along time: (nrec, nfreq_masked)
    fft_data = torch.fft.rfft(data_t, n=nfft, dim=1)[:, freq_mask]

    # Phase-only (avoid amplitude dominance)
    amp = torch.abs(fft_data)
    phase = fft_data / (amp + 1e-8)

    # Velocity batching heuristic
    if batch_size_v is None:
        # CUDA prefers smaller batches; CPU/MPS can tolerate larger
        if device.type == "cuda":
            batch_size_v = 64
        else:
            batch_size_v = nv
    batch_size_v = int(max(1, min(batch_size_v, nv)))

    fv_panel = torch.zeros((nv, nf), device=device, dtype=torch.float32)

    # Precompute common shapes
    f_mat = f_axis.unsqueeze(0)  # (1, nf)
    x = off_t.unsqueeze(0).unsqueeze(0)  # (1, 1, nrec)

    # Phase-shift integration, batched over velocities
    for v_start in range(0, nv, batch_size_v):
        v_end = min(v_start + batch_size_v, nv)
        v_batch = v_axis[v_start:v_end]         # (nv_b,)

        # k = 2π f / v : shapes -> f_axis(1 × nf), v_batch(nv_b × 1)
        # Vectorized phase-shift integration
        v_mat = v_batch.unsqueeze(1)            # (nv_b × 1)
        k = 2.0 * np.pi * f_mat / v_mat         # (nv_b × nf)

        # Phase kernel: exp(i k x); shapes:
        # x: (1, 1, nrec), k: (nv_b, nf, 1) -> kernel: (nv_b, nf, nrec)
        kernel = torch.exp(1j * k.unsqueeze(-1) * x)  
 
        # Integrate over receivers: Σ_r kernel[v,f,r] * phase[r,f]
        fv = torch.einsum('vfr,rf->vf', kernel, phase) # (nv_b, nf)

        # Magnitude
        fv_panel[v_start:v_end, :] = torch.abs(fv)

        # Free some temporary tensors (helps on tight GPUs)
        del v_batch, v_mat, k, kernel, fv
        if empty_cache and device.type == "cuda":
            torch.cuda.empty_cache()

    # Optional per-frequency normalization 
    if normalize:
        max_val = torch.amax(fv_panel, dim=0, keepdim=True)
        max_val = torch.where(max_val == 0, torch.ones_like(max_val), max_val)
        fv_panel = fv_panel / max_val

    return fv_panel, f_axis, v_axis

# =====================================================
# 2. Dispersion curve extraction (from f–v panel)
# =====================================================
def extr_disp(
    f_axis: ArrayLike,
    v_axis: ArrayLike,
    fv_panel: ArrayLike,
    *,
    f_ref_set: Optional[Sequence[float]] = None,
    vmax_set: Optional[Sequence[float]] = None,
    step: int = 5,
    ) -> np.ndarray:
    """
    Extract a dispersion curve from a frequency–velocity image by tracking
    local maxima, following Huajian Yao's MATLAB picking approach.

    This is a high-level wrapper that chooses between single-start
    (`AutoSearch`) and multi-start (`AutoSearchMultiplePoints`) picking.

    :param f_axis: Frequency axis (nf,).
    :param v_axis: Velocity axis (nv,).
    :param fv_panel: Dispersion image (nv, nf).
    :param f_ref_set: Optional list of reference start frequencies.
    :param vmax_set: Optional list of max velocities at corresponding f_ref_set.
    :param step: Vertical search step (velocity-index units).

    :return: Picked phase-velocity curve (nf,), numpy array.
    """
    # Convert to numpy for picking logic 
    f = convert_to_numpy(f_axis)
    v = convert_to_numpy(v_axis)
    disp = convert_to_numpy(fv_panel)
    
    nv, nf = disp.shape
    if f.shape[0] != nf or v.shape[0] != nv:
        raise ValueError("Axes mismatch: fv_panel must be (nv, nf) with matching v_axis/f_axis.")
    
    if f_ref_set is None:
        # Default: start at lowest usable frequency
        f_ref_set = [float(f[0])]
    if vmax_set is None:
        # Default: allow full velocity range
        vmax_set = [float(v[-1])] * len(f_ref_set)

    if len(f_ref_set) != len(vmax_set):
        raise ValueError("'f_ref_set' and 'vmax_set' must have the same length.")
    
    step = int(step)
    if step < 1:
        raise ValueError("step must be >= 1.")

    # Determine starting points (in index space)
    xpt: List[int] = []
    ypt: List[int] = []

    for f_ref, vmax in zip(f_ref_set, vmax_set):

        # Find closest frequency index ≥ f_ref
        idx_f_candidates = np.where(f >= float(f_ref))[0]
        if idx_f_candidates.size == 0:
            raise ValueError(f"No frequencies >= f_ref={f_ref} Hz.")
        idx_f = int(idx_f_candidates[0])

        disp_ref = disp[:, idx_f]

        # Restrict velocities to v < vmax
        mask_v = v < float(vmax)
        if not np.any(mask_v):
            raise ValueError(f"No velocities < vmax={vmax} m/s.")

        disp_sub = disp_ref[mask_v]
        v_sub = v[mask_v]

        # Find velocity index of maximum energy with the restricted window
        idx_v_local = int(np.argmax(disp_sub))
        v_ref = float(v_sub[idx_v_local])

        # Convert to full velocity index
        idx_v = int(np.abs(v - v_ref).argmin())

        xpt.append(idx_f)
        ypt.append(idx_v)

    xpt_arr = np.asarray(xpt, dtype=int)
    ypt_arr = np.asarray(ypt, dtype=int)

    # Single-start or multi-start picking
    if len(xpt) == 1:
        arr_pt = AutoSearch(int(ypt_arr[0]), int(xpt_arr[0]), disp, step=step)
    else:
        arr_pt = AutoSearchMultiplePoints(ypt_arr, xpt_arr, disp, step=step)
    
    return v[arr_pt]

def AutoSearch(initial_y: int, initial_x: int, image_data: np.ndarray, step: int = 5) -> np.ndarray:
    """
    Track a dispersion ridge (local maxima) from a single starting point
    in a 2D f–v image (velocity × frequency).

    This follows Huajian Yao's MATLAB strategy: at each frequency slice,
    search upward and downward in velocity to find the local maximum.

    :param initial_y: Initial velocity index (row index).
    :param initial_x: Initial frequency index (column index).
    :param image_data: 2D image array (nv, nf).
    :param step: Vertical search step (velocity index increment).

    :return: Indices of picked velocities for all frequencies (nf,).
    """
    y_size, x_size = image_data.shape
    arr_pt = np.zeros(x_size, dtype=int)

    # 1. Scan upward in frequency (from initial_x to high frequencies)
    current_y = int(initial_y)
    for i in range(int(initial_x), x_size):
        point_left = current_y
        point_right = current_y

        while True:
            new_left = max(0, point_left - step)
            if image_data[point_left, i] < image_data[new_left, i]:
                point_left = new_left
            else:
                point_left = new_left
                break

        # search downward (toward larger velocity index)
        while True:
            new_right = min(point_right + step, y_size - 1)
            if image_data[point_right, i] < image_data[new_right, i]:
                point_right = new_right
            else:
                point_right = new_right
                break

        idx_local = int(np.argmax(image_data[point_left : point_right + 1, i]))
        arr_pt[i] = idx_local + point_left
        current_y = int(arr_pt[i])

    # 2. Scan downward in frequency (from initial_x back to low frequencies)
    current_y = int(arr_pt[int(initial_x)])
    for i in range(int(initial_x) - 1, -1, -1):
        point_left = current_y
        point_right = current_y

        while True:
            new_left = max(0, point_left - step)
            if image_data[point_left, i] < image_data[new_left, i]:
                point_left = new_left
            else:
                point_left = new_left
                break

        # search downward
        while True:
            new_right = min(point_right + step, y_size - 1)
            if image_data[point_right, i] < image_data[new_right, i]:
                point_right = new_right
            else:
                point_right = new_right
                break

        idx_local = int(np.argmax(image_data[point_left : point_right + 1, i]))
        arr_pt[i] = idx_local + point_left
        current_y = int(arr_pt[i])

    return arr_pt

def AutoSearchMultiplePoints(
    ptY: np.ndarray, ptX: np.ndarray, image_data: np.ndarray, step: int = 5
    ) -> np.ndarray:
    """
    Track a dispersion ridge from multiple starting points in an f–v image,
    allowing extraction of more complex or multi-branch dispersion patterns.

    This generalizes AutoSearch by stitching together segments from
    several user-defined starting points.

    :param ptY: Initial velocity indices (npt,).
    :param ptX: Initial frequency indices (npt,).
    :param image_data: 2D image array (nv, nf).
    :param step: Vertical search step.

    :return: Indices of picked velocities for all frequencies (nf,).
    """
    ptY = np.asarray(ptY, dtype=int)
    ptX = np.asarray(ptX, dtype=int)

    n_pt = int(len(ptX))
    if n_pt < 1:
        raise ValueError("ptX/ptY must contain at least one point.")
    
    # Sort points by frequency index
    order = np.argsort(ptX)
    ptX = ptX[order]
    ptY = ptY[order]

    y_size, x_size = image_data.shape
    arr_pt = np.zeros(x_size, dtype=int)

    # 1. Scan from highest starting frequency to higher frequencies
    initial_x = int(ptX[-1])
    current_y = int(ptY[-1])
    for i in range(initial_x, x_size):
        point_left = current_y
        point_right = current_y

        # Up
        while True:
            new_left = max(0, point_left - step)
            if image_data[point_left, i] < image_data[new_left, i]:
                point_left = new_left
            else:
                point_left = new_left
                break

        # Down
        while True:
            new_right = min(point_right + step, y_size - 1)
            if image_data[point_right, i] < image_data[new_right, i]:
                point_right = new_right
            else:
                point_right = new_right
                break
        
        idx_local = int(np.argmax(image_data[point_left : point_right + 1, i]))
        arr_pt[i] = idx_local + point_left
        current_y = int(arr_pt[i])

    # 2. Scan toward lower frequencies, stitching through intermediate points
    current_y = int(arr_pt[initial_x])
    kk = 0
    mid_idx = n_pt - 2
    midX = int(ptX[mid_idx]) if mid_idx >= 0 else int(ptX[0])
    midY = int(ptY[mid_idx]) if mid_idx >= 0 else int(ptY[0])

    for i in range(initial_x, -1, -1):
        if i == midX:
            current_y = midY
            kk += 1
            if (n_pt - kk) > 1:
                midX = int(ptX[n_pt - kk - 2])
                midY = int(ptY[n_pt - kk - 2])

        point_left = current_y
        point_right = current_y

        # Up
        while True:
            new_left = max(0, point_left - step)
            if image_data[point_left, i] < image_data[new_left, i]:
                point_left = new_left
            else:
                point_left = new_left
                break

        # Down
        while True:
            new_right = min(point_right + step, y_size - 1)
            if image_data[point_right, i] < image_data[new_right, i]:
                point_right = new_right
            else:
                point_right = new_right
                break

        idx_local = int(np.argmax(image_data[point_left : point_right + 1, i]))
        arr_pt[i] = idx_local + point_left
        current_y = int(arr_pt[i])
    
    return arr_pt

# =====================================================
# 3. Compute dispersion directly from NCF matrix
# =====================================================
@torch.no_grad()
def compute_dispersion_from_ncf(
    ncf: ArrayLike,
    *,
    fs: float,
    dx: float = 8.16,
    fv_kwargs: Optional[Dict[str, Any]] = None,
    pick_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[np.ndarray]]:
    """
    Build offset + time vectors from an NCF and compute:
      - f–v panel
      - optional picks 

    **Important:** This function assumes `ncf` is "one-sided" (or folded).
    It constructs time as starting from 0 (causal).
    If you have a symmetric NCF, fold it before passing here.

    :param ncf: NCF matrix (nrec, nlag), symmetric around zero lag.
    :param fs: Sampling rate (Hz).
    :param dx: Spatial distance between adjacent channels (in meters). Default is 8.16.
    :param fv_kwargs: kwargs forwarded to dispersion_curve().
    :param pick_kwargs: kwargs forwarded to extr_disp(); if None -> no picking.

    :return: (fv_panel, f_axis, v_axis, picks_or_None)
    """
    ncf_arr = convert_to_numpy(ncf)
    if ncf_arr.ndim != 2:
        raise ValueError("'ncf' must be 2D (nrec, nlag).")
    
    nrec, nlag = ncf_arr.shape

    # 1. Build Offset Vector (0 to L)
    offset = np.arange(nrec, dtype=float) * float(dx)

    # 2. Build Time Vector (0 to T)
    # We assume 'disp_pick.py' has already folded/cut the data to be causal.
    t = np.arange(nlag, dtype=float) / float(fs)

    # 3. Compute Dispersion
    fv_kwargs = fv_kwargs or {}
    fv_panel, f_axis, v_axis = dispersion_curve(
        data=ncf_arr, offset=offset, t=t, **fv_kwargs)

    # 4. Picking
    picks = None
    if pick_kwargs is not None:
        picks = extr_disp(f_axis, v_axis, fv_panel, **pick_kwargs)

    return fv_panel, f_axis, v_axis, picks

# =====================================================
# 5. Regularize picks
# =====================================================
def regularize_dispersion_data(
    x_raw: Sequence[float] | np.ndarray, 
    y_raw: Sequence[float] | np.ndarray, 
    z_raw: Sequence[float] | np.ndarray, 
    f_min: float = 2.0, 
    f_max: float = 6.0, 
    f_step: float = 0.2
) -> tuple[list[float], list[float], list[float], dict[float, dict[str, np.ndarray]]]:
    """
    Regularizes scattered dispersion data onto a uniform frequency axis.
    Averages duplicate frequencies (combining S1 and S2) and interpolates missing values.
    
    :param x_raw: Raw spatial distances corresponding to the picks.
    :type x_raw: array_like
    :param y_raw: Raw frequency values corresponding to the picks.
    :type y_raw: array_like
    :param z_raw: Raw phase velocity values corresponding to the picks.
    :type z_raw: array_like
    :param f_min: Minimum frequency for the target uniform axis. Default is 2.0.
    :type f_min: float, optional
    :param f_max: Maximum frequency for the target uniform axis. Default is 6.0.
    :type f_max: float, optional
    :param f_step: Frequency step spacing for the target axis. Default is 0.2.
    :type f_step: float, optional
    :returns: A tuple containing flattened lists of regularized (x, y, z) data for plotting, 
              and a dictionary mapping each distance to its cleaned 'f' and 'v' arrays for inversion.
    :rtype: tuple[list[float], list[float], list[float], dict[float, dict[str, np.ndarray]]]
    """
    # 1. Define the standard inversion frequency axis
    # We add f_step/2 to f_max to ensure the final value is included in np.arange
    f_target = np.arange(f_min, f_max + (f_step / 2), f_step) 
    
    regularized_profiles = {}
    x_reg = []
    y_reg = []
    z_reg = []

    unique_distances = np.unique(x_raw)

    for dist in unique_distances:
        # Extract raw data for this specific distance
        mask = np.array(x_raw) == dist
        f_station = np.array(y_raw)[mask]
        v_station = np.array(z_raw)[mask]
        
        # --- Average duplicate frequencies (Combine S1 & S2) ---
        f_unique = np.unique(f_station)
        v_unique = np.array([np.mean(v_station[f_station == f_val]) for f_val in f_unique])
        
        # --- Mute/Drop data above f_max ---
        valid = f_unique <= f_max
        f_clean = f_unique[valid]
        v_clean = v_unique[valid]
        
        # Skip if a station somehow has fewer than 2 points left
        if len(f_clean) < 2:
            continue
            
        # 2. Build the Interpolator for this station
        interp_func = interp1d(
            f_clean, 
            v_clean, 
            kind='linear', 
            bounds_error=False, 
            # Pad missing edges with the nearest valid velocity
            fill_value=(v_clean[0], v_clean[-1]) 
        )
        
        # 3. Apply it to our standard target frequency axis
        v_target = interp_func(f_target)
        
        # Save to dictionary for inversion later
        regularized_profiles[dist] = {
            'f': f_target,
            'v': v_target
        }
        
        # Save to lists for plotting
        x_reg.extend([dist] * len(f_target))
        y_reg.extend(f_target)
        z_reg.extend(v_target)
        
    return x_reg, y_reg, z_reg, regularized_profiles

# =====================================================
# 6. Save regularized picks for inversion
# =====================================================
def export_inversion_inputs(
    profiles_dict: dict[float, dict[str, np.ndarray]], 
    output_dir: str
) -> None:
    """
    Exports regularized 1D dispersion curves into individual text files 
    formatted for standard 1D depth inversion software.

    :param profiles_dict: Dictionary mapping spatial distances to their corresponding 
                          'f' (frequency) and 'v' (velocity) arrays.
    :type profiles_dict: dict[float, dict[str, np.ndarray]]
    :param output_dir: Directory where the formatted text files will be saved.
    :type output_dir: str
    """
    # Create a folder to keep your directory clean
    os.makedirs(output_dir, exist_ok=True)
    
    for dist, data in profiles_dict.items():
        # Format the filename so they sort nicely (e.g., 'dispersion_0120m.txt')
        filename = f"dispersion_{int(dist):04d}m.txt" 
        filepath = os.path.join(output_dir, filename)
        
        # Stack the 1D arrays as two columns: Frequency, Velocity
        out_data = np.column_stack((data['f'], data['v']))
        
        # Save to a space- or tab-delimited text file
        # fmt='%.3f' keeps the numbers clean (3 decimal places)
        np.savetxt(
            filepath,
            out_data,
            fmt='%.3f',
            delimiter='\t',
            header='Frequency(Hz)\tPhaseVelocity(m/s)',
            comments='# '
        )

# =====================================================
# 7. Dispersion-based dv/v + depth mapping
# =====================================================
def compute_dvv_from_dispersion(
    ref_freq: np.ndarray,
    ref_vel: np.ndarray,
    cur_freq: np.ndarray,
    cur_vel: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the fractional velocity change (dv/v) between two dispersion curves.
    Interpolates the current curve onto the reference frequency axis for safe subtraction.

    :param ref_freq: Frequencies of the reference dispersion curve (Hz).
    :param ref_vel: Phase velocities of the reference curve (m/s).
    :param cur_freq: Frequencies of the current dispersion curve (Hz).
    :param cur_vel: Phase velocities of the current curve (m/s).
    :return: (common_freqs, dvv_array) where dvv_array is (v_cur - v_ref) / v_ref.
    """
    ref_freq = np.asarray(ref_freq, dtype=np.float64)
    ref_vel = np.asarray(ref_vel, dtype=np.float64)

    # Create an interpolator for the current curve.
    # bounds_error=False and fill_value=np.nan ensure we don't extrapolate garbage
    # if the current curve picked fewer frequencies than the reference curve.
    cur_interp_func = interp1d(
        np.asarray(cur_freq, dtype=np.float64),
        np.asarray(cur_vel, dtype=np.float64),
        kind='linear',
        bounds_error=False,
        fill_value=np.nan,
    )

    # Interpolate current velocities exactly at the reference frequencies
    cur_vel_interp = cur_interp_func(ref_freq)

    # Calculate dv/v (guard dead reference picks against division by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        dvv = (cur_vel_interp - ref_vel) / ref_vel
    dvv[~np.isfinite(dvv)] = np.nan

    # Mask out NaN values where interpolation fell out of bounds
    valid_mask = ~np.isnan(dvv)

    return ref_freq[valid_mask], dvv[valid_mask]


def approximate_depth_profile(
    freqs: np.ndarray,
    velocities: np.ndarray,
    factor: float = 0.333,
) -> np.ndarray:
    """
    Approximates the penetration depth of surface waves based on frequency and velocity.
    Standard Seismological Rule of Thumb: Depth ~ Wavelength / 3.

    :param freqs: Array of frequencies (Hz).
    :param velocities: Array of phase velocities (m/s).
    :param factor: The depth-to-wavelength scaling factor (0.333 for Scholte/Rayleigh).
    :return: Array of approximate depths (meters).
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)

    # Prevent division by zero if freq=0 is accidentally passed
    safe_freqs = np.where(freqs == 0, 1e-9, freqs)

    # Wavelength = Velocity / Frequency
    wavelengths = velocities / safe_freqs

    # Z = Wavelength * Factor
    return wavelengths * float(factor)


# =====================================================
# 8. Time-lapse dispersion wrapper
# =====================================================
_DISP_FILE_RE = re.compile(r"(\d{8})(?:_(\d{6}))?")


def _extract_datetime(filename: str) -> datetime:
    """Parse the leading YYYYMMDD[_HHMMSS] timestamp of a stack filename."""
    m = _DISP_FILE_RE.search(filename)
    if not m:
        raise ValueError(f"Could not parse datetime from filename: {filename}")
    date_str = m.group(1)
    time_str = m.group(2) if m.group(2) else "000000"
    return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")


def _pick_dispersion_curve(
    ncf: np.ndarray, cfg: Mapping[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fold a two-sided VSG and run the phase-shift imaging + AutoSearch picker
    (:func:`compute_dispersion_from_ncf`) with parameters from the
    ``dispersion:`` config block.

    Offset model: rows of the (sliced) VSG are assumed to start at the
    virtual source and increase monotonically away from it with spacing
    ``data.dx`` -- i.e. either export one-sided gathers beforehand (see
    ``src.ncf.export_targeted_ncfs``) or set ``dispersion.vs_row`` to the
    VSG row index of the virtual source so rows above it are dropped.

    :return: (f_axis, phase_velocity) as 1-D numpy arrays.
    """
    fs_raw = float(get_cfg(cfg, ["data", "fs_raw"], required=True))
    decimation = float(get_cfg(cfg, ["ingest", "decimation"], 1))
    fs = fs_raw / max(decimation, 1.0)
    dx = float(get_cfg(cfg, ["data", "dx"], 8.16))

    vs_row = int(get_cfg(cfg, ["dispersion", "vs_row"], 0))
    ncf = np.asarray(ncf)
    if not (0 <= vs_row < ncf.shape[0] - 2):
        raise ValueError(f"dispersion.vs_row={vs_row} out of range for VSG with {ncf.shape[0]} rows.")
    ncf = ncf[vs_row:, :]

    # Fold the symmetric two-sided lag axis (average causal + acausal).
    npts = ncf.shape[1]
    mid = npts // 2
    folded = (np.fliplr(ncf[:, :mid + 1]) + ncf[:, mid:]) / 2.0

    fv_kwargs = dict(
        vmin=float(get_cfg(cfg, ["dispersion", "vmin"], 100.0)),
        vmax=float(get_cfg(cfg, ["dispersion", "vmax"], 1500.0)),
        dv=float(get_cfg(cfg, ["dispersion", "dv"], 5.0)),
        fmin=float(get_cfg(cfg, ["dispersion", "fmin"], 1.0)),
        fmax=float(get_cfg(cfg, ["dispersion", "fmax"], 10.0)),
    )
    pick_kwargs = dict(step=int(get_cfg(cfg, ["dispersion", "pick_step"], 5)))

    _, f_axis, _, picks = compute_dispersion_from_ncf(
        folded, fs=fs, dx=dx, fv_kwargs=fv_kwargs, pick_kwargs=pick_kwargs,
    )
    return convert_to_numpy(f_axis).astype(np.float64), np.asarray(picks, dtype=np.float64)


@timeit
def run_timelapse_dispersion(cfg: Mapping[str, Any]) -> Optional[Path]:
    """
    Chronologically tracks dispersion-curve changes and maps them to depth.
    Produces a 2D matrix of dv/v over (time, depth) -- e.g. for monitoring a
    freezing front or seasonal velocity changes.

    Workflow per time step (stacked VSG file):
        1. Pick the dispersion curve via :func:`_pick_dispersion_curve`
           (phase-shift f-v panel + AutoSearch ridge tracking).
        2. dv/v against the reference (first) curve via
           :func:`compute_dvv_from_dispersion`.
        3. Map frequencies to depth with :func:`approximate_depth_profile`,
           using the REFERENCE velocities so the depth frame stays stable.
        4. Interpolate the scattered (depth, dv/v) points onto a fixed
           ``z_grid`` for the heatmap.

    Config keys (``dispersion:`` block unless noted):
        stacking.stacks_root (req) : root of the stacked NCFs
        stack_target               : which stack window to track (default "1d")
        vmin/vmax/dv/fmin/fmax     : f-v panel parameters
        pick_step                  : AutoSearch vertical step (default 5)
        vs_row                     : VSG row of the virtual source (default 0)
        max_depth_m / depth_step_m : depth grid (default 50.0 / 0.5)
        depth_factor               : depth ~ factor * wavelength (default 0.333)
        paths.disp_timelapse_root  : output dir (default ./data/disp_timelapse)

    Output ``.npz`` keys: ``dvv_heatmap`` (n_times, nz), ``z_grid`` (nz,),
    ``timestamps`` (n_times, POSIX seconds), ``files`` (n_times, str).

    :return: Path of the written .npz, or None if no inputs were found.
    """
    # 1. Configuration & paths. stacks_root canonically lives under
    # `stacking:`; accept legacy `paths.stacks_root` as fallback.
    stacks_root = get_cfg(cfg, ["stacking", "stacks_root"],
                          get_cfg(cfg, ["paths", "stacks_root"], None))
    if stacks_root is None:
        raise KeyError("Missing config key: stacking.stacks_root")
    data_root = Path(str(stacks_root)).expanduser()
    out_root = Path(str(get_cfg(cfg, ["paths", "disp_timelapse_root"],
                                "./data/disp_timelapse"))).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    stack_target = str(get_cfg(cfg, ["dispersion", "stack_target"],
                               get_cfg(cfg, ["dvv", "stack_target"], "1d")))
    search_dir = data_root / stack_target

    if not search_dir.exists():
        logger.error("Directory not found: %s", search_dir)
        return None

    # 2. Gather chronological files (cross-correlation stacks only).
    all_files = sorted(search_dir.rglob("*_cc_*.npy"),
                       key=lambda x: _extract_datetime(x.name))
    if not all_files:
        logger.error("No _cc_ files found in %s", search_dir)
        return None

    n_steps = len(all_files)
    logger.info("Found %d CC stacks for time-lapse dispersion.", n_steps)

    # 3. Fixed depth grid for the heatmap.
    max_depth = float(get_cfg(cfg, ["dispersion", "max_depth_m"], 50.0))
    dz = float(get_cfg(cfg, ["dispersion", "depth_step_m"], 0.5))
    depth_factor = float(get_cfg(cfg, ["dispersion", "depth_factor"], 0.333))
    z_grid = np.arange(0.0, max_depth + dz, dz)

    dvv_depth_heatmap = np.full((n_steps, len(z_grid)), np.nan, dtype=np.float32)
    timestamps = [_extract_datetime(p.name).timestamp() for p in all_files]

    # 4. Reference curve (first time step).
    logger.info("Extracting reference dispersion curve from %s", all_files[0].name)
    ref_freq, ref_vel = _pick_dispersion_curve(np.load(all_files[0]), cfg)
    ref_interp = interp1d(ref_freq, ref_vel, kind="linear",
                          bounds_error=False, fill_value=np.nan)

    # Row 0 is the reference vs itself: dv/v = 0 across its depth coverage.
    ref_depths = approximate_depth_profile(ref_freq, ref_vel, factor=depth_factor)
    order0 = np.argsort(ref_depths)
    dvv_depth_heatmap[0, :] = interp1d(
        ref_depths[order0], np.zeros_like(ref_depths[order0]),
        kind="linear", bounds_error=False, fill_value=np.nan,
    )(z_grid)

    # 5. Time-lapse loop.
    for t in tqdm(range(1, n_steps), desc="Time-lapse dispersion"):
        f_path = all_files[t]
        try:
            # A. Pick the current curve.
            cur_freq, cur_vel = _pick_dispersion_curve(np.load(f_path), cfg)

            # B. Strictly matched dv/v on the reference frequency axis.
            valid_freqs, valid_dvv = compute_dvv_from_dispersion(
                ref_freq, ref_vel, cur_freq, cur_vel,
            )
            if valid_freqs.size < 2:
                logger.warning("Too few overlapping picks for %s; skipping.", f_path.name)
                continue

            # C. Map frequencies to depth using the REFERENCE velocities so
            # the physical depth frame stays stable over time.
            stable_vels = ref_interp(valid_freqs)
            dynamic_depths = approximate_depth_profile(
                valid_freqs, stable_vels, factor=depth_factor,
            )

            # D. Interpolate scattered (depth, dv/v) onto the rigid z_grid.
            order = np.argsort(dynamic_depths)
            depth_interp = interp1d(
                dynamic_depths[order], valid_dvv[order],
                kind="linear", bounds_error=False, fill_value=np.nan,
            )
            dvv_depth_heatmap[t, :] = depth_interp(z_grid)

        except Exception as e:
            logger.warning("Failed to pick/track curve for %s: %s", f_path.name, e)

    # 6. Save the 2D matrix.
    time_str = datetime.now().strftime("%Y%m%d_%H%M")
    out_file = out_root / f"disp_timelapse_{stack_target}_{time_str}.npz"
    np.savez(
        out_file,
        dvv_heatmap=dvv_depth_heatmap,            # (time, depth)
        z_grid=z_grid,
        timestamps=np.asarray(timestamps),
        files=np.array([p.name for p in all_files]),
    )
    logger.info("Time-lapse depth matrix saved to %s", out_file)
    return out_file