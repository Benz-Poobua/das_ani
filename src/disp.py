"""
:module: src/disp.py
:author: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Dispersion imaging (f–v transform) and dispersion curve picking
          for DAS ambient noise interferometry.
"""
from __future__ import annotations

import dask
import logging 
import os
import torch
import numpy as np
from scipy.interpolate import interp1d
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.utils import convert_to_tensor, convert_to_numpy, nextpow2, timeit, parse_ncf_stack_filename, fk_filter

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
    :param dx: Channel spacing (m).
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
# 4. Spatial-Temporal Swap
# =====================================================
def prep_ncf(
    ncf: np.ndarray, 
    lag_axis: np.ndarray, 
    distance_axis: np.ndarray, 
    vs: str | int, 
    gauge_length: float = 8.16
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Separates the Noise Correlation Function (NCF) into causal/acausal parts 
    and performs spatial-temporal recombination to group energy by source direction.

    :param ncf: The 2D noise correlation function data array.
    :type ncf: np.ndarray
    :param lag_axis: 1D array of time lag values.
    :type lag_axis: np.ndarray
    :param distance_axis: 1D array of spatial distances along the cable.
    :type distance_axis: np.ndarray
    :param vs: Virtual source channel index or identifier.
    :type vs: str | int
    :param gauge_length: The spacing between channels in meters. Default is 8.16.
    :type gauge_length: float, optional
    :returns: A tuple containing (causal NCF, acausal NCF, causal lag axis, 
              source 1 recombined NCF, source 2 recombined NCF).
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    ncf = np.asarray(ncf)
    lag_axis = np.asarray(lag_axis)
    distance_axis = np.asarray(distance_axis)

    if ncf.shape[1] != lag_axis.size and ncf.shape[0] == lag_axis.size:
        ncf = ncf.T

    # 1. Basic Time Separation
    c_sel = lag_axis >= 0
    ncf_c = ncf[:, c_sel]
    new_lag_axis = lag_axis[c_sel]

    a_sel = lag_axis <= 0
    ncf_a = ncf[:, a_sel][:, ::-1]

    # 2. Spatial-Temporal Splitting
    position = int(vs) * gauge_length
    vs_idx = np.argmin(np.abs(distance_axis - position))

    A = ncf_c[:vs_idx, :] 
    B = ncf_c[vs_idx:, :] 
    C = ncf_a[:vs_idx, :] 
    D = ncf_a[vs_idx:, :] 

    # 3. Source-consistent Recombination
    ncf_s1 = np.vstack([A, D])
    ncf_s2 = np.vstack([C, B])

    return ncf_c, ncf_a, new_lag_axis, ncf_s1, ncf_s2


def clip_ncf_side(
    data: np.ndarray, 
    distance_axis: np.ndarray, 
    vs: str | int, 
    range_m: float, 
    side: str = "right", 
    pos_offset: float = 0.0,
    gauge_length: float = 8.16
) -> tuple[np.ndarray, np.ndarray]:
    """
    Clips NCF spatially to one side of the virtual source with an optional inner offset.

    :param data: The 2D NCF data to be clipped.
    :type data: np.ndarray
    :param distance_axis: 1D array of spatial distances along the cable.
    :type distance_axis: np.ndarray
    :param vs: Virtual source channel index or identifier.
    :type vs: str | int
    :param range_m: The maximum spatial range (in meters) to retain from the virtual source.
    :type range_m: float
    :param side: The direction to clip relative to the source ("left" or "right"). Default is "right".
    :type side: str, optional
    :param pos_offset: Inner spatial offset (in meters) to exclude near-source effects. Default is 0.0.
    :type pos_offset: float, optional
    :param gauge_length: The spacing between channels in meters. Default is 8.16.
    :type gauge_length: float, optional
    :returns: A tuple containing the clipped NCF data and the corresponding clipped distance axis.
    :rtype: tuple[np.ndarray, np.ndarray]
    :raises ValueError: If `side` is invalid or if `pos_offset` exceeds the available range.
    """
    position = int(vs) * gauge_length
    
    if side.lower() == "right":
        lower = position + pos_offset
        upper = position + range_m
    elif side.lower() == "left":
        lower = position - range_m
        upper = position - pos_offset
    else:
        raise ValueError("side must be 'left' or 'right'")

    lower = max(distance_axis.min(), lower)
    upper = min(distance_axis.max(), upper)

    if lower > upper:
        raise ValueError(f"pos_offset ({pos_offset}m) is larger than the available range.")
    
    idx_sel = (distance_axis >= lower) & (distance_axis <= upper)
    
    return data[idx_sel, :], distance_axis[idx_sel]

@dask.delayed
def process_and_save_subset(
    path: str, 
    lag_axis: np.ndarray, 
    distance_axis: np.ndarray, 
    dt: float, 
    dx: float, 
    vmin: float, 
    vmax: float, 
    target: str, 
    side: str, 
    pos_offset: float, 
    range_m: float, 
    out_dir: str = "../results/ncf_disp"
) -> str:
    """
    Dask-delayed function to process a single NCF file: 
    F-K filter -> Directional Swap -> Spatial Clip -> Flip (if left) -> Save.
    
    :param path: Path to the raw .npy NCF file.
    :type path: str
    :param lag_axis: 1D array of time lag values.
    :type lag_axis: np.ndarray
    :param distance_axis: 1D array of spatial distances.
    :type distance_axis: np.ndarray
    :param dt: Time sampling interval.
    :type dt: float
    :param dx: Spatial sampling interval (gauge length).
    :type dx: float
    :param vmin: Minimum velocity for the F-K filter.
    :type vmin: float
    :param vmax: Maximum velocity for the F-K filter.
    :type vmax: float
    :param target: Wavefield mapping target ("s1", "s2", "causal", or "acausal").
    :type target: str
    :param side: Side relative to the virtual source ("left" or "right"). If "left", data is flipped to be causal/positive-traveling.
    :type side: str
    :param pos_offset: Inner spatial offset to exclude near-source effects.
    :type pos_offset: float
    :param range_m: Total spatial range to clip.
    :type range_m: float
    :param out_dir: Directory to save the processed results. Default is "../results/ncf_disp".
    :type out_dir: str, optional
    :returns: The base filename of the saved output file.
    :rtype: str
    """
    # 1. Metadata and Load
    date, vs, window, v_mode = parse_ncf_stack_filename(path)
    ncf_raw = np.load(path)
    
    # Ensure (n_channel, n_time) orientation
    if ncf_raw.shape == (lag_axis.size, distance_axis.size):
        ncf_raw = ncf_raw.T
    
    # 2. F-K Filter (Extracting energy within velocity bounds)
    ncf_fk = fk_filter(ncf_raw, dt=dt, dx=dx, vmin=vmin, vmax=vmax, mode="extract")
    
    # 3. Spatial-Temporal Swap (prep_ncf assumed to be in same file)
    ncf_c, ncf_a, h_lag, s1, s2 = prep_ncf(ncf_fk, lag_axis, distance_axis, vs)
    
    # Select target wavefield (s1, s2, causal, or acausal)
    mapping = {"s1": s1, "s2": s2, "causal": ncf_c, "acausal": ncf_a}
    target_data = mapping[target.lower()]
    
    # 4. Spatial Clipping
    # Grabs the subset of channels on the chosen side of the VS
    final_data, final_dist = clip_ncf_side(
        target_data, distance_axis, vs, 
        range_m=range_m, side=side, pos_offset=pos_offset
    )

    # Calculate Relative Distance from Virtual Source
    # (e.g., if VS is at 800m, and channel is at 816m, dist_rel = 16m)
    dist_rel = final_dist - (int(vs) * 8.16)

    # --- 5. THE LEFT-SIDE FLIP LOGIC ---
    # For dispersion analysis, we want distance to increase away from the source (0 -> +Range).
    # On the left side, dist_rel is negative (e.g., -10, -20, -30).
    # We flip the array and take absolute distance so the phase-shift sees a 
    # positive-traveling wave.
    if side.lower() == "left":
        dist_rel = np.abs(dist_rel[::-1]) 
        final_data = final_data[::-1, :] 

    # 6. Save to results directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    out_name = f"{date}_cc_{vs}_{window}_{v_mode}_{target}_{side}.npy"
    out_path = os.path.join(out_dir, out_name)
    
    # Store as a dictionary for easy loading in dispersion loops
    np.save(out_path, {
        "data": final_data, 
        "dist_rel": dist_rel, 
        "lag": h_lag,
        "vs_m": int(vs) * 8.16,
        "side": side.lower()
    })
    
    return out_name

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