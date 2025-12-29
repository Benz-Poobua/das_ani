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

import logging 
import torch
import numpy as np

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.utils import convert_to_tensor, convert_to_numpy, nextpow2, timeit

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

    # FFT along time: (nrec, nf)
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
      - optional picks (if pick_kwargs provided)

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

    # Build offset vector
    offset = np.arange(nrec, dtype=float) * float(dx)

    # Build symmetric time vector centered at zero lag; assume nlag is odd and centered
    max_lag = (nlag - 1) // 2
    t = np.linspace(-max_lag / float(fs), max_lag / float(fs), nlag, dtype=float)

    # Compute dispersion
    fv_kwargs = fv_kwargs or {}
    fv_panel, f_axis, v_axis = dispersion_curve(
        data=ncf_arr, offset=offset, t=t, **fv_kwargs)

    # Pick fundamental dispersion curve
    picks = None
    if pick_kwargs is not None:
        picks = extr_disp(f_axis, v_axis, fv_panel, **pick_kwargs)

    return fv_panel, f_axis, v_axis, picks