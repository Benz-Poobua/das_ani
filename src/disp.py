"""
:module: src/disp.py
:author: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Dispersion imaging (f–v transform) and dispersion curve picking
          for DAS ambient noise interferometry.
"""
import logging 
import torch
import numpy as np
from utils import convert_to_tensor, convert_to_numpy, nextpow2

logger = logging.getLogger(__name__)

# 1. Dispersion image (f-v panel) via phase-shift method
# ================================================================================
def dispersion_curve(data, offset, t, vmin=200.0, vmax=4000.0, dv=10.0, 
                     fmin=0.1, fmax=50.0, normalize=True, device=None):
    
    """
    Compute the f-v (frequency-velocity) dispersion image using the phase-shift method (Park et al., 1998).

    This function is designed for DAS gathers or NCF virtual-source gathers with receivers along a 1D array.

    :param data: Seismic/DAS gather (nrec × nt). Each row is one channel.
    :type data: numpy.ndarray or torch.Tensor

    :param offset: Receiver offsets in meters, shape (nrec,).
    :type offset: numpy.ndarray or torch.Tensor

    :param t: Time axis in seconds, shape (nt,).
    :type t: numpy.ndarray or torch.Tensor

    :param vmin: Minimum phase velocity to search (m/s).
    :type vmin: float

    :param vmax: Maximum phase velocity to search (m/s).
    :type vmax: float

    :param dv: Velocity sampling interval (m/s).
    :type dv: float

    :param fmin: Minimum frequency for analysis (Hz).
    :type fmin: float

    :param fmax: Maximum frequency for analysis (Hz).
    :type fmax: float

    :param normalize: If True, normalize each frequency slice by its max.
    :type normalize: bool

    :param device: Torch device ('cuda', 'cpu', 'mps'). If None, auto-select.
    :type device: torch.device or None

    :return:
        - **fv_panel** (*torch.Tensor*) -- dispersion image, shape (nv, nf)
        - **f_axis** (*torch.Tensor*) -- frequency axis (Hz), shape (nf,)
        - **v_axis** (*torch.Tensor*) -- velocity axis (m/s), shape (nv,)
    :rtype: tuple
    """
    logger.info('Computing dispersion curve (phase-shift) method.')

    # Select device
    if device is None: 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    logger.info(f'Using device: {device}') 

    # Convert input to tensors
    data = convert_to_tensor(data, device=device)
    offset = convert_to_tensor(offset, device=device)
    t = convert_to_tensor(t, device=device)

    if data.ndim != 2:
        raise ValueError("'data' must be 2D: (nrec × nt).")
    if offset.ndim != 1 or offset.shape[0] != data.shape[0]:
        raise ValueError("'offset' must be 1D with length nrec.")
    if t.ndim != 1 or t.shape[0] != data.shape[1]:
        raise ValueError("'t' must be 1D with length nt.")
    
    nrec, nt = data.shape
    dt = float(t[1] - t[0])
    logger.debug(f'Gather shape: nrec={nrec}, nt={nt}, dt={dt:.6f}s')

    # Build velocity and frequency axes
    v_axis = torch.arange(vmin, vmax, dv, device=device, dtype=torch.float32)
    v_axis = v_axis[v_axis > 0]
    nv = v_axis.numel()

    # Next power-of-two FFT length
    nfft = int(nextpow2(torch.tensor(nt, device=device)).item())
    f_axis_full = torch.fft.rfftfreq(nfft, dt).to(device)
    freq_mask = (f_axis_full >= fmin) & (f_axis_full <= fmax)
    f_axis = f_axis_full[freq_mask]
    nf = f_axis.numel()

    logger.debug(f'Velocity axis: nv={nv}, vmin={v_axis.min().item():.1f}, vmax={v_axis.max().item():.1f}')
    logger.debug(f'Frequency axis: nf={nf}, fmin={f_axis.min().item():.3f}, fmax={f_axis.max().item():.3f}')

    # FFT along time (receiver × frequency)
    fft_data = torch.fft.rfft(data, n=nfft, dim=1)[:, freq_mask] # (nrec × nf)

    # Phase-only normalization (avoid amplitude dominance)
    amp = torch.abs(fft_data)
    phase_only = fft_data / (amp + 1e-8)

    # Vectorized phase-shift integration
    two_pi = 2.0 * np.pi
    f_mat = f_axis.unsqueeze(0)     # (1 × nf)
    v_mat = v_axis.unsqueeze(1)     # (nv × 1)
    k_mat = two_pi * f_mat / v_mat  # (nv × nf)

    x = offset                      # (nrec,)
    dx = torch.gradient(x)[0]       # (nrec,)

    # Phase kernel: (nv × nf × nrec)
    phase_kernel = torch.exp(1j * k_mat.unsqueeze(-1) * x)

    logger.debug('Performing vectorized phase-shift integration via einsum.')
    # fv_panel[v, f] = Σ_r (phase_kernel[v,f,r] * phase_only[r,f] * dx[r])
    fv_panel = torch.einsum('vfr,rf,r->vf', phase_kernel, phase_only, dx)

    # Amplitude 
    fv_panel = torch.abs(fv_panel)

    # Optional per-frequency normalization 
    if normalize:
        logger.debug('Normalizing each frequency slice of f-v panel.')
        max_val = torch.amax(fv_panel, dim=0, keepdim=True)
        max_val[max_val == 0] = 1.0
        fv_panel = fv_panel / max_val

    logger.info('Dispersion image computation completed.')
    return fv_panel, f_axis, v_axis

# 2. Dispersion curve extraction (from f–v panel)
# ================================================================================
def extr_disp(f_axis, v_axis, fv_panel, f_ref_set=None, vmax_set=None, step=5):
    """
    Extract a dispersion curve from a frequency–velocity image by tracking
    local maxima, following Huajian Yao's MATLAB picking approach.

    This is a high-level wrapper that chooses between single-start
    (`AutoSearch`) and multi-start (`AutoSearchMultiplePoints`) picking.

    :param f_axis: Frequency axis (Hz), length nf.
    :type f_axis: numpy.ndarray or torch.Tensor

    :param v_axis: Velocity axis (m/s), length nv.
    :type v_axis: numpy.ndarray or torch.Tensor

    :param fv_panel: Dispersion image, amplitude at each (v, f), shape (nv × nf).
    :type fv_panel: numpy.ndarray or torch.Tensor

    :param f_ref_set: List of reference frequencies for starting points.
    :type f_ref_set: list[float] or None

    :param vmax_set: List of max allowed velocities at corresponding f_ref_set.
    :type vmax_set: list[float] or None

    :param step: Vertical search step (in velocity-index units) for ridge tracking.
    :type step: int

    :return: 1D array of picked phase velocities vs frequency, shape (nf,).
    :rtype: numpy.ndarray
    """
    logger.info('Extracting dispersion curve from f-v panel.')

    # Convert to numpy for picking logic 
    f = convert_to_numpy(f_axis)
    v = convert_to_numpy(v_axis)
    disp = convert_to_numpy(fv_panel)

    nv, nf = disp.shape
    if disp.shape != (nv, nf):
        raise ValueError("'fv_panel' must have shape (nv, nf).")

    if f_ref_set is None:
        # Default: start at lowest usable frequency
        f_ref_set = [f[0]]

    if vmax_set is None:
        # Default: allow full velocity range
        vmax_set = [v[-1]] * len(f_ref_set)

    if len(f_ref_set) != len(vmax_set):
        raise ValueError("'f_ref_set' and 'vmax_set' must have same length.")
    
    # Determine starting points (in index space)
    xpt = []
    ypt = []
    for k in range(len(f_ref_set)):
        f_ref = f_ref_set[k]
        vmax = vmax_set[k]

        # Find closest frequency index ≥ f_ref
        idx_f = np.where(f >= f_ref)[0]
        if len(idx_f) == 0:
            raise ValueError(f'No frequencies >= f_ref = {f_ref} Hz.')
        idx_f = idx_f[0]

        f_ref_actual = f[idx_f]
        disp_ref = disp[:, idx_f]

        # Restrict velocities to v < vmax
        mask_v = v < vmax
        if not np.any(mask_v):
            raise ValueError(f'No velocities < vmax = {vmax} m/s.')
        v_sub = v[mask_v]
        disp_sub = disp_ref[mask_v]

        # Find velocity index of maximum energy with the restricted window
        idx_v_local = np.argmax(disp_sub)
        v_ref = v_sub[idx_v_local]

        # Convert to full velocity index
        idx_v = np.abs(v - v_ref).argmin()

        ypt.append(idx_v)
        xpt.append(idx_f)

        logger.debug(
            f'Start point {k}: f_ref={f_ref_actual:.3f} Hz, v_ref={v_ref:.1f} m/s '
            f'(idx_f = {idx_f}, idx_v={idx_v})'
        )
    
    xpt = np.array(xpt, dtype=int)
    ypt = np.array(ypt, dtype=int)

    # Single-start or multi-start picking
    if len(xpt) == 1:
        logger.debug('Using single-start AutoSearch for dispersion picking.')
        arr_pt = AutoSearch(ypt[0], xpt[0], disp, step=step)
    
    else:
        logger.debug('Using multi-start AutoSearchMultiplePoints for dispersion picking.')
        arr_pt = AutoSearchMultiplePoints(ypt, xpt, disp, step=step)
    
    voutput = v[arr_pt]
    logger.info('Dispersion curve extraction completed.')
    return voutput

def AutoSearch(initial_y, initial_x, image_data, step=5):
    """
    Track a dispersion ridge (local maxima) from a single starting point
    in a 2D f–v image (velocity × frequency).

    This follows Huajian Yao's MATLAB strategy: at each frequency slice,
    search upward and downward in velocity to find the local maximum.

    :param initial_y: Initial velocity index (row index).
    :type initial_y: int

    :param initial_x: Initial frequency index (column index).
    :type initial_x: int

    :param image_data: f–v image, shape (nv × nf).
    :type image_data: numpy.ndarray

    :param step: Vertical search step (velocity index increment).
    :type step: int

    :return: Indices of picked velocities for all frequencies, shape (nf,).
    :rtype: numpy.ndarray
    """
    YSize, XSize = image_data.shape
    ArrPt = np.zeros(XSize, dtype=int)

    # 1. Scan upward in frequency (from initial_x to high frequencies)
    current_y = initial_y
    for i in range(initial_x, XSize):
        point_left = current_y
        point_right = current_y
        # search upward (toward smaller velocity index)
        while True:
            point_left_new = max(0, point_left - step)
            if image_data[point_left, i] < image_data[point_left_new, i]:
                point_left = point_left_new
            else:
                point_left = point_left_new
                break
        # search downward (toward larger velocity index)
        while True:
            point_right_new = min(point_right + step, YSize - 1)
            if image_data[point_right, i] < image_data[point_right_new, i]:
                point_right = point_right_new
            else:
                point_right = point_right_new
                break

        idx_local = np.argmax(image_data[point_left:point_right + 1, i])
        ArrPt[i] = idx_local + point_left
        current_y = ArrPt[i]

    # 2. Scan downward in frequency (from initial_x back to low frequencies)
    current_y = ArrPt[initial_x]
    for i in range(initial_x - 1, -1, -1):
        point_left = current_y
        point_right = current_y
        # search upward
        while True:
            point_left_new = max(0, point_left - step)
            if image_data[point_left, i] < image_data[point_left_new, i]:
                point_left = point_left_new
            else:
                point_left = point_left_new
                break
        # search downward
        while True:
            point_right_new = min(point_right + step, YSize - 1)
            if image_data[point_right, i] < image_data[point_right_new, i]:
                point_right = point_right_new
            else:
                point_right = point_right_new
                break

        idx_local = np.argmax(image_data[point_left:point_right + 1, i])
        ArrPt[i] = idx_local + point_left
        current_y = ArrPt[i]

    return ArrPt

def AutoSearchMultiplePoints(ptY, ptX, image_data, step=5):
    """
    Track a dispersion ridge from multiple starting points in an f–v image,
    allowing extraction of more complex or multi-branch dispersion patterns.

    This generalizes AutoSearch by stitching together segments from
    several user-defined starting points.

    :param ptY: Array of initial velocity indices (row indices).
    :type ptY: numpy.ndarray

    :param ptX: Array of initial frequency indices (column indices).
    :type ptX: numpy.ndarray

    :param image_data: f–v image, shape (nv × nf).
    :type image_data: numpy.ndarray

    :param step: Vertical search step (velocity index increment).
    :type step: int

    :return: Indices of picked velocities for all frequencies, shape (nf,).
    :rtype: numpy.ndarray
    """
    ptY = np.asarray(ptY, dtype=int)
    ptX = np.asarray(ptX, dtype=int)

    nPt = len(ptX)
    if nPt == 0:
        raise ValueError('ptX/ptY must contain at least one point.')
    
    # Sort points by frequency index
    order = np.argsort(ptX)
    ptX = ptX[order]
    ptY = ptY[order]

    YSize, XSize = image_data.shape
    ArrPt = np.zeros(XSize, dtype=int)

    # 1. Scan from highest starting frequency to higher frequencies
    initial_x = ptX[-1]
    initial_y = ptY[-1]
    current_y = initial_y
    for i in range(initial_x, XSize):
        point_left = current_y
        point_right = current_y
        # Up
        while True:
            point_left_new = max(0, point_left - step)
            if image_data[point_left, i] < image_data[point_left_new, i]:
                point_left = point_left_new
            else:
                point_left = point_left_new
                break
        # Down
        while True:
            point_right_new = min(point_right + step, YSize - 1)
            if image_data[point_right, i] < image_data[point_right_new, i]:
                point_right = point_right_new
            else:
                point_right = point_right_new
                break
        
        idx_local = np.argmax(image_data[point_left:point_right + 1, i])
        ArrPt[i] = idx_local + point_left
        current_y = ArrPt[i]

    # 2. Scan toward lower frequencies, stitching through intermediate points
    initial_x = ptX[-1]
    current_y = ArrPt[initial_x]
    mid_idx = nPt - 2
    if mid_idx >= 0:
        midX = ptX[mid_idx]
        midY = ptY[mid_idx]
    else:
        midX = ptX[0]
        midY = ptY[0]

    kk = 0
    for i in range(initial_x, -1, -1):
        if i == midX:
            current_y = midY
            kk += 1
            if (nPt - kk) > 1:
                midX = ptX[nPt - kk - 2]
                midY = ptY[nPt - kk - 2]

        point_left = current_y
        point_right = current_y

        # Up
        while True:
            point_left_new = max(0, point_left - step)
            if image_data[point_left, i] < image_data[point_left_new, i]:
                point_left = point_left_new
            else:
                point_left = point_left_new
                break

        # Down
        while True:
            point_right_new = min(point_right + step, YSize - 1)
            if image_data[point_right, i] < image_data[point_right_new, i]:
                point_right = point_right_new
            else:
                point_right = point_right_new
                break

        idx_local = np.argmax(image_data[point_left:point_right + 1, i])
        ArrPt[i] = idx_local + point_left
        current_y = ArrPt[i]
    
    return ArrPt