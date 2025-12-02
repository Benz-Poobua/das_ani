"""
:module: src/disp.py
:author: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Dispersion imaging (f–v transform) and dispersion curve picking
          for DAS ambient noise interferometry.
"""
import os
import json
import logging 
import torch
import numpy as np
from src.utils import convert_to_tensor, convert_to_numpy, nextpow2, timeit

logger = logging.getLogger(__name__)

# 1. Dispersion image (f-v panel) via phase-shift method
# ================================================================================
@timeit
def dispersion_curve(data, 
                     offset, 
                     t,
                     vmin=200.0, 
                     vmax=4000.0, 
                     dv=10.0, 
                     fmin=0.1, 
                     fmax=50.0, 
                     normalize=True, 
                     device=None, 
                     batch_size_v=None):
    
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
    :param batch_size_v: Number of velocities per batch in the phase-shift
                         integration. If None, choose a heuristic based on nv.
    :type batch_size_v: int or None

    :return:
        - **fv_panel** (*torch.Tensor*) -- dispersion image, shape (nv, nf)
        - **f_axis** (*torch.Tensor*) -- frequency axis (Hz), shape (nf,)
        - **v_axis** (*torch.Tensor*) -- velocity axis (m/s), shape (nv,)
    :rtype: tuple
    """
    logger.info('Computing dispersion curve (phase-shift) method with batching.')

    # Select device
    if device is None: 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    logger.info(f'Using device: {device}') 

    # Convert input to tensors
    data = convert_to_tensor(data, device=device)
    offset = convert_to_tensor(offset, device=device)
    t = convert_to_tensor(t,device=device)

    if data.ndim != 2:
        raise ValueError("'data' must be 2D: (nrec × nt).")
    if offset.ndim != 1 or offset.shape[0] != data.shape[0]:
        raise ValueError("'offset' must be 1D with length nrec.")
    if t.ndim != 1 or t.shape[0] != data.shape[1]:
        raise ValueError("'t' must be 1D with length nt.")
    
    nrec, nt = data.shape
    if nt < 2:
        raise ValueError('Time axis must have at least 2 samples to compute dt.')

    dt = float(t[1] - t[0])
    logger.debug(f'Gather shape: nrec={nrec}, nt={nt}, dt={dt:.6f}s')

    # Build velocity 
    v_axis = torch.arange(vmin, vmax + dv, dv, device=device, dtype=torch.float32)
    nv = v_axis.numel()

    # FFT parameters
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
    amp   = torch.abs(fft_data)
    phase = fft_data / (amp + 1e-8)  # (nrec × nf)

    # Batching velocities
    if batch_size_v is None:
        # Heuristic: smaller batches on GPU, full nv on CPU
        batch_size_v = 64 if device.type == 'cuda' else nv
    batch_size_v = max(1, min(batch_size_v, nv))

    logger.info(f'Using batch_size_v = {batch_size_v} (nv = {nv})')

    fv_panel = torch.zeros((nv, nf), device=device, dtype=torch.float32)

    # Phase-shift integration, batched over velocities
    for v_start in range(0, nv, batch_size_v):
        v_end = min(v_start + batch_size_v, nv)
        v_batch = v_axis[v_start:v_end]         # (nv_b,)

        # k = 2π f / v : shapes -> f_axis(1 × nf), v_batch(nv_b × 1)
        # Vectorized phase-shift integration
        f_mat = f_axis.unsqueeze(0)             # (1 × nf)
        v_mat = v_batch.unsqueeze(1)            # (nv_b × 1)
        k = 2.0 * np.pi * f_mat / v_mat         # (nv_b × nf)

        # Phase kernel: exp(i k x); shapes:
        # x: (1, 1, nrec), k: (nv_b, nf, 1) -> kernel: (nv_b, nf, nrec)
        x = offset.unsqueeze(0).unsqueeze(0)          # (1 × 1 × nrec)
        kernel = torch.exp(1j * k.unsqueeze(-1) * x)  
 
        logger.debug(f'Velocity batch [{v_start}:{v_end}] -> kernel shape = {kernel.shape}')

        # Integrate over receivers: Σ_r kernel[v,f,r] * phase[r,f]
        fv = torch.einsum('vfr,rf->vf', kernel, phase) # (nv_b, nf)

        # Magnitude
        fv_panel[v_start:v_end, :] = torch.abs(fv)

        # Free some temporary tensors (helps on tight GPUs)
        del kernel, fv, v_batch, f_mat, v_mat, k
        if device.type == 'cuda':
            torch.cuda.empty_cache()

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

# 3. Compute dispersion directly from NCF matrix
# ================================================================================
def compute_dispersion_from_ncf(ncf, fs, dx=8.16, **kwargs):
    """
    Compute the dispersion image and picked curve directly from a single
    NCF (noise cross-correlation) matrix.

    This is a convenience wrapper that builds the offset and time axes
    from the NCF geometry and then calls :func:`dispersion_curve` and
    :func:`extr_disp`.

    :param ncf: Cross-correlation matrix with shape (nrec, nlag), where
                ``nrec`` is the number of channels and ``nlag`` is the
                number of lag samples (symmetric around zero lag).
    :type ncf: numpy.ndarray or torch.Tensor
    :param fs: Sampling rate (Hz) of the original time series.
    :type fs: float
    :param dx: Channel spacing in meters (receiver spacing along the fiber).
    :type dx: float
    :param kwargs: Additional keyword arguments forwarded to
                   :func:`dispersion_curve` (e.g., ``vmin``, ``vmax``,
                   ``dv``, ``fmin``, ``fmax``, ``device``, etc.).
    :type kwargs: dict

    :return: Tuple ``(fv_panel, f_axis, v_axis, picks)`` where
             - ``fv_panel`` is the dispersion image (nv, nf),
             - ``f_axis`` is the frequency axis (Hz),
             - ``v_axis`` is the velocity axis (m/s),
             - ``picks`` is the picked phase-velocity curve (nf,).
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, numpy.ndarray]
    """
    ncf_arr = convert_to_numpy(ncf)
    if ncf_arr.ndim != 2:
        raise ValueError("'ncf' must be 2D with shape (nrec, nlag).")
    
    nrec, nlag = ncf_arr.shape
    logger.info(f'Computing dispersion from NCF: nrec={nrec}, nlag={nlag}, fs={fs}, dx={dx}')

    # Build offset vector
    offset = np.arange(nrec, dtype=float) * float(dx)

    # Build symmetric time vector centered at zero lag
    max_lag = (nlag - 1) // 2
    t = np.linspace(-max_lag / fs, max_lag / fs, nlag, dtype=float)

    # Compute dispersion
    fv_panel, f_axis, v_axis = dispersion_curve(data=ncf_arr, offset=offset, t=t, **kwargs)

    # Pick fundamental dispersion curve
    picks = extr_disp(f_axis, v_axis, fv_panel)

    return fv_panel, f_axis, v_axis, picks

# 4. Load NCF file, compute dispersion, and save results
# ================================================================================
def load_ncf_and_compute_dispersion(ncf_path, 
                                    dx=8.16, 
                                    stack_window='daily', # 'daily', '7d', '15d', '30d'
                                    results_root='results/dispersion', 
                                    fs=250, 
                                    **disp_kwargs):
    """
    Load an NCF file from disk, compute the dispersion image and picked
    dispersion curve, and save all outputs to disk.

    The outputs are saved under::

        <results_root>/<stack_window>/

    with filenames derived from the input ``ncf_path`` base name.

    Files generated:

    - ``<base>_fv_panel.npy``  : dispersion image (nv, nf)
    - ``<base>_f_axis.npy``    : frequency axis (nf,)
    - ``<base>_v_axis.npy``    : velocity axis (nv,)
    - ``<base>_pick.npy``      : picked dispersion curve (nf,)
    - ``<base>_meta.json``     : metadata (paths, shapes, parameters)

    :param ncf_path: Path to the input NCF file (.npy) of shape (nrec, nlag).
    :type ncf_path: str
    :param dx: Channel spacing in meters (receiver spacing along the fiber).
    :type dx: float
    :param stack_window: Label of stack window used for this NCF, e.g.
                         ``'daily'``, ``'7d'``, ``'15d'``, ``'30d'``.
                         Only used for organizing output directories.
    :type stack_window: str
    :param results_root: Root directory where dispersion results will be
                         stored. Outputs are placed in a subdirectory
                         ``results_root/stack_window/``.
    :type results_root: str
    :param fs: Sampling rate (Hz) of the original time series.
    :type fs: float
    :param disp_kwargs: Additional keyword arguments forwarded to
                        :func:`compute_dispersion_from_ncf`, and then
                        to :func:`dispersion_curve`.
    :type disp_kwargs: dict

    :return: Dictionary containing in-memory results and output directory:
             ``{'fv_panel', 'f_axis', 'v_axis', 'picks', 'outdir'}``.
    :rtype: dict
    """
    logger.info(
        f'Loading NCF from {ncf_path} for dispersion analysis '
        f'(stack_window={stack_window}, dx={dx}, fs={fs})'
    )

    ncf = np.load(ncf_path)

    fv_panel, f_axis, v_axis, picks = compute_dispersion_from_ncf(ncf=ncf, fs=fs, dx=dx, **disp_kwargs)

    # Prepare output directory
    outdir = os.path.join(results_root, stack_window)
    os.makedirs(outdir, exist_ok=True)

    # Base name without extension
    base = os.path.basename(ncf_path).replace('.npy', '')

    # Save arrays
    np.save(os.path.join(outdir, f'{base}_fv_panel.npy'), convert_to_numpy(fv_panel))
    np.save(os.path.join(outdir, f'{base}_f_axis.npy'), convert_to_numpy(f_axis))
    np.save(os.path.join(outdir, f'{base}_v_axis.npy'), convert_to_numpy(v_axis))
    np.save(os.path.join(outdir, f'{base}_pick.npy'), picks)

    # Metadata
    meta = {
        'ncf_path': os.path.abspath(ncf_path),
        'dx': float(dx),
        'stack_window': stack_window,
        'shape_ncf': list(ncf.shape),
        'shape_fv_panel': list(convert_to_numpy(fv_panel).shape),
        'fs': float(fs),
        'dispersion_kwargs': {k: repr(v) for k, v in disp_kwargs.items()},
    }
    meta_path = os.path.join(outdir, f'{base}_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=4)

    logger.info(f'Saved dispersion results → {outdir}')

    return {
        'fv_panel': fv_panel,
        'f_axis': f_axis,
        'v_axis': v_axis,
        'picks': picks,
        'outdir': outdir
    }