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
from utils import convert_to_tensor, convert_to_numpy, runtime, nextpow2

logger = logging.getLogger(__name__)

# 1. Dispersion image (f-v panel) via phase-shift method
# ================================================================================
def dispersion_curve(data, offset, t, vmin=200.0, vmax=4000.0, dv=10.0, 
                     fmin=0.1, fmax=50.0, normalize=True, device=None):
    
    """
    Compute the f-v (frequency-velocity) dispersion image using the phase-shift method (Park et al., 1998).

    This function is designed for DAS gathers or NCF virtual-source gather with receivers along a 1D array.

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
    data = convert_to_tensor(data)
    offset = convert_to_tensor(offset, device)
    t = convert_to_tensor(t, device)

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
    nfft = int(nextpow2(torch.tensor([nt], device=device))[0].item())
    f_axis_full = torch.fft.rfftfreq(nfft, dt).to(device)
    freq_mask = (f_axis_full >= fmin) & (f_axis_full <= fmax)
    f_axis = f_axis_full[freq_mask]
    nf = f_axis.numel()

    logger.debug(f'Velocity axis: nv={nv}, vmin={v_axis.min().item():.1f}, vmax={v_axis.max().item():.1f}')
    logger.debug(f'Frequency axis: nf={nf}, fmin={f_axis.min().item():.3f}, fmax={f_axis.max().item():.3f}')

    # FFT along time (receiver × frequency)
    fft_data = torch.fft.rfft(data, n=nfft, dim=1)[:, freq_mask] # (nrec × nf)

    # Phase-only (avoid amplitude dominance)
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
    # fv_panel[v, f] = Σ_r ( phase_kernel[v,f,r] * phase_only[r,f] * dx[r] )
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









