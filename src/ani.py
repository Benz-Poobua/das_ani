"""
:module: src/ani.py
:auth: Benz Poobua 
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: This module contains DAS preprocessing (Bensen et al., 2007) and cross-correlation scripts.
:reference: The script is modified from Yan Yang 2022-07-10
"""
import torch
import logging
import numpy as np
from torch import nn
import scipy.signal as signal
from scipy.signal import butter, filtfilt, convolve, detrend
from utils import convert_to_numpy, convert_to_tensor, runtime

logger = logging.getLogger(__name__)

def bandpass_filter_tukey(data, fs, f1, f2, alpha=0.05):
    """
    Apply a Tukey-windowed bandpass filter to DAS data.

    :param data: 2D DAS array with shape (n_channels, n_samples)
    :type data: numpy.ndarray
    :param fs: sampling frequency in Hz
    :type fs: float
    :param f1: low-cut frequency in Hz
    :type f1: float
    :param f2: high-cut frequency in Hz
    :type f2: float
    :param alpha: Tukey window taper parameter (0–1), defaults to 0.05
    :type alpha: float, optional

    :return: filtered DAS array with same shape as input
    :rtype: numpy.ndarray
    """
    # Input validation 
    logger.debug(f'bandpass_filter_tukey(data.shape={data.shape}, fs={fs}, '
                 f'f1={f1}, f2={f2}, alpha={alpha})')

    if data.ndim != 2:
        logger.error(f"Input 'data' must be 2D (n_channels × n_samples). Got shape {data.shape}")
        raise ValueError('Input data must be 2D')
    
    if f1 <= 0 or f2 <= 0 or f2 <= f1:
        logger.error(f'Invalid bandpass frequencies f1={f1}, f2={f2}')
        raise ValueError('Frequencies must satisfy 0 < f1 < f2 < Nyquist')
    
    nyquist = fs / 2
    if f2 >= nyquist:
        logger.error(f'f2={f2} exceeds Nyquist frequency ({nyquist})')
        raise ValueError('High corner frequency is above Nyquist')
    
    # Create Tukey window
    n_samples = data.shape[1]
    window = signal.windows.tukey(n_samples, alpha=alpha)
    logger.debug(f'Tukey window generated (alpha={alpha}, length={n_samples})')

    # Butterworth bandpass filter
    low = f1 / nyquist 
    high = f2 / nyquist
    b, a = butter(4, [low, high], btype='bandpass')
    logger.debug(f'Butterworth filter designed: order=4, low={low}, high={high}')

    # Apply tapered bandpass
    try:
        tapered = data * window     # broadcast over channels
        filtered = filtfilt(b, a, tapered, axis=1)
        logger.info(
            f'Filtering completed: shape={filtered.shape}, '
            f'band=[{f1}-{f2}] Hz'
        )

    except Exception as e:
        logger.error(f'Filtering failed: {e}')
        raise

    return filtered

def running_absolute_mean(trace, nwin):
    """
    Compute a running absolute mean (RAM) of a 1D trace and return the trace
    normalized by its RAM.

    Padding strategy follows the NoisePy approach:
    - The absolute trace is padded on both sides by repeating the first and 
      last absolute values to avoid convolution edge effects.

    :param trace: Input 1D time series.
    :type trace: numpy.ndarray
    :param nwin: Window length (in samples) for the moving average.
    :type nwin: int

    :return: Trace normalized by running absolute mean.
    :rtype: numpy.ndarray
    """
    if trace.ndim != 1:
        raise ValueError("Input 'trace' must be a 1D array.")
    
    npts = len(trace)
    if nwin <= 1:
        logger.warning('nwin <= 1; returning original trace.') 
        return trace.copy()
    
    # Absolute values of trace
    abs_trace = np.abs(trace)

    # Prepare padded array: length = npts + 2*nwin
    padded = np.zeros(npts + 2 * nwin, dtype=trace.dtype)

    # Insert the central region
    padded[nwin:-nwin] = abs_trace

    # Pad front and back with boundary values
    padded[:nwin] = abs_trace[0]
    padded[-nwin:] = abs_trace[-1]

    # Moving average kernel 
    kernel = np.ones(nwin) / nwin

    # Convolve and remove padding
    ram = convolve(padded, kernel, mode='same')[nwin:-nwin]

    # Handle zeros to avoid division issues
    ram = np.where(ram == 0, np.nan, ram)

    return np.nan_to_num(trace / ram, nan=0.0)

def temporal_normalization(data, fs, window_time):
    """
    Apply temporal normalization to multichannel data using either:
    
    - **One-bit normalization** (if ``window_time == 0``)
    - **Running absolute mean (RAM) normalization** (if ``window_time > 0``)

    :param data: Multichannel time series array of shape (n_channels, n_samples).
    :type data: numpy.ndarray
    :param fs: Sampling frequency (Hz).
    :type fs: float
    :param window_time: Running window duration in seconds.  
                        - If 0 → performs one-bit normalization.  
                        - Recommended RAM window: ~½ of the longest period.
    :type window_time: float

    :return: Normalized data with the same shape as input.
    :rtype: numpy.ndarray
    """
    if data.ndim != 2:
        raise ValueError("Input 'data' must be 2D with shape (n_channels, n_samples).")
    
    nch, npts = data.shape

    # One-Bit Normalization 
    if window_time == 0:
        logger.info('Applying one-bit normalization.')
        return np.sign(data)
    
    # Running Absolute Mean (RAM) Normalization
    nwin = int(fs * window_time)
    if nwin < 1:
        logger.warning(f'Computed nwin={nwin}. Forcing nwin=1.')
        nwin = 1

    logger.info(f'Applying RAM normalization: window={window_time}s ({nwin} samples).')

    norm_data = data.copy()
    for i in range(nch):
        norm_data[i, :] = running_absolute_mean(norm_data[i, :], nwin)

    return norm_data

def spectral_whitening(rfftdata, df, window_freq, f1, f2):
    """
    GPU-accelerated spectral whitening. Supports two modes:

    - **Phase-only whitening** (if ``window_freq == 0``)  
      → Keeps only phase, sets amplitude to 1.

    - **Running Absolute Mean (RAM) spectral whitening** (if ``window_freq > 0``)  
      → Smooths amplitude using a moving window in the frequency domain.

    Additionally, a cosine taper is applied outside the [f1, f2] passband.

    :param rfftdata: Complex frequency-domain tensor of shape (nch, nfreq)
                     Must already be on GPU (CUDA or MPS).
    :type rfftdata: torch.Tensor
    :param df: Frequency bin spacing (Hz)
    :type df: float
    :param window_freq: RAM window length (Hz). If 0 → phase-only whitening.
    :type window_freq: float
    :param f1: Low-cut frequency (Hz)
    :type f1: float
    :param f2: High-cut frequency (Hz)
    :type f2: float
    :return: Whitened spectra, same shape as input
    :rtype: torch.Tensor
    """
    if not torch.is_tensor(rfftdata):
        raise TypeError('rfftdata must be a torch.Tensor.')
    if not rfftdata.is_complex():
        raise ValueError('rfftdata must be a complex-valued tensor.')

    device = rfftdata.device
    nch, nfreq = rfftdata.shape

    # Frequency indices
    idxf1 = int(f1 / df)
    idxf2 = idxf2 = int(torch.ceil(torch.tensor(f2/df, device=device)).item())  

    if idxf1 < 0 or idxf2 > nfreq:
        raise ValueError('f1 or f2 exceed available frequency bins.')
    
    mode = 'phase-only' if window_freq == 0 else 'RAM'
    logger.info(
        f'Spectral whitening ({mode}) | f1={f1}Hz f2={f2}Hz window={window_freq}Hz'
    )

    # 1. Phase-only Whitening
    # ========================================
    if window_freq == 0:
        phases = torch.angle(rfftdata)
        return torch.exp(1j * phases)
    
    # 2. Running Absolute Mean (RAM)
    # ========================================
    nwin = max(int(window_freq / df), 1)

    amp = torch.abs(rfftdata)       # (nch, nfreq)
    phases = torch.angle(rfftdata)

    # Running mean with 1D convolution (GPU)
    # conv1d expects shape (batch, channels, length)
    amp_3d = amp.unsqueeze(1)       # (nch, 1, nfreq)

    kernel = torch.ones((1, 1, nwin), device=device) / nwin 

    # Padding to maintain same length
    pad = nwin // 2
    amp_smooth = torch.nn.functional.conv1d(
        amp_3d, 
        kernel, 
        padding=pad
    ).squeeze(1)                    # (nch, nfreq)

    # Avoid division by zero
    amp_smooth = torch.where(amp_smooth == 0, torch.tensor(1.0, device=device), amp_smooth)

    # Rebuild whitened spectrum 
    rfftdata = torch.exp(1j * phases) * (amp / amp_smooth)

    # 3. Cosine Taper outside [f1, f2] 
    # ========================================
    if idxf1 > 0:
        taper1 = torch.cos(
            torch.linspace(torch.pi/2, torch.pi, idxf1, device=device)
        ) ** 2 
        rfftdata[:, :idxf1] *= taper1

    if idxf2 < nfreq:
        taper2 = torch.cos(
            torch.linspace(torch.pi, torch.pi/2, nfreq - idxf2, device=device)
        ) ** 2 
        rfftdata[:, idxf2:] *= taper2

    return rfftdata

def preprocess(x, fs_raw, f1, f2, decimation, diff, ram_win):
    """
    Preprocess a single DAS data chunk following the ambient noise workflow:
    differentiation → detrend → bandpass filter → decimation → 
    temporal normalization.

    :param x: Input DAS array of shape (n_channels × n_samples).
    :type x: numpy.ndarray or torch.Tensor
    :param fs_raw: Original sampling frequency (Hz).
    :type fs_raw: float
    :param f1: Low-cut frequency for bandpass filter (Hz).
    :type f1: float
    :param f2: High-cut frequency for bandpass filter (Hz).
    :type f2: float
    :param decimation: Decimation factor (integer > 0).
    :type decimation: int
    :param diff: Whether to take time derivative (∂/∂t).
    :type diff: bool
    :param ram_win: Window length (seconds) for running-absolute-mean
                    temporal normalization. If 0 → one-bit normalization.
    :type ram_win: float

    :return: Preprocessed DAS array (float32) with shape 
             (n_channels × n_samples/decimation).
    :rtype: numpy.ndarray
    """
    start_time = runtime()

    logger.info(
        f'Preprocess | shape={x.shape} | fs={fs_raw}Hz | '
        f'band=[{f1}, {f2}] Hz | decim={decimation} | '
        f'diff={diff} | RAM={ram_win}s'
    )

    # Track initial type to decide final output type
    is_tensor = torch.is_tensor(x)
    if is_tensor:
        logger.debug('Input is torch.Tensor → converting to numpy')
        x = convert_to_numpy(x) 

    # 1. Differentiation
    # ========================================
    if diff:
        logger.debug('Applying time derivative (np.gradient)')
        x = np.gradient(x, axis=-1) * fs_raw

    # 2. Detrend
    # ========================================
    logger.debug('Detrending time series')
    x = detrend(x, axis=-1)

    # 3. Bandpass filter (Butterworth + Tukey taper)
    # ========================================
    logger.debug(f'Applying bandpass filter: {f1}-{f2} Hz')
    x = bandpass_filter_tukey(x, fs_raw, f1, f2)

    # 4. Decimation
    # ========================================
    if decimation > 1:
        logger.debug(f'Decimation by factor {decimation}')
        x = x[:, ::decimation]

    fs_proc = fs_raw / decimation

    # 5. Remove channel-wise DC offset
    # ========================================
    logger.debug('Remove median trace offset')
    x -= np.median(x, axis=0)

    # 6. Temporal normalization (one-bit or RAM)
    # ========================================
    logger.debug(f'Applying temporal normalization | RAM window={ram_win}s')
    x = temporal_normalization(x, fs_proc, ram_win)

    # 7. Return float32
    # ========================================
    x = x.astype(np.float32)
    elasped = runtime() - start_time
    logger.info(f'Preprocess complete | output shape={x.shape} | {elasped:.3f}s')

    # 8. Optional return to torch
    # ========================================
    if is_tensor:
        logger.debug('Returning output as torch.Tensor on proper device')
        return convert_to_tensor(x)
    
    return x

def nextpow2(x):
    """
    Vectorized next power of 2 for tensor inputs, fully GPU-accelerated.

    :param x: 1D or multi-dim tensor of integers
    :type x: torch.Tensor

    :return: tensor of next powers of 2 (same shape as x)
    :rtype: torch.Tensor
    """
    x = x.to(torch.float32)
    return 2 ** torch.ceil(torch.log2(x))

def cross_correlation(signal_1, signal_2, is_spectral_whitening=False, whitening_params=None):
    """
    Compute multi-channel cross-correlation using FFT (GPU-accelerated if inputs are GPU tensors).

    :param signal_1: First input signal array (n_channels × n_samples).
    :type signal_1: torch.Tensor
    :param signal_2: Second input signal array (n_channels × n_samples).
    :type signal_2: torch.Tensor
    :param is_spectral_whitening: Whether to apply spectral whitening before CC.
    :type is_spectral_whitening: bool
    :param whitening_params: Tuple (fs, window_freq, f1, f2) for whitening.
    :type whitening_params: tuple or None
    :return: Cross-correlation of shape (n_channels × (2*n_samples - 1))
    :rtype: torch.Tensor
    """
    # Input validation 
    if signal_1.ndim != 2 or signal_2.ndim != 2:
        raise ValueError("Inputs must be 2D: (n_channels × n_samples).")
    
    if signal_1.shape != signal_2.shape:
        raise ValueError('signal_1 and signal_2 must have the same shape.')
    
    nch, npts = signal_1.shape
    device = signal_1.device

    # FFT size for full cross-correlation
    x_corr_len = 2 * npts - 1

    # nextpow2 returns tensor → convert to python int
    fast_length = int(nextpow2(torch.tensor(x_corr_len, device=device)).item())

    logger.info(f'Cross-correlation | FFT size = {fast_length}')

    # Forward FFTs
    fft_1 = torch.fft.rfft(signal_1, n=fast_length, dim=-1)
    fft_2 = torch.fft.rfft(signal_2, n=fast_length, dim=-1)

    # Optional spectral whitening
    if is_spectral_whitening:
        if whitening_params is None:
            raise ValueError('whitening_params must be provided when is_spectral_whitening=True')
        
        fs, window_freq, f1, f2 = whitening_params
        df = fs / fast_length

        logger.info('Applying spectral whitening before CC.')
        fft_1 = spectral_whitening(fft_1, df, window_freq, f1, f2)
        fft_2 = spectral_whitening(fft_2, df, window_freq, f1, f2)

    # Multiply with conjugate for CC spectrum 
    fft_prod = torch.conj(fft_1) * fft_2

    # Invert FFT → cross-correlation in time domain
    cc_full = torch.fft.irfft(fft_prod, n=fast_length, dim=-1)
 
    # Center the cross-correlation
    cc_full = torch.roll(cc_full, shifts=fast_length // 2, dims=-1)

    start = fast_length // 2 - (x_corr_len // 2)
    end = start + x_corr_len

    return cc_full[:, start:end]

class TorchCrossCorrelation(nn.Module):
    """
    PyTorch module wrapper for GPU-accelerated multi-channel cross-correlation.

    This class makes the `cross_correlation()` function behave like a PyTorch layer
    that integrates seamlessly with `.to(device)`, model containers, and autograd.

    :param is_spectral_whitening: Whether to apply spectral whitening.
    :type is_spectral_whitening: bool
    :param whitening_params: Whitening parameters (fs, window_freq, f1, f2).
                             Required if ``is_spectral_whitening=True``.
    :type whitening_params: tuple or None
    """
    def __init__(self, *, is_spectral_whitening=False, whitening_params=None):
        super().__init__()

        if is_spectral_whitening and whitening_params is None:
            raise ValueError(
                'whitening_params must be provided when is_spectral_whitening=True'
            )
            
        self.is_spectral_whitening = is_spectral_whitening
        self.whitening_params = whitening_params

        logger.info(
            f'TorchCrossCorrelation initialized | '
            f'whitening={is_spectral_whitening} | params={whitening_params}'
        )

    def forward(self, data1, data2):
        """
        Compute multi-channel cross-correlation using FFT.

        :param data1: First input signal of shape (n_channels × n_samples).
        :type data1: torch.Tensor
        :param data2: Second input signal, same shape as ``data1``.
        :type data2: torch.Tensor
        :return: Cross-correlation result with shape (n_channels × (2*n_samples - 1)).
        :rtype: torch.Tensor
        """
        logger.debug(
            f'Cross-correlation forward() | '
            f'data1.shape={tuple(data1.shape)} | '
            f'whitening={self.is_spectral_whitening}'
        )

        # Direct call to function 
        cc = cross_correlation(
            signal_1=data1, 
            signal_2=data2, 
            is_spectral_whitening=self.is_spectral_whitening, 
            whitening_params=self.whitening_params
        )

        return cc
    
def cross_correlation_full(data, ich1, ich2, 
                           is_spectral_whitening=False, 
                           whitening_params=None):
    """
    Compute multi-channel cross-correlation between a selected channel group 
    and the entire DAS array using FFT (GPU-accelerated if inputs are GPU tensors).

    :param data: Full DAS matrix of shape (n_channels × n_samples).
    :type data: torch.Tensor
    :param ich1: Starting channel index for selection.
    :type ich1: int
    :param ich2: Ending channel index (exclusive).
    :type ich2: int
    :param is_spectral_whitening: Whether to apply spectral whitening.
    :type is_spectral_whitening: bool
    :param whitening_params: Whitening parameters (fs, window_freq, f1, f2).
    :type whitening_params: tuple or None

    :return: Cross-correlation output with shape 
             (ich2 - ich1) × (2*n_samples - 1)
    :rtype: torch.Tensor
    """
    # Input validation
    if not torch.is_tensor(data):
        raise TypeError("Input 'data' must be a torch.Tensor.")
    
    if data.ndim != 2:
        raise ValueError("Input 'data' must be 2D: (n_channels × n_samples).")
    
    if ich1 < 0 or ich2 > data.shape[0] or ich1 >= ich2:
        raise ValueError("Invalid channel range ich1-ich2.")
    
    if is_spectral_whitening and whitening_params is None:
        raise ValueError('whitening_params must be provided when whitening=True.')
    
    device = data.device
    n_total, npts = data.shape
    n_sel = ich2 - ich1

    logger.info(
        f'CrossCorrelationFull | data={tuple(data.shape)} | '
        f'selected channels=({ich1}:{ich2}) → {n_sel} channels | '
        f'whitening={is_spectral_whitening}'
    )

    # Select subset of channels
    signal_sel = data[ich1:ich2, :]                                 # (Nsel × npts)
    signal_all = data                                               # (Ntotal × npts)

    # FFT size for full CC
    x_corr_len = 2 * npts - 1
    fast_length = int(nextpow2(torch.tensor(x_corr_len, device=device)).item())
    logger.info(
        f'FFT length: npts={npts} → xcorr_len={x_corr_len} → fast_length={fast_length}'
    )

    # Forward FFT
    fft_sel  = torch.fft.rfft(signal_sel,  n=fast_length, dim=-1)   # (Nsel × Nfreq)
    fft_all  = torch.fft.rfft(signal_all,  n=fast_length, dim=-1)   # (Ntotal × Nfreq)
    logger.debug(
        f'FFT shapes: fft_sel={fft_sel.shape}, fft_all={fft_all.shape}'
    )

    # Optional spectral whitening 
    if is_spectral_whitening:
        fs, window_freq, f1, f2 = whitening_params
        df = fs / fast_length

        logger.info(
            f'Applying spectral whitening | df={df:.6f} | '
            f'f1={f1}Hz f2={f2}Hz | window_freq={window_freq}Hz'
        )

        fft_sel = spectral_whitening(fft_sel, df, window_freq, f1, f2)
        fft_all = spectral_whitening(fft_all, df, window_freq, f1, f2)

    # Broadcasting for pairwise CC:
    # (Nsel, 1, Nfreq) × (1, Ntotal, Nfreq)
    fft_sel_exp  = fft_sel.unsqueeze(1)                             # (Nsel × 1 × Nfreq)
    fft_all_exp  = fft_all.unsqueeze(0)                             # (1 × Ntotal × Nfreq)

    # Multiply by conjugate for CC
    fft_prod = torch.conj(fft_sel_exp) * fft_all_exp                # (Nsel × Ntotal × Nfreq)

    logger.debug(f'fft_prod shape = {fft_prod.shape}')

    # Back to time domain
    cc_full = torch.fft.irfft(fft_prod, n=fast_length, dim=-1)

    # Center the zero lag
    cc_full = torch.roll(cc_full, shifts=fast_length // 2, dims=-1)

    # Slice only valid CC part
    start = fast_length // 2 - (x_corr_len // 2)
    end   = start + x_corr_len

    cc_out = cc_full[:, :, start:end]

    logger.info(
        f'Output CC shape = {tuple(cc_out.shape)} | '
        f'(Nsel={n_sel}, Ntotal={n_total}, CClen={x_corr_len})'
    )
    return cc_out