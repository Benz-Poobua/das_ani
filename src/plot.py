"""
:module: src/plot.py
:author: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Plotting utilities for DAS, NCFs, and dispersion images
"""
from __future__ import annotations

import glob
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation

from pathlib import Path
from tqdm.auto import tqdm
from typing import Tuple, List, Literal, Callable, Any, Union, Optional
from scipy import signal

from src.utils import parse_ncf_stack_filename, fk_filter, fk_transform
from src.ncf import get_vs_number, process_single_file, prep_ncf

# ===========================================================================
# Plotting Configuration
# ===========================================================================
params = {
    'savefig.dpi': 300,
    'axes.labelsize': 14,
    'axes.titlesize': 18,
    'font.size': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex': False,
    'figure.figsize': [12, 6],
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans']
}
rcParams.update(params)

# ===========================================================================
# 1. Plot DAS
# ===========================================================================
def plot_das_wavefield(
    data: np.ndarray,
    fs: float,
    dx: float,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
    start_chan: int | None = None,
    end_chan: int | None = None,
    pclip: float = 99.0,
    wave_clim: Optional[Tuple[float, float]] = None,
    cmap: str = "seismic",
    clabel: str = "Amplitude",
    figsize: Tuple[float, float] = (12, 6),
    title: str | None = None
) -> None:
    """
    Plots a static 2D space-time wavefield slice with selectable time and channel ranges.

    Processing Pipeline:
    1. Temporal & Spatial Slicing: Isolates the specific time window and spatial subset of the array.
    2. Color Clipping: Uses `wave_clim` for strict absolute amplitude limits. If `wave_clim` is None, 
       falls back to statistical symmetric clipping using `pclip` to dynamically accommodate the data range.

    :param data: 2D array of DAS data (Channels x Time Samples).
    :param fs: Sampling frequency in Hz.
    :param dx: Spatial channel spacing in meters.
    :param start_sec: Start time of the slice in seconds. Default is 0.0.
    :param duration_sec: Total time duration in seconds. If None, plots until the end of the data.
    :param start_chan: First channel index to plot. Default is 0.
    :param end_chan: Last channel index to plot. If None, plots to the end of the array.
    :param pclip: Percentile (0 to 100) used for dynamic symmetric color clipping if `wave_clim` is None.
                  Defaults to 99.0 (clips the top 1% of extreme absolute values).
    :param wave_clim: Optional tuple (vmin, vmax) to manually set absolute physical amplitude limits for the colorbar. 
                      If provided, overrides `pclip` to ensure true amplitude scaling across different datasets.
    :param cmap: Matplotlib colormap string. Default is "seismic".
    :param clabel: Label for the colorbar (e.g., 'Nano strain rate'). Default is "Amplitude".
    :param figsize: Tuple defining figure dimensions in inches. Default is (12, 6).
    :param title: Optional custom title for the plot. If None, auto-generates based on channels and duration.
    :returns: None
    """
    # 1. Handle Time Slicing dynamically
    start_sample = int(start_sec * fs)
    
    if duration_sec is None:
        end_sample = data.shape[1]
    else:
        end_sample = int((start_sec + duration_sec) * fs)
        end_sample = min(end_sample, data.shape[1])
        
    actual_duration_sec = (end_sample - start_sample) / fs

    # 2. Handle Channel Slicing
    s_ch = start_chan if start_chan is not None else 0
    e_ch = end_chan if end_chan is not None else data.shape[0]
    e_ch = min(e_ch, data.shape[0])

    # 3. Slice the data
    data_subset = data[s_ch:e_ch, start_sample:end_sample]

    # 4. Construct physical axes
    t_axis = np.arange(start_sample, end_sample) / fs
    # Distance axis reflects the actual position on the cable
    x_axis_km = np.arange(s_ch, e_ch) * dx / 1000.0  

    # 5. Calculate Color Limits (Manual Override vs. Statistical)
    if wave_clim is not None:
        vmin, vmax = wave_clim
    else:
        das_clip = float(np.percentile(np.abs(data_subset), pclip))
        vmin, vmax = -das_clip, das_clip

    # 6. Plotting
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    im = ax.imshow(
        data_subset, 
        aspect='auto',
        extent=[t_axis[0], t_axis[-1], x_axis_km[-1], x_axis_km[0]],
        vmin=vmin, vmax=vmax,
        cmap=cmap
    )

    fig.colorbar(im, ax=ax, label=clabel) 
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Distance along cable [km]')
    
    if title is None:
        chan_info = f"Chans {s_ch}–{e_ch}"
        title = f'DAS Wavefield ({actual_duration_sec:.0f}s Window, {chan_info})'
    ax.set_title(title)

    plt.show()

def animate_das_wavefield(
    das_data: dict,
    window_size_sec: float = 10.0,
    step_sec: float = 1.0,
    start_chan: int | None = None,
    end_chan: int | None = None,
    fps: int = 10,
    save_path: str | None = None,
    cmap: str = "seismic",
    clabel: str = "Amplitude",
    pclip: float = 99.0,
    figsize: Tuple[float, float] = (12, 6)
) -> FuncAnimation:
    """
    Animates a sliding time window across a spatially-selectable 2D DAS wavefield.

    This function is unit-agnostic and can be used to visualize strain, 
    strain rate, or optical phase shift data.

    Note: For large animations rendered in Jupyter, ensure you increase the
    embed limit in your notebook: `matplotlib.rcParams['animation.embed_limit'] = 100.0`

    :param das_data: Dictionary containing 'data' (2D array), 't_axis' (1D array), 
                     and 'x_axis' (1D array).
    :type das_data: dict
    :param window_size_sec: How many seconds of data to show in a single frame. Default is 10.0.
    :type window_size_sec: float, optional
    :param step_sec: How many seconds to move forward per frame. Default is 1.0.
    :type step_sec: float, optional
    :param start_chan: First channel index to include. Default is 0.
    :type start_chan: int | None, optional
    :param end_chan: Last channel index to include. Default is the total channel count.
    :type end_chan: int | None, optional
    :param fps: Frames per second for the playback. Default is 10.
    :type fps: int, optional
    :param save_path: If provided, saves the animation to disk (e.g., 'movie.mp4'). Default is None.
    :type save_path: str | None, optional
    :param cmap: Matplotlib colormap to use. Default is "seismic".
    :type cmap: str, optional
    :param clabel: Label for the colorbar (e.g., 'Strain', 'Strain Rate'). Default is "Amplitude".
    :type clabel: str, optional
    :param pclip: Percentile limit for global color scaling. Default is 99.0.
    :type pclip: float, optional
    :param figsize: Tuple defining the figure dimensions. Default is (12, 6).
    :type figsize: Tuple[float, float], optional
    :returns: The constructed animation object ready for rendering or display.
    :rtype: FuncAnimation
    :raises ValueError: If the dataset is too short for the requested window size.
    """
    # 1. Handle Spatial Slicing
    s_ch = start_chan if start_chan is not None else 0
    e_ch = end_chan if end_chan is not None else das_data['data'].shape[0]
    e_ch = min(e_ch, das_data['data'].shape[0])

    # Subset the data and x_axis once for efficiency
    spatial_subset = das_data['data'][s_ch:e_ch, :]
    x_subset = das_data['x_axis'][s_ch:e_ch]

    # Setup the figure with the new figsize argument
    fig, ax = plt.subplots(figsize=figsize)
    
    # 2. Calculate Time Indices
    dt = das_data['t_axis'][1] - das_data['t_axis'][0]
    fs = 1.0 / dt
    
    window_samples = int(window_size_sec * fs)
    step_samples = int(step_sec * fs)
    total_samples = len(das_data['t_axis'])
    num_frames = (total_samples - window_samples) // step_samples
    
    if num_frames <= 0:
        raise ValueError("Dataset is too short for the requested window size.")

    # 3. Compute Global Color Limits (based on the spatial subset)
    print(f"Computing global color limits for channels {s_ch}–{e_ch}...")
    das_clip = float(np.percentile(np.abs(spatial_subset), pclip))
    
    # 4. Initial Frame Setup
    start_idx = 0
    end_idx = window_samples
    
    im = ax.imshow(
        spatial_subset[:, start_idx:end_idx], 
        aspect='auto',
        extent=[das_data['t_axis'][start_idx], das_data['t_axis'][end_idx-1],
                x_subset[-1], x_subset[0]],
        vmin=-das_clip, vmax=das_clip,
        cmap=cmap
    )
                   
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Distance along cable [km]')
    title = ax.set_title(
        f"DAS Wavefield - Time: {das_data['t_axis'][start_idx]:.1f}s "
        f"to {das_data['t_axis'][end_idx-1]:.1f}s (Ch {s_ch}–{e_ch})"
    )
    fig.colorbar(im, ax=ax, label=clabel)
    fig.tight_layout()

    # 5. Update Function
    def update(frame):
        start = frame * step_samples
        end = start + window_samples
        
        im.set_data(spatial_subset[:, start:end])
        im.set_extent([
            das_data['t_axis'][start], das_data['t_axis'][end-1],
            x_subset[-1], x_subset[0]
        ])
                       
        title.set_text(
            f"DAS Wavefield - Time: {das_data['t_axis'][start]:.1f}s "
            f"to {das_data['t_axis'][end-1]:.1f}s (Ch {s_ch}–{e_ch})"
        )
        return im, title

    # 6. Build Animation
    print(f"Generating animation with {num_frames} frames...")
    
    def frame_generator():
        yield from tqdm(range(num_frames), desc="Rendering Subset Video")

    anim = FuncAnimation(
        fig, update, frames=frame_generator, 
        save_count=num_frames, interval=1000//fps, blit=False
    )
    
    if save_path:
        print(f"Saving to {save_path}...")
        anim.save(save_path, writer='ffmpeg', fps=fps)
        print("Save complete.")
        
    plt.close(fig) 
    return anim

def plot_das_psd(
    data: np.ndarray, 
    fs: float, 
    dx: float, 
    start_chan: int = 0, 
    end_chan: int | None = None, 
    nperseg: int = 4096,
    flim: Optional[Tuple[float | None, float | None]] = (0.01, None),
    psd_ylim: Optional[Tuple[float, float]] = (0.0, 60.0),
    xscale: str = "log",
    ylabel: str = "PSD (dB)",
    figsize: Tuple[float, float] = (10, 5),
    title: str | None = None
) -> None:
    """
    Plots the Mean Power Spectral Density (1D) for a specific spatial range.
    
    Processing Pipeline:
    1. Channel Slicing: Isolates the specific spatial subset of the array for analysis.
    2. Welch's Method: Computes the Power Spectral Density across the time axis for every channel, 
       then averages them spatially into a single 1D mean PSD curve.
    3. Axis Alignment: Applies `flim` to the X-axis (Frequency) and `psd_ylim` to the Y-axis (Power) 
       to ensure visual and absolute amplitude consistency with external 2D spectrograms.

    :param data: 2D array (Channels x Samples) of raw or continuous DAS data.
    :param fs: Sampling frequency in Hz.
    :param dx: Spatial channel spacing in meters.
    :param start_chan: First channel index to include. Default is 0.
    :param end_chan: Last channel index to include. If None, uses all remaining channels.
    :param nperseg: Segment length for Welch's method. Default is 4096.
    :param flim: Frequency limits (min_Hz, max_Hz) for the X-axis. If None, auto-scales. 
                 If max_Hz is None, defaults to the Nyquist frequency (fs/2). Default is (0.01, None).
    :param psd_ylim: Physical dB limits (min_dB, max_dB) for the Y-axis. If None, auto-scales. 
                     Locking this matches the `psd_clim` of 2D plots. Default is (0.0, 60.0).
    :param xscale: Scale of the X-axis ('linear' or 'log'). Default is "log".
    :param ylabel: Label for the Y-axis. Default is "PSD (dB)".
    :param figsize: Tuple defining figure dimensions in inches. Default is (10, 5).
    :param title: Optional custom title for the plot. If None, auto-generates based on channels.
    """
    # 1. Handle Channel Slicing
    s_ch = max(0, start_chan)
    if end_chan is None:
        e_ch = data.shape[0]
    else:
        e_ch = min(end_chan, data.shape[0])

    data_section = data[s_ch:e_ch, :]

    # 2. Calculate physical distance for the title
    start_km = (s_ch * dx) / 1000.0
    end_km = ((e_ch - 1) * dx) / 1000.0

    # 3. Compute PSD using Welch's method
    freqs, psd_values = signal.welch(data_section, fs=fs, nperseg=nperseg, axis=1)
    mean_psd = np.mean(psd_values, axis=0)

    # 4. Convert to dB (avoiding log of zero)
    psd_db = 10 * np.log10(mean_psd + 1e-12)

    # 5. Plotting
    plt.figure(figsize=figsize, layout="constrained")
    plt.plot(freqs, psd_db, color='black', linewidth=1)
    
    # Apply chosen scale
    plt.xscale(xscale)

    # 6. Handle Dynamic Frequency Limits (flim)
    if flim is not None:
        f_low = flim[0] if flim[0] is not None else freqs[1]  # Avoid 0 if log scale
        f_high = flim[1] if flim[1] is not None else fs / 2.0
        plt.xlim(f_low, f_high)
        
    # 7. Handle Dynamic Y-Axis Limits (psd_ylim)
    if psd_ylim is not None:
        plt.ylim(psd_ylim)

    # 8. Formatting
    if title is not None:
        plt.title(title)
    else:
        title_str = f'Mean PSD (Chans {s_ch}-{e_ch} | {start_km:.2f}-{end_km:.2f} km)'
        plt.title(title_str)
        
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(ylabel)
    plt.grid(True, which="both", color='gray', linestyle='--', alpha=0.3)

    plt.show()

def plot_das_psd_2d(
    data: np.ndarray, 
    fs: float, 
    dx: float, 
    start_chan: int = 0, 
    end_chan: int | None = None, 
    pclip: float = 95.0,
    psd_clim: Optional[Tuple[float, float]] = None,
    nperseg: int = 4096,
    flim: Tuple[float, float] | None = (0.0, 25.0),
    cmap: str = "jet",
    figsize: Tuple[float, float] = (12, 6),
    title: str | None = None
) -> None:
    """
    Plots the 2D Space-Frequency Power Spectral Density (PSD) spectrogram 
    to analyze localized spatial variations in noise across the DAS array.
    
    Processing Pipeline:
    1. Channel Slicing: Isolates the specific spatial subset of the array for analysis.
    2. Welch's Method: Computes the Power Spectral Density across the time axis for every channel.
    3. Frequency Cropping: Masks the output to only include the frequency band specified by `flim`.
    4. Color Clipping: Uses `psd_clim` for strict physical dB limits. If `psd_clim` is None, 
       falls back to statistical global clipping using `pclip` to dynamically drop extreme outliers.

    :param data: 2D array (Channels x Samples) of raw or continuous DAS data.
    :param fs: Sampling frequency in Hz.
    :param dx: Spatial channel spacing in meters.
    :param start_chan: First channel index to include. Default is 0.
    :param end_chan: Last channel index to include. If None, uses all remaining channels.
    :param pclip: Percentile (0 to 100) used for dynamic color clipping if `psd_clim` is None. 
                  Defaults to 95.0 (clips the top and bottom 5% of extreme values).
    :param psd_clim: Optional tuple (vmin, vmax) to manually set absolute physical dB limits for the colorbar. 
                     If provided, overrides `pclip` to ensure true amplitude scaling across different datasets.
    :param nperseg: Segment length for Welch's method. Higher values yield finer 
                    frequency resolution but higher variance. Default is 4096.
    :param flim: Frequency limits (min_Hz, max_Hz) for the Y-axis. The data is cropped to this band 
                 *before* color limits are calculated to prevent out-of-band noise from skewing the colormap.
    :param cmap: Matplotlib colormap string. Default is "jet".
    :param figsize: Tuple defining the figure dimensions in inches. Default is (12, 6).
    :param title: Optional custom title for the plot. If None, auto-generates based on channels.
    """
    # 1. Handle Channel Slicing
    s_ch = max(0, start_chan)
    if end_chan is None:
        e_ch = data.shape[0]
    else:
        e_ch = min(end_chan, data.shape[0])

    data_section = data[s_ch:e_ch, :]

    # 2. Create Spatial Axis (converted to kilometers for clean plotting)
    x_axis_km = np.arange(s_ch, e_ch) * dx / 1000.0

    # 3. Compute PSD using Welch's method
    freqs, psd_values = signal.welch(data_section, fs=fs, nperseg=nperseg, axis=1)

    # 4. Convert to Decibels (dB), adding a tiny epsilon to avoid log(0)
    power_db = 10 * np.log10(psd_values + 1e-12)

    # 5. Apply Frequency Mask (Y-axis bounds)
    if flim is not None:
        fmin, fmax = flim
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        freqs_plot = freqs[freq_mask]
        power_plot = power_db[:, freq_mask]
    else:
        freqs_plot = freqs
        power_plot = power_db
        fmin, fmax = freqs[0], freqs[-1]

    # 6. Calculate Color Limits (Manual Override vs. Statistical)
    if psd_clim is not None:
        vmin, vmax = psd_clim
    else:
        vmin = np.percentile(power_plot, 100.0 - pclip)
        vmax = np.percentile(power_plot, pclip)

    # 7. Create Meshgrid and Plot
    X, Y = np.meshgrid(x_axis_km, freqs_plot)

    plt.figure(figsize=figsize, layout="constrained")
    
    # Transpose power_plot (.T) so axes align: (len(freqs_plot), len(channels))
    # Uses gouraud shading to prevent blocky artifacts in the spectrogram output
    mesh = plt.pcolormesh(X, Y, power_plot.T, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)

    # 8. Formatting
    plt.colorbar(mesh, label='Power Spectral Density (dB/Hz)')
    plt.ylim(fmin, fmax)
    plt.xlim(x_axis_km[0], x_axis_km[-1])
    
    plt.xlabel('Distance along cable (km)')
    plt.ylabel('Frequency (Hz)')
    
    start_km = (s_ch * dx) / 1000.0
    end_km = ((e_ch - 1) * dx) / 1000.0
    
    if title is not None:
        plt.title(title, pad=15)
    else:
        title_str = f'DAS Space-Frequency PSD (Chans {s_ch}-{e_ch} | {start_km:.2f}-{end_km:.2f} km)'
        plt.title(title_str, pad=15)

    plt.grid(True, which='both', color='white', linestyle='--', alpha=0.3)
    plt.show()

def plot_das_spectrogram(
    file_paths: List[Union[str, Path]],
    fs: float,
    dx: float,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
    start_chan: int = 0,
    end_chan: int | None = None,
    pclip: float = 98.0,
    psd_clim: Optional[Tuple[float, float]] = None,
    nperseg: int = 4096,
    flim: Tuple[float, float] = (0.1, 15.0),
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (14, 6),
    interval: int = 1000
) -> animation.FuncAnimation:
    """
    Creates an animated Time-Frequency Spectrogram across a sequence of DAS files.
    
    Processing Pipeline:
    1. Spatial Stacking: Averages the specified channel range (start_chan to end_chan) 
       into a single 1D time-series trace to significantly boost the Signal-to-Noise Ratio (SNR).
    2. STFT: Computes the Short-Time Fourier Transform using SciPy's spectrogram.
    3. Frequency Cropping: Masks the output to only include the frequency band specified by `flim`.
    4. Color Clipping: Uses `psd_clim` for strict physical dB limits. If `psd_clim` is None, 
       falls back to statistical global clipping (`pclip`) based on the first file in the sequence. 
       This prevents "flashing" during animation and ensures colors represent absolute energy consistently.

    :param file_paths: List of file paths to the raw DAS data arrays (.npy or .npz).
    :param fs: Sampling frequency of the DAS instrument in Hz.
    :param dx: Spatial channel spacing in meters.
    :param start_sec: Temporal offset from the beginning of each file to start processing (seconds).
    :param duration_sec: Length of the time window to process per file (seconds). If None, processes the whole file.
    :param start_chan: Minimum channel index to include in the spatial average. Default is 0.
    :param end_chan: Maximum channel index to include in the spatial average. If None, processes to the end of the cable.
    :param pclip: Percentile (0 to 100) used for dynamic color clipping if `psd_clim` is None.
    :param psd_clim: Optional tuple (vmin, vmax) to manually set absolute physical dB limits for the colorbar. 
                     If provided, overrides `pclip` to ensure true amplitude scaling across different datasets.
    :param nperseg: Number of points per segment for the STFT. Controls the Time-Frequency Uncertainty trade-off. 
                    Larger values (e.g., 4096) give high frequency resolution but low temporal resolution.
    :param flim: Frequency limits (min_Hz, max_Hz) for the Y-axis. The data is cropped to this band 
                 *before* color limits are calculated to prevent out-of-band noise from skewing the colormap.
    :param cmap: Matplotlib colormap string. Default is "viridis".
    :param figsize: Tuple defining the figure dimensions in inches.
    :param interval: Delay between animated frames in milliseconds. Default is 1000 (1 second).
    
    :returns: A Matplotlib FuncAnimation object ready to be rendered (e.g., via HTML(ani.to_jshtml())).
    """
    if not file_paths:
        raise ValueError("file_paths list cannot be empty.")

    # 1. Helper to load and crop a single file
    def load_and_stack(filepath: Union[str, Path]) -> np.ndarray:
        data = np.load(filepath)
        if isinstance(data, np.lib.npyio.NpzFile):
            keys = data.files
            arr = data['data'] if 'data' in keys else data[keys[0]]
        else:
            arr = data
            
        e_ch = arr.shape[0] if end_chan is None else min(end_chan, arr.shape[0])
        s_ch = max(0, start_chan)
        
        s_samp = int(start_sec * fs)
        e_samp = arr.shape[1] if duration_sec is None else int((start_sec + duration_sec) * fs)
        e_samp = min(e_samp, arr.shape[1])
        
        trace = np.mean(arr[s_ch:e_ch, s_samp:e_samp], axis=0)
        return trace, s_ch, e_ch

    # 2. Setup initial frame to configure the plot axes
    first_trace, s_ch, e_ch = load_and_stack(file_paths[0])
    
    freqs, times, Sxx = signal.spectrogram(first_trace, fs=fs, nperseg=nperseg)
    power_db = 10 * np.log10(Sxx + 1e-12)
    
    fmin, fmax = flim
    f_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs_plot = freqs[f_mask]
    power_plot = power_db[f_mask, :]
    
    # Calculate Color Limits (Manual Override vs. Statistical)
    if psd_clim is not None:
        vmin, vmax = psd_clim
    else:
        vmin = np.percentile(power_plot, 100.0 - pclip)
        vmax = np.percentile(power_plot, pclip)

    # 3. Initialize Figure
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    
    X, Y = np.meshgrid(times, freqs_plot)
    mesh = ax.pcolormesh(X, Y, power_plot, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
    fig.colorbar(mesh, ax=ax, label='Power (dB/Hz)')
    
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time within file (s)')
    ax.set_ylim(fmin, fmax)
    
    start_km = (s_ch * dx) / 1000.0
    end_km = ((e_ch - 1) * dx) / 1000.0
    chan_str = f"Chans {s_ch}-{e_ch} ({start_km:.2f}-{end_km:.2f} km)"

    # 4. Animation Update Function
    def update(frame_idx: int):
        filepath = file_paths[frame_idx]
        trace, _, _ = load_and_stack(filepath)
        
        _, _, Sxx_new = signal.spectrogram(trace, fs=fs, nperseg=nperseg)
        power_db_new = 10 * np.log10(Sxx_new + 1e-12)
        power_plot_new = power_db_new[f_mask, :]
        
        mesh.set_array(power_plot_new.ravel())
        
        filename = Path(filepath).name
        ax.set_title(f"Spectrogram | {chan_str}\nFile [{frame_idx+1}/{len(file_paths)}]: {filename}")
        
        return mesh,

    # 5. Create and return the animation
    ani = FuncAnimation(
        fig, 
        update, 
        frames=tqdm(range(len(file_paths)), desc="Rendering Animation", unit="frame"), 
        interval=interval, 
        blit=False 
    )
    
    plt.close(fig) 
    return ani

def animate_das_dashboard(
    file_paths: List[Union[str, Path]],
    fs: float,
    dx: float,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
    start_chan: int = 0,
    end_chan: int | None = None,
    pclip: float = 98.0,
    wave_clim: Optional[Tuple[float, float]] = None,
    psd_clim: Optional[Tuple[float, float]] = None,
    nperseg: int = 4096,
    flim: Tuple[float, float] = (0.1, 15.0),
    psd_ylim: Tuple[float, float] | None = (0, 70),
    clabel: str = "Amplitude",  
    figsize: Tuple[float, float] = (16, 12),
    interval: int = 1000
) -> animation.FuncAnimation:
    """
    Generates a synchronized 2x2 dashboard animation of DAS data using global color clipping.
    
    Panels:
    1. (Top-Left) DAS Wavefield: Distance vs. Time.
    2. (Top-Right) Mean PSD: 1D Frequency plot.
    3. (Bottom-Left) Space-Frequency PSD: Distance vs. Frequency.
    4. (Bottom-Right) Time-Frequency Spectrogram: Time vs. Frequency (spatially stacked).

    Processing Pipeline:
    1. Temporal & Spatial Slicing: Isolates the specific time window and spatial subset of the array for all panels.
    2. Signal Processing: Applies linear detrending, Welch's method (Mean and Space-Freq PSD), and STFT (Spectrogram).
    3. Frequency Cropping: Masks the PSD and Spectrogram outputs to only include the frequency band specified by `flim`.
    4. Color Clipping: Uses `wave_clim` and `psd_clim` for strict absolute amplitude/dB limits. If None, falls back to 
       statistical global clipping using `pclip` based on the first file to ensure consistent animation frames without flashing.

    :param file_paths: List of file paths to the raw DAS data arrays (.npy or .npz).
    :param fs: Sampling frequency of the DAS instrument in Hz.
    :param dx: Spatial channel spacing in meters.
    :param start_sec: Temporal offset from the beginning of each file to start processing (seconds).
    :param duration_sec: Length of the time window to process per file (seconds). If None, processes the whole file.
    :param start_chan: Minimum channel index to include in the analysis. Default is 0.
    :param end_chan: Maximum channel index to include in the analysis. If None, processes to the end of the cable.
    :param pclip: Percentile (0 to 100) used for dynamic color clipping if manual limits are not provided.
    :param wave_clim: Optional tuple (vmin, vmax) to manually set absolute limits for the Wavefield colorbar.
    :param psd_clim: Optional tuple (vmin, vmax) to manually set absolute physical dB limits for the PSD/Spectrogram colorbars.
    :param nperseg: Number of points per segment for the STFT and Welch's method. Default is 4096.
    :param flim: Frequency limits (min_Hz, max_Hz) for the Y-axis across all spectral plots.
    :param psd_ylim: Fixed Y-axis limits (dB) for the 1D Mean PSD plot. Set to None for auto-scaling.
    :param clabel: Label for the wavefield colorbar (e.g., 'Nano strain rate'). Default is "Amplitude".
    :param figsize: Tuple defining the figure dimensions in inches. Default is (16, 12).
    :param interval: Delay between animated frames in milliseconds. Default is 1000 (1 second).
    
    :returns: A Matplotlib FuncAnimation object ready to be rendered.
    """
    if not file_paths:
        raise ValueError("file_paths cannot be empty.")

    # Robust file loader with memory leak prevention
    def _get_array(filepath: Union[str, Path]) -> np.ndarray:
        data = np.load(filepath)
        if isinstance(data, np.lib.npyio.NpzFile):
            keys = data.files
            arr = data['data'] if 'data' in keys else data[keys[0]]
            data.close()  # <-- PATCH 1: Closes file handle to prevent OS crash on large loops
            return arr
        return data

    # ==========================================
    # 1. Global Setup & Pre-Flight Scan
    # ==========================================
    first_path = file_paths[0]
    raw_array = _get_array(first_path)
    data_detrended = signal.detrend(raw_array, type='linear', axis=1)
    
    start_sample = int(start_sec * fs)
    end_sample = int((start_sec + duration_sec) * fs) if duration_sec else data_detrended.shape[1]
    end_sample = min(end_sample, data_detrended.shape[1])  # <-- PATCH 2: Prevent time-axis out-of-bounds
    
    s_ch = max(0, start_chan)
    e_ch = end_chan if end_chan is not None else data_detrended.shape[0]
    e_ch = min(e_ch, data_detrended.shape[0])
    
    first_subset = data_detrended[s_ch:e_ch, start_sample:end_sample]
    
    t_axis = np.arange(start_sample, end_sample) / fs
    x_axis_km = np.arange(s_ch, e_ch) * dx / 1000.0

    # --- Wavefield Limits ---
    if wave_clim is not None:
        vmin_wave, vmax_wave = wave_clim
    else:
        vclip_wave = np.percentile(np.abs(first_subset), pclip)
        vmin_wave, vmax_wave = -vclip_wave, vclip_wave

    # --- PSD/Spectrogram Limits ---
    freqs, psd_values = signal.welch(first_subset, fs=fs, nperseg=nperseg, axis=1)
    power_db = 10 * np.log10(psd_values + 1e-12)
    f_mask = (freqs >= flim[0]) & (freqs <= flim[1])
    v_2d = power_db[:, f_mask]
    
    if psd_clim is not None:
        vmin_psd, vmax_psd = psd_clim
    else:
        vmin_psd = np.percentile(v_2d, 100.0 - pclip)
        vmax_psd = np.percentile(v_2d, pclip)

    # ==========================================
    # 2. Initialize the 2x2 Figure
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=figsize, layout="constrained")
    (ax1, ax2), (ax3, ax4) = axes

    # Panel 1: Wavefield
    im1 = ax1.imshow(
        first_subset, aspect='auto', cmap='seismic', vmin=vmin_wave, vmax=vmax_wave,
        extent=[t_axis[0], t_axis[-1], x_axis_km[-1], x_axis_km[0]]
    )
    fig.colorbar(im1, ax=ax1, label=clabel)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Distance [km]")

    # Panel 2: Mean PSD
    mean_psd_db = 10 * np.log10(np.mean(psd_values, axis=0) + 1e-12)
    line2, = ax2.plot(freqs, mean_psd_db, color='black', lw=1.5)
    ax2.set(xscale='log', xlim=flim)
    if psd_ylim: ax2.set_ylim(psd_ylim)
    ax2.set_title("Mean PSD")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("PSD (dB)")
    ax2.grid(True, which='both', alpha=0.3)

    # Panel 3: Space-Frequency PSD
    X3, Y3 = np.meshgrid(x_axis_km, freqs[f_mask])
    im3 = ax3.pcolormesh(X3, Y3, v_2d.T, cmap='jet', shading='gouraud', vmin=vmin_psd, vmax=vmax_psd)
    fig.colorbar(im3, ax=ax3, label='PSD (dB/Hz)')
    ax3.set_title("Space-Frequency PSD")
    ax3.set_xlabel("Distance [km]")
    ax3.set_ylabel("Frequency [Hz]")
    ax3.set_ylim(flim)

    # Panel 4: Time-Frequency Spectrogram (Spatially Stacked)
    _, times_stft, Sxx = signal.spectrogram(np.mean(first_subset, axis=0), fs=fs, nperseg=nperseg)
    power_stft_db = 10 * np.log10(Sxx + 1e-12)
    X4, Y4 = np.meshgrid(times_stft, freqs[f_mask])
    im4 = ax4.pcolormesh(X4, Y4, power_stft_db[f_mask, :], cmap='viridis', shading='gouraud', vmin=vmin_psd, vmax=vmax_psd)
    fig.colorbar(im4, ax=ax4, label='PSD (dB/Hz)')
    ax4.set_title("Time-Frequency Spectrogram (Stacked)")
    ax4.set_xlabel("Time within file [s]")
    ax4.set_ylabel("Frequency [Hz]")
    ax4.set_ylim(flim)

    # ==========================================
    # 3. Fast Update Function
    # ==========================================
    def update(frame_idx: int):
        file_path = file_paths[frame_idx]
        
        # Load and slice
        raw_arr = _get_array(file_path)
        d_detrend = signal.detrend(raw_arr, type='linear', axis=1)
        
        # Apply the safe clamps here as well
        s_idx = int(start_sec * fs)
        e_idx = int((start_sec + duration_sec) * fs) if duration_sec else d_detrend.shape[1]
        e_idx = min(e_idx, d_detrend.shape[1])
        
        subset = d_detrend[s_ch:e_ch, s_idx:e_idx]
        
        # Update 1: Wavefield
        im1.set_array(subset)
        ax1.set_title(f"Wavefield: {Path(file_path).name}")

        # Update 2: Mean PSD
        _, psd_vals = signal.welch(subset, fs=fs, nperseg=nperseg, axis=1)
        mean_db = 10 * np.log10(np.mean(psd_vals, axis=0) + 1e-12)
        line2.set_ydata(mean_db)

        # Update 3: Space-Freq
        pow_db = 10 * np.log10(psd_vals + 1e-12)
        im3.set_array(pow_db[:, f_mask].T.ravel())

        # Update 4: Time-Freq Spectrogram
        _, _, Sxx_new = signal.spectrogram(np.mean(subset, axis=0), fs=fs, nperseg=nperseg)
        pow_stft_db = 10 * np.log10(Sxx_new + 1e-12)
        im4.set_array(pow_stft_db[f_mask, :].ravel())

        return im1, line2, im3, im4

    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=tqdm(range(len(file_paths)), desc="Rendering Dashboard", unit="frame"), 
        interval=interval, 
        blit=False
    )
    
    plt.close(fig)
    return ani

# ===========================================================================
# 2. Plot NCF
# ===========================================================================
def animate_ncf_section_mesh(
    pattern: str,
    *,
    lag_axis: np.ndarray,
    distance_axis: np.ndarray,
    mode: str = "causal",
    unit: str = "m", 
    clip: float | None = 0.05,
    pclip: float | None = None,
    cmap: str = "seismic",
    dx: float = 8.16,
    range_m: float = 500.0,
    clip_lim: bool = True,
    interval_ms: int = 200,
    repeat_delay_ms: int = 1000,
    sort_by_vs: bool = True,
    vs_start: int | None = None,
    vs_end: int | None = None,
    max_lag: float | None = None,
    figsize: Tuple[float, float] | None = None,
) -> FuncAnimation:
    """
    Animates Noise Cross-Correlation Function (NCF) sections dynamically over a 
    sequence of virtual sources using Matplotlib's pcolormesh. 
    
    This function loads NCF matrices from disk, applies temporal windowing (causal, 
    acausal, or all lags), applies robust amplitude clipping, and optionally tracks 
    the active virtual source with a rolling spatial window. 

    :param pattern: Glob pattern matching the precomputed NCF files (e.g., "*.npy" or "*.npz").
    :param lag_axis: 1D array of time lags (in seconds) corresponding to the data matrix.
    :param distance_axis: 1D array of spatial distances along the array.
    :param mode: Temporal windowing mode. Options: "causal" (lag >= 0), "acausal" 
                 (lag <= 0 mapped to positive absolute time), or "all". Default is "causal".
    :param unit: Spatial distance unit for the x-axis ("m" or "km"). Default is "m".
    :param clip: Absolute amplitude limit for colorbar scaling. Used only if `pclip` is None.
    :param pclip: Percentile for dynamic amplitude clipping (e.g., 99.0). If provided, computes 
                  a global median percentile across all frames for stable animation scaling.
    :param cmap: Matplotlib colormap to use. Default is "seismic".
    :param dx: Spatial distance between adjacent channels (in meters). Default is 8.16.
    :param range_m: Spatial window size (in meters) to display around the active virtual source. 
                    Only applied if `clip_lim=True`.
    :param clip_lim: If True, dynamically updates the x-axis limits to track the active virtual 
                     source as it moves down the array.
    :param interval_ms: Delay between animation frames in milliseconds. Default is 200.
    :param repeat_delay_ms: Delay in milliseconds before the animation loops.
    :param sort_by_vs: If True, sorts the loaded files numerically by their virtual source index.
    :param vs_start: Optional minimum virtual source index to include in the animation.
    :param vs_end: Optional maximum virtual source index to include in the animation.
    :param max_lag: Optional maximum absolute lag time (seconds) to display on the y-axis.
    :param figsize: Optional tuple defining figure dimensions (width, height) in inches. If None, 
                   defaults to (6, 6) when tracking the virtual source, or (10, 6) for the full array.
    :returns: The constructed Matplotlib FuncAnimation object ready for rendering or display.
    """
    mode, unit = mode.lower().strip(), unit.lower().strip()
    dist_scale = 1000.0 if unit == "km" else 1.0
    lag_axis = np.asarray(lag_axis)
    plot_distance, plot_range = np.asarray(distance_axis) / dist_scale, range_m / dist_scale

    paths = glob.glob(pattern)
    parsed = []
    for p in paths:
        date, vs, window, xmode = parse_ncf_stack_filename(p) 
        vs_idx = int(vs)
        if (vs_start is not None and vs_idx < vs_start) or (vs_end is not None and vs_idx > vs_end):
            continue
        parsed.append((p, date, vs, window, xmode))

    if sort_by_vs: parsed.sort(key=lambda x: int(x[2]))

    title_prefix = f"Window: {parsed[0][3]}"

    if mode == "all":
        y, sel = lag_axis, slice(None)
    elif mode == "causal":
        sel = lag_axis >= 0
        y = lag_axis[sel]
    else: 
        sel = lag_axis <= 0
        y_raw = np.abs(lag_axis[sel])
        order = np.argsort(y_raw)
        y = y_raw[order]

    if pclip is not None:
        per_file_clips = []
        for path_info in tqdm(parsed, desc="Scanning for global pclip"):
            temp_ncf = np.load(path_info[0])
            if temp_ncf.shape == (lag_axis.size, distance_axis.size):
                temp_ncf = temp_ncf.T
            temp_data = temp_ncf if mode == "all" else temp_ncf[:, sel]
            per_file_clips.append(np.percentile(np.abs(temp_data), pclip))
        c0 = float(np.median(per_file_clips)) 
    else:
        c0 = float(clip if clip is not None else 1.0)

    # ---------------------------------------------------------
    # Apply dynamic figsize fallback
    # ---------------------------------------------------------
    if figsize is None:
        figsize = (6, 6) if clip_lim else (10, 6)
        
    fig, ax = plt.subplots(figsize=figsize)
    # ---------------------------------------------------------
    
    ax.invert_yaxis()
    ax.set_xlabel(f"Distance along array ({unit})")
    ax.set_ylabel("|Lag time| (s)" if mode == "acausal" else "Lag time (s)")

    if max_lag is not None:
        ax.set_ylim(max_lag, -max_lag) if mode == "all" else ax.set_ylim(max_lag, 0)

    ncf0 = np.load(parsed[0][0])
    if ncf0.shape == (lag_axis.size, distance_axis.size): ncf0 = ncf0.T
    data0 = ncf0 if mode == "all" else ncf0[:, sel]
    if mode == "acausal": data0 = data0[:, order]

    mesh = ax.pcolormesh(plot_distance, y, data0.T, shading="gouraud", cmap=cmap, vmin=-c0, vmax=c0)
    pos0_plot = (int(parsed[0][2]) * dx) / dist_scale
    vline = ax.axvline(x=pos0_plot, color="black", linestyle="--", linewidth=1.2, alpha=0.6)
    
    fig.colorbar(mesh, ax=ax, label="Correlation amplitude")
    
    dec = 2 if unit == "km" else 1
    ax.set_title(f"{os.path.basename(parsed[0][0])}\n{title_prefix} | VS={parsed[0][2]} ({pos0_plot:.{dec}f} {unit})", pad=15)
    fig.tight_layout()

    def update(frame_idx):
        path, _, vs, _, _ = parsed[frame_idx]
        ncf = np.load(path)
        if ncf.shape == (lag_axis.size, distance_axis.size): ncf = ncf.T
        data = ncf if mode == "all" else ncf[:, sel]
        if mode == "acausal": data = data[:, order]

        mesh.set_array(data.T.ravel())
        pos_plot = (int(vs) * dx) / dist_scale
        vline.set_xdata([pos_plot, pos_plot])

        if clip_lim:
            ax.set_xlim(max(plot_distance.min(), pos_plot - plot_range), min(plot_distance.max(), pos_plot + plot_range))

        ax.set_title(f"{os.path.basename(path)}\n{title_prefix} | VS={vs} ({pos_plot:.{dec}f} {unit})", pad=15)
        return mesh, vline

    def init(): return mesh, vline
    def frame_generator(): yield from tqdm(range(len(parsed)), desc="Rendering Video")

    ani = FuncAnimation(fig, update, init_func=init, frames=frame_generator, save_count=len(parsed), interval=interval_ms, blit=False, repeat=True)
    plt.close(fig) 
    return ani

def animate_directional_ncf_section_mesh(
    pattern: str,
    *,
    lag_axis: np.ndarray,
    distance_axis: np.ndarray,
    target: Literal["causal", "acausal", "s1", "s2"] = "s1",
    unit: str = "m",
    clip: float | None = 0.05,
    pclip: float | None = None,
    cmap: str = "seismic",
    dx: float = 8.16,
    range_m: float = 500.0,
    clip_lim: bool = True,
    view_side: Literal["both", "left", "right"] = "both",  
    pos_offset: float = 0.0,                               
    interval_ms: int = 200,
    sort_by_vs: bool = True,
    vs_start: int | None = None,
    vs_end: int | None = None,
    max_lag: float | None = None,
) -> FuncAnimation:
    """
    Animates directionally folded Noise Cross-Correlation Function (NCF) sections dynamically 
    over a sequence of virtual sources using Matplotlib.

    Leverages `src.disp.prep_ncf` to extract specific wavefield components (causal, acausal, 
    or directionally folded S1/S2 modes). Features dynamic spatial windowing to track the active 
    virtual source, with options to isolate specific array sides and apply positional offsets to 
    mitigate near-field noise.

    :param pattern: Glob pattern matching the precomputed NCF files (e.g., "*.npy" or "*.npz").
    :param lag_axis: 1D array of time lags (in seconds) corresponding to the raw data matrix.
    :param distance_axis: 1D array of spatial distances along the array.
    :param target: The specific wavefield component to animate. Options: "causal", "acausal", 
                   "s1" (e.g., forward-propagating), or "s2" (e.g., backward-propagating). Default is "s1".
    :param unit: Spatial distance unit for the x-axis ("m" or "km"). Default is "m".
    :param clip: Absolute amplitude limit for colorbar scaling. Used only if `pclip` is None.
    :param pclip: Percentile for dynamic amplitude clipping (e.g., 99.0). Computes a global median 
                  percentile across all frames for stable animation scaling.
    :param cmap: Matplotlib colormap to use. Default is "seismic".
    :param dx: Spatial distance between adjacent channels (in meters). Default is 8.16.
    :param range_m: Spatial window size (in meters) to display around the active virtual source. 
                    Only applied if `clip_lim=True`.
    :param clip_lim: If True, dynamically updates the x-axis limits to track the active virtual 
                     source as it moves down the array.
    :param view_side: Determines which side of the virtual source to display when tracking 
                      ("both", "left", or "right").
    :param pos_offset: Spatial exclusion offset (in meters) from the virtual source to clip out 
                       near-source noise or autocorrelation artifacts.
    :param interval_ms: Delay between animation frames in milliseconds. Default is 200.
    :param sort_by_vs: If True, sorts the loaded files numerically by their virtual source index.
    :param vs_start: Optional minimum virtual source index to include in the animation.
    :param vs_end: Optional maximum virtual source index to include in the animation.
    :param max_lag: Optional maximum absolute lag time (seconds) to display on the y-axis.
    :returns: The constructed Matplotlib FuncAnimation object ready for rendering or display.
    """
    target, unit = target.lower().strip(), unit.lower().strip()
    dist_scale = 1000.0 if unit == "km" else 1.0
    plot_distance, plot_range, plot_offset = np.asarray(distance_axis)/dist_scale, range_m/dist_scale, pos_offset/dist_scale

    paths = glob.glob(pattern)
    parsed = []
    for p in paths:
        date, vs, window, xmode = parse_ncf_stack_filename(p) 
        if (vs_start is not None and int(vs) < vs_start) or (vs_end is not None and int(vs) > vs_end):
            continue
        parsed.append((p, date, vs, window, xmode))

    if sort_by_vs: parsed.sort(key=lambda x: int(x[2]))

    title_prefix = f"{parsed[0][1]} | {parsed[0][3]} | {parsed[0][4]}"
    target_title_suffix = f"Target: {target.upper()}"

    def process_frame_data(path: str, vs_str: str) -> Tuple[np.ndarray, np.ndarray]:
        ncf_raw = np.load(path)
        if ncf_raw.shape == (lag_axis.size, distance_axis.size): ncf_raw = ncf_raw.T 
        ncf_c, ncf_a, new_lag, s1, s2 = prep_ncf(ncf_raw, lag_axis, distance_axis, vs=vs_str, dx=dx)
        return {"causal": ncf_c, "acausal": ncf_a, "s1": s1, "s2": s2}[target], new_lag

    if pclip is not None:
        c0 = float(np.median([np.percentile(np.abs(process_frame_data(p[0], p[2])[0]), pclip) for p in tqdm(parsed, desc=f"Scanning global pclip")])) 
    else:
        c0 = float(clip if clip is not None else 1.0)

    fig, ax = plt.subplots(figsize=(6, 6) if clip_lim else (10, 6))
    ax.invert_yaxis()
    ax.set_xlabel(f"Distance along array ({unit})"); ax.set_ylabel("Lag time (s)")
    if max_lag is not None: ax.set_ylim(max_lag, 0)

    data0, new_lag_axis = process_frame_data(parsed[0][0], parsed[0][2])
    mesh = ax.pcolormesh(plot_distance, new_lag_axis, data0.T, shading="gouraud", cmap=cmap, vmin=-c0, vmax=c0)
    pos0_plot = (int(parsed[0][2]) * dx) / dist_scale
    vline = ax.axvline(x=pos0_plot, color="black", linestyle="--", linewidth=1.2, alpha=0.6)
    fig.colorbar(mesh, ax=ax, label="Correlation amplitude")
    
    dec = 2 if unit == "km" else 1
    ax.set_title(f"{os.path.basename(parsed[0][0])}\n{title_prefix} | {target_title_suffix}\nVS={parsed[0][2]} ({pos0_plot:.{dec}f} {unit})", pad=15)
    fig.tight_layout()

    def update(frame_idx):
        path, _, vs, _, _ = parsed[frame_idx]
        data, _ = process_frame_data(path, vs)
        mesh.set_array(data.T.ravel())

        pos_plot = (int(vs) * dx) / dist_scale
        vline.set_xdata([pos_plot, pos_plot])

        if clip_lim:
            if view_side == "both": left, right = pos_plot - plot_range, pos_plot + plot_range
            elif view_side == "right": left, right = pos_plot + plot_offset, pos_plot + plot_range
            else: left, right = pos_plot - plot_range, pos_plot - plot_offset
            ax.set_xlim(max(plot_distance.min(), left), min(plot_distance.max(), right))

        ax.set_title(f"{os.path.basename(path)}\n{title_prefix} | {target_title_suffix}\nVS={vs} ({pos_plot:.{dec}f} {unit})", pad=15)
        return mesh, vline

    def init(): return mesh, vline
    def frame_generator(): yield from tqdm(range(len(parsed)), desc=f"Rendering {target.upper()} Video")

    ani = FuncAnimation(fig, update, init_func=init, frames=frame_generator, save_count=len(parsed), interval=interval_ms, blit=False, repeat=True)
    plt.close(fig)
    return ani

def animate_preprocessed_ncf(
    files: List[str],
    *,
    unit: str = "m",
    clip: float | None = 0.05,
    pclip: float | None = None,
    cmap: str = "seismic",
    range_m: float = 4000.0,
    clip_lim: bool = True,
    view_side: Literal["both", "left", "right"] = "both",
    pos_offset: float = 0.0,
    interval_ms: int = 200,
    save_vs: List[int] | None = None,
    save_dir: str = "./saved_figures",
    save_fmt: str = "png",
    save_dpi: int = 300,
) -> FuncAnimation:
    """
    Animates spatially and temporally pre-processed Noise Cross-Correlation Function (NCF) 
    gathers using Matplotlib's pcolormesh.

    This function seamlessly handles variable receiver geometries (e.g., urban datasets where 
    nodes drop offline) by dynamically redrawing the spatial grid for each frame. It features 
    dynamic percentile clipping, directional viewing, near-field offset masking, and selective 
    frame saving. Accepts an explicit list of pre-sorted .npz files for easy slicing.

    :param files: List of file paths to the pre-processed NCF numpy archives.
    :param unit: Spatial distance unit for the x-axis ("m" or "km"). Default is "m".
    :param clip: Absolute amplitude limit for colorbar scaling. Used only if `pclip` is None.
    :param pclip: Percentile for dynamic amplitude clipping (e.g., 99.0). Computes a global 
                  median percentile across early frames for stable animation scaling.
    :param cmap: Matplotlib colormap to use. Default is "seismic".
    :param range_m: Maximum spatial distance (in meters or km based on `unit`) to display.
    :param clip_lim: If True, strictly limits the x-axis bounds based on `range_m`, `view_side`, 
                     and `pos_offset`.
    :param view_side: Determines which side of the virtual source gather to display 
                      ("both", "left", or "right"). Default is "both".
    :param pos_offset: Spatial exclusion offset from the virtual source to clip out near-source noise.
    :param interval_ms: Delay between animation frames in milliseconds. Default is 200.
    :param save_vs: Optional list of Virtual Source (VS) numbers to save as static high-res images.
    :param save_dir: Directory where the static frames will be saved. Default is "./saved_figures".
    :param save_fmt: Image format for the saved frames (e.g., "png"). Default is "png".
    :param save_dpi: Resolution for the saved frames. Default is 300.
    :returns: The constructed Matplotlib FuncAnimation object ready for rendering or display.
    """
    if not files:
        raise ValueError("Provided file list is empty!")

    view_side, unit = view_side.lower().strip(), unit.lower().strip()
    dist_scale = 1000.0 if unit == "km" else 1.0
    plot_range, plot_offset = range_m / dist_scale, pos_offset / dist_scale

    parsed = []
    for p in files:
        date, vs, window, xmode = parse_ncf_stack_filename(p)
        parsed.append((p, date, vs, xmode))

    if pclip is not None:
        c0 = float(np.median([np.percentile(np.abs(np.load(p[0])['data']), pclip) 
                              for p in tqdm(parsed[:50], desc="Scanning global pclip")])) 
    else:
        c0 = float(clip if clip is not None else 1.0)

    if view_side == "both": left_bound, right_bound = -plot_range, plot_range
    elif view_side == "right": left_bound, right_bound = plot_offset, plot_range
    else: left_bound, right_bound = -plot_range, -plot_offset

    # Figure Layout using Constrained Layout
    fig, ax = plt.subplots(figsize=(8, 6) if clip_lim else (10, 6), layout="constrained")
    ax.invert_yaxis()
    ax.set_xlabel(f"Offset from Virtual Source ({unit})")
    ax.set_ylabel("Lag time (s)")
    if clip_lim: 
        ax.set_xlim(left_bound, right_bound)

    # Load Initial Frame
    archive0 = np.load(parsed[0][0])
    current_offset0 = archive0['offset'] / dist_scale
    lag_axis0 = archive0['lag']
    data0 = archive0['data'].T
    
    mesh = ax.pcolormesh(current_offset0, lag_axis0, data0, shading="gouraud", cmap=cmap, vmin=-c0, vmax=c0)
    vline = ax.axvline(x=0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.6)
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04).set_label("Correlation amplitude")
    
    date0, vs0, xmode0 = parsed[0][1], parsed[0][2], parsed[0][3]
    title_text = ax.set_title(f"NCF Gather (VS={vs0} | {date0} | {xmode0})")

    # Animation tracking variables
    pbar_container = []
    processed_frames = set()
    saved_frames = set()
    total_frames = len(parsed)

    def update(frame_idx):
        nonlocal mesh  # Declare nonlocal so we can overwrite the mesh
        
        if not pbar_container:
            pbar_container.append(tqdm(total=total_frames, desc="Rendering Video"))

        path, date, vs, xmode = parsed[frame_idx]
        archive = np.load(path)
        dA = archive['data'].T
        
        # Load the dynamic offset axis for this specific frame
        current_offset = archive['offset'] / dist_scale
        lag_axis = archive['lag']
        

        mesh.remove()
        mesh = ax.pcolormesh(current_offset, lag_axis, dA, shading="gouraud", cmap=cmap, vmin=-c0, vmax=c0)
        # ------------------------------------------
        
        title_text.set_text(f"NCF Gather (VS={vs} | {date} | {xmode})")
        
        # --- LOGIC: Save Specific Frames ---
        if save_vs is not None and int(vs) in save_vs:
            if frame_idx not in saved_frames:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"NCF_Gather_VS_{vs}.{save_fmt}")
                fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight", facecolor="white")
                saved_frames.add(frame_idx)
        # -----------------------------------
        
        if frame_idx not in processed_frames:
            pbar_container[0].update(1)
            processed_frames.add(frame_idx)
        if len(processed_frames) == total_frames:
            pbar_container[0].close()

        return mesh, vline, title_text

    ani = FuncAnimation(fig, update, frames=total_frames, interval=interval_ms, blit=False)
    plt.close(fig)
    return ani

def animate_preprocessed_fk(
    files: List[str],
    *,
    unit: str = "m",
    clip: float | None = None,
    pclip: float | None = 99.0,
    cmap: str = "inferno",
    view_side: Literal["both", "left", "right"] = "right",
    pos_offset: float = 0.0,
    klim: Tuple[float, float] | None = None,
    figsize: Tuple[float, float] = (8, 6),
    vmin: float | None = None,                 
    vmax: float | None = None,                 
    interval_ms: int = 200,
    save_vs: List[int] | None = None,
    save_dir: str = "./saved_figures",
    save_fmt: str = "png",
    save_dpi: int = 300,
) -> FuncAnimation:
    """
    Animates the normalized 2D frequency-wavenumber (f-k) power spectrum of 
    pre-processed Noise Cross-Correlation Function (NCF) gathers.

    This function dynamically recalculates the spatial grid and f-k transform for 
    every frame, safely handling variable receiver geometries (e.g., dropping nodes). 
    It features dynamic power normalization, directional wavefield isolation, and 
    optional phase-velocity reference overlays. Accepts explicit lists of pre-sorted 
    .npz files to allow for easy slicing and includes hooks for exporting static frames.

    :param files: List of file paths to the pre-processed NCF numpy archives.
    :param unit: Spatial distance unit used for the x-axis ("m" or "km"). Default is "m".
    :param clip: Absolute amplitude limit for power normalization. Used only if `pclip` is None.
    :param pclip: Percentile for dynamic amplitude clipping (e.g., 99.0). Computes a global 
                  median percentile across early frames for stable animation scaling.
    :param cmap: Matplotlib colormap to use. Default is "inferno" (standard for power spectra).
    :param view_side: Determines which side of the spatial array to process and which wavenumbers 
                      to display ("both", "left", or "right"). 
    :param pos_offset: Spatial exclusion offset from the virtual source. Data within this distance 
                       is excluded before the f-k transform to prevent near-source spatial aliasing.
    :param klim: Optional tuple (kmin, kmax) specifying the wavenumber (x-axis) limits. 
                 Automatically flips sign if `view_side` is "left".
    :param figsize: Tuple specifying the figure dimensions. Default is (8, 6).
    :param vmin: Optional minimum phase velocity (m/s). Plots a cyan dashed reference line.
    :param vmax: Optional maximum phase velocity (m/s). Plots a lime dashed reference line.
    :param interval_ms: Delay between animation frames in milliseconds. Default is 200.
    :param save_vs: Optional list of Virtual Source (VS) numbers to save as static high-res images.
    :param save_dir: Directory where the static frames will be saved. Default is "./saved_figures".
    :param save_fmt: Image format for the saved frames (e.g., "png"). Default is "png".
    :param save_dpi: Resolution for the saved frames. Default is 300.
    :returns: The constructed Matplotlib FuncAnimation object ready for rendering or display.
    """
    if not files:
        raise ValueError("Provided file list is empty!")

    view_side, unit = view_side.lower().strip(), unit.lower().strip()
    dist_scale = 1000.0 if unit == "km" else 1.0

    parsed = []
    for p in files:
        date, vs, window, xmode = parse_ncf_stack_filename(p)
        parsed.append((p, date, vs, xmode))

    def process_fk_frame(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        archive = np.load(path)
        data, lag_axis, offset_axis = archive['data'], archive['lag'], archive['offset']
        
        # Spatial Masking
        mask = np.abs(offset_axis) >= pos_offset
        if view_side == "right": mask &= (offset_axis >= 0)
        elif view_side == "left": mask &= (offset_axis <= 0)
            
        f_axis, k_axis_raw, fk_complex_raw = fk_transform(data[mask, :], lag_axis[1]-lag_axis[0], offset_axis[1]-offset_axis[0])
        sort_idx = np.argsort(-k_axis_raw)
        k_axis, fk_power = -k_axis_raw[sort_idx], np.abs(fk_complex_raw[sort_idx, :])
        
        pos_f_mask = f_axis >= 0
        f_axis, fk_power = f_axis[pos_f_mask], fk_power[:, pos_f_mask]
        
        if view_side == "right": k_mask = k_axis >= 0
        elif view_side == "left": k_mask = k_axis <= 0
        else: k_mask = np.ones_like(k_axis, dtype=bool)
            
        return k_axis[k_mask], f_axis, fk_power[k_mask, :].T

    # Calculate global clip over a subset to save time
    if pclip is not None:
        c0 = float(np.median([np.percentile(process_fk_frame(p[0])[2], pclip) 
                              for p in tqdm(parsed[:50], desc="Scanning global f-k pclip")])) 
    else:
        c0 = float(clip if clip is not None else 1.0)
    c0 = c0 if c0 > 0 else 1.0

    # Layout Setup
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.set_xlabel(f"Wavenumber k (cycles/{unit})")
    ax.set_ylabel("Frequency f (Hz)")

    if klim is not None:
        ax.set_xlim(-klim[1], -klim[0]) if (view_side == "left" and klim[0] >= 0) else ax.set_xlim(*klim)

    # Initial Frame
    k0, f0, data0 = process_fk_frame(parsed[0][0])
    plot_k0 = k0 * dist_scale

    mesh = ax.pcolormesh(plot_k0, f0, data0/c0, shading="gouraud", cmap=cmap, vmin=0, vmax=1.0)
    ax.set_ylim(0, np.max(f0))
    
    # Velocity Overlay Lines
    if vmin is not None: ax.plot(plot_k0, vmin * np.abs(plot_k0 / dist_scale), color="cyan", linestyle="--", linewidth=1.8, label=f"vmin = {vmin} m/s")
    if vmax is not None: ax.plot(plot_k0, vmax * np.abs(plot_k0 / dist_scale), color="lime", linestyle="--", linewidth=1.8, label=f"vmax = {vmax} m/s")
    if vmin is not None or vmax is not None:
        ax.legend(loc="upper left", fontsize=10, framealpha=0.7, facecolor="black", edgecolor="white", labelcolor="white")

    vline = ax.axvline(x=0.0, color="white", linestyle=":", linewidth=1.0, alpha=0.4) if (view_side == "both" or (klim and klim[0] <= 0 <= klim[1])) else None
        
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04).set_label("Normalized Power")
    
    date0, vs0, xmode0 = parsed[0][1], parsed[0][2], parsed[0][3]
    title_text = ax.set_title(f"F-K Spectrum (VS={vs0} | {date0} | View: {view_side.upper()})")

    pbar_container = []
    processed_frames = set()
    saved_frames = set()
    total_frames = len(parsed)

    def update(frame_idx):
        nonlocal mesh # Declare nonlocal to overwrite the grid safely
        
        if not pbar_container:
            pbar_container.append(tqdm(total=total_frames, desc="Rendering f-k Video"))

        path, date, vs, xmode = parsed[frame_idx]
        
        # Recalculate axes for dynamic geometry
        k_axis, f_axis, fk_power = process_fk_frame(path)
        plot_k = k_axis * dist_scale
        
        mesh.remove()
        mesh = ax.pcolormesh(plot_k, f_axis, fk_power/c0, shading="gouraud", cmap=cmap, vmin=0, vmax=1.0)
        # -----------------------------------------
        
        title_text.set_text(f"F-K Spectrum (VS={vs} | {date} | View: {view_side.upper()})")
        
        # --- LOGIC: Save Specific Frames ---
        if save_vs is not None and int(vs) in save_vs:
            if frame_idx not in saved_frames:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"FK_Spectrum_VS_{vs}.{save_fmt}")
                fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight", facecolor="white")
                saved_frames.add(frame_idx)
        # -----------------------------------

        if frame_idx not in processed_frames:
            pbar_container[0].update(1)
            processed_frames.add(frame_idx)
        if len(processed_frames) == total_frames:
            pbar_container[0].close()
            
        return (mesh, vline, title_text) if vline is not None else (mesh, title_text)

    ani = FuncAnimation(fig, update, frames=total_frames, interval=interval_ms, blit=False)
    plt.close(fig)
    return ani

def animate_pipeline(
    file_list: List[str], 
    out_dir: str | None = None, 
    vmin: float = 150.0, 
    vmax: float = 2000.0, 
    vmax_time: float | None = None,
    fmax_plot: float = 10.0,
    pos_offset: float = 100.0,
    inner_taper: float = 50.0,
    range_m: float = 4000.0,
    sigma: float = 1.0,
    pclip: float = 98.0, 
    interval: int = 500, 
    buffer_start_s: float = 0.2, 
    buffer_end_s: float = 1.0, 
    top_flat_m: float = 100.0,
    max_lag: float | None = None  # NEW: Truncates the lag axis to hide artifacts
) -> FuncAnimation:
    """
    Animates a 6-panel 2D validation dashboard to visualize the multi-stage 
    Noise Cross-Correlation Function (NCF) processing pipeline.

    The dashboard dynamically displays the data at three distinct stages:
    1. RAW: The pre-processed cross-correlation gathers.
    2. FK-FILTER: The data after applying a frequency-wavenumber phase velocity filter.
    3. POLISHED: The final data after applying f-k filtering, inner/far spatial 
       tapers, and a targeted time-domain mute.
       
    Each stage is plotted side-by-side in both the f-k power spectrum domain 
    (left column) and the time-space lag-offset domain (right column).

    :param file_list: List of file paths to the raw NCF arrays to be processed and animated.
    :param out_dir: Optional directory path to save the intermediate or polished numpy arrays.
    :param vmin: Minimum phase velocity (m/s) used for f-k filtering and plotting the cyan reference line. Default is 150.0.
    :param vmax: Maximum phase velocity (m/s) used for f-k filtering and plotting the lime reference line. Default is 2000.0.
    :param vmax_time: Optional phase velocity (m/s) specifically used to calculate the time-domain ballistic mute. If None, falls back to `vmax`.
    :param fmax_plot: Maximum frequency (Hz) to display on the y-axis of the f-k panels. Default is 10.0.
    :param pos_offset: Spatial exclusion offset (meters). Data inside this offset is tapered to mitigate near-field noise. Default is 100.0.
    :param inner_taper: The width (meters) of the spatial taper applied around the `pos_offset`. Default is 50.0.
    :param range_m: Maximum spatial range (meters) to process and display. Default is 4000.0.
    :param sigma: Gaussian smoothing/tapering parameter used within the spatial or temporal mutes. Default is 1.0.
    :param pclip: Percentile (e.g., 98.0) used to dynamically calculate the colorbar clipping bounds (`vmin`/`vmax`) for both domains.
    :param interval: Delay between animation frames in milliseconds. Default is 500.
    :param buffer_start_s: Time buffer (seconds) added before the theoretical arrival time for the temporal mute window. Default is 0.2.
    :param buffer_end_s: Time buffer (seconds) added after the theoretical arrival time for the temporal mute window. Default is 1.0.
    :param top_flat_m: The spatial width (meters) of the flat-top region at the apex of the time-domain mute window. Default is 100.0.
    :param max_lag: Optional maximum lag time (seconds) to display. Useful for cropping out edge artifacts introduced by temporal mutes. If None, uses the full lag axis.
    :returns: The constructed Matplotlib FuncAnimation object ready for rendering or display.
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    plt.close(fig) 

    def update(frame_idx):
        path = file_list[frame_idx]
        
        # Pass all exposed parameters directly to the processing core
        coords, t_data_list, fk_data_list = process_single_file(
            path, 
            out_dir=out_dir, 
            vmin=vmin, 
            vmax=vmax,
            vmax_time=vmax_time,
            fmax_plot=fmax_plot,
            pos_offset=pos_offset,
            inner_taper=inner_taper,
            range_m=range_m,
            sigma=sigma, 
            buffer_start_s=buffer_start_s, 
            buffer_end_s=buffer_end_s,
            top_flat_m=top_flat_m,
            max_lag=max_lag # Passed down if needed internally
        )
        
        lag, offset, _, _ = coords
        data_raw, data_fk, data_polish = t_data_list
        k_ax, f_p, p_raw, p_fk, p_polish, lim_f = fk_data_list

        c0_fk, c0_t = np.percentile(p_raw, pclip), np.percentile(np.abs(data_raw), pclip)
        k_line = np.linspace(k_ax.min(), k_ax.max(), 500)

        stages = [
            (p_raw, data_raw, "1. RAW"), 
            (p_fk, data_fk, f"2. FK-FILTER (v={vmin}-{vmax})"), 
            (p_polish, data_polish, "3. FK + INNER TAPER + TIME MUTE + FAR TAPER")
        ]

        for i, (p_data, t_data, title) in enumerate(stages):
            ax_f, ax_t = axes[i, 0], axes[i, 1]
            ax_f.clear(); ax_t.clear()

            # --- Left Column: F-K Domain ---
            ax_f.pcolormesh(k_ax, f_p, p_data, shading='gouraud', cmap='magma', vmin=0, vmax=c0_fk)
            ax_f.plot(k_line, vmin * np.abs(k_line), 'cyan', ls='--', lw=1, alpha=0.7)
            ax_f.plot(k_line, vmax * np.abs(k_line), 'lime', ls='--', lw=1, alpha=0.7)
            ax_f.set_title(title, fontweight='bold', loc='left')
            ax_f.set_ylim(0, lim_f); ax_f.set_xlim(k_ax.min(), k_ax.max())
            ax_f.set_xlabel("k (cycles/m)"); ax_f.set_ylabel("Frequency (Hz)")

            if i == 2:
                ax_f.text(
                    0.5, 0.9, "Artifacts caused by Muting\n(Use Row 2 for true spectrum)", 
                    transform=ax_f.transAxes, ha='center', va='top', color='white', 
                    fontsize=10, bbox=dict(facecolor='red', alpha=0.5, edgecolor='none')
                )

            # --- Right Column: Time-Lag Domain ---
            ax_t.pcolormesh(offset, lag, t_data.T, shading='gouraud', cmap='seismic', vmin=-c0_t, vmax=c0_t)
            
            # Apply max_lag limits
            if max_lag is not None:
                ax_t.set_ylim(max_lag, 0) # Inverted Y-axis
            else:
                ax_t.invert_yaxis()
                
            ax_t.set_title(f"{title} (Time-Lag)", fontweight='bold', loc='left')
            ax_t.set_xlabel("Offset (m)"); ax_t.set_ylabel("Lag (s)")

        fig.suptitle(f"Processing VS: {os.path.basename(path)} [{frame_idx+1}/{len(file_list)}]", fontsize=16, y=0.98)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        return axes.flatten()

    anim = FuncAnimation(
        fig, update, frames=tqdm(range(len(file_list)), desc="Rendering Video & Saving Data"), 
        interval=interval, blit=False, cache_frame_data=False
    )
    return anim

# ===========================================================================
# 3. Plot Dispersion Images
# ===========================================================================
def animate_fv(
    fv_files: List[str],
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    cmap: str = "viridis",
    interval_ms: int = 300,
    figsize: Tuple[float, float] = (10, 6)
) -> FuncAnimation:
    """
    Animates precomputed frequency-velocity (f-v) panels using imshow for fast, 
    lightweight rendering. Bypasses raw data computation by loading previously 
    saved .npz archives containing the f-v matrix and frequency/velocity axes.

    :param fv_files: List of file paths to precomputed numpy archives (.npz). 
                     Archives are expected to contain 'fv', 'f_axis', and 'v_axis' keys.
    :param xmin: Minimum frequency (x-axis) bound to display.
    :param xmax: Maximum frequency (x-axis) bound to display.
    :param ymin: Minimum velocity (y-axis) bound to display.
    :param ymax: Maximum velocity (y-axis) bound to display.
    :param cmap: Matplotlib colormap to use. Default is "viridis".
    :param interval_ms: Delay between frames in milliseconds. Default is 300.
    :param figsize: Tuple defining figure dimensions. Default is (10, 6).
    :returns: The constructed animation object ready for rendering or display.
    """
    if not fv_files:
        raise ValueError("No files found! Check your file path or glob pattern.")

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    
    # Load first frame to setup the grid
    first_frame = np.load(fv_files[0])
    fv_data = first_frame["fv"]
    f_axis = first_frame["f_axis"]
    v_axis = first_frame["v_axis"]
    
    extent = [f_axis.min(), f_axis.max(), v_axis.min(), v_axis.max()]
    
    # interpolation='bicubic' for the smooth resolution
    mesh = ax.imshow(
        fv_data, 
        extent=extent, 
        aspect='auto', 
        origin='lower', 
        cmap=cmap,
        interpolation='bicubic',  
        vmin=np.nanmin(fv_data),
        vmax=np.nanmax(fv_data)
    )
    
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase velocity (m/s)")
    plt.colorbar(mesh, ax=ax, label="Normalized Amplitude")

    # Apply limits exactly like old code
    if xmin is not None and xmax is not None: ax.set_xlim(xmin, xmax)
    elif xmin is not None: ax.set_xlim(left=xmin)
    elif xmax is not None: ax.set_xlim(right=xmax)

    if ymin is not None and ymax is not None: ax.set_ylim(ymin, ymax)
    elif ymin is not None: ax.set_ylim(bottom=ymin)
    elif ymax is not None: ax.set_ylim(top=ymax)

    # Variables for robust progress tracking  
    pbar_container = []
    processed_frames = set()
    total_frames = len(fv_files)

    def update(frame_idx):
        if not pbar_container:
            pbar_container.append(
                tqdm(total=total_frames, desc="Loading Precomputed F-V Images")
            )
            
        filepath = fv_files[frame_idx]
        data = np.load(filepath)
        
        mesh.set_data(data["fv"])
        
        fname = os.path.basename(filepath).replace(".npz", "").replace("_fv", "")
        ax.set_title(f"Dispersion Images: {fname}", pad=15)
        
        if frame_idx not in processed_frames:
            pbar_container[0].update(1)
            processed_frames.add(frame_idx)
            
        if len(processed_frames) == total_frames:
            pbar_container[0].close()
            
        return mesh, 

    ani = FuncAnimation(
        fig, update, frames=total_frames, interval=interval_ms, blit=False
    )
    
    plt.close(fig)
    return ani

# ===========================================================================
# 4. Plot Comparison
# ===========================================================================
def animate_ncf_comparison(
    files_A: List[str],
    files_B: List[str],
    label_A: str = "Conventional",
    label_B: str = "v1",
    *,
    unit: str = "m",
    clip: float | None = None,
    pclip: float | None = 99.0,
    clip_diff: float | None = None,
    pclip_diff: float | None = 99.0,
    cmap: str = "seismic",
    range_m: float = 100.0,
    clip_lim: bool = True,
    view_side: Literal["both", "left", "right"] = "both",
    pos_offset: float = 0.0,
    interval_ms: int = 200,
    save_vs: List[int] | None = None,
    save_dir: str = "./saved_figures",
    save_fmt: str = "png",
    save_dpi: int = 300,
) -> FuncAnimation:
    """
    Animates a 3-panel, side-by-side comparison of two pre-processed Noise Cross-Correlation 
    Function (NCF) datasets and their residual difference. 

    This function pairs files from two different processing pipelines based on their Virtual 
    Source (VS) index. It handles variable receiver geometries (e.g., dropping urban nodes) 
    by dynamically redrawing the spatial grid for each frame. It computes global median 
    percentile clipping independently for stable scaling and includes hooks for saving frames.

    :param files_A: List of file paths for the first set of pre-processed NCF files.
    :param files_B: List of file paths for the second set of pre-processed NCF files.
    :param label_A: Title for the first panel (default: "Conventional").
    :param label_B: Title for the second panel (default: "v1").
    :param unit: Spatial distance unit for the x-axis ("m" or "km"). Default is "m".
    :param clip: Absolute amplitude limit for the main panels' colorbars.
    :param pclip: Percentile for dynamic amplitude clipping of the main panels (e.g., 99.0).
    :param clip_diff: Absolute amplitude limit for the residual panel's colorbar.
    :param pclip_diff: Percentile for dynamic amplitude clipping of the residual panel (e.g., 99.0).
    :param cmap: Matplotlib colormap to use for all three panels. Default is "seismic".
    :param range_m: Maximum spatial distance (in meters or kilometers based on `unit`) to display.
    :param clip_lim: If True, strictly limits the x-axis bounds based on `range_m`, `view_side`, and `pos_offset`.
    :param view_side: Determines which side of the virtual source gather to display ("both", "left", or "right").
    :param pos_offset: Spatial exclusion offset from the virtual source to clip out near-source noise.
    :param interval_ms: Delay between animation frames in milliseconds. Default is 200.
    :param save_vs: Optional list of Virtual Source (VS) numbers to save as static high-res images.
    :param save_dir: Directory where the static frames will be saved. Default is "./saved_figures".
    :param save_fmt: Image format for the saved frames (e.g., "png", "pdf"). Default is "png".
    :param save_dpi: Resolution for the saved frames. Default is 300.
    :returns: The constructed Matplotlib FuncAnimation object ready for rendering or display.
    """
    if len(files_A) != len(files_B):
        raise ValueError(f"File count mismatch: {len(files_A)} vs {len(files_B)}")
    if not files_A:
        raise ValueError("Provided file lists are empty!")

    view_side, unit = view_side.lower().strip(), unit.lower().strip()
    dist_scale = 1000.0 if unit == "km" else 1.0
    plot_range, plot_offset = range_m / dist_scale, pos_offset / dist_scale

    parsed_pairs = []
    for pA, pB in zip(files_A, files_B):
        date, vs, window, xmode = parse_ncf_stack_filename(pA)
        parsed_pairs.append((pA, pB, vs)) 

    if pclip is not None:
        c0 = float(np.median([np.percentile(np.abs(np.load(pA)['data']), pclip) 
                              for pA, _, _ in tqdm(parsed_pairs[:50], desc="Scanning Global Clip")]))
    else:
        c0 = float(clip if clip is not None else 1.0)

    if pclip_diff is not None:
        c0_diff = float(np.median([np.percentile(np.abs(np.load(pA)['data'] - np.load(pB)['data']), pclip_diff) 
                                   for pA, pB, _ in tqdm(parsed_pairs[:50], desc="Scanning Residual Clip")]))
    else:
        c0_diff = float(clip_diff if clip_diff is not None else c0 / 10.0)

    if view_side == "both": left_bound, right_bound = -plot_range, plot_range
    elif view_side == "right": left_bound, right_bound = plot_offset, plot_range
    else: left_bound, right_bound = -plot_range, -plot_offset

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True, layout="constrained")
    for ax in axes:
        ax.invert_yaxis()
        ax.set_xlabel(f"Offset from Virtual Source ({unit})")
        if clip_lim: 
            ax.set_xlim(left_bound, right_bound)

    axes[0].set_ylabel("Lag time (s)")

    # Load Initial Frame
    pA0, pB0, vs0 = parsed_pairs[0]
    archive_A = np.load(pA0)
    archive_B = np.load(pB0)
    
    current_offset = archive_A['offset'] / dist_scale
    lag_axis = archive_A['lag']
    data_A = archive_A['data'].T
    data_B = archive_B['data'].T
    data_diff = data_A - data_B

    # Initialize Meshes
    mesh_A = axes[0].pcolormesh(current_offset, lag_axis, data_A, shading="gouraud", cmap=cmap, vmin=-c0, vmax=c0)
    mesh_B = axes[1].pcolormesh(current_offset, lag_axis, data_B, shading="gouraud", cmap=cmap, vmin=-c0, vmax=c0)
    mesh_diff = axes[2].pcolormesh(current_offset, lag_axis, data_diff, shading="gouraud", cmap=cmap, vmin=-c0_diff, vmax=c0_diff)

    vlines = [ax.axvline(x=0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.6) for ax in axes]

    fig.colorbar(mesh_A, ax=axes[0], fraction=0.046, pad=0.04).set_label("Amplitude")
    fig.colorbar(mesh_B, ax=axes[1], fraction=0.046, pad=0.04).set_label("Amplitude")
    fig.colorbar(mesh_diff, ax=axes[2], fraction=0.046, pad=0.04).set_label("Residual Amplitude")

    axes[0].set_title(label_A)
    axes[1].set_title(label_B)
    axes[2].set_title(f"Residual ({label_A} - {label_B})")

    title_text = fig.suptitle(f"NCF Comparison (VS={vs0})", fontsize=16)

    pbar_container = []
    processed_frames = set()
    saved_frames = set() 
    total_frames = len(parsed_pairs)

    def update(frame_idx):
        nonlocal mesh_A, mesh_B, mesh_diff  # Enable variable-geometry overwriting
        
        if not pbar_container:
            pbar_container.append(tqdm(total=total_frames, desc="Rendering Video"))

        pA, pB, vs = parsed_pairs[frame_idx]
        archive_A = np.load(pA)
        archive_B = np.load(pB)
        
        dA = archive_A['data'].T
        dB = archive_B['data'].T
        dDiff = dA - dB
        
        # Load the dynamic offset specifically for this frame
        current_offset = archive_A['offset'] / dist_scale
        lag_axis = archive_A['lag']
        
        mesh_A.remove()
        mesh_B.remove()
        mesh_diff.remove()
        
        mesh_A = axes[0].pcolormesh(current_offset, lag_axis, dA, shading="gouraud", cmap=cmap, vmin=-c0, vmax=c0)
        mesh_B = axes[1].pcolormesh(current_offset, lag_axis, dB, shading="gouraud", cmap=cmap, vmin=-c0, vmax=c0)
        mesh_diff = axes[2].pcolormesh(current_offset, lag_axis, dDiff, shading="gouraud", cmap=cmap, vmin=-c0_diff, vmax=c0_diff)
        
        title_text.set_text(f"NCF Comparison (VS={vs})")
        
        if save_vs is not None and int(vs) in save_vs:
            if frame_idx not in saved_frames:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"NCF_Comparison_VS_{vs}.{save_fmt}")
                fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight", facecolor="white")
                saved_frames.add(frame_idx)
        
        if frame_idx not in processed_frames:
            pbar_container[0].update(1)
            processed_frames.add(frame_idx)
        if len(processed_frames) == total_frames:
            pbar_container[0].close()
            
        return mesh_A, mesh_B, mesh_diff, *vlines, title_text

    ani = FuncAnimation(fig, update, frames=total_frames, interval=interval_ms, blit=False)
    plt.close(fig)
    return ani

def animate_fv_ssim_tracking(
    files_A: List[str],
    files_B: List[str],
    ssim_func: Callable,
    label_A: str = "Conventional",
    label_B: str = "v1",
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    cmap: str = "viridis",
    interval_ms: int = 300,
    save_vs: List[int] | None = None,
    save_dir: str = "./saved_figures",
    save_fmt: str = "png",
    save_dpi: int = 300,
    **ssim_kwargs
) -> FuncAnimation:
    """
    Animates a 3-panel diagnostic dashboard comparing two sets of precomputed 
    frequency-velocity (f-v) dispersion images, featuring a dynamic rolling line 
    chart that tracks the Structural Similarity Index (SSIM) across the array.

    This function pre-computes the SSIM scores for all provided file pairs before 
    rendering to establish the historical tracking path. It utilizes bicubic 
    interpolation for smooth, publication-quality dispersion panels and a constrained 
    layout to prevent colorbars from compressing the plots. Accepts explicit lists 
    of pre-sorted .npz files to allow for easy slicing (e.g., passing `files_A[:30]`).

    :param files_A: List of file paths to the reference precomputed f-v numpy archives 
                    (e.g., Conventional processing). Archives must contain 'fv', 'f_axis', and 'v_axis'.
    :param files_B: List of file paths to the test precomputed f-v numpy archives. Must match `files_A`.
    :param ssim_func: A callable function used to compute the structural similarity.
    :param label_A: Title string for the first f-v panel. Default is "Conventional".
    :param label_B: Title string for the second f-v panel. Default is "v1".
    :param xmin: Minimum frequency (x-axis) bound to display.
    :param xmax: Maximum frequency (x-axis) bound to display.
    :param ymin: Minimum phase velocity (y-axis) bound to display.
    :param ymax: Maximum phase velocity (y-axis) bound to display.
    :param cmap: Matplotlib colormap to use for the dispersion images. Default is "viridis".
    :param interval_ms: Delay between animation frames in milliseconds. Default is 300.
    :param save_vs: Optional list of Virtual Source (VS) numbers. If provided, the animation 
                    will save high-resolution static frames of these specific VS indices to disk.
    :param save_dir: Directory where the static frames will be saved. Default is "./saved_figures".
    :param save_fmt: Image format for the saved frames (e.g., "png", "pdf"). Default is "png".
    :param save_dpi: Resolution for the saved frames. Default is 300 (publication quality).
    :param **ssim_kwargs: Additional keyword arguments passed directly into the `ssim_func`.
    :returns: The constructed Matplotlib FuncAnimation object ready for rendering or display.
    """
    if len(files_A) != len(files_B):
        raise ValueError(f"File count mismatch: A={len(files_A)}, B={len(files_B)}")
    if not files_A:
        raise ValueError("Provided file lists are empty!")

    # 1. Pre-compute SSIM Scores for the Entire Timeline
    scores = []
    vs_numbers = []
    
    for pA, pB in tqdm(zip(files_A, files_B), total=len(files_A), desc="Pre-computing SSIM"):
        dA = np.load(pA)["fv"]
        dB = np.load(pB)["fv"]
        
        score = ssim_func(D_ref=dA, D_test=dB, **ssim_kwargs)
        scores.append(score)
        
        try:
            vs = get_vs_number(pA)
        except NameError:
            vs = len(vs_numbers)
        vs_numbers.append(vs)
        
    scores = np.array(scores)

    # 2. Figure Layout using Constrained Layout
    fig = plt.figure(figsize=(18, 6), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1) 
    ax3 = fig.add_subplot(gs[0, 2])                         
    
    # 3. Load Initial Frame for Grid Setup
    first_A = np.load(files_A[0])
    f_axis = first_A["f_axis"]
    v_axis = first_A["v_axis"]
    extent = [f_axis.min(), f_axis.max(), v_axis.min(), v_axis.max()]
    
    # Init Panel 1 (Conventional)
    mesh_A = ax1.imshow(first_A["fv"], extent=extent, origin='lower', aspect='auto', 
                        cmap=cmap, interpolation='bicubic', 
                        vmin=np.nanmin(first_A["fv"]), vmax=np.nanmax(first_A["fv"]))
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Phase velocity (m/s)")
    ax1.set_title(label_A)
    fig.colorbar(mesh_A, ax=ax1, fraction=0.046, pad=0.04)

    # Init Panel 2 (v1)
    first_B = np.load(files_B[0])
    mesh_B = ax2.imshow(first_B["fv"], extent=extent, origin='lower', aspect='auto', 
                        cmap=cmap, interpolation='bicubic',
                        vmin=np.nanmin(first_B["fv"]), vmax=np.nanmax(first_B["fv"]))
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_title(label_B)
    fig.colorbar(mesh_B, ax=ax2, fraction=0.046, pad=0.04)
    
    # Apply F-V Limits
    if xmin is not None and xmax is not None: ax1.set_xlim(xmin, xmax)
    elif xmin is not None: ax1.set_xlim(left=xmin)
    elif xmax is not None: ax1.set_xlim(right=xmax)

    if ymin is not None and ymax is not None: ax1.set_ylim(ymin, ymax)
    elif ymin is not None: ax1.set_ylim(bottom=ymin)
    elif ymax is not None: ax1.set_ylim(top=ymax)

    # Init Panel 3 (The SSIM Tracker)
    ax3.set_title("Structural Similarity Index (SSIM)")
    ax3.set_xlabel("Virtual Source Index (VS)")
    ax3.set_ylabel("SSIM Score")
    
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")
    
    y_min = max(0.0, np.min(scores) - 0.05)
    ax3.set_ylim(y_min, 1.02)
    ax3.grid(True, linestyle="--", alpha=0.6)
    
    ax3.plot(range(len(scores)), scores, color='gray', alpha=0.4, linewidth=2)
    tracker_dot, = ax3.plot([0], [scores[0]], marker='o', color='red', markersize=8, zorder=5)

    title_text = fig.suptitle(
        f"Dispersion Images Comparison (VS = {vs_numbers[0]} | SSIM: {scores[0]:.4f})", 
        fontsize=16
    )

    # 4. Animation Logic
    pbar_container = []
    processed_frames = set()
    saved_frames = set()  # Track which frames have been saved to disk
    total_frames = len(files_A)

    def update(frame_idx):
        if not pbar_container:
            pbar_container.append(tqdm(total=total_frames, desc="Rendering F-V & SSIM Video"))
            
        dA = np.load(files_A[frame_idx])["fv"]
        dB = np.load(files_B[frame_idx])["fv"]
        mesh_A.set_data(dA)
        mesh_B.set_data(dB)
        
        tracker_dot.set_data([frame_idx], [scores[frame_idx]])
        vs_current = vs_numbers[frame_idx]
        
        title_text.set_text(
            f"Dispersion Images Comparison (VS = {vs_current} | SSIM: {scores[frame_idx]:.4f})"
        )
        
        # --- LOGIC: Save Specific Frames ---
        if save_vs is not None and int(vs_current) in save_vs:
            if frame_idx not in saved_frames:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"FV_SSIM_Comparison_VS_{vs_current}.{save_fmt}")
                fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight", facecolor="white")
                saved_frames.add(frame_idx)
        # -----------------------------------
        
        if frame_idx not in processed_frames:
            pbar_container[0].update(1)
            processed_frames.add(frame_idx)
        if len(processed_frames) == total_frames:
            pbar_container[0].close()
            
        return mesh_A, mesh_B, tracker_dot, title_text

    ani = FuncAnimation(fig, update, frames=total_frames, interval=interval_ms, blit=False)
    plt.close(fig)
    return ani