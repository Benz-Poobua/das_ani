"""
:module: src/plot.py
:author: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Plotting utilities for dispersion imaging (f–v panels),
          dispersion curve picks, and related DAS diagnostics.
"""
from __future__ import annotations

import glob
import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.ticker import ScalarFormatter
from matplotlib.animation import FuncAnimation

from joblib import Parallel, delayed
from tqdm.auto import tqdm
from typing import Tuple, List, Literal, Callable, Any
from scipy.interpolate import griddata

from disba import depthplot, surf96
from disba._common import ifunc

from src.utils import parse_ncf_stack_filename, fk_filter
from src.disp import prep_ncf

# ===========================================================================
# 1. Plot NCF
# ===========================================================================
def plot_ncf_section_mesh(
    ncf: np.ndarray,
    lag_axis: np.ndarray,
    distance_axis: np.ndarray,
    mode: str = "all",
    clip: float | None = 0.05,
    pclip: float | None = None,
    cmap: str = "seismic",
    title: str | None = None,
    filename: str | None = None,
    vs: str = "0",
    gauge_length: float = 8.16,
    range_m: float = 500.0,
    clip_lim: bool = True
) -> None:
    """
    Plots a Noise Correlation Function (NCF) section mesh for a single virtual source.

    :param ncf: The 2D noise correlation function data array.
    :type ncf: np.ndarray
    :param lag_axis: 1D array of time lag values.
    :type lag_axis: np.ndarray
    :param distance_axis: 1D array of distance values along the array.
    :type distance_axis: np.ndarray
    :param mode: The type of NCF to plot ('all', 'causal', 'acausal'). Default is "all".
    :type mode: str, optional
    :param clip: Absolute limit for color scaling. Default is 0.05.
    :type clip: float | None, optional
    :param pclip: Percentile limit for color scaling (overrides clip if set). Default is None.
    :type pclip: float | None, optional
    :param cmap: Matplotlib colormap to use. Default is "seismic".
    :type cmap: str, optional
    :param title: Custom title for the plot. If None, generated automatically. Default is None.
    :type title: str | None, optional
    :param filename: Filename to include in the automatic title. Default is None.
    :type filename: str | None, optional
    :param vs: Virtual source identifier. Default is "0".
    :type vs: str, optional
    :param gauge_length: Spatial separation between channels in meters. Default is 8.16.
    :type gauge_length: float, optional
    :param range_m: Distance in meters around the virtual source to display. Default is 500.0.
    :type range_m: float, optional
    :param clip_lim: Whether to clip the x-axis to the specified range_m. Default is True.
    :type clip_lim: bool, optional
    :returns: None
    :raises ValueError: If an invalid `mode` is provided or if `ncf` shape mismatches the axes.
    """
    mode = mode.lower().strip()
    if mode not in {"all", "causal", "acausal"}:
        raise ValueError("mode must be one of: 'all', 'causal', 'acausal'")

    ncf = np.asarray(ncf)
    lag_axis = np.asarray(lag_axis)
    distance_axis = np.asarray(distance_axis)

    # Expect (n_distance, n_lag). Auto-fix common transpose.
    if ncf.shape == (lag_axis.size, distance_axis.size):
        ncf = ncf.T
    if ncf.shape != (distance_axis.size, lag_axis.size):
        raise ValueError(f"Shape mismatch: ncf.shape={ncf.shape}")

    # --- Position Calculation ---
    position = int(vs) * gauge_length

    # --- Select mode and slice data ---
    if mode == "all":
        y = lag_axis
        data = ncf
        ylabel = "Lag time (s)"
    elif mode == "causal":
        sel = lag_axis >= 0
        y = lag_axis[sel]
        data = ncf[:, sel]
        ylabel = "Lag time (s)"
    else:  # acausal
        sel = lag_axis <= 0
        y_raw = np.abs(lag_axis[sel])
        order = np.argsort(y_raw)
        y = y_raw[order]
        data = ncf[:, sel][:, order]
        ylabel = "|Lag time| (s)"

    # --- Color clip ---
    if pclip is not None:
        c = np.percentile(np.abs(data), pclip)
    elif clip is not None:
        c = float(clip)
    else:
        c = np.max(np.abs(data)) if data.size else 1.0

    # --- Figure Setup ---
    fig_size = (6, 6) if clip_lim else (10, 6)
    fig, ax = plt.subplots(figsize=fig_size)

    # --- Rendering with pcolormesh ---
    # pcolormesh(X, Y, Z) where Z is (n_y, n_x)
    # We pass data.T which is (n_lag, n_dist) to match X(dist) and Y(lag)
    img = ax.pcolormesh(
        distance_axis, 
        y, 
        data.T, 
        shading='gouraud', # or `auto`
        cmap=cmap, 
        vmin=-c, 
        vmax=c
    )

    # --- Virtual Source Marker ---
    ax.axvline(x=position, color='black', linestyle='--', linewidth=1.2, alpha=0.6, label=f"VS {vs}")

    # --- Title ---
    vs_info = f" (VS={vs} @ {position:.1f}m)"
    if title is None:
        base = os.path.basename(filename) if filename else ""
        title = f"NCF {mode}: {base}{vs_info}".strip().rstrip(":")
    else:
        title = f"NCF {mode}: {title}{vs_info}"

    # --- Final Touches ---
    if clip_lim:
        # Calculate theoretical limits
        left_lim = position - range_m
        right_lim = position + range_m

        # Clamp to the actual boundaries of the data
        ax.set_xlim(
            max(distance_axis.min(), left_lim), 
            min(distance_axis.max(), right_lim)
        )

    ax.invert_yaxis() # Traditional seismic view: time increases downward
    ax.set_xlabel("Distance along array (m)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=15)
    
    fig.colorbar(img, ax=ax, label="Correlation amplitude")
    fig.tight_layout()
            
    plt.show()

def animate_ncf_section_mesh(
    pattern: str,
    *,
    lag_axis: np.ndarray,
    distance_axis: np.ndarray,
    mode: str = "causal",
    clip: float | None = 0.05,
    pclip: float | None = None,
    cmap: str = "seismic",
    gauge_length: float = 8.16,
    range_m: float = 500.0,
    clip_lim: bool = True,
    interval_ms: int = 200,
    repeat_delay_ms: int = 1000,
    sort_by_vs: bool = True,
) -> FuncAnimation:
    """
    Animates NCF sections dynamically using Matplotlib's FuncAnimation.

    Includes robust global color scaling, dynamic camera panning, 
    visible titles with filenames, and tqdm progress bars.

    :param pattern: Glob pattern to match NCF file paths (e.g., 'data/*.npy').
    :type pattern: str
    :param lag_axis: 1D array of time lag values.
    :type lag_axis: np.ndarray
    :param distance_axis: 1D array of distance values along the array.
    :type distance_axis: np.ndarray
    :param mode: The type of NCF to animate ('all', 'causal', 'acausal'). Default is "causal".
    :type mode: str, optional
    :param clip: Absolute limit for color scaling. Default is 0.05.
    :type clip: float | None, optional
    :param pclip: Percentile limit for color scaling across all files. Default is None.
    :type pclip: float | None, optional
    :param cmap: Matplotlib colormap to use. Default is "seismic".
    :type cmap: str, optional
    :param gauge_length: Spatial separation between channels in meters. Default is 8.16.
    :type gauge_length: float, optional
    :param range_m: Distance in meters around the virtual source to display. Default is 500.0.
    :type range_m: float, optional
    :param clip_lim: Whether to dynamically pan the x-axis bounds per frame. Default is True.
    :type clip_lim: bool, optional
    :param interval_ms: Delay between frames in milliseconds. Default is 200.
    :type interval_ms: int, optional
    :param repeat_delay_ms: Delay before repeating the animation in milliseconds. Default is 1000.
    :type repeat_delay_ms: int, optional
    :param sort_by_vs: Whether to sort the files by their virtual source index. Default is True.
    :type sort_by_vs: bool, optional
    :returns: The constructed animation object ready for rendering or display.
    :rtype: FuncAnimation
    :raises ValueError: If an invalid `mode` is provided.
    :raises FileNotFoundError: If the `pattern` does not match any files.
    """
    
    mode = mode.lower().strip()
    if mode not in {"all", "causal", "acausal"}:
        raise ValueError("mode must be one of: 'all', 'causal', 'acausal'")

    lag_axis = np.asarray(lag_axis)
    distance_axis = np.asarray(distance_axis)

    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    # Parse and sort metadata
    parsed: List[Tuple[str, str, str, str, str]] = []
    for p in paths:
        date, vs, window, xmode = parse_ncf_stack_filename(p) 
        parsed.append((p, date, vs, window, xmode))

    if sort_by_vs:
        parsed.sort(key=lambda x: int(x[2]))

    _, date0, _, window0, xmode0 = parsed[0]
    title_prefix = f"{date0} | {window0} | {xmode0}"

    # --- Pre-calculate static Y-axis slice based on mode ---
    if mode == "all":
        y = lag_axis
        sel = slice(None)
    elif mode == "causal":
        sel = lag_axis >= 0
        y = lag_axis[sel]
    else:  # acausal
        sel = lag_axis <= 0
        y_raw = np.abs(lag_axis[sel])
        order = np.argsort(y_raw)
        y = y_raw[order]

    # --- Pre-scan files for Global Clip Limit with tqdm ---
    if pclip is not None:
        per_file_clips = []
        for path_info in tqdm(parsed, desc="Scanning for global pclip"):
            temp_ncf = np.load(path_info[0])
            
            # Fix transpose if necessary
            if temp_ncf.shape == (lag_axis.size, distance_axis.size):
                temp_ncf = temp_ncf.T
            
            # Slice data based on mode
            temp_data = temp_ncf if mode == "all" else temp_ncf[:, sel]
            
            # Calculate and store the percentile for this specific file
            per_file_clips.append(np.percentile(np.abs(temp_data), pclip))
        
        # Use the median of all calculated clips to ignore extreme outliers
        c0 = float(np.median(per_file_clips)) 
    else:
        c0 = float(clip if clip is not None else 1.0)
        
    print(f"Global clip limit safely locked to: +/- {c0:.5f}")

    # --- Set up Figure ---
    fig_size = (6, 6) if clip_lim else (10, 6)
    fig, ax = plt.subplots(figsize=fig_size)

    ylabel = "|Lag time| (s)" if mode == "acausal" else "Lag time (s)"
    ax.invert_yaxis()
    ax.set_xlabel("Distance along array (m)")
    ax.set_ylabel(ylabel)

    # --- Initialize First Frame ---
    ncf0 = np.load(parsed[0][0])
    if ncf0.shape == (lag_axis.size, distance_axis.size):
        ncf0 = ncf0.T
        
    data0 = ncf0 if mode == "all" else ncf0[:, sel]
    if mode == "acausal":
        data0 = data0[:, order]

    mesh = ax.pcolormesh(
        distance_axis, y, data0.T,
        shading="gouraud", cmap=cmap, vmin=-c0, vmax=c0
    )
    
    pos0 = int(parsed[0][2]) * gauge_length
    vline = ax.axvline(x=pos0, color="black", linestyle="--", linewidth=1.2, alpha=0.6)
    
    fig.colorbar(mesh, ax=ax, label="Correlation amplitude")
    
    # --- Set initial title BEFORE tight_layout() so Matplotlib leaves space ---
    filename0 = os.path.basename(parsed[0][0])
    ax.set_title(f"{filename0}\n{title_prefix} | VS={parsed[0][2]} ({pos0:.1f} m)", pad=15)
    
    fig.tight_layout()

    # --- Update Function for Animation ---
    def update(frame_idx):
        path, _, vs, _, _ = parsed[frame_idx]
        
        ncf = np.load(path)
        if ncf.shape == (lag_axis.size, distance_axis.size):
            ncf = ncf.T
            
        data = ncf if mode == "all" else ncf[:, sel]
        if mode == "acausal":
            data = data[:, order]

        # Update mesh data in-place 
        mesh.set_array(data.T.ravel())

        # Update marker line
        position = int(vs) * gauge_length
        vline.set_xdata([position, position])

        # Dynamically pan the camera
        if clip_lim:
            left_lim = position - range_m
            right_lim = position + range_m
            ax.set_xlim(
                max(distance_axis.min(), left_lim),
                min(distance_axis.max(), right_lim),
            )

        # Update title with filename
        filename = os.path.basename(path)
        ax.set_title(f"{filename}\n{title_prefix} | VS={vs} ({position:.1f} m)", pad=15)
        
        return mesh, vline

    # --- Setup dummy init to prevent early evaluation ---
    def init():
        return mesh, vline

    # --- Progress Bar Generator ---
    def frame_generator():
        yield from tqdm(range(len(parsed)), desc="Rendering JSHTML Video")

    # --- Create Animation ---
    ani = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=frame_generator,
        save_count=len(parsed),
        interval=interval_ms,
        blit=False,
        repeat=True,
    )

    plt.close(fig) 
    
    return ani


def animate_fk_filtered_ncf_section_mesh(
    pattern: str,
    *,
    lag_axis: np.ndarray,
    distance_axis: np.ndarray,
    dt: float,
    dx: float,
    vmin: float,
    vmax: float,
    fk_mode: Literal["eliminate", "extract"] = "extract",
    fk_smooth: Literal["no", "gaussian", "uniform"] = "gaussian",
    fk_sigma: float = 2.0,
    mode: str = "causal",
    clip: float | None = 0.05,
    pclip: float | None = None,
    cmap: str = "seismic",
    gauge_length: float = 8.16,
    range_m: float = 500.0,
    clip_lim: bool = True,
    interval_ms: int = 200,
    sort_by_vs: bool = True,
) -> FuncAnimation:
    """
    Animates NCF sections dynamically with an applied f-k velocity filter.

    :param pattern: Glob pattern to match NCF file paths.
    :type pattern: str
    :param lag_axis: 1D array of time lag values.
    :type lag_axis: np.ndarray
    :param distance_axis: 1D array of distance values along the array.
    :type distance_axis: np.ndarray
    :param dt: Time step size.
    :type dt: float
    :param dx: Spatial step size.
    :type dx: float
    :param vmin: Minimum velocity for the f-k filter.
    :type vmin: float
    :param vmax: Maximum velocity for the f-k filter.
    :type vmax: float
    :param fk_mode: Filter operation mode. Default is "extract".
    :type fk_mode: Literal["eliminate", "extract"], optional
    :param fk_smooth: Smoothing applied to the f-k mask. Default is "gaussian".
    :type fk_smooth: Literal["no", "gaussian", "uniform"], optional
    :param fk_sigma: Standard deviation for the gaussian filter. Default is 2.0.
    :type fk_sigma: float, optional
    :param mode: The type of NCF to animate ('all', 'causal', 'acausal'). Default is "causal".
    :type mode: str, optional
    :param clip: Absolute limit for color scaling. Default is 0.05.
    :type clip: float | None, optional
    :param pclip: Percentile limit for global color scaling. Default is None.
    :type pclip: float | None, optional
    :param cmap: Matplotlib colormap to use. Default is "seismic".
    :type cmap: str, optional
    :param gauge_length: Spatial separation between channels in meters. Default is 8.16.
    :type gauge_length: float, optional
    :param range_m: Distance in meters around the virtual source to display. Default is 500.0.
    :type range_m: float, optional
    :param clip_lim: Whether to dynamically pan the x-axis bounds per frame. Default is True.
    :type clip_lim: bool, optional
    :param interval_ms: Delay between frames in milliseconds. Default is 200.
    :type interval_ms: int, optional
    :param sort_by_vs: Whether to sort the files by virtual source index. Default is True.
    :type sort_by_vs: bool, optional
    :returns: The constructed animation object ready for rendering or display.
    :rtype: FuncAnimation
    :raises ValueError: If an invalid `mode` is provided.
    :raises FileNotFoundError: If the `pattern` does not match any files.
    """
    
    mode = mode.lower().strip()
    if mode not in {"all", "causal", "acausal"}:
        raise ValueError("mode must be one of: 'all', 'causal', 'acausal'")

    lag_axis = np.asarray(lag_axis)
    distance_axis = np.asarray(distance_axis)

    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    # Parse and sort metadata
    parsed: List[Tuple[str, str, str, str, str]] = []
    for p in paths:
        date, vs, window, xmode = parse_ncf_stack_filename(p) 
        parsed.append((p, date, vs, window, xmode))

    if sort_by_vs:
        parsed.sort(key=lambda x: int(x[2]))

    _, date0, _, window0, xmode0 = parsed[0]
    title_prefix = f"{date0} | {window0} | {xmode0}"
    fk_title_suffix = f"f–k vel: {vmin}-{vmax} m/s"

    # --- Pre-calculate static Y-axis slice based on mode ---
    if mode == "all":
        y = lag_axis
        sel = slice(None)
    elif mode == "causal":
        sel = lag_axis >= 0
        y = lag_axis[sel]
    else:  # acausal
        sel = lag_axis <= 0
        y_raw = np.abs(lag_axis[sel])
        order = np.argsort(y_raw)
        y = y_raw[order]

    # --- Pre-scan files for Global Clip Limit (WITH F-K FILTER) ---
    if pclip is not None:
        per_file_clips = []
        for path_info in tqdm(parsed, desc="f-k Scanning for global pclip"):
            temp_ncf = np.load(path_info[0])
            
            # Ensure shape is (nx, nt) for the fk_filter
            if temp_ncf.shape == (lag_axis.size, distance_axis.size):
                temp_ncf = temp_ncf.T
            
            # Apply the f-k filter to the FULL data before slicing
            temp_fk = fk_filter(
                temp_ncf, dt=dt, dx=dx, vmin=vmin, vmax=vmax, 
                mode=fk_mode, smooth=fk_smooth, sigma=fk_sigma
            )
            
            # Slice data based on mode (causal/acausal/all)
            temp_data = temp_fk if mode == "all" else temp_fk[:, sel]
            
            per_file_clips.append(np.percentile(np.abs(temp_data), pclip))
        
        c0 = float(np.median(per_file_clips)) 
    else:
        c0 = float(clip if clip is not None else 1.0)
        
    print(f"Global f-k clip limit safely locked to: +/- {c0:.5f}")

    # --- Set up Figure ---
    fig_size = (6, 6) if clip_lim else (10, 6)
    fig, ax = plt.subplots(figsize=fig_size)

    ylabel = "|Lag time| (s)" if mode == "acausal" else "Lag time (s)"
    ax.invert_yaxis()
    ax.set_xlabel("Distance along array (m)")
    ax.set_ylabel(ylabel)

    # --- Initialize First Frame ---
    ncf0 = np.load(parsed[0][0])
    if ncf0.shape == (lag_axis.size, distance_axis.size):
        ncf0 = ncf0.T
        
    # Apply f-k filter to frame 0
    ncf0_fk = fk_filter(
        ncf0, dt=dt, dx=dx, vmin=vmin, vmax=vmax, 
        mode=fk_mode, smooth=fk_smooth, sigma=fk_sigma
    )
        
    data0 = ncf0_fk if mode == "all" else ncf0_fk[:, sel]
    if mode == "acausal":
        data0 = data0[:, order]

    mesh = ax.pcolormesh(
        distance_axis, y, data0.T,
        shading="gouraud", cmap=cmap, vmin=-c0, vmax=c0
    )
    
    pos0 = int(parsed[0][2]) * gauge_length
    vline = ax.axvline(x=pos0, color="black", linestyle="--", linewidth=1.2, alpha=0.6)
    
    fig.colorbar(mesh, ax=ax, label="Correlation amplitude")
    
    # --- Set initial title ---
    filename0 = os.path.basename(parsed[0][0])
    ax.set_title(f"{filename0}\n{title_prefix} | {fk_title_suffix}\nVS={parsed[0][2]} ({pos0:.1f} m)", pad=15)
    
    fig.tight_layout()

    # --- Update Function for Animation ---
    def update(frame_idx):
        path, _, vs, _, _ = parsed[frame_idx]
        
        ncf = np.load(path)
        if ncf.shape == (lag_axis.size, distance_axis.size):
            ncf = ncf.T
            
        # Apply f-k filter on the fly
        ncf_fk = fk_filter(
            ncf, dt=dt, dx=dx, vmin=vmin, vmax=vmax, 
            mode=fk_mode, smooth=fk_smooth, sigma=fk_sigma
        )
            
        data = ncf_fk if mode == "all" else ncf_fk[:, sel]
        if mode == "acausal":
            data = data[:, order]

        mesh.set_array(data.T.ravel())

        position = int(vs) * gauge_length
        vline.set_xdata([position, position])

        if clip_lim:
            left_lim = position - range_m
            right_lim = position + range_m
            ax.set_xlim(
                max(distance_axis.min(), left_lim),
                min(distance_axis.max(), right_lim),
            )

        filename = os.path.basename(path)
        ax.set_title(f"{filename}\n{title_prefix} | {fk_title_suffix}\nVS={vs} ({position:.1f} m)", pad=15)
        
        return mesh, vline

    def init():
        return mesh, vline

    def frame_generator():
        yield from tqdm(range(len(parsed)), desc="Rendering f-k JSHTML Video")

    ani = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=frame_generator,
        save_count=len(parsed),
        interval=interval_ms,
        blit=False,
        repeat=True,
    )

    plt.close(fig) 
    
    return ani

def animate_directional_fk_ncf_section_mesh(
    pattern: str,
    *,
    lag_axis: np.ndarray,
    distance_axis: np.ndarray,
    dt: float,
    dx: float,
    vmin: float,
    vmax: float,
    target: Literal["causal", "acausal", "s1", "s2"] = "s1",
    fk_mode: Literal["eliminate", "extract"] = "extract",
    fk_smooth: Literal["no", "gaussian", "uniform"] = "gaussian",
    fk_sigma: float = 2.0,
    clip: float | None = 0.05,
    pclip: float | None = None,
    cmap: str = "seismic",
    gauge_length: float = 8.16,
    range_m: float = 500.0,
    clip_lim: bool = True,
    view_side: Literal["both", "left", "right"] = "both",  
    pos_offset: float = 0.0,                               
    interval_ms: int = 200,
    sort_by_vs: bool = True,
) -> FuncAnimation:
    """
    Animates directional NCF sections, relying on src.disp.prep_ncf for 
    spatial-temporal swapping and src.utils.fk_filter for velocity filtering.

    :param pattern: Glob pattern to match NCF file paths.
    :type pattern: str
    :param lag_axis: 1D array of time lag values.
    :type lag_axis: np.ndarray
    :param distance_axis: 1D array of distance values along the array.
    :type distance_axis: np.ndarray
    :param dt: Time step size.
    :type dt: float
    :param dx: Spatial step size.
    :type dx: float
    :param vmin: Minimum velocity for the f-k filter.
    :type vmin: float
    :param vmax: Maximum velocity for the f-k filter.
    :type vmax: float
    :param target: Target slice to render from prep_ncf output. Default is "s1".
    :type target: Literal["causal", "acausal", "s1", "s2"], optional
    :param fk_mode: Filter operation mode. Default is "extract".
    :type fk_mode: Literal["eliminate", "extract"], optional
    :param fk_smooth: Smoothing applied to the f-k mask. Default is "gaussian".
    :type fk_smooth: Literal["no", "gaussian", "uniform"], optional
    :param fk_sigma: Standard deviation for the gaussian filter. Default is 2.0.
    :type fk_sigma: float, optional
    :param clip: Absolute limit for color scaling. Default is 0.05.
    :type clip: float | None, optional
    :param pclip: Percentile limit for global color scaling. Default is None.
    :type pclip: float | None, optional
    :param cmap: Matplotlib colormap to use. Default is "seismic".
    :type cmap: str, optional
    :param gauge_length: Spatial separation between channels in meters. Default is 8.16.
    :type gauge_length: float, optional
    :param range_m: Total viewing window around the virtual source. Default is 500.0.
    :type range_m: float, optional
    :param clip_lim: Whether to dynamically pan the x-axis bounds per frame. Default is True.
    :type clip_lim: bool, optional
    :param view_side: Restricts viewing side relative to the virtual source. Default is "both".
    :type view_side: Literal["both", "left", "right"], optional
    :param pos_offset: Exclusion zone offset from the virtual source. Default is 0.0.
    :type pos_offset: float, optional
    :param interval_ms: Delay between frames in milliseconds. Default is 200.
    :type interval_ms: int, optional
    :param sort_by_vs: Whether to sort the files by virtual source index. Default is True.
    :type sort_by_vs: bool, optional
    :returns: The constructed animation object ready for rendering or display.
    :rtype: FuncAnimation
    :raises FileNotFoundError: If the `pattern` does not match any files.
    :raises ValueError: If an invalid `view_side` is provided.
    """
    target = target.lower().strip()
    
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    parsed: List[Tuple[str, str, str, str, str]] = []
    for p in paths:
        date, vs, window, xmode = parse_ncf_stack_filename(p) 
        parsed.append((p, date, vs, window, xmode))

    if sort_by_vs:
        parsed.sort(key=lambda x: int(x[2]))

    title_prefix = f"{parsed[0][1]} | {parsed[0][3]} | {parsed[0][4]}"
    fk_title_suffix = f"f–k: {vmin}-{vmax} m/s | Target: {target.upper()}"

    # --- CLEANED UP: Helper function calling your external tools ---
    def process_frame_data(path: str, vs_str: str) -> Tuple[np.ndarray, np.ndarray]:
        ncf_raw = np.load(path)
        if ncf_raw.shape == (lag_axis.size, distance_axis.size):
            ncf_raw = ncf_raw.T 
            
        # 1. Filter using imported function
        ncf_fk = fk_filter(
            ncf_raw, dt=dt, dx=dx, vmin=vmin, vmax=vmax, 
            mode=fk_mode, smooth=fk_smooth, sigma=fk_sigma
        )
        
        # 2. Swap and slice using imported function
        ncf_c, ncf_a, new_lag, s1, s2 = prep_ncf(
            ncf_fk, lag_axis, distance_axis, vs=vs_str, gauge_length=gauge_length
        )
        
        # 3. Return target
        target_map = {"causal": ncf_c, "acausal": ncf_a, "s1": s1, "s2": s2}
        return target_map[target], new_lag

    # --- Pre-scan files for Global Clip Limit ---
    if pclip is not None:
        per_file_clips = []
        for path_info in tqdm(parsed, desc=f"Scanning {target.upper()} for global pclip"):
            temp_data, _ = process_frame_data(path_info[0], path_info[2])
            per_file_clips.append(np.percentile(np.abs(temp_data), pclip))
        c0 = float(np.median(per_file_clips)) 
    else:
        c0 = float(clip if clip is not None else 1.0)

    # --- Set up Figure ---
    fig_size = (6, 6) if clip_lim else (10, 6)
    fig, ax = plt.subplots(figsize=fig_size)

    ax.invert_yaxis()
    ax.set_xlabel("Distance along array (m)")
    ax.set_ylabel("Lag time (s)")

    # --- Initialize First Frame ---
    data0, new_lag_axis = process_frame_data(parsed[0][0], parsed[0][2])

    mesh = ax.pcolormesh(
        distance_axis, new_lag_axis, data0.T,
        shading="gouraud", cmap=cmap, vmin=-c0, vmax=c0
    )
    
    pos0 = int(parsed[0][2]) * gauge_length
    vline = ax.axvline(x=pos0, color="black", linestyle="--", linewidth=1.2, alpha=0.6)
    fig.colorbar(mesh, ax=ax, label="Correlation amplitude")
    
    filename0 = os.path.basename(parsed[0][0])
    ax.set_title(f"{filename0}\n{title_prefix} | {fk_title_suffix}\nVS={parsed[0][2]} ({pos0:.1f} m)", pad=15)
    fig.tight_layout()

    # --- Update Function ---
    def update(frame_idx):
        path, _, vs, _, _ = parsed[frame_idx]
        
        data, _ = process_frame_data(path, vs)
        mesh.set_array(data.T.ravel())

        position = int(vs) * gauge_length
        vline.set_xdata([position, position])

        # --- NEW: Dynamic visual clipping ---
        if clip_lim:
            if view_side.lower() == "both":
                left_bound = position - range_m
                right_bound = position + range_m
            elif view_side.lower() == "right":
                left_bound = position + pos_offset
                right_bound = position + range_m
            elif view_side.lower() == "left":
                left_bound = position - range_m
                right_bound = position - pos_offset
            else:
                raise ValueError("view_side must be 'both', 'left', or 'right'")
                
            ax.set_xlim(
                max(distance_axis.min(), left_bound),
                min(distance_axis.max(), right_bound),
            )

        filename = os.path.basename(path)
        ax.set_title(f"{filename}\n{title_prefix} | {fk_title_suffix}\nVS={vs} ({position:.1f} m)", pad=15)
        return mesh, vline

    def init(): return mesh, vline
    def frame_generator(): yield from tqdm(range(len(parsed)), desc=f"Rendering {target.upper()} Video")

    ani = FuncAnimation(
        fig, update, init_func=init, frames=frame_generator,
        save_count=len(parsed), interval=interval_ms, blit=False, repeat=True
    )

    plt.close(fig) 
    return ani

# ===========================================================================
# 2. Plot Dispersion Images + Picks
# ===========================================================================
def animate_fv(
    processed_files: list[str],
    calc_fv_func,
    fv_kwargs: dict,
    cmap: str = "viridis",
    interval_ms: int = 300,
) -> FuncAnimation:
    """
    Animates frequency-velocity (f-v) panels from raw data files.

    :param processed_files: List of file paths to processed numpy data files.
    :type processed_files: list[str]
    :param calc_fv_func: Function to compute the f-v panel. Expected to return (fv, f_axis, v_axis).
    :type calc_fv_func: Callable
    :param fv_kwargs: Additional keyword arguments passed to `calc_fv_func`.
    :type fv_kwargs: dict
    :param cmap: Matplotlib colormap to use. Default is "viridis".
    :type cmap: str, optional
    :param interval_ms: Delay between frames in milliseconds. Default is 300.
    :type interval_ms: int, optional
    :returns: The constructed animation object ready for rendering or display.
    :rtype: FuncAnimation
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Initialize first frame
    item0 = np.load(processed_files[0], allow_pickle=True).item()
    fv0, f_axis0, v_axis0 = calc_fv_func(
        data=item0["data"], offset=item0["dist_rel"], t=item0["lag"], **fv_kwargs
    )
    
    f_np = f_axis0.cpu().numpy() if hasattr(f_axis0, 'cpu') else np.asarray(f_axis0)
    v_np = v_axis0.cpu().numpy() if hasattr(v_axis0, 'cpu') else np.asarray(v_axis0)
    fv_np = fv0.cpu().numpy() if hasattr(fv0, 'cpu') else np.asarray(fv0)

    # 2. Setup Plot Elements
    mesh = ax.pcolormesh(f_np, v_np, fv_np, shading='gouraud', cmap=cmap, snap=True)
    
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase velocity (m/s)")
    plt.colorbar(mesh, ax=ax, label="Normalized Amplitude")

    def update(frame_idx):
        fpath = processed_files[frame_idx]
        item = np.load(fpath, allow_pickle=True).item()
        
        fv, f_axis, v_axis = calc_fv_func(
            data=item["data"], offset=item["dist_rel"], t=item["lag"], **fv_kwargs
        )
        
        # Update Mesh
        fv_np_cur = fv.cpu().numpy() if hasattr(fv, 'cpu') else np.asarray(fv)
        mesh.set_array(fv_np_cur.ravel())
        
        fname = os.path.basename(fpath).replace(".npy", "")
        ax.set_title(f"Dispersion Viewer: {fname}")
        
        return mesh, # Note the comma: FuncAnimation expects an iterable of artists

    ani = FuncAnimation(
        fig, update, frames=len(processed_files), interval=interval_ms, blit=False
    )
    
    plt.close(fig)
    return ani

def animate_fv_pick(
    fv_files: list[str],
    picks_dir: str,
    cmap: str = "viridis",
    interval_ms: int = 400,
) -> FuncAnimation:
    """
    Instantly animates pre-computed f-v panels and overlays saved picks if they exist.

    :param fv_files: List of file paths to pre-computed frequency-velocity panels.
    :type fv_files: list[str]
    :param picks_dir: Directory path containing the pre-picked `.npy` files.
    :type picks_dir: str
    :param cmap: Matplotlib colormap to use. Default is "viridis".
    :type cmap: str, optional
    :param interval_ms: Delay between frames in milliseconds. Default is 400.
    :type interval_ms: int, optional
    :returns: The constructed animation object ready for rendering or display.
    :rtype: FuncAnimation
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Initialize first frame
    item0 = np.load(fv_files[0], allow_pickle=True).item()
    f_axis0, v_axis0, fv0 = item0["f_axis"], item0["v_axis"], item0["fv"]
    
    # Move to CPU/NumPy if they are tensors
    f_np = f_axis0.cpu().numpy() if hasattr(f_axis0, 'cpu') else np.asarray(f_axis0)
    v_np = v_axis0.cpu().numpy() if hasattr(v_axis0, 'cpu') else np.asarray(v_axis0)
    fv_np = fv0.cpu().numpy() if hasattr(fv0, 'cpu') else np.asarray(fv0)

    # 2. Setup Plot Elements
    mesh = ax.pcolormesh(f_np, v_np, fv_np, shading='gouraud', cmap=cmap, snap=True)
    scat = ax.scatter([], [], color='red', s=30, edgecolors='white', linewidth=0.5, label="Saved Picks", zorder=10)
    
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase velocity (m/s)")
    ax.legend(loc='upper right')
    plt.colorbar(mesh, ax=ax, label="Normalized Amplitude")

    def update(frame_idx):
        # Load FV Panel
        fpath = fv_files[frame_idx]
        fname_full = os.path.basename(fpath)
        data = np.load(fpath, allow_pickle=True).item()
        
        fv_np_cur = data["fv"].cpu().numpy() if hasattr(data["fv"], 'cpu') else np.asarray(data["fv"])
        mesh.set_array(fv_np_cur.ravel())
        
        # Check for matching Pick file
        pick_fname = fname_full.replace("_fv.npy", "_pick.npy")
        pick_path = os.path.join(picks_dir, pick_fname)
        
        if os.path.exists(pick_path):
            pick_data = np.load(pick_path, allow_pickle=True).item()
            points = np.column_stack((pick_data['f'], pick_data['v']))
            scat.set_offsets(points)
            scat.set_visible(True)
            status = ""
        else:
            scat.set_visible(False)
            
        # Title
        display_name = fname_full.replace("_fv.npy", "")
        ax.set_title(f"Dispersion Viewer: {display_name}")
        
        return mesh, scat

    ani = FuncAnimation(
        fig, update, frames=len(fv_files), interval=interval_ms, blit=False
    )
    
    plt.close(fig)
    return ani

# ===========================================================================
# 3. Plot Picks
# ===========================================================================
def plot_scatter_section(
    x: np.ndarray | list, 
    y: np.ndarray | list, 
    z: np.ndarray | list, 
    title: str = '2D Dispersion Pseudo-Section', 
    cmap: str = 'turbo', 
    y_max: float | None = None
) -> None:
    """
    Plots a scatter pseudo-section of dispersion data.

    :param x: Array of x-coordinates (e.g., Virtual Shot Distance).
    :type x: array_like
    :param y: Array of y-coordinates (e.g., Frequency).
    :type y: array_like
    :param z: Array of z-values used for colormapping (e.g., Phase Velocity).
    :type z: array_like
    :param title: Title of the plot. Default is '2D Dispersion Pseudo-Section'.
    :type title: str, optional
    :param cmap: Matplotlib colormap to use. Default is 'turbo'.
    :type cmap: str, optional
    :param y_max: Maximum value for the y-axis, used to lock limits. Default is None.
    :type y_max: float | None, optional
    :returns: None
    """
    plt.figure(figsize=(12, 6))
    
    sc = plt.scatter(
        x, y, c=z, 
        cmap=cmap, 
        s=30, 
        edgecolors='black', 
        linewidth=0.2, 
        alpha=0.9
    )

    # Formatting
    plt.colorbar(sc, label='Phase Velocity (m/s)')
    plt.xlabel('Virtual Shot Distance (m)')
    plt.ylabel('Frequency (Hz)')
    plt.title(title)

    # Axis handling: set bounds and invert to mimic a depth section
    if y_max is not None:
        plt.ylim(np.min(y), y_max)

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_interpolated_section(
    x: np.ndarray | list, 
    y: np.ndarray | list, 
    z: np.ndarray | list, 
    title: str = 'Interpolated 2D Dispersion Pseudo-Section', 
    cmap: str = 'turbo', 
    y_max: float | None = None, 
    grid_res: int = 200
) -> None:
    """
    Interpolates scattered dispersion data onto a regular grid and plots a filled contour map.

    :param x: Array of x-coordinates (e.g., Virtual Shot Distance).
    :type x: array_like
    :param y: Array of y-coordinates (e.g., Frequency).
    :type y: array_like
    :param z: Array of z-values to interpolate (e.g., Phase Velocity).
    :type z: array_like
    :param title: Title of the plot. Default is 'Interpolated 2D Dispersion Pseudo-Section'.
    :type title: str, optional
    :param cmap: Matplotlib colormap to use. Default is 'turbo'.
    :type cmap: str, optional
    :param y_max: Maximum value for the y-axis. Default is None.
    :type y_max: float | None, optional
    :param grid_res: Resolution of the interpolation grid (NxN points). Default is 200.
    :type grid_res: int, optional
    :returns: None
    """
    x_np, y_np, z_np = np.array(x), np.array(y), np.array(z)

    # 1. Define the regular grid
    xi = np.linspace(x_np.min(), x_np.max(), grid_res)
    yi = np.linspace(y_np.min(), y_np.max(), grid_res)
    xi_mesh, yi_mesh = np.meshgrid(xi, yi)

    # 2. Interpolate the data
    zi_mesh = griddata(
        points=(x_np, y_np), 
        values=z_np, 
        xi=(xi_mesh, yi_mesh), 
        method='linear'
    )

    # 3. Plotting
    plt.figure(figsize=(12, 6))
    
    contour = plt.contourf(
        xi_mesh, yi_mesh, zi_mesh, 
        levels=50, 
        cmap=cmap, 
        extend='both'
    )

    # Overlay constraints
    plt.scatter(
        x_np, y_np, 
        c='black', 
        s=5, 
        alpha=0.4, 
        label='Data Constraints'
    )

    # Formatting
    plt.colorbar(contour, label='Phase Velocity (m/s)')
    plt.xlabel('Virtual Shot Distance (m)')
    plt.ylabel('Frequency (Hz)')
    plt.title(title)

    # Axis handling
    if y_max is not None:
        plt.ylim(y_np.min(), y_max)

    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.3, color='white')
    plt.tight_layout()
    plt.show()

def plot_pcolormesh_section(
    x: np.ndarray | list, 
    y: np.ndarray | list, 
    z: np.ndarray | list, 
    title: str = 'Smooth 2D Dispersion Pseudo-Section', 
    cmap: str = 'turbo', 
    y_max: float | None = None, 
    grid_res: int = 200
) -> None:
    """
    Interpolates scattered dispersion data onto a regular grid and plots a smooth, continuous mesh.

    :param x: Array of x-coordinates (e.g., Virtual Shot Distance).
    :type x: array_like
    :param y: Array of y-coordinates (e.g., Frequency).
    :type y: array_like
    :param z: Array of z-values to interpolate (e.g., Phase Velocity).
    :type z: array_like
    :param title: Title of the plot. Default is 'Smooth 2D Dispersion Pseudo-Section'.
    :type title: str, optional
    :param cmap: Matplotlib colormap to use. Default is 'turbo'.
    :type cmap: str, optional
    :param y_max: Maximum value for the y-axis. Default is None.
    :type y_max: float | None, optional
    :param grid_res: Resolution of the interpolation grid (NxN points). Default is 200.
    :type grid_res: int, optional
    :returns: None
    """
    x_np, y_np, z_np = np.array(x), np.array(y), np.array(z)

    # 1. Define the regular grid
    xi = np.linspace(x_np.min(), x_np.max(), grid_res)
    yi = np.linspace(y_np.min(), y_np.max(), grid_res)
    xi_mesh, yi_mesh = np.meshgrid(xi, yi)

    # 2. Interpolate the data
    zi_mesh = griddata(
        points=(x_np, y_np), 
        values=z_np, 
        xi=(xi_mesh, yi_mesh), 
        method='linear'
    )

    # 3. Plotting
    plt.figure(figsize=(12, 6))
    
    # Use pcolormesh with gouraud shading for a continuous color gradient
    mesh = plt.pcolormesh(
        xi_mesh, yi_mesh, zi_mesh, 
        cmap=cmap, 
        shading='gouraud'
    )

    # Overlay constraints
    plt.scatter(
        x_np, y_np, 
        c='black', 
        s=5, 
        alpha=0.4, 
        label='Data Constraints'
    )

    # Formatting
    plt.colorbar(mesh, label='Phase Velocity (m/s)')
    plt.xlabel('Virtual Shot Distance (m)')
    plt.ylabel('Frequency (Hz)')
    plt.title(title)

    # Axis handling
    if y_max is not None:
        plt.ylim(y_np.min(), y_max)

    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.3, color='white')
    plt.tight_layout()
    plt.show()