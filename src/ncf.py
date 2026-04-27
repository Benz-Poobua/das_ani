"""
:module: src/ncf.py
:author: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: NCF processing utilities for dispersion imaging (f–v panels),
          dispersion curve picks, and related DAS diagnostics.
"""
from __future__ import annotations

import os
import sys
import glob
from typing import List, Literal

import numpy as np
import dask
from tqdm.auto import tqdm
from scipy.signal.windows import tukey

sys.path.append('..')
from src.utils import parse_ncf_stack_filename, fk_filter, fk_transform
from src.disp import prep_ncf

# =============================================================================
# 1. Core Utilities & Spatial/Temporal Muting
# =============================================================================
def get_vs_number(filepath: str) -> int:
    """
    Extracts the Virtual Source integer from a standard NCF filename for proper sorting.
    
    :param filepath: Path to the NCF file.
    :returns: The integer value of the virtual source.
    """
    filename = os.path.basename(filepath)
    return int(filename.split('_')[2])

def apply_inner_spatial_mute(data: np.ndarray, offset: np.ndarray, cut_m: float = 100.0, taper_m: float = 50.0) -> np.ndarray:
    """
    Zeros out the near-offset traces and applies a smooth cosine taper to the transition 
    zone to prevent spectral ringing (Gibbs phenomenon) in the f-k domain.
    
    :param data: 2D numpy array of seismic data.
    :param offset: 1D array of spatial offsets.
    :param cut_m: The absolute distance (in meters) to hard-mute.
    :param taper_m: The distance (in meters) over which to fade the data back in.
    :returns: Muted and tapered 2D numpy array.
    """
    dist = np.abs(offset)
    weights = np.ones_like(dist)
    weights[dist <= cut_m] = 0.0
    
    transition = (dist > cut_m) & (dist < cut_m + taper_m)
    norm_dist = (dist[transition] - cut_m) / taper_m
    weights[transition] = 0.5 * (1 - np.cos(np.pi * norm_dist))
    
    return data * weights[:, np.newaxis]

def apply_time_distance_mute(
    data: np.ndarray, 
    offset: np.ndarray, 
    lag: np.ndarray, 
    vmin: float, 
    vmax_time: float,            
    buffer_start_s: float = 0.2, 
    buffer_end_s: float = 1.0, 
    top_flat_m: float = 100.0
) -> np.ndarray:
    """
    Zeros out data outside the physical travel-time cone.
    
    Uses independent buffers for the fast leading edge and the slow trailing coda 
    to preserve both the early arrivals and the ringing tail of the wave packets.

    :param data: 2D array containing the seismic data (e.g., cross-correlations).
    :param offset: 1D array of absolute or relative spatial offsets in meters.
    :param lag: 1D array of time lags in seconds.
    :param vmin: Minimum expected phase velocity in m/s (defines the trailing edge).
    :param vmax_time: Maximum group velocity in m/s (defines the leading edge).
    :param buffer_start_s: Time buffer in seconds to keep before the fastest wave arrival.
    :param buffer_end_s: Time buffer in seconds to keep after the slowest wave arrival.
    :param top_flat_m: Distance in meters from the source where the upper mute remains flat.
    :returns: A muted 2D array where data outside the calculated time-cone is set to zero.
    """
    nx, nt = data.shape
    abs_offset = np.abs(offset)
    mask = np.zeros_like(data)
    
    for i in range(nx):
        # 1. Top wedge (Fast leading edge) with a flat top
        # This keeps effective_x at 0 until we pass your threshold
        effective_x = max(0, abs_offset[i] - top_flat_m)
        t_start = max(0, (effective_x / vmax_time) - buffer_start_s)   
        
        # 2. Bottom wedge (Slow trailing coda)
        t_end = (abs_offset[i] / vmin) + buffer_end_s
        
        mask[i, (lag >= t_start) & (lag <= t_end)] = 1.0
        
    return data * mask

def apply_spatial_taper(data: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """
    Applies a Tukey (tapered cosine) window to the spatial axis of the data to gently 
    fade the far-offset edges to zero.
    
    :param data: 2D numpy array of seismic data.
    :param alpha: Fraction of the window inside the cosine tapered region (0 to 1).
    :returns: Tapered 2D numpy array.
    """
    if alpha <= 0:
        return data
    nx = data.shape[0]
    window = tukey(nx, alpha=alpha)
    return data * window[:, np.newaxis]

# =============================================================================
# 2. Batch Processing & Dask Pipelines
# =============================================================================
def process_single_file(
    path: str, 
    out_dir: str | None = None, 
    vmin: float = 150.0, 
    vmax: float = 2000.0, 
    vmax_time: float | None = None,
    fmax_plot: float = 10.0, 
    pos_offset: float = 100.0, 
    inner_taper: float = 50.0, 
    range_m: float = 4000.0, 
    sigma: float = 3.0, 
    buffer_start_s: float = 0.2, 
    buffer_end_s: float = 1.0, 
    top_flat_m: float = 100.0,
    max_lag: float | None = None
) -> tuple:    
    """
    Executes the golden processing pipeline on a single NCF file: F-K Filter -> 
    Inner Mute -> Time Cone Mute -> Far Taper. Optionally crops edge artifacts and saves the output.
    
    :param path: Path to the raw `.npz` NCF file.
    :param out_dir: Directory to save the polished `.npz` file. If None, it is not saved.
    :param vmin: Minimum phase velocity for the F-K filter and time mute.
    :param vmax: Maximum phase velocity for the F-K filter (Phase velocity).
    :param vmax_time: Maximum group velocity to use for the time mute. If None, it defaults to `vmax`.
    :param fmax_plot: Maximum frequency (Hz) to compute for outgoing f-k power plots.
    :param pos_offset: Near-offset distance (m) to hard mute.
    :param inner_taper: Distance (m) to fade the near-offset in smoothly.
    :param range_m: Maximum offset (m) to retain.
    :param sigma: Gaussian smoothing parameter for the F-K filter.
    :param buffer_start_s: Time buffer (s) to keep before the fastest arrival in the time mute.
    :param buffer_end_s: Time buffer (s) to keep after the slowest arrival in the time mute.
    :param top_flat_m: Distance in meters from the source where the upper mute remains flat.
    :param max_lag: Optional maximum absolute lag time (s) to retain. Physically crops the 
                    data arrays after processing to remove edge artifacts before saving and plotting.
    :returns: A tuple of (coords, t_data_list, fk_data_list) used for animation frames.
    """
    # 1. Handle decoupled velocity logic
    if vmax_time is None:
        vmax_time = vmax

    # 2. Load data
    archive = np.load(path)
    data_raw, lag, offset = archive['data'], archive['lag'], archive['offset']
    dt, dx = lag[1] - lag[0], np.abs(offset[1] - offset[0])

    # 3. Stage 1: F-K Filter (Uses standard vmax for Phase Velocity)
    data_fk = fk_filter(
        data_raw, dt=dt, dx=dx, vmin=vmin, vmax=vmax, 
        mode="extract", direction="both", smooth="gaussian", sigma=sigma
    )

    # 4. Stage 2: Cascaded Time and Spatial Mutes
    data_polish = data_fk.copy()
    dist = np.abs(offset)
    
    data_polish = apply_inner_spatial_mute(data_polish, offset, cut_m=pos_offset, taper_m=inner_taper)
    data_polish[dist > range_m, :] = 0.0
    
    # Use vmax_time for the time-domain Group Velocity mute
    data_polish = apply_time_distance_mute(
        data_polish, offset, lag, vmin, vmax_time, 
        buffer_start_s=buffer_start_s, buffer_end_s=buffer_end_s, top_flat_m=top_flat_m
    )
    
    data_polish = apply_spatial_taper(data_polish, alpha=0.1)

    # Crop Edge Artifacts 
    if max_lag is not None:
        # We use absolute lag to safely handle both causal (positive) and non-causal (two-sided) data
        lag_mask = np.abs(lag) <= max_lag
        lag = lag[lag_mask]
        data_raw = data_raw[:, lag_mask]
        data_fk = data_fk[:, lag_mask]
        data_polish = data_polish[:, lag_mask]

    # 5. Optional Saving (Now saves the artifact-free cropped data)
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.basename(path)
        name_base, ext = os.path.splitext(filename)
        new_filename = f"{name_base}_fk_{int(vmin)}_{int(vmax)}{ext}"
        save_path = os.path.join(out_dir, new_filename)
        
        np.savez_compressed(
            save_path, data=data_polish, lag=lag, offset=offset, vmin=vmin, vmax=vmax, pos_offset=pos_offset
        )

    # 6. Outgoing Power Helper (Now calculates clean FK spectra without edge ringing)
    def get_outgoing_power(data_matrix):
        f, k_raw, _ = fk_transform(data_matrix, dt, dx)
        k_ax = -k_raw
        idx = np.argsort(k_ax)
        k_ax_s = k_ax[idx]
        limit_f = min(fmax_plot, f.max())
        f_m = (f >= 0) & (f <= limit_f)
        f_p = f[f_m]
        p_out = np.zeros((len(k_ax_s), len(f_p)))
        
        for mask, side_sign in [(offset >= 0, 1), (offset <= 0, -1)]:
            if np.any(mask):
                padded = np.zeros_like(data_matrix)
                padded[mask, :] = data_matrix[mask, :]
                _, _, fk_side = fk_transform(padded, dt, dx)
                fk_side = fk_side[idx, :]
                k_m = (k_ax_s * side_sign) >= 0
                p_out[k_m, :] += np.abs(fk_side[k_m, :][:, f_m])
        return k_ax_s, f_p, p_out.T, limit_f

    # 7. Compute Plotting Matrices
    k_ax, f_p, p_raw, lim_f = get_outgoing_power(data_raw)
    _, _, p_fk, _ = get_outgoing_power(data_fk)
    _, _, p_polish, _ = get_outgoing_power(data_polish)

    return (lag, offset, dt, dx), (data_raw, data_fk, data_polish), (k_ax, f_p, p_raw, p_fk, p_polish, lim_f)


@dask.delayed
def split_ncf_sides_for_dispersion(path: str, out_dir: str) -> List[str] | None:
    """
    Dask-delayed function that loads pre-processed F-K data, splits it into positive (right) 
    and negative (left) spatial offsets, and saves them independently.
    
    :param path: Path to the pre-processed `.npz` file.
    :param out_dir: Output directory for the split files.
    :returns: A list of the saved file paths, or None if an error occurred.
    """
    try:
        archive = np.load(path)
        data = archive['data']      
        lag = archive['lag']
        offset = archive['offset']
        
        vmin = archive['vmin'] if 'vmin' in archive else None
        vmax = archive['vmax'] if 'vmax' in archive else None
        
        filename = os.path.basename(path)
        name_base, _ = os.path.splitext(filename)
        saved_paths = []

        # Process Right Side (+ Offsets)
        mask_right = offset > 0
        if np.sum(mask_right) > 5:
            save_path_r = os.path.join(out_dir, f"{name_base}_right.npz")
            np.savez_compressed(
                save_path_r, data=data[mask_right, :], lag=lag, offset=offset[mask_right], side="right", vmin=vmin, vmax=vmax
            )
            saved_paths.append(save_path_r)

        # Process Left Side (- Offsets)
        mask_left = offset < 0
        if np.sum(mask_left) > 5:
            save_path_l = os.path.join(out_dir, f"{name_base}_left.npz")
            np.savez_compressed(
                save_path_l, data=data[mask_left, :], lag=lag, offset=offset[mask_left], side="left", vmin=vmin, vmax=vmax
            )
            saved_paths.append(save_path_l)
            
        return saved_paths
        
    except Exception as e:
        print(f"Error splitting {os.path.basename(path)}: {e}")
        return None

def export_targeted_ncfs(
    pattern: str,
    out_dir: str,
    *,
    lag_axis: np.ndarray,
    distance_axis: np.ndarray,
    target: Literal["causal", "acausal", "s1", "s2"] = "s1",
    gauge_length: float = 10.0,
    range_m: float = 4000.0,
    view_side: Literal["both", "left", "right"] = "both",
    pos_offset: float = 0.0,
    vs_start: int | None = None,
    vs_end: int | None = None,
    max_lag: float | None = None,
    taper_alpha: float = 0.1, 
):
    """
    Extracts, crops, spatially tapers, and exports specific wavefield components 
    (e.g., directionally folded S1/S2 modes) from raw Noise Cross-Correlation Function (NCF) gathers.

    This function leverages `src.disp.prep_ncf` to isolate targeted wavefields. It then applies 
    strict temporal cropping, spatial windowing (including directional isolation and near-field 
    masking), and a spatial Tukey taper to reduce edge artifacts. The finalized arrays are saved 
    as compressed `.npz` archives ready for f-k analysis or dispersion extraction.

    :param pattern: Glob pattern matching the raw NCF files to be processed (e.g., "*.npy" or "*.npz").
    :param out_dir: Directory path where the processed `.npz` archives will be saved. Directory is created if it does not exist.
    :param lag_axis: 1D array of time lags (in seconds) corresponding to the raw data matrix.
    :param distance_axis: 1D array of spatial distances (in meters) along the array.
    :param target: The specific wavefield component to extract. Options: "causal", "acausal", 
                   "s1" (e.g., forward-propagating), or "s2" (e.g., backward-propagating). Default is "s1".
    :param gauge_length: Physical distance between adjacent channels (in meters) used to calculate the absolute virtual source position. Default is 10.0.
    :param range_m: Maximum spatial distance (in meters) to keep from the virtual source. Default is 4000.0.
    :param view_side: Determines which side of the spatial gather to export ("both", "left", or "right"). Default is "both".
    :param pos_offset: Spatial exclusion offset (in meters) from the virtual source. Data inside this distance is clipped out to mitigate near-field noise. Default is 0.0.
    :param vs_start: Optional minimum virtual source index to process. Allows for batching or restarting interrupted jobs.
    :param vs_end: Optional maximum virtual source index to process.
    :param max_lag: Optional maximum absolute lag time (seconds) to keep. Crops the temporal axis to speed up downstream computations.
    :param taper_alpha: Shape parameter (alpha) for the Tukey spatial taper applied to the outer edges of the final spatial window. Set to 0.0 to bypass tapering. Default is 0.1.
    """
    target = target.lower().strip()
    view_side = view_side.lower().strip()
    out_path_dir = os.path.abspath(out_dir)
    os.makedirs(out_path_dir, exist_ok=True)

    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    valid_files = []
    for p in paths:
        date, vs_str, window, xmode = parse_ncf_stack_filename(p)
        vs_idx = int(vs_str)
        if (vs_start is not None and vs_idx < vs_start) or (vs_end is not None and vs_idx > vs_end):
            continue
        valid_files.append((p, vs_str))
        
    if not valid_files:
        print("No files to process after applying vs_start/vs_end filters.")
        return

    for path, vs_str in tqdm(valid_files, desc=f"Saving {target.upper()} gathers"):
        ncf_raw = np.load(path)
        if ncf_raw.shape == (lag_axis.size, distance_axis.size):
            ncf_raw = ncf_raw.T 
            
        ncf_c, ncf_a, new_lag, s1, s2 = prep_ncf(ncf_raw, lag_axis, distance_axis, vs=vs_str, gauge_length=gauge_length)
        target_map = {"causal": ncf_c, "acausal": ncf_a, "s1": s1, "s2": s2}
        data = target_map[target]
        
        if max_lag is not None:
            time_mask = (new_lag >= 0) & (new_lag <= max_lag)
            data, final_lag = data[:, time_mask], new_lag[time_mask]
        else:
            final_lag = new_lag

        vs_pos_m = int(vs_str) * gauge_length
        offset_axis = distance_axis - vs_pos_m
        space_mask = (np.abs(offset_axis) <= range_m) & (np.abs(offset_axis) >= pos_offset)
        
        if view_side == "right": space_mask &= (offset_axis >= 0)
        elif view_side == "left": space_mask &= (offset_axis <= 0)
            
        data, final_offset = data[space_mask, :], offset_axis[space_mask]

        if taper_alpha > 0:
            data = apply_spatial_taper(data, alpha=taper_alpha)
        
        base_name = os.path.basename(path)
        name_no_ext = os.path.splitext(base_name)[0]
        out_filename = f"{name_no_ext}_{target}.npz"
        
        np.savez_compressed(
            os.path.join(out_path_dir, out_filename), data=data, lag=final_lag, offset=final_offset, taper_alpha=taper_alpha
        )

def fast_batch_processor(
    file_list: List[str],
    out_dir: str,
    vmin: float = 150.0,
    vmax: float = 2000.0,
    vmax_time: float | None = None,
    pos_offset: float = 100.0,
    inner_taper: float = 50.0,
    range_m: float = 4000.0,
    sigma: float = 1.0, 
    buffer_start_s: float = 0.2, 
    buffer_end_s: float = 1.5, 
    top_flat_m: float = 100.0,
    max_lag: float | None = None
) -> None:
    """
    High-speed, headless pipeline execution for Noise Correlation Functions (NCFs).
    
    Strips out all plotting logic to maximize processing speed. This function reads raw NCF 
    data, applies a Gaussian-smoothed f-k filter, cleans the arrays using a cascaded 
    series of spatial and temporal mutes/tapers, and optionally crops edge artifacts. 
    The resulting arrays are safely saved to disk.

    :param file_list: List of absolute or relative file paths to the raw `.npz` files.
    :param out_dir: Destination directory where the processed `.npz` files will be saved.
    :param vmin: Minimum expected phase velocity in m/s. Used for both the f-k filter 
                 boundaries and the time-distance travel cone mute. Defaults to 150.0.
    :param vmax: Maximum expected phase velocity in m/s (F-K Phase Velocity). Defaults to 2000.0.
    :param vmax_time: Maximum group velocity in m/s to use for the time-distance mute. If None, defaults to `vmax`.
    :param pos_offset: Absolute distance in meters from the virtual source to completely 
                       zero out (hard mute). Prevents near-offset artifacting. Defaults to 100.0.
    :param inner_taper: Distance in meters over which to smoothly fade the data back in 
                        after the `pos_offset` cut, preventing spectral ringing. Defaults to 50.0.
    :param range_m: Maximum absolute distance in meters to retain. Traces beyond this 
                    offset are completely zeroed out. Defaults to 4000.0.
    :param sigma: Standard deviation for the F-K filter's Gaussian smoothing. Higher 
                  values create smoother filter edges. Defaults to 1.0.
    :param buffer_start_s: Time buffer in seconds to keep before the fastest wave arrival in the time-distance mute. Defaults to 0.2s.
    :param buffer_end_s: Time buffer in seconds to keep after the slowest wave arrival in the time-distance mute. Defaults to 1.5s.
    :param top_flat_m: Distance in meters from the source where the upper mute remains flat, preserving early arrivals. Defaults to 100.0m.
    :param max_lag: Optional maximum absolute lag time (s) to retain. Physically crops the 
                    data arrays after processing to remove edge artifacts before saving.
    :returns: None. The function saves processed `.npz` files directly to `out_dir` with 
              the naming convention: `[original_name]_fk_[vmin]_[vmax].npz`.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Handle decoupled velocity logic
    if vmax_time is None:
        vmax_time = vmax
        
    for path in tqdm(file_list, desc="Processing & Saving NCFs"):
        try:
            # 2. Load Data
            archive = np.load(path)
            data = archive['data']   
            lag = archive['lag']
            offset = archive['offset']
            dt = lag[1] - lag[0]
            dx = np.abs(offset[1] - offset[0])
            
            # 3. Stage 1: F-K Filter (Uses Phase Velocity: vmax)
            data_clean = fk_filter(
                data, dt=dt, dx=dx, vmin=vmin, vmax=vmax,
                mode="extract", direction="both", smooth="gaussian", sigma=sigma
            )

            # 4. Stage 2: Cascaded Mutes and Tapers
            dist = np.abs(offset)
            data_clean = apply_inner_spatial_mute(data_clean, offset, cut_m=pos_offset, taper_m=inner_taper)
            data_clean[dist > range_m, :] = 0.0
            
            # Use Group Velocity (vmax_time) and Asymmetric Buffers for the Time Cone!
            data_clean = apply_time_distance_mute(
                data_clean, offset, lag, vmin, vmax_time, 
                buffer_start_s=buffer_start_s, buffer_end_s=buffer_end_s, top_flat_m=top_flat_m
            )
            data_clean = apply_spatial_taper(data_clean, alpha=0.1)

            # Crop Edge Artifacts 
            if max_lag is not None:
                lag_mask = np.abs(lag) <= max_lag
                lag = lag[lag_mask]
                data_clean = data_clean[:, lag_mask]

            # 5. Save with exact requested naming convention
            filename = os.path.basename(path)
            name_base, ext = os.path.splitext(filename)
            new_filename = f"{name_base}_fk_{int(vmin)}_{int(vmax)}{ext}"
            save_path = os.path.join(out_dir, new_filename)
            
            np.savez_compressed(
                save_path,
                data=data_clean,
                lag=lag,        
                offset=offset,
                vmin=vmin,
                vmax=vmax,
                vmax_time=vmax_time,   
                pos_offset=pos_offset
            )
            
        except Exception as e:
            print(f"\nError processing {os.path.basename(path)}: {e}")