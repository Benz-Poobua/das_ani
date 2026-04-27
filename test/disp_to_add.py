# (Append these to the bottom of src/disp.py)

import numpy as np
from scipy.interpolate import interp1d
from typing import Tuple

def compute_dvv_from_dispersion(
    ref_freq: np.ndarray,
    ref_vel: np.ndarray,
    cur_freq: np.ndarray,
    cur_vel: np.ndarray
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
    # Create an interpolator for the current curve.
    # bounds_error=False and fill_value=np.nan ensure we don't extrapolate garbage
    # if the current curve picked fewer frequencies than the reference curve.
    cur_interp_func = interp1d(
        cur_freq, cur_vel, 
        kind='linear', 
        bounds_error=False, 
        fill_value=np.nan
    )
    
    # Interpolate current velocities exactly at the reference frequencies
    cur_vel_interp = cur_interp_func(ref_freq)
    
    # Calculate dv/v
    dvv = (cur_vel_interp - ref_vel) / ref_vel
    
    # Mask out NaN values where interpolation fell out of bounds
    valid_mask = ~np.isnan(dvv)
    
    return ref_freq[valid_mask], dvv[valid_mask]


def approximate_depth_profile(
    freqs: np.ndarray, 
    velocities: np.ndarray, 
    factor: float = 0.333
) -> np.ndarray:
    """
    Approximates the penetration depth of surface waves based on frequency and velocity.
    Standard Seismological Rule of Thumb: Depth ~ Wavelength / 3.
    
    :param freqs: Array of frequencies (Hz).
    :param velocities: Array of phase velocities (m/s).
    :param factor: The depth-to-wavelength scaling factor (0.333 for Scholte/Rayleigh).
    :return: Array of approximate depths (meters).
    """
    # Prevent division by zero if freq=0 is accidentally passed
    safe_freqs = np.where(freqs == 0, 1e-9, freqs)
    
    # Wavelength = Velocity / Frequency
    wavelengths = velocities / safe_freqs
    
    # Z = Wavelength * Factor
    depths = wavelengths * factor
    
    return depths


# =====================================================
# Time-Lapse Dispersion Wrapper (Tool 2)
# =====================================================
import re
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import Any, Mapping

# Assuming get_cfg and timeit are imported at the top of disp.py
from src.utils import timeit, get_cfg, load_config
import logging

logger = logging.getLogger(__name__)
_DISP_FILE_RE = re.compile(r"(\d{8})(?:_(\d{6}))?")

def _extract_datetime(filename: str) -> datetime:
    m = _DISP_FILE_RE.search(filename)
    if not m:
        raise ValueError(f"Could not parse datetime from filename: {filename}")
    date_str = m.group(1)
    time_str = m.group(2) if m.group(2) else "000000"
    return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")

@timeit
def run_timelapse_dispersion(cfg: Mapping[str, Any]) -> None:
    """
    Chronologically tracks dispersion curve changes and maps them to depth.
    Outputs a 2D Heatmap of Time vs. Depth for a specific channel or stacked array.
    """
    # 1. Configuration & Paths
    data_root = Path(get_cfg(cfg, ["paths", "stacks_root"], required=True)).expanduser()
    out_root = Path(get_cfg(cfg, ["paths", "disp_timelapse_root"], "./data/disp_timelapse")).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    
    stack_target = str(get_cfg(cfg, ["dvv", "stack_target"], "1h"))
    search_dir = data_root / stack_target
    
    if not search_dir.exists():
        logger.error(f"Directory not found: {search_dir}")
        return

    # 2. Gather Chronological Files
    # For dispersion, we typically need Cross-Correlation (cc) files
    all_files = sorted(search_dir.rglob("*_cc_*.npy"), key=lambda x: _extract_datetime(x.name))
    
    if not all_files:
        logger.error(f"No _cc_ files found in {search_dir}")
        return
        
    n_steps = len(all_files)
    logger.info(f"Found {n_steps} CC files for Time-Lapse Dispersion.")

    # 3. Define the Fixed Depth Grid (for the final Heatmap)
    max_depth = float(get_cfg(cfg, ["dispersion", "max_depth_m"], 50.0))
    dz = float(get_cfg(cfg, ["dispersion", "depth_step_m"], 0.5))
    z_grid = np.arange(0, max_depth + dz, dz)
    
    # Initialize the Heatmap Matrix: Shape (Time, Depth)
    dvv_depth_heatmap = np.full((n_steps, len(z_grid)), np.nan, dtype=np.float32)
    timestamps = []

    # 4. Extract Reference Curve (Hour 1)
    # NOTE: Replace `extract_dispersion_curve` with your actual f-k/AutoSearch caller
    logger.info("Extracting Reference Dispersion Curve...")
    ref_data = np.load(all_files[0])
    
    # pseudo-code: ref_freq, ref_vel = your_existing_fk_picker(ref_data, cfg)
    # ref_freq, ref_vel = extract_dispersion_curve(ref_data, cfg) 
    
    timestamps.append(_extract_datetime(all_files[0].name).timestamp())

    # 5. The Time-Lapse Loop
    for t in tqdm(range(1, n_steps), desc="Tracking Freezing Front"):
        f_path = all_files[t]
        timestamps.append(_extract_datetime(f_path.name).timestamp())
        
        cur_data = np.load(f_path)
        
        try:
            # A. Pick the current curve
            # pseudo-code: cur_freq, cur_vel = your_existing_fk_picker(cur_data, cfg)
            # cur_freq, cur_vel = extract_dispersion_curve(cur_data, cfg)
            
            # B. Calculate strictly matched dv/v (From Batch 1)
            valid_freqs, valid_dvv = compute_dvv_from_dispersion(
                ref_freq, ref_vel, cur_freq, cur_vel
            )
            
            # C. Map the valid frequencies to Depth (From Batch 1)
            # We use the REFERENCE velocity for depth to keep the physical frame stable
            # Interp1d again to get the velocities at `valid_freqs`
            stable_vels = interp1d(ref_freq, ref_vel)(valid_freqs)
            dynamic_depths = approximate_depth_profile(valid_freqs, stable_vels)
            
            # D. Interpolate the scattered (Depth, dv/v) points onto the Rigid z_grid
            # We drop points that fall outside our picked depth range
            depth_interp = interp1d(
                dynamic_depths, valid_dvv, 
                kind='linear', bounds_error=False, fill_value=np.nan
            )
            
            dvv_depth_heatmap[t, :] = depth_interp(z_grid)
            
        except Exception as e:
            logger.warning(f"Failed to pick/track curve for {f_path.name}: {e}")

    # 6. Save the 2D Matrix
    time_str = datetime.now().strftime("%Y%m%d_%H%M")
    out_file = out_root / f"disp_timelapse_{stack_target}_{time_str}.npz"
    
    np.savez(
        out_file,
        dvv_heatmap=dvv_depth_heatmap,  # Shape: (Time, Depth)
        z_grid=z_grid,
        timestamps=np.array(timestamps)
    )
    logger.info(f"Time-Lapse Depth Matrix saved to {out_file}")