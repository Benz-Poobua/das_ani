"""
:module: src/dvv.py
:author: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Coda Wave Interferometry (CWI) and Ballistic Phase-Shift engine.
          Calculates fractional velocity changes (dv/v) via GPU-accelerated 
          time-axis stretching.
"""
from __future__ import annotations

import os
import glob
import re
import gc
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import deque
from typing import Tuple, List, Optional, Union
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from tqdm import tqdm

from src.utils import timeit, load_config, get_cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class TorchTimeStretcher(nn.Module):
    """
    High-performance PyTorch module for time-lapse wave stretching.
    
    Replaces traditional CPU-bound SciPy loops by utilizing PyTorch's 
    grid_sample engine to interpolate hundreds of stretched reference traces 
    in a single batched GPU operation.
    """
    def __init__(
        self,
        min_stretch: float = -0.05,
        max_stretch: float = 0.05,
        num_stretches: int = 201,
        device: torch.device = torch.device("cpu")
    ):
        """
        :param min_stretch: Minimum stretch ratio (e.g., -0.05 for -5%).
        :param max_stretch: Maximum stretch ratio (e.g., +0.05 for +5%).
        :param num_stretches: Number of discrete steps in the stretch grid.
        :param device: Target compute device.
        """
        super().__init__()
        self.device = device
        self.num_stretches = int(num_stretches)
        
        # 1D vector of epsilon values (dv/v)
        self.epsilons = torch.linspace(
            min_stretch, max_stretch, steps=self.num_stretches, device=self.device
        )
        self.d_eps = float(self.epsilons[1] - self.epsilons[0])

    @torch.no_grad()
    def forward(self, ref: torch.Tensor, cur: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stretches the reference trace against the current trace and computes the optimal dv/v.
        
        :param ref: Reference waveform tensor of shape (L,).
        :param cur: Current waveform tensor of shape (L,).
        :return: (best_epsilon, max_correlation_coefficient)
        """
        if ref.shape != cur.shape:
            raise ValueError(f"Shape mismatch: ref {ref.shape} vs cur {cur.shape}")
        
        L = ref.shape[-1]
        
        # -----------------------------------------------------------
        # 1. Construct the 1D Batched Sampling Grid
        # -----------------------------------------------------------
        # PyTorch grid_sample maps coordinates from [-1, 1].
        # A time stretch t_new = t_old * (1 + eps) implies that to find the 
        # amplitude at t_new, we must sample the original signal at t_old = t_new / (1 + eps).
        
        base_grid = torch.linspace(-1, 1, steps=L, device=self.device) # (L,)
        
        # grid shape needed for 1D grid_sample: (Batch, H_out, W_out, 2) -> (N, 1, L, 2)
        grid = torch.zeros((self.num_stretches, 1, L, 2), device=self.device)
        
        # X-coordinates: scale the base grid by 1 / (1 + eps)
        # unsqueeze trickery aligns dimensions for broadcasting
        scaling_factors = 1.0 / (1.0 + self.epsilons)
        grid[..., 0] = base_grid.unsqueeze(0) * scaling_factors.unsqueeze(1)
        
        # Y-coordinates remain 0 (it's a 1D signal disguised as a 2D image row)
        grid[..., 1] = 0.0
        
        # -----------------------------------------------------------
        # 2. Execute GPU Image Warp (The Stretch)
        # -----------------------------------------------------------
        # Reshape ref to (Batch=1, Channels=1, Height=1, Width=L)
        ref_view = ref.view(1, 1, 1, L)
        
        # grid_sample automatically broadcasts the batch dimension of `ref_view` 
        # to match the `self.num_stretches` batch dimension of the `grid`.
        stretched_refs = F.grid_sample(
            ref_view.expand(self.num_stretches, -1, -1, -1), # Expand to avoid deprecation warnings
            grid,
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=True
        ).squeeze() # Output shape: (num_stretches, L)

        # -----------------------------------------------------------
        # 3. Vectorized Pearson Correlation
        # -----------------------------------------------------------
        # Zero-mean the vectors for pure shape correlation
        stretched_refs_zm = stretched_refs - stretched_refs.mean(dim=1, keepdim=True)
        cur_zm = cur - cur.mean()
        
        # Numerator: Dot product (Batch x L) @ (L)
        cov = torch.matmul(stretched_refs_zm, cur_zm)
        
        # Denominator: product of L2 norms
        norm_ref = torch.norm(stretched_refs_zm, dim=1)
        norm_cur = torch.norm(cur_zm)
        
        # Prevent division by zero
        denom = norm_ref * norm_cur + 1e-12
        cc_curve = cov / denom # Shape: (num_stretches,)
        
        # -----------------------------------------------------------
        # 4. Sub-grid Parabolic Interpolation
        # -----------------------------------------------------------
        best_idx = torch.argmax(cc_curve)
        max_cc = cc_curve[best_idx]
        
        if best_idx == 0 or best_idx == self.num_stretches - 1:
            # Hit the grid boundary, cannot interpolate
            return self.epsilons[best_idx], max_cc
            
        # Fit a parabola to the top 3 points for exact peak location
        y1 = cc_curve[best_idx - 1]
        y2 = cc_curve[best_idx]
        y3 = cc_curve[best_idx + 1]
        
        # Vertex of a parabola passing through 3 evenly spaced points
        # dx_peak is the fractional shift from the center point (-0.5 to +0.5)
        dx_peak = 0.5 * (y1 - y3) / (y1 - 2*y2 + y3 + 1e-12)
        
        best_epsilon = self.epsilons[best_idx] + (dx_peak * self.d_eps)
        
        return best_epsilon, max_cc
    
class DvVProcessor:
    """
    Manages the time-lapse processing of a single channel's Coda or Ballistic wave.
    
    Handles Reference State routing (fixed, step, rolling) and enforces 
    dynamic peak tracking to physically prevent cycle-skipping.
    """
    def __init__(
        self,
        stretcher: TorchTimeStretcher,
        mode: str = "step",
        rolling_window_steps: int = 5,
        max_jump_eps: float = 0.01, # e.g., max physical jump of 1% per time step
    ):
        """
        :param stretcher: Instantiated TorchTimeStretcher engine.
        :param mode: "fixed", "step", or "rolling".
        :param rolling_window_steps: Number of prior steps to average if mode="rolling".
        :param max_jump_eps: The maximum allowable dv/v change between consecutive time steps.
                             Acts as a strict physical guardrail against cycle-skipping.
        """
        valid_modes = {"fixed", "step", "rolling"}
        if mode.lower() not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}")
            
        self.stretcher = stretcher
        self.mode = mode.lower()
        self.rolling_window_steps = int(rolling_window_steps)
        self.max_jump_eps = float(max_jump_eps)

    def _dynamic_peak_tracker(
        self, 
        cc_curve: torch.Tensor, 
        prior_eps: float, 
        is_step_mode: bool
    ) -> Tuple[float, float]:
        """
        The Cycle-Skipping Guardrail.
        Masks out unrealistic stretch values before finding the peak correlation.
        """
        # If we are in 'step' mode, we are comparing T to T-1.
        # The expected eps is 0.0, and it shouldn't jump by more than max_jump_eps.
        # If 'fixed', we are comparing T to 0. The expected eps is the previous hour's total eps.
        center_eps = 0.0 if is_step_mode else prior_eps
        
        # Create a boolean mask: 1 if within physical limits, 0 if impossible
        valid_mask = torch.abs(self.stretcher.epsilons - center_eps) <= self.max_jump_eps
        
        # Apply mask: Set invalid stretches to -1.0 (worst possible correlation)
        masked_cc = torch.where(valid_mask, cc_curve, torch.tensor(-1.0, device=cc_curve.device))
        
        # Find the peak ONLY within the valid physical window
        best_idx = torch.argmax(masked_cc)
        max_cc = masked_cc[best_idx]
        
        if max_cc == -1.0:
            logger.warning(f"No valid correlation found within max_jump_eps limit ({self.max_jump_eps}).")
            return center_eps, 0.0
            
        # Standard Sub-grid Parabolic Interpolation (from Batch 1)
        if best_idx == 0 or best_idx == self.stretcher.num_stretches - 1:
            return float(self.stretcher.epsilons[best_idx].item()), float(max_cc.item())
            
        y1 = masked_cc[best_idx - 1]
        y2 = masked_cc[best_idx]
        y3 = masked_cc[best_idx + 1]
        
        dx_peak = 0.5 * (y1 - y3) / (y1 - 2*y2 + y3 + 1e-12)
        best_epsilon = self.stretcher.epsilons[best_idx] + (dx_peak * self.stretcher.d_eps)
        
        return float(best_epsilon.item()), float(max_cc.item())

    @torch.no_grad()
    def process_timeline(
        self, 
        time_series_data: torch.Tensor, 
        fixed_ref: Optional[torch.Tensor] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Processes a sequential matrix of waveforms for a single channel.
        
        :param time_series_data: Tensor of shape (n_time_steps, n_samples).
        :param fixed_ref: Optional pre-computed anchor trace. If None, uses index 0.
        :return: (dvv_history_array, cc_history_array)
        """
        n_steps = time_series_data.shape[0]
        dvv_history = np.zeros(n_steps, dtype=np.float32)
        cc_history = np.ones(n_steps, dtype=np.float32)
        
        if n_steps < 2:
            return dvv_history, cc_history
            
        # --- State Initialization ---
        cumulative_dvv = 0.0
        prior_tracked_eps = 0.0
        rolling_buffer = deque(maxlen=self.rolling_window_steps)
        
        if self.mode == "fixed":
            current_ref = fixed_ref if fixed_ref is not None else time_series_data[0]
        else:
            current_ref = time_series_data[0]
            rolling_buffer.append(current_ref)

        # --- The Time Loop ---
        for t in range(1, n_steps):
            cur_trace = time_series_data[t]
            
            # 1. Run the GPU Stretcher
            # (Note: Assumes TorchTimeStretcher returns the raw cc_curve tensor)
            cc_curve = self.stretcher(current_ref, cur_trace)
            
            # 2. Apply the Physical Guardrail
            step_eps, max_cc = self._dynamic_peak_tracker(
                cc_curve, 
                prior_eps=prior_tracked_eps, 
                is_step_mode=(self.mode != "fixed")
            )
            
            # 3. Update the Math based on Mode
            if self.mode == "fixed":
                cumulative_dvv = step_eps           # The stretch IS the total change
                prior_tracked_eps = step_eps        # Update tracker center
            elif self.mode == "step":
                cumulative_dvv += step_eps          # Integrate the tiny steps
                prior_tracked_eps = cumulative_dvv  # Not strictly used for step masking, but kept for state
            elif self.mode == "rolling":
                cumulative_dvv += step_eps          
                prior_tracked_eps = cumulative_dvv
            
            # 4. Save metrics (dv/v = -epsilon)
            dvv_history[t] = -cumulative_dvv
            cc_history[t] = max_cc
            
            # 5. Advance the Reference Buffer
            if self.mode == "step":
                current_ref = cur_trace
            elif self.mode == "rolling":
                rolling_buffer.append(cur_trace)
                current_ref = torch.stack(list(rolling_buffer)).mean(dim=0)

        return dvv_history, cc_history

# Regex to parse the datetime from our upgraded stack.py outputs
_DVV_FILE_RE = re.compile(r"(\d{8})(?:_(\d{6}))?")

def _extract_datetime(filename: str) -> datetime:
    m = _DVV_FILE_RE.search(filename)
    if not m:
        raise ValueError(f"Could not parse datetime from filename: {filename}")
    date_str = m.group(1)
    time_str = m.group(2) if m.group(2) else "000000"
    return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")

@timeit
def process_dvv(cfg: Mapping[str, Any]) -> None:
    """
    Main execution loop for dv/v processing.
    Loads stacked correlation files chronologically, extracts the target window,
    and computes the spatiotemporal dv/v heatmap.
    """
    # 1. Load Configurations
    data_root = Path(get_cfg(cfg, ["paths", "stacks_root"], required=True)).expanduser()
    out_root = Path(get_cfg(cfg, ["paths", "dvv_root"], "./data/dvv_results")).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    
    fs = float(get_cfg(cfg, ["data", "fs_raw"], required=True)) / float(get_cfg(cfg, ["preprocess", "decimation"], 1))
    
    # dvv specific params
    target_window = get_cfg(cfg, ["dvv", "coda_window_sec"], [2.0, 10.0])
    mode = str(get_cfg(cfg, ["dvv", "reference_mode"], "step")).lower()
    max_jump = float(get_cfg(cfg, ["dvv", "max_jump_eps"], 0.01))
    rolling_steps = int(get_cfg(cfg, ["dvv", "rolling_window_steps"], 5))
    
    # target stack directory (e.g., "1h", "1d", "7d")
    stack_target = str(get_cfg(cfg, ["dvv", "stack_target"], "1h"))
    search_dir = data_root / stack_target
    
    if not search_dir.exists():
        logger.error(f"Stack directory not found: {search_dir}")
        return

    # 2. Gather and Chronologically Sort Files
    # Assuming we are processing auto-correlations for CWI
    file_pattern = "*_auto_*.npy" 
    all_files = sorted(search_dir.rglob(file_pattern), key=lambda x: _extract_datetime(x.name))
    
    if not all_files:
        logger.error(f"No files matching {file_pattern} found in {search_dir}")
        return
        
    n_steps = len(all_files)
    logger.info(f"Found {n_steps} chronological files for dv/v processing. Mode: {mode}")

    # 3. Determine Coda Window Indices
    # Assuming the symmetric stacked array has center at index N
    # e.g., Shape = (n_channels, 2*N + 1)
    sample_arr = np.load(all_files[0], mmap_mode='r')
    n_channels, total_lags = sample_arr.shape
    center_idx = total_lags // 2
    
    idx_start = center_idx + int(target_window[0] * fs)
    idx_end = center_idx + int(target_window[1] * fs)
    n_coda_samples = idx_end - idx_start
    
    logger.info(f"Extracting window {target_window}s (Indices {idx_start} to {idx_end})")

    # 4. Initialize PyTorch Engines
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    stretcher = TorchTimeStretcher(
        min_stretch=float(get_cfg(cfg, ["dvv", "min_stretch"], -0.05)),
        max_stretch=float(get_cfg(cfg, ["dvv", "max_stretch"], 0.05)),
        num_stretches=int(get_cfg(cfg, ["dvv", "num_stretches"], 201)),
        device=device
    )
    
    processor = DvVProcessor(
        stretcher=stretcher,
        mode=mode,
        rolling_window_steps=rolling_steps,
        max_jump_eps=max_jump
    )

    # 5. Process in Memory Chunks (Channel by Channel or Batched)
    # To save RAM, we load the full timeline for a chunk of channels at once
    chunk_size = int(get_cfg(cfg, ["runtime", "dvv_chunk_size"], 100))
    
    dvv_heatmap = np.zeros((n_steps, n_channels), dtype=np.float32)
    cc_heatmap = np.zeros((n_steps, n_channels), dtype=np.float32)
    
    # Store timestamps for the output
    timestamps = [_extract_datetime(f.name).timestamp() for f in all_files]

    for ch_start in tqdm(range(0, n_channels, chunk_size), desc="Processing Channel Chunks"):
        ch_end = min(ch_start + chunk_size, n_channels)
        
        # Load timeline for this chunk
        # Shape: (n_steps, n_channels_in_chunk, n_coda_samples)
        chunk_timeline = np.zeros((n_steps, ch_end - ch_start, n_coda_samples), dtype=np.float32)
        
        for t, f_path in enumerate(all_files):
            arr = np.load(f_path, mmap_mode='r')
            chunk_timeline[t, :, :] = arr[ch_start:ch_end, idx_start:idx_end]
            
        # Convert chunk to tensor
        chunk_tensor = torch.from_numpy(chunk_timeline).to(device)
        
        # Process each channel in the chunk sequentially through the tracker
        for local_ch in range(ch_end - ch_start):
            global_ch = ch_start + local_ch
            
            # Extract 1D timeline for this specific channel: Shape (n_steps, n_coda_samples)
            ch_data = chunk_tensor[:, local_ch, :]
            
            dvv_history, cc_history = processor.process_timeline(ch_data)
            
            dvv_heatmap[:, global_ch] = dvv_history
            cc_heatmap[:, global_ch] = cc_history
            
        # Free GPU memory
        del chunk_tensor
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
    # 6. Save the 2D Heatmaps
    time_str = datetime.now().strftime("%Y%m%d_%H%M")
    out_file = out_root / f"dvv_heatmap_{mode}_{stack_target}_{time_str}.npz"
    
    np.savez(
        out_file,
        dvv=dvv_heatmap,         # Shape: (n_time_steps, n_channels)
        cc=cc_heatmap,           # Shape: (n_time_steps, n_channels)
        timestamps=np.array(timestamps),
        coda_window=np.array(target_window)
    )
    
    logger.info(f"Successfully saved 2D spatiotemporal dv/v heatmap to {out_file}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run PyTorch Coda Wave Interferometry (dv/v)")
    p.add_argument("--config", type=str, required=True, help="Path to config file (.yaml)")
    args = p.parse_args()
    
    config = load_config(args.config)
    process_dvv(config)