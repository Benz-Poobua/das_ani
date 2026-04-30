"""
:module: src/npz_to_zarr.py
:author: Benz Poobua 
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Convert legacy DAS array formats (.npz) into high-performance, 
          chunk-aligned Zarr stores. Supports recursive directory traversal 
          for nested daily datasets (e.g., urban arrays) and safely handles 
          pickled metadata. Automatically aligns Zarr chunk dimensions to 
          match the cross-correlation segments defined in the master YAML config 
          to eliminate I/O bottlenecks during parallel processing.
"""
import numpy as np
import zarr
import logging
from pathlib import Path
from tqdm import tqdm
import argparse

from src.utils import load_config, get_cfg, timeit, write_runlog

# ==========================================
# 1. Logging Configuration
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("conversion_history.log"), 
        logging.StreamHandler()                        
    ]
)
logger = logging.getLogger(__name__)

def load_chunking_from_config(config_path: str | Path) -> int:
    """Reads the YAML config using src.utils to determine perfect chunk alignment."""
    try:
        cfg = load_config(config_path)
        
        fs = float(get_cfg(cfg, ['data', 'fs_raw'], required=True))
        xcorr_win_sec = float(get_cfg(cfg, ['xcorr', 'xcorr_seg_sec'], required=True))
        
        chunk_samples = int(xcorr_win_sec * fs)
        logger.info(f"Loaded config: fs={fs}Hz, win={xcorr_win_sec}s -> {chunk_samples} samples")
        return chunk_samples

    except Exception as e:
        logger.error(f"Failed to extract chunking from config {config_path}: {e}")
        return None

def calculate_smart_chunks(num_channels: int, target_mb: float = 25.0) -> int:
    """Fallback: Calculates temporal chunk size for a target MB."""
    bytes_per_sample = 4
    target_samples = (target_mb * 1024 * 1024) / (num_channels * bytes_per_sample)
    return int(max(1000, round(target_samples, -3)))

@timeit
def convert_folder(folder_name: str, config_path: str | None = None) -> None:
    
    # Path Resolution
    data_root = Path('data')
    input_dir = data_root / folder_name
    output_dir = data_root / f"{folder_name}_zarr"

    if not input_dir.exists():
        input_dir = Path('..') / 'data' / folder_name
        output_dir = Path('..') / 'data' / f"{folder_name}_zarr"

    if not input_dir.exists():
        logger.error(f"Directory not found: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = sorted(list(input_dir.rglob('*.npz')))

    if not files:
        logger.warning(f"No files found recursively in {input_dir}")
        return

    # ==========================================
    # 2. Determine Chunking Strategy
    # ==========================================
    try:
        with np.load(files[0], allow_pickle=True) as first_npz:
            num_chans, num_samples = first_npz['data'].shape
            dt = float(first_npz.get('dt', 1.0 / 10.0))
            dx = float(first_npz.get('dx', 10.0))
    except Exception as e:
        logger.error(f"Could not read metadata from {files[0]}: {e}")
        return

    time_chunk = None
    if config_path and Path(config_path).exists():
        logger.info(f"Attempting to align chunks using {config_path}")
        time_chunk = load_chunking_from_config(config_path)
    
    if time_chunk is None:
        logger.warning("No config found or read failed. Falling back to 25MB Smart Chunking.")
        time_chunk = calculate_smart_chunks(num_chans)

    chunk_shape = (num_chans, time_chunk)
    
    logger.info("="*50)
    logger.info(f"STARTING ZARR CONVERSION: {folder_name}")
    logger.info(f"Input Directory:  {input_dir}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Files to process: {len(files)}")
    logger.info(f"Aligned Chunk:    {chunk_shape}")
    logger.info("="*50)

    # ==========================================
    # 3. Conversion Loop
    # ==========================================
    success_count = 0
    total_bytes = 0

    for f_path in tqdm(files, desc=f"Converting", unit="file"):
        try:
            rel_path = f_path.relative_to(input_dir)
            target_path = output_dir / rel_path.with_suffix('.zarr')
            
            # Ensure the specific daily subfolder exists
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            with np.load(f_path, allow_pickle=True) as npz:
                data = npz['data'].astype(np.float32)
                total_bytes += data.nbytes

            root = zarr.group(store=str(target_path), overwrite=True)
            z_array = root.create_array(name='data', data=data, chunks=chunk_shape, overwrite=True)
            z_array.attrs['dt'] = dt
            z_array.attrs['dx'] = dx
            success_count += 1
            
        except Exception as e:
            logger.error(f"Failed to convert {f_path.parent.name}/{f_path.name}: {e}")

    processed_gb = total_bytes / (1024**3)
    summary_msg = f"CONVERSION COMPLETE: {success_count} files | {processed_gb:.2f} GB processed."
    logger.info(summary_msg)
    write_runlog(summary_msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DAS NPZ to Zarr.")
    parser.add_argument("folder", help="Input folder name (e.g., raw_offshore)")
    parser.add_argument("--config", help="Path to YAML config file", default=None)
    
    args = parser.parse_args()
    convert_folder(args.folder, config_path=args.config)

# Example
# python -m src.npz_to_zarr raw_bridge --config configs/bridge_cc.yaml