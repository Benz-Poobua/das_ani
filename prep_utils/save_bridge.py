import os
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration
input_dir = Path("../data/raw_bridge_unready")
output_dir = Path("../data/raw_bridge")  
output_dir.mkdir(parents=True, exist_ok=True)

# Define Regex to extract metadata
bridge_re = re.compile(r"DAS_(\d{8})_(\d{6})_dx([\d\.]+)m_dt([\d\.]+)s\.npy")

def process_single_file(file_path):
    """Worker function to process and then delete a single DAS file."""
    try:
        # A. Parse the filename
        m = bridge_re.search(file_path.name)
        if m:
            date_str, time_str, dx_str, dt_str = m.groups()
            dx, dt = float(dx_str), float(dt_str)
        else:
            m_fallback = re.search(r"DAS_(\d{8})_(\d{6})", file_path.name)
            if not m_fallback:
                return f"Skipped {file_path.name}: Unrecognized format."
            date_str, time_str = m_fallback.groups()
            dx, dt = 2.45, 0.004  # Default bridge metadata
            
        # B. Load the array
        raw_data = np.load(file_path)
        
        # C. Standardize dimensions to (Channels x Time)
        if raw_data.shape[0] > raw_data.shape[1]:
            raw_data = raw_data.T
            
        # D. Define standard output name and save
        out_filename = output_dir / f"{date_str}_{time_str}_bridge.npz"
        
        np.savez_compressed(
            out_filename,
            data=raw_data.astype(np.float32),
            dt=dt,
            dx=dx,
            start_sample=0,
            end_sample=raw_data.shape[1]
        )
        
        # E. Safely delete the original file to conserve local storage
        file_path.unlink()
        
        return f"Success & Deleted: {file_path.name}"
        
    except Exception as e:
        return f"Error processing {file_path.name}: {e}"

def main():
    # Gather Files
    raw_files = sorted(input_dir.glob("*.npy"))
    print(f"Found {len(raw_files)} files ready for conversion.")

    if not raw_files:
        print("No files to process. Exiting.")
        return

    # Automatically determine the number of cores to use.
    max_workers = max(1, os.cpu_count() - 1)
    print(f"Spinning up a pool of {max_workers} workers...")

    # Run multiprocessing with the progress bar
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the pool
        futures = [executor.submit(process_single_file, fp) for fp in raw_files]
        
        # Wrap as_completed with tqdm to keep the progress bar updating properly
        for future in tqdm(as_completed(futures), total=len(raw_files), desc="Converting & Purging"):
            pass

    print("Batch conversion and cleanup successfully completed!")

if __name__ == "__main__":
    main()