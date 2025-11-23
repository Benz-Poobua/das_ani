"""
:module: src/fake.py
:auth: Benz Poobua 
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: This script generates synthetic DAS data (low‐rank + noise) for testing. It also serves as a prototype.
"""
import os
import glob
import torch
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Choose device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logger.info(f'Using device: {device}')

# Input DAS files 
input_dir = os.path.join('..', 'data', 'preprocessed', '20210901')
das_paths = sorted(glob.glob(os.path.join(input_dir, '*.npz')))


# Output base folder (data subfoler will be created)
script_dir = os.path.dirname(os.path.abspath(__file__))
outpath_base = os.path.normpath(os.path.join(script_dir, '..', 'data', 'synthetic'))
os.makedirs(outpath_base, exist_ok=True)

# Choose rank
k_svd = 50

def rsvd_torch(A: torch.Tensor, k: int, n_oversample: int = 10, n_power_iter: int = 2):
    """
    Randomized SVD in PyTorch (GPU‐accelerated if available).
    Returns U (m×k), S (k,), Vh (k×n) such that A ≈ U diag(S) Vh.
    """
    m, n = A.shape
    dtype = A.dtype
    device = A.device

    # Step 1: random test matrix
    P = torch.randn(n, k + n_oversample, device=device, dtype=dtype)

    # Step 2: sample columns
    Z = A @ P

    # Step 3: power iterations (improve accuracy)
    for _ in range(n_power_iter):
        Z = A @ (A.T @ Z)

    # Step 4: orthonormal basis Q
    Q, _ = torch.linalg.qr(Z, mode='reduced')

    # Step 5: project into smaller space
    B = Q.T @ A  # shape (k + n_oversample) × n

    # Step 6: full SVD on smaller matrix
    Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)

    # Step 7: lift U back
    U = Q @ Ub

    # Step 8: truncate to k
    return U[:, :k], S[:k], Vh[:k, :]

for path in das_paths:
    try: 
        data = np.load(path)
    except Exception as e:
        logger.error(f'Failed loading {path}: {e}')
        continue

    if 'data' not in data:
        logger.error(f"No key 'data' in {path}")
        continue

    das_array = data['data']
    dt = data.get('dt', None)
    if dt is None:
        logger.warning(f'No dt in {path}; assume default dt=0.004s')
        dt = 0.004

    # Select first 60 seconds of data
    samples_test = int(60.0 / dt)
    if das_array.shape[1] < samples_test:
        samples_test = das_array.shape[1]

    A_np = das_array[:, :samples_test].astype(np.float32)
    m, n = A_np.shape
    logger.info(f'Loaded {path} → shape {A_np.shape}')

    # Convert to torch tensor
    A = torch.from_numpy(A_np).to(device)

    # Compute low-rank approx via rSVD
    Ur, Sr, Vhr = rsvd_torch(A, k_svd)
    logger.info(f'rSVD shapes: U={Ur.shape}, S={Sr.shape}, Vh={Vhr.shape}')

    # Reconstruct synthetic
    A_rec = Ur @ torch.diag(Sr) @ Vhr

    # Add Gaussian noise
    noise_level = 0.05   # 5% noise relative to standard deviation
    A_rec_cpu = A_rec.cpu().numpy()
    noise = noise_level * np.std(A_rec_cpu) * np.random.randn(*A_rec_cpu.shape).astype(np.float32)
    A_synth = A_rec_cpu + noise 

    # Save synthetic array
    basename = os.path.basename(path)
    data_str = basename.split('_')[0]       # e.g., '20210901
    output_subdir = os.path.join(outpath_base, data_str)
    os.makedirs(output_subdir, exist_ok=True)
    out_name = basename.replace('.npz', '_fake.npy')
    out_path = os.path.join(output_subdir, out_name)
    np.save(out_path, A_synth)
    logger.info(f'Saved synthetic DAS to {out_path}')

logger.info('All done')