"""
:module: src/fake.py
:auth: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Generate synthetic DAS data using low‐rank rSVD only (no noise),
          saving .npz files compatible with the real DAS processing pipeline.
"""
import os
import glob
import logging
import torch
import numpy as np

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Device selection
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logger.info(f'Using device: {device}')

# Input DAS directory (real data)
input_dir = os.path.join('..', 'data', 'preprocessed', '20210901')
das_paths = sorted(glob.glob(os.path.join(input_dir, '*.npz')))

if not das_paths:
    logger.warning(f'No .npz files found in {input_dir}')

# Output directory base for synthetic data
script_dir = os.path.dirname(os.path.abspath(__file__))
out_base = os.path.normpath(os.path.join(script_dir, '..', 'data', 'synthetic'))
os.makedirs(out_base, exist_ok=True)

# Parameters
k_svd = 50             # target rank for low-rank reconstruction
seed = 1234            # reproducibility
np.random.seed(seed)
torch.manual_seed(seed)

# Randomized SVD
def rsvd_torch(A: torch.Tensor, k: int, n_oversample: int = 10, n_power_iter: int = 2):
    """
    Randomized SVD in PyTorch (GPU if available).
    Returns U (m×k), S (k,), Vh (k×n) such that A ≈ U diag(S) Vh.

    :param A: Input matrix (m×n)
    :type A: torch.Tensor
    :param k: Target rank
    :type k: int
    :param n_oversample: Oversampling parameter
    :type n_oversample: int
    :param n_power_iter: Number of power iterations to improve accuracy
    :type n_power_iter: int
    """
    m, n = A.shape
    dtype = A.dtype
    dev = A.device

    # Step 1: random test matrix
    P = torch.randn(n, k + n_oversample, device=dev, dtype=dtype)

    # Step 2: sample the column space
    Z = A @ P

    # Step 3: power iterations
    for _ in range(n_power_iter):
        Z = A @ (A.T @ Z)

    # Step 4: orthonormal basis
    Q, _ = torch.linalg.qr(Z, mode='reduced')

    # Step 5: small matrix
    B = Q.T @ A   # (k + r) × n

    # Step 6: SVD on small matrix
    Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)

    # Step 7: lift back to original space
    U = Q @ Ub

    # Step 8: truncate to rank k
    return U[:, :k], S[:k], Vh[:k, :]

# Main synthetic generation loop
for path in das_paths:
    # Load real DAS file
    try:
        data = np.load(path)
    except Exception as e:
        logger.error(f'Could not load {path}: {e}')
        continue

    if 'data' not in data:
        logger.error(f"Missing key 'data' in {path}, skipping.")
        continue

    A_np = data['data'].astype(np.float32)
    dt = float(data.get("dt", 0.004))

    m, n = A_np.shape
    logger.info(f'Loaded {path} → shape {A_np.shape}, dt={dt}')

    # Use FULL record
    # A_np is already (nch × npts_full)

    # Convert to torch
    A = torch.from_numpy(A_np).to(device)

    # If k_svd > min(m, n), cap it to the valid range
    k_eff = min(k_svd, m, n)
    if k_eff < k_svd:
        logger.warning(
            f'k_svd={k_svd} is larger than min(m, n)={min(m, n)} for {path}. '
            f'Using k={k_eff} instead.'
        )

    U, S, Vh = rsvd_torch(A, k=k_eff)
    A_lr = U @ torch.diag(S) @ Vh
    A_synth = A_lr.cpu().numpy().astype(np.float32)

    # Output directory per day
    basename = os.path.basename(path)
    datestamp = basename.split('_')[0]     # e.g., '20210901'
    out_dir = os.path.join(out_base, datestamp)
    os.makedirs(out_dir, exist_ok=True)

    out_name = basename.replace('.npz', '_fake.npz')
    out_path = os.path.join(out_dir, out_name)

    # Save in .npz format expected by cc.py
    # - 'data': synthetic DAS array (nch × npts_full)
    # - 'dt'  : sampling interval
    np.savez_compressed(
        out_path,
        data=A_synth,
        dt=dt,
        start_time='synthetic_lowrank_full',
        comment=f'Pure low-rank synthetic (rank={k_eff})'
    )

    logger.info(f'Saved synthetic DAS → {out_path} (shape={A_synth.shape})')

logger.info('All synthetic DAS generation completed (no noise, full record).')