#!/bin/bash
#SBATCH --job-name=ani_cc_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out

# Production cross-correlation (GPU).
#
# Reads configs/urban_cc_gpu.yaml (runtime.use_gpu: true):
#   preprocess.mode: hybrid     -> scipy detrend/bandpass on CPU, single H2D
#                                  transfer, CUDA median/temporal-norm.
#   preprocess.mode: pure_torch -> fully CUDA-resident preprocessing
#                                  (fastest; rFFT-domain bandpass).
#
# Keep runtime.njobs: 1 on a single GPU -- VS mode does not parallelize
# across nn.DataParallel; for multi-GPU runs launch separate SLURM tasks
# pinned to different GPUs via CUDA_VISIBLE_DEVICES.
#
# Output naming/shape identical to the CPU script:
#   <input_basename>_cc_<vs:03d>_<mode>.npy, shape (nch, 2*M+1).

# Always start from repo root
cd /home/groups/biondo/spoobua/das_ani

# Load environment
source sherlock_setup_gpu.sh

# 1. Cross-correlation
python3 -m src.cc --config configs/urban_cc_gpu.yaml --verbose

# 2. Stacking (CPU-bound; controlled by the stacking: block of the YAML)
python3 -m src.stack --config configs/urban_cc_gpu.yaml --verbose
