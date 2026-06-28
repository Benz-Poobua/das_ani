#!/bin/bash
#SBATCH --job-name=ani_eval_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out

# Benchmark suite (GPU): preprocessing backends + lag sweep (no scaling).
#
# Partition / resources:
#   --partition=gpu, --gres=gpu:1 : request the physical GPU
#   --cpus-per-task=1             : one core is enough to feed a V100
#
# Config must be the _gpu.yaml (use_gpu: true, whiten_chunk_nch: 4096,
# preprocess.mode: hybrid). The preprocess experiment then compares, on the
# GPU device:
#   pure_numpy : all-CPU benchmark (reference)
#   hybrid     : scipy filters on CPU -> CUDA median/temporal-norm
#   pure_torch : fully CUDA-resident chain incl. rFFT-domain bandpass
#
# njobs_complexity stays at 1 to prevent device-lock crashes under SLURM
# Exclusive-Process mode.
#
# NOTE: keep this command free of inline comments -- a `#` after a
# line-continuation backslash breaks the command in bash.

# Always start from repo root
cd /home/groups/biondo/spoobua/das_ani

# Load environment
source sherlock_setup_gpu.sh

python3 -m src.eval \
  --cc_config configs/urban_cc_gpu.yaml \
  --outdir data/benchmarks/urban_gpu \
  --n_files 16 \
  --repeats 4 \
  --skip_scaling \
  --window_sec 60 \
  --njobs_complexity 1 \
  --lags 0.5 1 2 3 4 5 6 \
  --preprocess_modes hybrid pure_torch pure_numpy \
  --cleanup
