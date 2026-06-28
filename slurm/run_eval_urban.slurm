#!/bin/bash
#SBATCH --job-name=ani_eval
#SBATCH --partition=serc
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out

# Benchmark suite (CPU): preprocessing backends + strong scaling + lag sweep.
#
# The preprocess experiment times the three backends on the golden subset and
# reports fidelity vs the pure_numpy benchmark:
#   pure_numpy : legacy all-CPU chain (correctness reference)
#   hybrid     : scipy detrend/bandpass + torch median/temporal-norm
#   pure_torch : all-torch chain incl. rFFT-domain bandpass (torch CPU here)
# Skip it with --skip_preprocess; restrict modes with --preprocess_modes.

# Always start from repo root
cd /home/groups/biondo/spoobua/das_ani

# Load environment
source sherlock_setup.sh

# Thread control is managed inside Python via _set_thread_env() in cc.py.
# Setting OMP/MKL variables here would be immediately overridden.

# Run workload
python3 -m src.eval \
  --cc_config configs/urban_cc.yaml \
  --outdir data/benchmarks/urban \
  --n_files 16 \
  --repeats 4 \
  --cores 1 2 4 8 16 \
  --window_sec 60 \
  --njobs_complexity 1 \
  --lags 0.5 1 2 3 4 5 6 \
  --preprocess_modes hybrid pure_torch pure_numpy \
  --cleanup
