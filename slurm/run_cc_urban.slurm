#!/bin/bash
#SBATCH --job-name=ani_cc
#SBATCH --partition=serc
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out

# Production cross-correlation (CPU).
#
# Reads configs/urban_cc.yaml:
#   runtime.njobs        : worker processes (file-level parallelism). Threads
#                          per worker are derived from SLURM_CPUS_PER_TASK
#                          inside cc.py (_set_thread_env), so raise njobs in
#                          the YAML to use the 16 cores requested here.
#   preprocess.mode      : pure_numpy (benchmark) | hybrid | pure_torch
#   xcorr.mode           : v1 (block-wise short-lag) | conventional
#
# Outputs one VSG per (file, virtual source):
#   data/ncf_raw_urban_test/<input_basename>_cc_<vs:03d>_<mode>.npy
# with shape (nch, 2*M+1), M = round(max_lag_sec * fs_proc). Auto-resume
# state is kept in <input_basename>_cc_state_<mode>.json next to the VSGs.

# Always start from repo root
cd /home/groups/biondo/spoobua/das_ani

# Load environment
source sherlock_setup.sh

# Thread control is managed inside Python via _set_thread_env() in cc.py.

# 1. Cross-correlation
python3 -m src.cc --config configs/urban_cc.yaml --verbose

# 2. Stacking (controlled by the stacking: block of the same YAML)
python3 -m src.stack --config configs/urban_cc.yaml --verbose
