#!/bin/bash
#SBATCH --job-name=ani_eval
#SBATCH --partition=serc
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out

# Always start from repo root
cd /home/groups/biondo/spoobua/das_ani

# Load environment
source sherlock_setup.sh

# Avoid BLAS/OpenMP oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Run workload
python3 -m src.eval_robustness \
  --cc_config configs/bridge_cc.yaml \
  --outdir data/benchmarks/bridge \
  --n_files 16 \
  --repeats 4 \
  --cores 1 2 4 8 16 \
  --window_sec 10 \
  --njobs_complexity 16 \
  --lags 0.1 0.5 1 2 
  # --auto_cc