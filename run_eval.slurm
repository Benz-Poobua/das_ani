#!/bin/bash
#SBATCH --job-name=ani_eval
#SBATCH --partition=serc
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --output=logs/%x_%j.out

# Always start from repo root
cd /home/groups/biondo/spoobua/das_ani

# Load environment
source sherlock_setup.sh

# Thread control is managed inside Python via _set_thread_env() in cc.py.
# Setting these here would be immediately overridden and is misleading.

# Run workload
python3 -m src.eval \
  --cc_config configs/offshore_cc.yaml \
  --outdir data/benchmarks/offshore \
  --n_files 16 \
  --repeats 4 \
  --cores 1 2 4 8 16 \
  --window_sec 204.8 \
  --njobs_complexity 1 \
  --lags 4 6 8 10 12 16 \
  --cleanup
