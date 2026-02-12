# Sherlock HPC Setup Guide
DAS Ambient Noise Interferometry (ANI) Framework  

Author: Benz Poobua  
Platform: Stanford Sherlock HPC  
Location: /home/groups/biondo/spoobua/das_ani  

---

# 1. Overview
This document describes the reproducible setup for running the DAS ANI framework on Sherlock HPC.

The project uses Sherlock's module system (NOT pip-installed scientific stack) to avoid:

- GLIBCXX runtime errors
- NumPy ABI mismatches
- Torch dependency conflicts
- CUDA misconfiguration
- Mixed pip/module environments

---

# 2. Project Directory Architecture (HPC Best Practice)
We separate code and heavy data storage.

## Code (stable storage)

```bash
/home/groups/biondo/spoobua/das_ani
```

## Data (high-performance scratch)

```bash
/scratch/groups/biondo/spoobua/das_ani/data
```
Inside the repo, `data/` is a symbolic link:

```bash
data -> /scratch/groups/biondo/spoobua/das_ani/data
```
This keeps:

- Repository clean
- I/O fast
- GROUP_HOME quota safe
- HOME quota untouched

GROUP_SCRATCH is purged 90 days after last modification.

---

# 3. Where To Run Code
## Method 1: Run Directly From GROUP_HOME

```bash
cd /home/groups/biondo/spoobua/das_ani
export PYTHONPATH="$PWD:$PYTHONPATH"
```

## Method 2: Use a Symlink in HOME

```bash
cd /home/users/spoobua
ln -s /home/groups/biondo/spoobua/das_ani das_ani
```

# 4. Required Module Stack (CPU Version)
Run this at every login or inside every Slurm job:

```bash
module reset
module load devel math
module load python/3.12.1
module load py-pandas/2.2.1_py312
module load py-pytorch/2.4.1_py312
module load py-scikit-image/0.24.0_py312
```
Always match `_py312` suffix with Python 3.12.

Do NOT load default module versions.

---

# 5. Add Project to PYTHONPATH
From repo root:

```bash 
cd /home/groups/biondo/spoobua/das_ani
export PYTHONPATH="$PWD:$PYTHONPATH"
```
This enables:

```python 
from src.cc import process_single_file
```

---

# 6. Verify Environment

```bash
python3 -c "import torch, numpy, pandas, skimage; \
print('Python:', __import__('sys').version.split()[0]); \
print('NumPy:', numpy.__version__); \
print('Pandas:', pandas.__version__); \
print('Torch:', torch.__version__); \
print('CUDA available:', torch.cuda.is_available()); \
print('scikit-image:', skimage.__version__)"
```

---

# 7. Running Benchmarks

```bash
python3 src/eval_robustness.py \
  --cc_config configs/cc.yaml \
  --outdir data/benchmarks/final \
  --n_files 16 \
  --repeats 1 \
  --cores 1 2 4 8 16 \
  --window_sec 60 \
  --njobs_complexity 16 \
  --lags 0.5 1 2 4 5 10 20
```

---

# 8. Slurm Script Template (CPU)
```bash
#!/bin/bash
#SBATCH --job-name=ani
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

module reset
module load devel math
module load python/3.12.1
module load py-pandas/2.2.1_py312
module load py-pytorch/2.4.1_py312
module load py-scikit-image/0.24.0_py312

cd /home/groups/biondo/spoobua/das_ani
export PYTHONPATH="$PWD:$PYTHONPATH"

python3 src/eval_robustness.py ...
```

---

# 9. Storage & Quota Monitoring
Check disk usage:

```bash
sh_quota
```
|Directory|Purpose|
|-|-|
|HOME|Small (15GB), avoid large files|
|GROUP_HOME|Long-lived shared storage|
|SCRATCH|Temporary user scratch|
|GROUP_SCRATCH|Large project scratch (purged after 90 days)|

## Inspect Group Home
```bash
cd $GROUP_HOME
ls
```

## Inspect Scratch Paths
```bash
echo $SCRATCH
echo $GROUP_SCRATCH
```

---

# 10. Job Monitoring
Check running jobs:

```bash
squeue -u spoobua
```

---

# 11. Partition Overview
```bash 
sh_part
```

---

# 12. Data Storage Strategy (GROUP_SCRATCH Symlink Architecture)

- Code lives in GROUP_HOME

- Large data lives in GROUP_SCRATCH

- The repository keeps a symlink to scratch

## Current Layout
Repository location:

```bash
/home/groups/biondo/spoobua/das_ani
```

Data physical location:
```bash
/scratch/groups/biondo/spoobua/das_ani/data
```

Inside the repository:
```bash
data -> /scratch/groups/biondo/spoobua/das_ani/data
```
This allows the code to use `data/` normally while storing large files on high-performance scratch.

## How To Recreate The Symlink (If Needed)
If the repository is copied or the symlink is broken:

```bash
cd /home/groups/biondo/spoobua/das_ani
mkdir -p /scratch/groups/biondo/spoobua/das_ani
mv data /scratch/groups/biondo/spoobua/das_ani/
ln -s /scratch/groups/biondo/spoobua/das_ani/data data
```

Verify:
```bash
ls -l
```

We should see:
```bash
data -> /scratch/groups/biondo/spoobua/das_ani/data
```

## How To Safe Final Results
```bash
mkdir -p /home/groups/biondo/spoobua/das_ani_results
cp -r data/benchmarks/final /home/groups/biondo/spoobua/das_ani_results/
```

---

# 13. Checking Available Modules and Versions
Sherlock uses the Lmod module system.

## List Available Modules
To search for a specific package (example: PyTorch):

```bash
# Check available versions
module avail py-pytorch

# Load matching version
module load py-pytorch/2.4.1_py312

# Verify
python3 -c "import torch; print(torch.__version__)"
```

## Check What Is Currently Loaded
```bash 
module list
```

## Inspect Python Version
```bash 
python3 --version
```












