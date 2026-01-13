# DAS-ANI: Distributed Acoustic Sensing Preprocessing & Ambient Noise Interferometry Tools

## Overview

This repository provides a modular, configuration-driven framework for **Distributed Acoustic Sensing (DAS)** preprocessing and **Ambient Noise Interferometry (ANI)** workflows.

The primary goals are:

- Preprocessing large-scale DAS datasets

- Efficient computation of noise cross-correlations (NCFs)

- Temporal stacking of NCFs (daily, 7d, 15d, 30d)

- Dispersion imaging (f–v panels) and dispersion curve picking

- Exploration of data compression strategies (e.g., sketching, low-rank methods) and their impact on virtual source gathers and dispersion results

---

## Installation

### **Option 1: Install as editable package (pip)**
```bash
pip install -e .
```

### **Option 2: Conda environment (recommended)**
```bash
# Create environment 
conda env create -f environment.yml

# Activate 
conda activate das_ani-env
```
---

## Downloading DAS Data from Google Cloud

Large DAS datasets (e.g., preprocessed windows or NCF products) are typically hosted on Google Cloud Storage (GCS).

### **Requirements**
- `gsutil` installed
- Authenticated GCP account with read permission
  
Official documentation: https://cloud.google.com/storage/docs/gsutil

### Example
```bash
gsutil -m cp -n -r gs://path/to/data .

# or
gsutil -m \ -o "GSUtil:parallel_process_count=1" \ -o "GSUtil:parallel_thread_count=16" \ cp -r "gs://path/to/data" .
```
### Flag explanation
1. `gsutil`
Google Cloud Storage command-line tool.
2. `-m` (multi-threading)
Enables parallel transfers for faster downloads.
3. `cp`
Copy command (similar to Unix `cp`), works cloud ↔ local.
4. `-n` (no-clobber)
Skip files that already exist locally.
5. `-r` (recursive)
Copy entire folders.
6. `gs://path/to/data`
Source path inside a Google Cloud Storage bucket.
7. `.`
Destination = current directory.
---

## Repository Structure
```text
.
├── .gitignore
├── LICENSE
├── README.md
├── environment.yml
├── pyproject.toml
├── Makefile
├── configs/                     # YAML configuration files
│   ├── cc.yaml                  # Cross-correlation parameters
│   └── disp.yaml                # Dispersion + picking parameters
│
├── geometry/
│   └── .gitkeep                 # Geometry-related inputs (offsets, maps)
│
├── data/
│   ├── preprocessed/            # Preprocessed DAS time windows (.npz)
│   ├── ncf_raw/                 # Raw noise cross-correlations (.npy)
│   └── ncf_stacks/              # Stacked NCFs (.npy)
│       ├── daily/
│       ├── 7d/
│       ├── 15d/
│       └── 30d/
│
├── notebooks/
│   ├── geometry.ipynb        # Fiber geometry visualization
│   ├── processing.ipynb      # End-to-end DAS processing demo
│   ├── dispersion.ipynb      # Dispersion imaging & picking
│   └── kernel.ipynb          # Tensor Sketch / kernel demos
│
└── src/
    ├── utils.py                 # I/O, config helpers, diagnostics
    ├── ani.py                   # ANI preprocessing + correlation kernels
    ├── cc.py                    # Cross-correlation workflow
    ├── stack.py                 # NCF stacking (daily, multi-day)
    ├── disp.py                  # Dispersion imaging + picking algorithms
    ├── disp_pick.py             # Dispersion workflow (main script)
    └── plot.py                  # Plotting utilities
```
---
## Workflow Overview
All scripts are config-driven via YAML files in configs/.
You generally do not need to modify Python code for parameter changes.
### 1. Cross-correlation (NCF generation)
```bash
make cc_only
```
Produces:
```bash
data/ncf_raw/*.npy
```
Raw NCFs can be large. It is safe to delete them after stacking if storage is limited.
### 2. Stacking (daily, 7d, 15d, 30d)
```bash
make stack_only
```
Produces:
```bash
data/ncf_stacks/<window>/*.npy
```
Stack windows are controlled in configs/cc.yaml.
### 3. Dispersion imaging & picking
```bash
make disp_only
```
Produces:
```bash
results/dispersion/<window>/VS_xxx/
```
Each virtual source folder contains:
- f–v panel
- frequency & velocity axes
- picked dispersion curve
- metadata (`.json`)
To process multiple windows:
```bash
make disp_only DISP_WINDOWS="daily 7d 15d 30d" NJOBS=12
```
### 4. Full pipeline
```bash
make all
```
Runs:
```bash
cc → stack → dispersion
```
---
## License
This project is licensed under the MIT License.
See the `LICENSE` file for full text.
