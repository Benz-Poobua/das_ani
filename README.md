# DAS-ANI: Distributed Acoustic Sensing Preprocessing & Ambient Noise Interferometry Tools

## Purpose

This repository contains tools, workflows, and utilities for preprocessing and processing **Distributed Acoustic Sensing (DAS)** data.  
The focus is on **shallow subsurface imaging** using **ambient noise interferometry (ANI)**. This project also introduces a new framework for compressing DAS data and evaluating its impact on virtual source gathers (VSGs) and dispersion curve construction.

---

## Installation

You may install dependencies using **pip** or **conda**.

### **Using pip**

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```
### **Using conda**
```bash
# Create the environment from YAML
conda env create -f environment.yml

# Activate the environment
conda activate das_ani-env
```
---

## Downloading DAS Data from Google Cloud

To download large datasets (e.g., preprocessed DAS windows or NCF results), you need:

- gsutil installed
- Google Cloud authentication (with proper permissions)
  
Official documentation: https://cloud.google.com/storage/docs/gsutil

### Example: Download DAS data from a GCS bucket
```bash
gsutil -m cp -n -r gs://path/to/data .
```
### Explanation of flags
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
├── requirements.txt
│
├── geometry/
│   └── .gitkeep             # originally stores: geometry_offset.csv, map_data.npy, map_data.txt
│
├── data/
│   ├── ncf_raw/*.npy            # Noise cross-correlation results
│   ├── ncf_stacks               # Stacked noise cross-correlation results
│   │   └──daily/*_daily.npz 
│   └── preprocessed             # Preprocessed DAS time windows
│       └──day/*.npz 
├── notebooks/
│   ├── das_geometry.ipynb            # Fiber geometry visualization
│   └── das_processing_demo.ipynb     # End-to-end DAS processing demo
│
└── src/
    ├── utils.py              # Helper functions for I/O, conversions, utilities
    ├── ani.py                # Ambient Noise Interferometry algorithms
    ├── cc.py                 # Cross-correlation workflow for DAS
    ├── stack.py              # Stacking algorithm
    ├── disp.py               # Dispersion images and picking algorithms
    ├── disp_pick.py          # Dispersion curves workflow
    ├── plot.py               # Plotting utilities
    └── fake.py (optional)    # Synthetic DAS generator (will be removed soon)
```
---
## Tutorial
After downloading the data, ensure that the directory `./data/preprocessed` exists.
You can then run the processing cross-correlation workflow with:
```bash
python -m src.cc --data_root ./data/preprocessed --output_root ./data/ncf_raw --njobs 4 --use_gpu --verbose
```
Next, you can stack NCFs using: 
```bash
python -m src.stack --raw_root ./data/ncf_raw --stacks_root ./data/ncf_stacks
```
The last step is to compute dispersion images and pick dispersion curves from stacked NCFs:
```bash
python -m src.disp_pick --ncf_root ./data/ncf_stacks/daily --results_root ./results/dispersion --stack_window daily --njobs 4
```
---
## License
This project is licensed under the MIT License.
See the `LICENSE` file for full text.
