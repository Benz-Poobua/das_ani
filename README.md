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
│   ├── ncf/                 # Noise cross-correlation results
│   └── preprocessed/        # Preprocessed DAS time windows
│
├── notebooks/
│   ├── das_geometry.ipynb            # Fiber geometry visualization
│   └── das_processing_demo.ipynb     # End-to-end DAS processing demo
│
└── src/
    ├── utils.py              # Helper functions for I/O, conversions, utilities
    ├── ani.py                # Ambient Noise Interferometry algorithms
    ├── cc.py                 # Cross-correlation workflow for DAS
    └── fake.py (optional)    # Synthetic DAS generator (used for testing)
```
---
## License
This project is licensed under the MIT License.
See the `LICENSE` file for full text.
