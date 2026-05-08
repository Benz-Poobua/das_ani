#!/usr/bin/env bash
# sherlock_setup.sh
# Usage:
#   source sherlock_setup.sh
# Must be sourced (not executed).

echo "========================================"
echo " Setting up Sherlock DAS ANI environment"
echo "========================================"

# -------------------------
# Load Modules
# -------------------------
module reset
module load devel math
module load python/3.12.1
module load py-pandas/2.2.1_py312
module load py-pytorch/2.4.1_py312
module load py-scikit-image/0.24.0_py312
module load py-scipy/1.12.0_py312

# -------------------------
# Set Project Root (Dynamic)
# -------------------------
# Automatically detects the directory where this script is located
PROJ="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$PROJ"

# Avoid duplicate PYTHONPATH entries
case ":${PYTHONPATH-}:" in
  *":$PROJ:"*) ;;
  *) export PYTHONPATH="$PROJ:${PYTHONPATH-}" ;;
esac

# -------------------------
# Print Environment Info
# -------------------------
echo
echo "Project directory: $PWD"
echo "PYTHONPATH: $PYTHONPATH"
echo
echo "Loaded modules:"
module list
echo

# -------------------------
# Verify Python Environment
# -------------------------
python3 - << 'EOF'
import sys
import numpy as np
import pandas as pd
import torch
import skimage
import scipy
import src

print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("scikit-image:", skimage.__version__)
print("SciPy:", scipy.__version__)

# Verify our new I/O architecture is present
try:
    import zarr
    import numcodecs
    print("Zarr:", zarr.__version__)
    print("Numcodecs:", numcodecs.__version__)
except ImportError as e:
    print(f"⚠️ WARNING: Missing I/O dependency: {e}")
    print("   Run: python3 -m pip install --user zarr numcodecs")

print("src imported from:", src.__file__)
EOF

echo "✅ Sherlock environment ready."