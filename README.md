# DAS-ANI: Distributed Acoustic Sensing Preprocessing & Ambient Noise Interferometry Tools

## Overview

This repository provides a modular, configuration-driven framework for **Distributed Acoustic Sensing (DAS)** preprocessing and **Ambient Noise Interferometry (ANI)** workflows.

The core goals are:

- Preprocessing of large-scale DAS datasets (out-of-core ingestion, spatial batching, in-place memory-bounded normalization and filtering)
- Efficient computation of noise cross-correlations (NCFs) using either a conventional FFT correlator or an optimized **block-by-block short-lag correlator** (Zhang, 2026)
- Temporal stacking of NCFs (1 h, 1 d, 7 d, 15 d, …) and basic QC
- Dispersion imaging via the **Multichannel Analysis of Surface Waves (MASW)** phase-shift method and automated dispersion-curve picking
- CPU and GPU execution paths, single-node and HPC-scale orchestration

---

## Why a block-wise correlator?

Conventional FFT-based cross-correlation pads two length-$N_{\text{win}}$ sequences to $2N_{\text{win}}$ before transforming, even though ambient-noise interferometry only needs lags in a small window $|m| \le M$ corresponding to the maximum inter-channel travel time. That makes the conventional approach scale as $\mathcal{O}(N_{\text{win}} \log N_{\text{win}})$ even though only $2M+1 \ll 2 N_{\text{win}}$ output samples are kept.

The block-wise scheme of [Zhang (2026)](https://doi.org/10.1016/j.dsp.2025.105509) partitions the long input into blocks of length $K$, performs FFTs of size $K + 2M$ on each block, and accumulates the spectral products before a single inverse FFT. With the optimal block size obtained analytically via the Lambert $W$-function,

$$
K^{\star} = 2M\,\bigl(\,-W_{-1}\!\bigl(-\tfrac{1}{4eM}\bigr)\,-\,1\,\bigr),
$$

the asymptotic cost becomes $\mathcal{O}\!\bigl(N_{\text{win}} \log(4eM \ln 4eM)\bigr)$, which is substantially cheaper than the conventional baseline whenever $M \ll N_{\text{win}}$ — the standard regime for ANI.

This code implements both correlators and lets you select between them through a single config option (see below).

---

## Installation

### Install as an editable package (pip)
```bash
# Create virtual environment
python -m venv das_ani
source das_ani/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install as editable package
pip install -e .
```

Required runtime dependencies are listed in `pyproject.toml`. GPU acceleration requires a PyTorch build matching your local CUDA version (see the PyTorch installation matrix). `torch compile` is optionally used to JIT-fuse the spectral kernels (PyTorch ≥ 2.0).

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
├── pyproject.toml
├── Makefile
├── configs/                     # YAML configuration files
│   └── cc.yaml                  # Cross-correlation parameters + stacking parameters
│
├── data/
│   ├── preprocessed/            # Preprocessed DAS time windows (.npz)
│   ├── ncf_raw/                 # Raw noise cross-correlations (.npy)
│   └── ncf_stacks/              # Stacked NCFs (.npy)
│       ├── 1h/
│       ├── 1d/
│       └── 7d/
│
└── src/
    ├── utils.py                 # I/O, config helpers, diagnostics
    ├── ani.py                   # ANI preprocessing + correlation kernels
    ├── cc.py                    # Cross-correlation workflow
    ├── stack.py                 # NCF stacking (hours, daily, multi-day)
    ├── disp.py                  # Dispersion imaging + picking algorithms
    └── plot.py                  # Plotting utilities
```
---
## Workflow Overview

All scripts are config-driven via YAML files in `configs/`. You should not need to modify Python code for parameter changes — only the YAML.

### 1. Cross-correlation (NCF generation)

```bash
make cc
# or, equivalently:
python -m src.cc --config configs/urban_cc.yaml --verbose
```

Produces:
```text
data/ncf_raw_<deployment>/*.npz
```

#### Config structure

All cross-correlation parameters live in a single YAML file. The seven top-level sections are:

| Section      | Purpose |
|--------------|---------|
| `paths`      | Input data root and NCF output root |
| `runtime`    | Parallelism, GPU toggle, memory budget, JIT |
| `data`       | Sampling rate, channel range, spacing, virtual-source stride |
| `preprocess` | Bandpass filter, decimation, optional differentiation, whitening chunk size |
| `xcorr`      | Correlator mode, lag window, segment length, whitening |
| `perf`       | Runtime logging for the benchmark CSV |
| `stacking`   | Optional in-line stacking of the freshly produced NCFs |

#### Selecting the correlator

The correlator is chosen with `xcorr.mode`:

```yaml
xcorr:
  mode: v1            # Block-wise short-lag correlator (Zhang 2026).
                      # Recommended for ANI where max_lag_sec is much
                      # shorter than xcorr_seg_sec_v1.

  mode: conventional  # Conventional full-lag FFT correlator.
                      # Use when the lag window is comparable to the
                      # segment length, or as a fidelity baseline.

  max_lag_sec: 2.0    # M in seconds (the lag half-window)
  xcorr_seg_sec: 60.0      # N_win for conventional mode
  xcorr_seg_sec_v1: 60.0   # N_win for v1 mode

  # v1-specific knobs:
  v1_fft_snap_pow2: true   # snap (K + 2M) to a power-of-two FFT length
  v1_fallback: v1_2M       # block-size strategy if Lambert-W is skipped:
                           #   "v1_2M"  -> K = 2M
                           #   "v1_Mp1" -> K = M + 1

  # additional knobs:
  is_spectral_whitening: true   # spectral whitening before correlation
  window_freq_hz: 0.0           # Hz half-width for whitening smoothing
  auto_cc: false                # true => autocorrelation only (CWI / ACF)
```

#### Auto-correlation mode (for CWI / ACF)

Setting `xcorr.auto_cc: true` switches the workflow from inter-channel cross-correlation to single-channel autocorrelation. This is the configuration used for Coda Wave Interferometry (CWI) and autocorrelation-based reflection imaging (ACF). The block-wise (`v1`) correlator handles autocorrelation identically; only the channel pairing changes.

#### Execution platform: CPU or GPU

Platform selection is in `runtime`. The same algorithmic modes (`conventional`, `v1`) work on either backend without code changes.

**CPU (default, recommended for archive processing on shared nodes):**
```yaml
runtime:
  njobs: 1            # number of concurrent worker processes
  use_gpu: false      # CPU-only execution
  mmap: true          # memory-mapped file ingestion (out-of-core)
  frac_mem: 0.25      # per-worker memory budget as a fraction of node RAM
  min_chunk: 64       # smallest allowed spatial-channel chunk
  max_chunk: 4096     # largest allowed spatial-channel chunk
  torch_compile: false
  compile_mode: reduce-overhead
```

**GPU (PyTorch + CUDA):**
```yaml
runtime:
  njobs: 1
  use_gpu: true       # CUDA execution via PyTorch tensors
  mmap: true
  frac_mem: 0.5       # GPU-VRAM fraction; raise to 0.6 on 24 GB+ devices
  min_chunk: 64
  max_chunk: 8192     # larger SIMD-friendly chunks on GPU
  torch_compile: false # optional: enable JIT-fusion of spectral kernels
  compile_mode: reduce-overhead

preprocess:
  whiten_chunk_nch: 4096   # GPU-specific: channels per whitening batch
```

`torch_compile: true` enables `torch.compile` JIT fusion of the forward-FFT / multiply / accumulate / inverse-FFT kernel chain (PyTorch ≥ 2.0); leave it off until your dispatch overhead is non-trivial relative to per-kernel cost.

#### HPC / multi-node scaling

The pipeline scales by **file-level parallelism**: each continuous DAS file is independent and is dispatched to one worker process. To run on an HPC cluster, set `njobs` to the number of concurrent workers per node and dispatch the configs across nodes via a Slurm job array (or similar). Each worker handles its own out-of-core ingestion, spatial batching, and correlation; the `frac_mem`, `min_chunk`, `max_chunk` knobs in `runtime` keep per-worker memory inside the allocation.

For CPU jobs, set `OMP_NUM_THREADS` and `MKL_NUM_THREADS` (or equivalents) at the Slurm-script level to a disjoint subset of cores per worker so that BLAS / FFTW do not over-subscribe across workers.

#### Example: switching CPU ↔ GPU on the same dataset

The only differences between `configs/urban_cc.yaml` (CPU) and `configs/urban_cc_gpu.yaml` (GPU) are:

```yaml
# CPU                          # GPU
runtime:                       runtime:
  use_gpu: false                 use_gpu: true
  frac_mem: 0.25                 frac_mem: 0.5
  max_chunk: 4096                max_chunk: 8192

preprocess:                    preprocess:
  (no whiten_chunk_nch)          whiten_chunk_nch: 4096

perf:                          perf:
  out_path: ./data/runlogs/      out_path: ./data/runlogs/
    perf_cc.csv                    perf_cc_gpu.csv
```

Everything else — `xcorr.mode`, `max_lag_sec`, segment lengths, preprocessing, stacking — is identical.

### 2. Stacking (1 h, 1 d, 7 d, …)

Stacking is configured in the same YAML file as cross-correlation, in the `stacking` block:

```yaml
stacking:
  enabled: true                       # set false to skip stacking
  raw_root: ./data/ncf_raw            # where to read raw NCFs from
  stacks_root: ./data/ncf_stacks      # where to write stacked NCFs
  overwrite: false

  base_stack: 1d                      # fundamental stack unit (1h, 1d, …)

  windows:                            # which longer windows to also build
    7d:  true
    15d: true
    30d: true
```

```bash
make stack
# or:
python -m src.stack --config configs/urban_cc.yaml --verbose
```

Produces:
```text
data/ncf_stacks/<window>/*.npy
```

---

## Citation

If you use this codebase in your research, please cite the algorithm reference

> Zhang, W.-Q. (2026). *Accelerating cross-correlation for long
> sequences with short lag constraints: An optimized block-wise
> approach.* **Digital Signal Processing**, 168, 105509.
> <https://doi.org/10.1016/j.dsp.2025.105509>

and acknowledge this repository.

---
## License

This project is licensed under the MIT License. See the `LICENSE` file for full text.