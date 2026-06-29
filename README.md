# DAS-ANI: Distributed Acoustic Sensing Preprocessing & Ambient Noise Interferometry Tools

## Overview

This repository provides a modular, configuration-driven framework for **Distributed Acoustic Sensing (DAS)** preprocessing and **Ambient Noise Interferometry (ANI)** workflows.

The core goals are:

- Preprocessing of large-scale DAS datasets with a **selectable backend** (`pure_numpy` benchmark, `hybrid` scipy+torch, `pure_torch` GPU-resident) — see [Preprocessing backends](#preprocessing-backends-preprocessmode)
- Efficient computation of noise cross-correlations (NCFs) using either a conventional FFT correlator or an optimized **block-by-block short-lag correlator** (Zhang, 2026)
- Temporal stacking of NCFs (1 h, 1 d, 7 d, 15 d, …) and basic QC
- Dispersion imaging via the **Multichannel Analysis of Surface Waves (MASW)** phase-shift method and automated dispersion-curve picking
- dv/v monitoring (stretching + cross-correlation methods) and **Multi-Channel Cross-Correlation (MCCC)** relative arrival times
- CPU and GPU execution paths, single-node and HPC-scale (SLURM) orchestration, with a built-in benchmark/fidelity suite (`src/eval.py`) and a pytest test suite

---

## Why a block-wise correlator?

Conventional FFT-based cross-correlation pads two length- $N_{\text{win}}$ sequences to $2N_{\text{win}}$ before transforming, even though ambient-noise interferometry only needs lags in a small window $|m| \le M$ corresponding to the maximum inter-channel travel time. That makes the conventional approach scale as $\mathcal{O}(N_{\text{win}} \log N_{\text{win}})$ even though only $2M+1 \ll 2 N_{\text{win}}$ output samples are kept.

The block-wise scheme of [Zhang (2026)](https://doi.org/10.1016/j.dsp.2025.105509) partitions the long input into blocks of length $K$, performs FFTs of size $K + 2M$ on each block, and accumulates the spectral products before a single inverse FFT. With the optimal block size obtained analytically via the Lambert $W$-function,

$$K^* = 2M \left( -W_{-1} \left( -\frac{1}{4eM} \right) - 1 \right),$$

the asymptotic cost becomes $\mathcal{O}\\Bigl(N_{\text{win}}\\log_2\\bigl(4eM\\ ln(4eM)\bigr)\Bigr)$, which is substantially cheaper than the conventional baseline whenever $M \ll N_{\text{win}}$ — the standard regime for ANI.

This code implements both correlators and lets you select between them through a single config option (see below). The accompanying benchmark study across three DAS deployments (urban, offshore, bridge) is described in Poobua, Li & Biondi (SEP-199), *Minimum-Effort DAS Cross-Correlation*.

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

# Optional extras:
pip install -e ".[dev]"          # ruff, black, mypy, pytest
pip install -e ".[viz]"          # matplotlib, seaborn, scikit-image (SSIM), plotly
pip install -e ".[postprocess]"  # dask (parallel NCF batch pipelines in src/ncf.py)
```

Required runtime dependencies are listed in `pyproject.toml`. GPU acceleration requires a PyTorch build matching your local CUDA version (see the PyTorch installation matrix). `torch.compile` is optionally used to JIT-fuse the spectral kernels (PyTorch ≥ 2.0).

### Run the test suite
```bash
pip install -e ".[dev]"
pytest            # or: make test
```

The tests cover, among other things, numerical parity of the three preprocessing backends and the fidelity of the block-wise (`v1`) correlator against the conventional one.

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
├── sherlock_setup.sh            # HPC module environment (CPU nodes)
├── sherlock_setup_gpu.sh        # HPC module environment (GPU nodes)
│
├── configs/                     # YAML configuration files
│   ├── urban_cc.yaml            # Urban deployment, CPU (preprocess.mode: pure_numpy)
│   └── urban_cc_gpu.yaml        # Urban deployment, GPU (preprocess.mode: hybrid)
│
├── slurm/                       # SLURM batch scripts (Sherlock)
│   ├── run_cc_urban.slurm       # production CC + stacking, CPU
│   ├── run_cc_urban_gpu.slurm   # production CC + stacking, GPU
│   ├── run_eval_urban.slurm     # benchmark suite, CPU
│   └── run_eval_urban_gpu.slurm # benchmark suite, GPU
│
├── data/
│   ├── raw_urban/               # Continuous DAS windows (.npz / .zarr)
│   ├── ncf_raw*/                # Raw noise cross-correlations / VSGs (.npy)
│   └── ncf_stacks*/             # Stacked NCFs (.npy): 1d/, 7d/, 15d/, 30d/
│
├── src/
│   ├── utils.py                 # I/O, config helpers, FK filter, diagnostics
│   ├── ani.py                   # Preprocessing backends + correlation kernels
│   ├── cc.py                    # Cross-correlation workflow (VSG generation)
│   ├── stack.py                 # NCF stacking (hours, daily, multi-day)
│   ├── eval.py                  # Benchmarks: preprocess fidelity, scaling, lag sweep
│   ├── error.py                 # Fidelity metrics (Frobenius, cosine, SSIM, picks)
│   ├── ncf.py                   # NCF post-processing (FK, mutes, gather export)
│   ├── disp.py                  # Dispersion imaging + picking algorithms
│   ├── dvv.py                   # dv/v monitoring (stretching, xcorr, attenuation)
│   └── mccc.py                  # Multi-channel cross-correlation (dt/t)
│
└── tests/                       # pytest suite
```
---

## Input data format

Continuous DAS recordings are ingested from `.npz` archives (or `.zarr` stores) found recursively under `paths.data_root`.

### Filename convention

```text
YYYYMMDD_HHMMSS_<tag>.npz        e.g.  20250722_025000_bridge.npz
```

The leading datetime is **required** — the stacking engine (`src/stack.py`) parses it to group NCFs into 1 h / 1 d / multi-day windows. `<tag>` is a free-form deployment label (`urban`, `bridge`, `offshore`, …).

### Mandatory keys inside each `.npz`

| Key            | Type / shape          | Meaning |
|----------------|-----------------------|---------|
| `data`         | float32 `(nch, nt)`   | Continuous DAS strain or strain-rate window (channels × samples) |
| `dt`           | scalar (s)            | Temporal sampling interval (`fs_raw = 1/dt`) |
| `dx`           | scalar (m)            | Channel spacing along the fiber |
| `start_sample` | scalar (int)          | Index of the first sample relative to the original continuous record |
| `end_sample`   | scalar (int)          | Index one past the last sample relative to the original record |

`data` and `dt` are hard requirements of the CC engine; `dx`, `start_sample`, `end_sample` are required by the file specification (provenance + downstream geometry) and `load_data` logs a warning when they are absent so legacy archives keep working.

For `.zarr` stores, the group must contain a 2-D `data` array with a `dt` attribute.

---

## Workflow Overview

All scripts are config-driven via YAML files in `configs/`. You should not need to modify Python code for parameter changes — only the YAML.

### 1. Cross-correlation (VSG generation)

```bash
make cc
# or, equivalently:
python -m src.cc --config configs/urban_cc.yaml --verbose
```

#### Config structure

All cross-correlation parameters live in a single YAML file. The eight top-level sections are:

| Section      | Purpose |
|--------------|---------|
| `paths`      | Input data root and NCF output root |
| `runtime`    | Parallelism, GPU toggle, memory budget, JIT |
| `data`       | Sampling rate, channel range, spacing, virtual-source stride |
| `ingest`     | Once-per-file operations: anti-aliased decimation, strain→strain-rate differentiation |
| `preprocess` | Per-window stage: backend (`mode`), bandpass corners, temporal normalization, whitening chunk size |
| `xcorr`      | Correlator mode, lag window, segment length, whitening |
| `perf`       | Runtime logging for the benchmark CSV |
| `stacking`   | Optional in-line stacking of the freshly produced NCFs |

> Note: `decimation` and `diff` live under `ingest:` (they run once per file, before per-window preprocessing). The legacy `preprocess.decimation` / `preprocess.diff` locations are still accepted with a deprecation warning.

#### Preprocessing backends (`preprocess.mode`)

Every window passes through the same four stages — detrend → Tukey-tapered zero-phase Butterworth bandpass → per-sample median removal → temporal normalization (RAM or 1-bit) — but you choose **where** they execute:

```yaml
preprocess:
  mode: hybrid        # pure_numpy | hybrid | pure_torch
  f1: 1.0             # bandpass corners (Hz)
  f2: 10.0
  ram_win_sec: 0.0    # 0.0 = 1-bit normalization
```

| `mode` | Detrend + Bandpass | Median + Temporal norm | Returns | Use when |
|--------|--------------------|------------------------|---------|----------|
| `pure_numpy` | CPU (scipy `sosfiltfilt`) | CPU (numpy/scipy) | `np.ndarray` | Legacy, **bit-exact benchmark**; archive processing on CPU nodes |
| `hybrid` | CPU (same scipy routines as `pure_numpy`) | torch on target device | `torch.Tensor` on device | **Numerical fidelity** on CPU *or* GPU — matches `pure_numpy` to float32 round-off (Option A) |
| `pure_torch` | torch on device (rFFT-domain `\|H(f)\|²` mask) | torch on device | `torch.Tensor` on device | **GPU-tailored speed** — fully device-resident, no host round-trip |

Fidelity notes:

- `hybrid` uses `torch.quantile(q=0.5)` for the median (numpy-parity even for even channel counts) and a numpy-parity running-absolute-mean, so it reproduces `pure_numpy` to single-precision epsilon.
- `pure_torch` replaces `sosfiltfilt` with its analytic zero-phase response `|H(f)|²` applied in the rFFT domain. With the Tukey taper zeroing the trace edges the two agree closely in the passband, but they are not bit-identical. The deviation for *your* band is quantified by the `preprocess` experiment in `src/eval.py` (and asserted in `tests/test_preprocess.py`).
- If `mode` is omitted, the legacy mapping applies: `use_gpu: false` → `pure_numpy`, `use_gpu: true` → `hybrid`.

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

Platform selection is in `runtime`. The same algorithmic modes (`conventional`, `v1`) and preprocessing backends work on either platform without code changes.

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

preprocess:
  mode: pure_numpy
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
  mode: hybrid             # or pure_torch for a fully GPU-resident chain
  whiten_chunk_nch: 4096   # GPU-specific: channels per whitening batch
```

`torch_compile: true` enables `torch.compile` JIT fusion of the forward-FFT / multiply / accumulate / inverse-FFT kernel chain (PyTorch ≥ 2.0); leave it off until your dispatch overhead is non-trivial relative to per-kernel cost.

#### Example: switching CPU ↔ GPU on the same dataset

The only differences between `configs/urban_cc.yaml` (CPU) and `configs/urban_cc_gpu.yaml` (GPU) are:

```yaml
# CPU                          # GPU
runtime:                       runtime:
  use_gpu: false                 use_gpu: true
  frac_mem: 0.25                 frac_mem: 0.5
  max_chunk: 4096                max_chunk: 8192

preprocess:                    preprocess:
  mode: pure_numpy               mode: hybrid   # or pure_torch
  (no whiten_chunk_nch)          whiten_chunk_nch: 4096

perf:                          perf:
  out_path: ./data/runlogs/      out_path: ./data/runlogs/
    perf_cc.csv                    perf_cc_gpu.csv
```

Everything else — `xcorr.mode`, `max_lag_sec`, segment lengths, stacking — is identical.

#### Output format: Virtual Source Gathers (VSGs)

For each input file and each virtual source (every `data.src_stride`-th channel), `src/cc.py` writes one VSG:

```text
<input_basename>_cc_<vs:03d>_<xcorr.mode>.npy
e.g.  20250722_025000_bridge_cc_080_v1.npy
```

| Property | Value |
|----------|-------|
| dtype    | `float32` |
| shape    | `(nch, 2*M + 1)` with `nch = last_chan - first_chan + 1` and `M = round(max_lag_sec * fs_proc)` |
| row `i`  | receiver channel `first_chan + i` (absolute cable index) |
| lag axis | `np.arange(-M, M + 1) / fs_proc` seconds; positive lag = propagation **from** the virtual source **to** the receiver |
| `<vs>`   | virtual-source index **relative to** `first_chan`, zero-padded to 3 digits |
| scaling  | mean over the `nseg` correlation segments of the file |

Auto-correlation runs produce `<input_basename>_auto_<mode>.npy` of the same shape. Auto-resume state per file is kept in `<input_basename>_cc_state_<mode>.json` (delete it to force recomputation; individual VSGs are also validated by shape on restart).

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

Produces (named after the **end date** of each window):
```text
data/ncf_stacks/<window>/YYYYMMDD[_HHMMSS]_cc_<vs:03d>_<window>_<method>.npy
e.g.  data/ncf_stacks/7d/20250728_cc_080_7d_v1.npy
```

### 3. Benchmarks & fidelity (`src/eval.py`)

```bash
make eval
# or, with full control:
python -m src.eval \
  --cc_config configs/urban_cc.yaml \
  --outdir data/benchmarks/urban \
  --n_files 16 --repeats 4 \
  --cores 1 2 4 8 16 \
  --window_sec 60 --njobs_complexity 1 \
  --lags 0.5 1 2 3 4 5 6 \
  --preprocess_modes hybrid pure_torch pure_numpy \
  --cleanup
```

Three experiments, each skippable (`--skip_preprocess`, `--skip_scaling`, `--skip_complexity`):

| Experiment   | What it measures |
|--------------|------------------|
| `preprocess` | Per-file wall time of each preprocessing backend **and** its fidelity vs the `pure_numpy` benchmark (`rel_fro`, `max_abs`, per-channel cosine similarity). Expect `hybrid` ≈ float32 epsilon; `pure_torch` shows the rFFT-bandpass approximation for your band. |
| `scaling`    | Strong scaling of the CC engine (conventional vs `v1`) over worker counts |
| `complexity` | Lag sweep: runtime of conventional vs `v1` and NCF fidelity (`v1` vs conventional) per lag |

Results accumulate in `benchmark_results.csv` (one row per run; the `experiment`, `mode`, and `note` columns identify each row) plus a `run_manifest.json` recording the exact configuration.

### 4. Post-processing modules

- `src/ncf.py` — FK filtering, spatial/temporal mutes, directional (causal/acausal, S1/S2) gather export for dispersion work.
- `src/disp.py` — phase-shift f–v dispersion imaging and automated ridge picking, dispersion-based dv/v + wavelength/3 depth mapping, a config-driven time-lapse dispersion wrapper (`run_timelapse_dispersion`, writes a dv/v (time, depth) heatmap `.npz`), and export for 1-D inversion.
- `src/dvv.py` — dv/v monitoring: stretching method (returns the raw stretching factor ε in %, with `dv/v = -ε`), cross-correlation peak-shift method (returns dv/v directly), and spectral-ratio Q estimation.
- `src/mccc.py` — Multi-Channel Cross-Correlation (VanDecar & Crosson, 1990) for sub-sample relative arrival times / dt/t across the array.

---

## Running on HPC (SLURM)

Ready-to-submit Sherlock scripts live in `slurm/` (adjust the `cd` path, partition names, and resource requests to your cluster):

```bash
sbatch slurm/run_cc_urban.slurm        # production CC + stacking, CPU
sbatch slurm/run_cc_urban_gpu.slurm    # production CC + stacking, GPU
sbatch slurm/run_eval_urban.slurm      # benchmark suite, CPU
sbatch slurm/run_eval_urban_gpu.slurm  # benchmark suite, GPU
```

Conventions baked into the scripts:

- Each script `cd`s to the repo root and sources `sherlock_setup.sh` (CPU) or `sherlock_setup_gpu.sh` (GPU) to load the module environment and put the repo on `PYTHONPATH`.
- **Thread control is handled inside Python** (`_set_thread_env()` in `src/cc.py`, driven by `SLURM_CPUS_PER_TASK`); do not export `OMP_NUM_THREADS` etc. in the batch script — it would be overridden.
- GPU jobs request `--partition=gpu --gres=gpu:1 --cpus-per-task=1` and keep `njobs: 1` / `--njobs_complexity 1` to avoid device-lock crashes under SLURM Exclusive-Process mode. For multi-GPU throughput, submit one task per GPU pinned via `CUDA_VISIBLE_DEVICES` (VS mode does not parallelize across `nn.DataParallel`).
- The pipeline scales by **file-level parallelism**: each continuous DAS file is independent, so scale out with a job array over date ranges and scale up with `runtime.njobs` workers per node. `frac_mem` / `min_chunk` / `max_chunk` keep per-worker memory inside the allocation, and the auto-resume state makes requeued jobs idempotent.

---

## Citation

If you use this codebase in your research, please cite the algorithm reference

> Zhang, W.-Q. (2026). *Accelerating cross-correlation for long
> sequences with short lag constraints: An optimized block-wise
> approach.* **Digital Signal Processing**, 168, 105509.
> <https://doi.org/10.1016/j.dsp.2025.105509>

the application/benchmark report

> Poobua, S., Li, H., & Biondi, B. L. *Minimum-Effort DAS
> Cross-Correlation:* SEP Report 199, Stanford University.

and acknowledge this repository.

---
## License

This project is licensed under the MIT License. See the `LICENSE` file for full text.
