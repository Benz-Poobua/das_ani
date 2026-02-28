# Sherlock Run Guide

**DAS Ambient Noise Interferometry (ANI) Framework**

Author: Benz Poobua
Platform: Stanford Sherlock HPC

---

## 1. Login to Sherlock

```bash
ssh <SUNetID>@login.sherlock.stanford.edu
```

After login, move to the project directory (GROUP_HOME location):

```bash
cd /home/groups/biondo/spoobua/das_ani
```

> 🔹 Always work from `GROUP_HOME`, not `/home/users`, to avoid the 15 GB HOME quota limit.

---

## 2. Load Environment (Every New Session)

Run the setup script to load the correct module stack and Python path:

```bash
source sherlock_setup.sh
```

This ensures:

* Correct Python (3.12)
* Matching NumPy / Pandas / PyTorch ABI
* No mixed pip/module environment
* `src/` is visible via `PYTHONPATH`

---

## 3. Submit a Compute Job (Recommended Way)

Run workloads using Slurm — **not interactive mode** — for stability and reproducibility.

```bash
sbatch run_eval.slurm
```

Why this is better than interactive mode:

* ✅ Survives logout / laptop shutdown
* ✅ Runs on dedicated compute node
* ✅ Scheduler-managed resources
* ✅ No idle-session timeout
* ✅ Required for multiprocessing workloads

---

## 4. Check Job Status

```bash
squeue -u $USER
```

Job states:

* `PD` → Pending (waiting in queue)
* `R`  → Running
* `CG` → Completing
* disappears → Finished

---

## 5. Monitor Output Logs

Logs are written automatically (as defined in `run_eval.slurm`):

```bash
tail -f logs/ani_eval_<JOBID>.out
```

This lets you watch progress in real time.

---

## 6. Cancel a Job (If Needed)

```bash
scancel <JOBID>
```

Example:

```bash
scancel 15851173
```

---

## 7. Editing Code vs Running Jobs (Important Distinction)

| Task                            | Where to Do It |
| ------------------------------- | -------------- |
| Edit `.py` files                | Login node ✅   |
| Git operations                  | Login node ✅   |
| Small syntax tests              | Login node ✅   |
| Run ANI pipeline                | Slurm job ✅    |
| Multiprocessing / FFT workloads | Slurm job ✅    |

> ❗ Never run heavy scripts on the login node — this can get throttled or killed.

---

## 8. Project Storage Layout (HPC-Optimized)

| Component    | Location                                      |
| ------------ | --------------------------------------------- |
| Code         | `/home/groups/biondo/spoobua/das_ani`         |
| Large data   | `/scratch/groups/biondo/spoobua/das_ani/data` |
| Repo `data/` | Symlink → scratch                             |

Check the symlink:

```bash
ls -l data
```

You should see:

```bash
data -> /scratch/groups/biondo/spoobua/das_ani/data
```

> ⚠️ `GROUP_SCRATCH` is purged after **90 days of inactivity**.

---

## 9. Check Storage Usage

```bash
sh_quota
```

---

## 10. Inspect Available Resources (Partitions)

```bash
sh_part
```

Useful when deciding:

* how many CPUs to request
* memory limits
* max runtime allowed

---

## 11. Typical Workflow

```bash
ssh sherlock
cd /home/groups/biondo/spoobua/das_ani

# Load environment
source sherlock_setup.sh

# Submit job
sbatch run_eval.slurm

# Monitor
squeue -u $USER
tail -f logs/ani_eval_<jobid>.out
```

---
## 12. Download data to local machine

```bash
scp -r <SUNetID>@login.sherlock.stanford.edu:/home/groups/biondo/spoobua/das_ani/data/benchmarks/rob_w120/plots ~/Downloads/
```

---