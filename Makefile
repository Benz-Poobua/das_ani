# ============================================================
# DAS Ambient Noise Interferometry Pipeline
# Author: Benz Poobua
# ============================================================

# -----------------------
# Config files
# -----------------------
CC_CFG      ?= configs/cc.yaml
DISP_CFG    ?= configs/disp.yaml

# -----------------------
# Python executable
# -----------------------
VENV        ?= das_ani
PYTHON      ?= $(VENV)/bin/python

# -----------------------
# Defaults / overrides
# -----------------------
NJOBS        ?= 4

# Dispersion overrides
DISP_WINDOWS ?=         # e.g., "daily 7d"
DISP_MODE    ?=         # e.g., both, causal, acausal
FK           ?=         # Set FK=1 to force enable F-K filter

# Eval overrides
EVAL_OUTDIR  ?= data/runlogs/eval_modes
EVAL_NJOBS   ?= 4
EVAL_TITLE   ?=
EVAL_LOGY    ?= 0

# Robustness benchmark overrides (NEW)
ROB_TAG           ?= rob1                # folder name inside ROB_ROOT
ROB_ROOT          ?= data/benchmarks     # parent folder
ROB_OUTDIR        ?= $(ROB_ROOT)/$(ROB_TAG)

ROB_NFILES        ?= 3
ROB_REPEATS       ?= 2
ROB_WINDOW_SEC    ?= 60
ROB_NJOBS_COMP    ?= $(NJOBS)

ROB_MAX_CORES     ?=
ROB_CORES         ?=    # e.g., 1 2 4 8
ROB_LAGS          ?=    # e.g., 0.5 1 2 3 5 10 20
ROB_CLEANUP       ?= 0

# -----------------------
# Default target
# -----------------------
.DEFAULT_GOAL := help

# -----------------------
# Phony targets (not files)
# -----------------------
.PHONY: help \
	cc cc_only \
	stack stack_only \
	disp disp_only \
	eval eval_only \
	robustness robust_only rob_replot \
	all \
	clean clean_cc_raw clean_stacks clean_disp clean_eval clean_robust clean_all \
	paths

# ============================================================
# HELP
# ============================================================
help:
	@echo ""
	@echo "DAS ANI Pipeline (Makefile)"
	@echo "---------------------------"
	@echo "Dependency targets:"
	@echo "  make cc         Run cross-correlation"
	@echo "  make stack      Run stacking (depends on cc)"
	@echo "  make disp       Run dispersion (depends on stack)"
	@echo "  make eval       Run eval_modes (depends on disp)"
	@echo "  make all        Run full pipeline: cc -> stack -> disp -> eval"
	@echo ""
	@echo "No-dependency targets:"
	@echo "  make cc_only         Run cc only (no deps)"
	@echo "  make stack_only      Run stack only (no deps)"
	@echo "  make disp_only       Run dispersion only (no deps)"
	@echo "  make eval_only       Run eval_modes only (no deps)"
	@echo "  make robust_only     Run eval_robustness only (no deps)"
	@echo "  make rob_replot      Replot robustness from CSV only (no rerun)"
	@echo ""
	@echo "Cleaning targets:"
	@echo "  make clean           Remove all generated outputs (keep dirs)"
	@echo "  make clean_cc_raw    Remove raw NCF outputs (data/ncf_raw)"
	@echo "  make clean_stacks    Remove stacked NCFs (data/ncf_stacks)"
	@echo "  make clean_disp      Remove dispersion outputs (results/dispersion)"
	@echo "  make clean_eval      Remove eval outputs ($(EVAL_OUTDIR))"
	@echo "  make clean_robust    Remove robustness outputs ($(ROB_ROOT))"
	@echo "  make clean_all       Remove output directories entirely"
	@echo ""
	@echo "Override examples:"
	@echo "  make cc_only CC_CFG=configs/cc_fast.yaml"
	@echo "  make disp_only DISP_MODE=both FK=1 NJOBS=8"
	@echo "  make disp_only DISP_WINDOWS=\"daily 7d\""
	@echo "  make eval_only EVAL_TITLE=\"Run 3: FK Filtered\""
	@echo ""
	@echo "Robustness examples:"
	@echo "  make robust_only ROB_TAG=rob1 ROB_NFILES=4 ROB_REPEATS=3 ROB_WINDOW_SEC=60 ROB_NJOBS_COMP=6"
	@echo "  make robust_only ROB_TAG=rob2 ROB_CORES=\"1 2 4 8\" ROB_LAGS=\"0.5 1 2 5 10 20\""
	@echo "  make rob_replot ROB_TAG=rob2"
	@echo ""

# ============================================================
# Sanity print
# ============================================================
paths:
	@echo "CC_CFG         = $(CC_CFG)"
	@echo "DISP_CFG       = $(DISP_CFG)"
	@echo "DISP_WINDOWS   = $(DISP_WINDOWS)"
	@echo "DISP_MODE      = $(DISP_MODE)"
	@echo "FK             = $(FK)"
	@echo "NJOBS          = $(NJOBS)"
	@echo "EVAL_OUTDIR    = $(EVAL_OUTDIR)"
	@echo "EVAL_NJOBS     = $(EVAL_NJOBS)"
	@echo "ROB_TAG        = $(ROB_TAG)"
	@echo "ROB_ROOT       = $(ROB_ROOT)"
	@echo "ROB_OUTDIR     = $(ROB_OUTDIR)"
	@echo "ROB_NFILES     = $(ROB_NFILES)"
	@echo "ROB_REPEATS    = $(ROB_REPEATS)"
	@echo "ROB_WINDOW_SEC = $(ROB_WINDOW_SEC)"
	@echo "ROB_NJOBS_COMP = $(ROB_NJOBS_COMP)"
	@echo "ROB_MAX_CORES  = $(ROB_MAX_CORES)"
	@echo "ROB_CORES      = $(ROB_CORES)"
	@echo "ROB_LAGS       = $(ROB_LAGS)"
	@echo "ROB_CLEANUP    = $(ROB_CLEANUP)"
	@echo "PYTHON         = $(PYTHON)"

# ============================================================
# CROSS-CORRELATION
# ============================================================
cc_only:
	@echo ">>> Running cross-correlation (no deps)"
	@mkdir -p data/ncf_raw
	$(PYTHON) -m src.cc --config $(CC_CFG)

cc: cc_only

# ============================================================
# STACKING (depends on cc)
# ============================================================
stack_only:
	@echo ">>> Running stacking (no deps)"
	@mkdir -p data/ncf_stacks
	$(PYTHON) -m src.stack --config $(CC_CFG)

stack: cc stack_only

# ============================================================
# DISPERSION (depends on stack)
#
# Flexible usage:
#   make disp_only DISP_MODE=causal FK=1
#   make disp_only DISP_WINDOWS="daily 30d"
# ============================================================
disp_only:
	@echo ">>> Running dispersion (no deps)"
	@mkdir -p results/dispersion
	@ARGS=""; \
	if [ -n "$(DISP_WINDOWS)" ]; then ARGS="$$ARGS --stack_windows $(DISP_WINDOWS)"; fi; \
	if [ -n "$(DISP_MODE)" ]; then ARGS="$$ARGS --mode $(DISP_MODE)"; fi; \
	if [ "$(FK)" = "1" ]; then ARGS="$$ARGS --fk"; fi; \
	echo "    Args: --njobs $(NJOBS) $$ARGS"; \
	$(PYTHON) -m src.disp_pick --config $(DISP_CFG) --njobs $(NJOBS) $$ARGS

disp: stack disp_only

# ============================================================
# EVALUATION (depends on disp)
# ============================================================
eval_only:
	@echo ">>> Running eval_modes (no deps) outdir=$(EVAL_OUTDIR)"
	@mkdir -p $(EVAL_OUTDIR)
	@LOGY_FLAG=""; \
	if [ "$(EVAL_LOGY)" = "1" ]; then LOGY_FLAG="--logy_runtime"; fi; \
	TITLE_FLAG=""; \
	if [ -n "$(EVAL_TITLE)" ]; then TITLE_FLAG="--title \"$(EVAL_TITLE)\""; fi; \
	eval "$(PYTHON) -m src.eval_modes \
		--cc_config $(CC_CFG) \
		--disp_config $(DISP_CFG) \
		--outdir $(EVAL_OUTDIR) \
		--mmap \
		--plots \
		$$LOGY_FLAG \
		--njobs $(EVAL_NJOBS) \
		$$TITLE_FLAG"

eval: disp eval_only

# ============================================================
# ROBUSTNESS / BENCHMARKING
# ============================================================
robust_only:
	@echo ">>> Running eval_robustness (no deps) outdir=$(ROB_OUTDIR)"
	@mkdir -p $(ROB_OUTDIR)
	@ARGS=""; \
	if [ -n "$(ROB_MAX_CORES)" ]; then ARGS="$$ARGS --max_cores $(ROB_MAX_CORES)"; fi; \
	if [ -n "$(ROB_CORES)" ]; then ARGS="$$ARGS --cores $(ROB_CORES)"; fi; \
	if [ -n "$(ROB_LAGS)" ]; then ARGS="$$ARGS --lags $(ROB_LAGS)"; fi; \
	if [ "$(ROB_CLEANUP)" = "1" ]; then ARGS="$$ARGS --cleanup"; fi; \
	echo "    Args: $$ARGS"; \
	$(PYTHON) -m src.eval_robustness \
		--cc_config $(CC_CFG) \
		--outdir $(ROB_OUTDIR) \
		--n_files $(ROB_NFILES) \
		--repeats $(ROB_REPEATS) \
		--window_sec $(ROB_WINDOW_SEC) \
		--njobs_complexity $(ROB_NJOBS_COMP) \
		$$ARGS

robustness: robust_only

# Replot robustness (no rerun): reads benchmark_results.csv and regenerates plots
rob_replot:
	@echo ">>> Replotting robustness results from CSV in $(ROB_OUTDIR)"
	@$(PYTHON) - << 'EOF' ;\
import pandas as pd ;\
from pathlib import Path ;\
from src.eval_robustness import plot_results ;\
out_dir = Path("$(ROB_OUTDIR)") ;\
csv_path = out_dir / "benchmark_results.csv" ;\
df = pd.read_csv(csv_path) ;\
plot_results(df, out_dir) ;\
print("Replotted:", out_dir / "plots") ;\
EOF

# ============================================================
# FULL PIPELINE
# ============================================================
all: cc stack disp eval
	@echo ">>> Full pipeline completed"

# ============================================================
# CLEANING
# ============================================================
clean_cc_raw:
	@echo ">>> Cleaning raw NCF outputs (data/ncf_raw)"
	@rm -rf data/ncf_raw/*
	@mkdir -p data/ncf_raw

clean_stacks:
	@echo ">>> Cleaning stacked NCF outputs (data/ncf_stacks)"
	@rm -rf data/ncf_stacks/*
	@mkdir -p data/ncf_stacks

clean_disp:
	@echo ">>> Cleaning dispersion outputs (results/dispersion)"
	@rm -rf results/dispersion/*
	@mkdir -p results/dispersion

clean_eval:
	@echo ">>> Cleaning eval outputs ($(EVAL_OUTDIR))"
	@rm -rf $(EVAL_OUTDIR)/*
	@mkdir -p $(EVAL_OUTDIR)

clean_robust:
	@echo ">>> Cleaning robustness outputs ($(ROB_ROOT))"
	@rm -rf $(ROB_ROOT)/*
	@mkdir -p $(ROB_ROOT)

clean: clean_cc_raw clean_stacks clean_disp clean_eval clean_robust
	@echo ">>> Clean complete"

clean_all:
	@echo ">>> Removing output directories entirely"
	@rm -rf data/ncf_raw data/ncf_stacks results/dispersion $(EVAL_OUTDIR) $(ROB_ROOT)
	@echo ">>> Removed. Recreate by running targets."