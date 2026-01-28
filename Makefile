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
NJOBS      ?= 4
DISP_WINDOWS ?=
EVAL_OUTDIR ?= data/runlogs/eval_modes
EVAL_NJOBS  ?= 4
EVAL_TITLE  ?=
EVAL_LOGY   ?= 0

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
	all \
	clean clean_cc_raw clean_stacks clean_disp clean_eval clean_all \
	paths

# ============================================================
# HELP
# ============================================================
help:
	@echo ""
	@echo "DAS ANI Pipeline (Makefile)"
	@echo "---------------------------"
	@echo "Dependency targets:"
	@echo "  make cc        Run cross-correlation"
	@echo "  make stack     Run stacking (depends on cc)"
	@echo "  make disp      Run dispersion (depends on stack)"
	@echo "  make eval      Run eval_modes (depends on disp)"
	@echo "  make all       Run full pipeline: cc → stack → disp → eval"
	@echo ""
	@echo "No-dependency targets:"
	@echo "  make cc_only       Run cc only (no deps)"
	@echo "  make stack_only    Run stack only (no deps)"
	@echo "  make disp_only     Run dispersion only (no deps)"
	@echo "  make eval_only     Run eval_modes only (no deps)"
	@echo ""
	@echo "Cleaning targets:"
	@echo "  make clean         Remove all generated outputs (keep dirs)"
	@echo "  make clean_cc_raw  Remove raw NCF outputs only (data/ncf_raw)"
	@echo "  make clean_stacks  Remove stacked NCF outputs only (data/ncf_stacks)"
	@echo "  make clean_disp    Remove dispersion outputs only (results/dispersion)"
	@echo "  make clean_eval    Remove eval outputs only ($(EVAL_OUTDIR))"
	@echo "  make clean_all     Remove output directories entirely"
	@echo ""
	@echo "Override examples:"
	@echo "  make cc_only CC_CFG=configs/cc.yaml"
	@echo "  make disp_only DISP_WINDOWS=\"daily 7d 15d\" NJOBS=12"
	@echo "  make eval_only EVAL_NJOBS=8 EVAL_LOGY=1 EVAL_TITLE=\"GPU test\""
	@echo ""

# ============================================================
# Sanity print
# ============================================================
paths:
	@echo "CC_CFG      = $(CC_CFG)"
	@echo "DISP_CFG    = $(DISP_CFG)"
	@echo "DISP_WINDOWS= $(DISP_WINDOWS)"
	@echo "NJOBS       = $(NJOBS)"
	@echo "EVAL_OUTDIR = $(EVAL_OUTDIR)"
	@echo "EVAL_NJOBS  = $(EVAL_NJOBS)"
	@echo "EVAL_TITLE  = $(EVAL_TITLE)"
	@echo "EVAL_LOGY   = $(EVAL_LOGY)"
	@echo "PYTHON      = $(PYTHON)"

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
# DISP_WINDOWS can be overridden on CLI:
#   make disp_only DISP_WINDOWS="daily 7d 15d" NJOBS=12
# ============================================================
disp_only:
	@echo ">>> Running dispersion (no deps) windows='$(DISP_WINDOWS)' njobs=$(NJOBS)"
	@mkdir -p results/dispersion
	@if [ -z "$(DISP_WINDOWS)" ]; then \
		$(PYTHON) -m src.disp_pick --config $(DISP_CFG) --njobs $(NJOBS); \
	else \
		$(PYTHON) -m src.disp_pick --config $(DISP_CFG) --stack_windows $(DISP_WINDOWS) --njobs $(NJOBS); \
	fi

disp: stack disp_only

# ============================================================
# EVALUATION (depends on disp)
# Adds the command you requested:
#   python -m src.eval_modes --cc_config ... --disp_config ... --outdir ... --mmap --plots --logy_runtime
# Extra knobs:
#   EVAL_NJOBS (pair-eval parallelism), EVAL_TITLE, EVAL_LOGY (0/1)
# ============================================================
eval_only:
	@echo ">>> Running eval_modes (no deps) outdir=$(EVAL_OUTDIR) njobs=$(EVAL_NJOBS)"
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

clean: clean_cc_raw clean_stacks clean_disp clean_eval
	@echo ">>> Clean complete"

clean_all:
	@echo ">>> Removing output directories entirely"
	@rm -rf data/ncf_raw data/ncf_stacks results/dispersion $(EVAL_OUTDIR)
	@echo ">>> Removed. Recreate by running targets."