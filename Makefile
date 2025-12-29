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
PYTHON      ?= python

# -----------------------
# Defaults / overrides
# -----------------------
DISP_STACK ?= daily
NJOBS      ?= 4

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
	all \
	clean clean_cc_raw clean_stacks clean_disp clean_all \
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
	@echo "  make all       Run full pipeline: cc → stack → disp"
	@echo ""
	@echo "No-dependency targets:"
	@echo "  make cc_only       Run cc only (no deps)"
	@echo "  make stack_only    Run stack only (no deps)"
	@echo "  make disp_only     Run disp only (no deps)"
	@echo ""
	@echo "Cleaning targets:"
	@echo "  make clean         Remove all generated outputs (keep dirs)"
	@echo "  make clean_cc_raw  Remove raw NCF outputs only (data/ncf_raw)"
	@echo "  make clean_stacks  Remove stacked NCF outputs only (data/ncf_stacks)"
	@echo "  make clean_disp    Remove dispersion outputs only (results/dispersion)"
	@echo "  make clean_all     Remove output directories entirely"
	@echo ""
	@echo "Override examples:"
	@echo "  make cc_only CC_CFG=configs/cc.yaml"
	@echo "  make disp_only DISP_STACK=30d NJOBS=12"
	@echo ""

# ============================================================
# Sanity print
# ============================================================
paths:
	@echo "CC_CFG     = $(CC_CFG)"
	@echo "DISP_CFG   = $(DISP_CFG)"
	@echo "DISP_STACK = $(DISP_STACK)"
	@echo "NJOBS      = $(NJOBS)"
	@echo "PYTHON     = $(PYTHON)"

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

# DISPERSION
# ============================================================
# DISP_WINDOWS can be overridden on CLI:
#   make disp_only DISP_WINDOWS="daily 7d 15d" NJOBS=12
DISP_WINDOWS ?=

disp_only:
	@echo ">>> Running dispersion (no deps) windows='$(DISP_WINDOWS)' njobs=$(NJOBS)"
	@mkdir -p results/dispersion
	@if [ -z "$(DISP_WINDOWS)" ]; then \
		$(PYTHON) -m src.disp_pick --config $(DISP_CFG) --njobs $(NJOBS); \
	else \
		$(PYTHON) -m src.disp_pick --config $(DISP_CFG) --stack_windows $(DISP_WINDOWS) --njobs $(NJOBS); \
	fi

disp: stack disp_onlys

# ============================================================
# FULL PIPELINE
# ============================================================
all: cc stack disp
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

clean: clean_cc_raw clean_stacks clean_disp
	@echo ">>> Clean complete"

clean_all:
	@echo ">>> Removing output directories entirely"
	@rm -rf data/ncf_raw data/ncf_stacks results/dispersion
	@echo ">>> Removed. Recreate by running targets."