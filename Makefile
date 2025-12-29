# ============================================================
# DAS Ambient Noise Interferometry Pipeline
# Author: Benz Poobua
# ============================================================

# -----------------------
# Config files
# -----------------------
CC_CFG      := cc.yaml
DISP_CFG    := disp.yaml

# -----------------------
# Python executable
# -----------------------
PYTHON      := python

# -----------------------
# Default target
# -----------------------
.DEFAULT_GOAL := help

# -----------------------
# Phony targets (not files)
# -----------------------
.PHONY: help cc stack disp all clean

# ============================================================
# HELP
# ============================================================
help:
	@echo ""
	@echo "DAS ANI Pipeline (Makefile)"
	@echo "---------------------------"
	@echo "Targets:"
	@echo "  make cc        Run cross-correlation (NCF generation)"
	@echo "  make stack     Run stacking (daily / 7d / 15d / 30d)"
	@echo "  make disp      Run dispersion imaging + picking"
	@echo "  make all       Run full pipeline: cc → stack → disp"
	@echo "  make clean     Remove generated outputs"
	@echo ""
	@echo "Override examples:"
	@echo "  make disp DISP_STACK=30d NJOBS=12"
	@echo ""

# ============================================================
# CROSS-CORRELATION
# ============================================================
cc:
	@echo ">>> Running cross-correlation"
	$(PYTHON) -m src.cc --config $(CC_CFG)

# ============================================================
# STACKING (depends on cc)
# ============================================================
stack: cc
	@echo ">>> Running stacking"
	$(PYTHON) -m src.stack --config $(CC_CFG)

# ============================================================
# DISPERSION (depends on stack)
# ============================================================
# Optional overrides:
#   make disp DISP_STACK=30d NJOBS=12
DISP_STACK ?= daily
NJOBS      ?= 4

disp: stack
	@echo ">>> Running dispersion (stack=$(DISP_STACK), njobs=$(NJOBS))"
	$(PYTHON) -m src.disp_pick \
		--config $(DISP_CFG) \
		--stack_window $(DISP_STACK) \
		--njobs $(NJOBS)

# ============================================================
# FULL PIPELINE
# ============================================================
all: cc stack disp
	@echo ">>> Full pipeline completed"

# ============================================================
# CLEAN
# ============================================================
clean:
	@echo ">>> Cleaning generated data"
	rm -rf data/ncf_raw/*
	rm -rf data/ncf_stacks/*
	rm -rf results/dispersion/*
