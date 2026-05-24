# ============================================================
# DAS Ambient Noise Interferometry Pipeline
# Author: Benz Poobua
# ============================================================

# -----------------------
# Config files
# -----------------------
CC_CFG      ?= configs/cc.yaml

# -----------------------
# Python executable
# -----------------------
VENV        ?= das_ani
PYTHON      ?= $(VENV)/bin/python

# -----------------------
# Default target
# -----------------------
.DEFAULT_GOAL := help

# -----------------------
# Phony targets
# -----------------------
.PHONY: help cc stack paths

# ============================================================
# HELP
# ============================================================
help:
	@echo ""
	@echo "DAS ANI Pipeline (Lightweight Makefile)"
	@echo "---------------------------------------"
	@echo "Available targets:"
	@echo "  make cc         Run cross-correlation (standalone)"
	@echo "  make stack      Run stacking (standalone)"
	@echo "  make paths      Print current variable configuration"
	@echo ""
	@echo "Override examples:"
	@echo "  make cc CC_CFG=configs/cc_fast.yaml"
	@echo ""

# ============================================================
# Sanity print
# ============================================================
paths:
	@echo "CC_CFG = $(CC_CFG)"
	@echo "PYTHON = $(PYTHON)"

# ============================================================
# CROSS-CORRELATION
# ============================================================
cc:
	@echo ">>> Running cross-correlation"
	@mkdir -p data/ncf_raw
	$(PYTHON) -m src.cc --config $(CC_CFG) --verbose

# ============================================================
# STACKING
# ============================================================
stack:
	@echo ">>> Running stacking"
	@mkdir -p data/ncf_stacks
	$(PYTHON) -m src.stack --config $(CC_CFG) --verbose