# ============================================================
# DAS Ambient Noise Interferometry Pipeline
# Author: Benz Poobua
# ============================================================

# -----------------------
# Config files
# -----------------------
CC_CFG      ?= configs/urban_cc.yaml
EVAL_OUT    ?= data/benchmarks/urban

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
.PHONY: help cc stack eval test paths

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
	@echo "  make eval       Run the benchmark suite (preprocess/scaling/lag)"
	@echo "  make test       Run the pytest suite"
	@echo "  make paths      Print current variable configuration"
	@echo ""
	@echo "Override examples:"
	@echo "  make cc CC_CFG=configs/urban_cc_gpu.yaml"
	@echo "  make eval EVAL_OUT=data/benchmarks/urban_gpu"
	@echo ""

# ============================================================
# Sanity print
# ============================================================
paths:
	@echo "CC_CFG   = $(CC_CFG)"
	@echo "EVAL_OUT = $(EVAL_OUT)"
	@echo "PYTHON   = $(PYTHON)"

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

# ============================================================
# BENCHMARKS (preprocess backends + scaling + lag sweep)
# ============================================================
eval:
	@echo ">>> Running benchmark suite"
	@mkdir -p $(EVAL_OUT)
	$(PYTHON) -m src.eval --cc_config $(CC_CFG) --outdir $(EVAL_OUT)

# ============================================================
# TESTS
# ============================================================
test:
	@echo ">>> Running pytest"
	$(PYTHON) -m pytest tests -q
