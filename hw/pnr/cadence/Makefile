# Copyright © 2019-2023
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#=======================================================================
# Vortex GPGPU — Cadence Innovus PnR Makefile
# Target: NanGate45 stdcells + bsg_fakeram SRAM macros
#
# Two PnR targets mirror the synthesis flow in hw/syn/synopsys/Makefile:
#
#   single-core : PnR of VX_socket_top (one VX_core + its L1 I$/D$ +
#                 lmem + register file). Macros are bsg_fakeram only.
#                 Produces the database that the LEF/LIB extraction step
#                 turns into VX_socket_top.lef / VX_socket_top.lib.
#
#   full-vortex : PnR of Vortex (4 × VX_socket_top blackboxes + shared
#                 L2 cache + top-level glue). Requires the core .lef/.lib
#                 from the single-core step to be present in libs/.
#
# Usage:
#   make <config> STAGE=<number>           use just the stage number
#   make <config> STAGE=<full_name>        or the full stage name
#
#   e.g.  make single-core STAGE=1        init design
#         make single-core STAGE=3        floorplan
#         make full-vortex STAGE=5 THREADS=8
#         make single-core STAGE=3 UTIL=0.65
#
# Stages (run in order, one at a time):
#    1  01_init_design      2  02_power_connect    3  03_floorplan
#    4  04_power_plan       5  05_placement        6  06_pre_cts_opt
#    7  07_cts              8  08_post_cts_opt     9  09_routing
#   10  10_post_route_opt  11  11_reports         12  12_outputs
#
# Each stage loads the checkpoint saved by the previous stage and opens
# the Innovus GUI so you can inspect the result after it completes.
# The run fails immediately if the preceding checkpoint is missing.
#
# After a stage finishes you can also open it any time with:
#   ./open_checkpoint.sh <config> <stage>
#
# Variables:
#   STAGE       — (required) stage to run
#   THREADS     — CPU threads for Innovus (default: 8)
#   EFFORT      — optimization effort: low | medium | high (default: low)
#   UTIL        — stdcell utilization target (default: 0.70)
#   MARGIN_PCT  — floorplan guard-band margin (default: 0.10)
#   ASPECT      — die aspect ratio W/H (default: 1.0)
#   CORE_MARGIN — core-to-die margin in µm (default: 5.0)
#=======================================================================

SCRIPT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
override SYN_DIR  := $(shell cd "$(SCRIPT_DIR)../../syn/synopsys" 2>/dev/null && pwd)
override LIBS_DIR := $(SYN_DIR)/libs

SCRIPTS_DIR := $(SCRIPT_DIR)../../scripts

UTIL        ?= 0.70
MARGIN_PCT  ?= 0.10
ASPECT      ?= 1.0
CORE_MARGIN ?= 5.0
EFFORT      ?= low
THREADS     ?= 8
STAGE       ?=

#-----------------------------------------------------------------------
# Per-configuration area-budget script and Innovus top-module name.
#
# AREA_SCRIPT_<cfg> : Python script that emits runs/<cfg>/floorplan_budget.tcl
# TOP_<cfg>         : top cell passed through to Innovus as $TOP; also
#                     drives the <TOP>_netlist.v / <TOP>.sdc filenames that
#                     00_config.tcl looks up in $SYN_DIR/runs/<cfg>/results/.
#-----------------------------------------------------------------------

CONFIGS := single-core full-vortex

AREA_SCRIPT_single-core := $(SCRIPTS_DIR)/compute_single_core_area_budget.py
TOP_single-core         := VX_socket_top

AREA_SCRIPT_full-vortex := $(SCRIPTS_DIR)/compute_full_vortex_area_budget.py
TOP_full-vortex         := Vortex

#-----------------------------------------------------------------------
# Hard-macro export knobs (used by the extract-macro target below).
#
# MACRO_CELL : name recorded inside the emitted LEF MACRO / LIB cell.
#              Keep it aligned with the single-core top so the full-
#              vortex netlist instantiation resolves.
# VIEW_NAME  : MMMC analysis view(s) to extract timing from. Defaults
#              to the single view created by mmmc.tcl.
# EXPORT_DIR : where the generated .lef/.lib land. This is the same
#              path that the full-vortex config (01_init_design.tcl,
#              mmmc.tcl) looks in before falling back to $LIBS_DIR.
#-----------------------------------------------------------------------
MACRO_CELL ?= VX_socket_top
VIEW_NAME  ?= typView
EXPORT_DIR := $(SCRIPT_DIR)export/single-core

.PHONY: clean extract-macro export-blackbox $(CONFIGS)

define PNR_RULES

.PHONY: $(1)

$(1):
	@if [ -z "$(STAGE)" ]; then \
	    echo ""; \
	    echo "ERROR: STAGE is required.  e.g.  make $(1) STAGE=1"; \
	    echo ""; \
	    echo "   1  01_init_design      2  02_power_connect    3  03_floorplan"; \
	    echo "   4  04_power_plan       5  05_placement        6  06_pre_cts_opt"; \
	    echo "   7  07_cts              8  08_post_cts_opt     9  09_routing"; \
	    echo "  10  10_post_route_opt  11  11_reports         12  12_outputs"; \
	    echo ""; \
	    exit 1; \
	fi
	mkdir -p runs/$(1)/logs runs/$(1)/reports runs/$(1)/results
	python3 $(AREA_SCRIPT_$(1)) \
	    --runs_dir    $(SYN_DIR)/runs \
	    --libs_dir    $(LIBS_DIR) \
	    --utilization $(UTIL) \
	    --margin_pct  $(MARGIN_PCT) \
	    --aspect      $(ASPECT) \
	    --core_margin $(CORE_MARGIN) \
	    --emit_tcl    runs/$(1)/floorplan_budget.tcl
	@echo ""
	@echo "================================================================"
	@echo " Stage: $(STAGE)   Config: $(1)   Top: $(TOP_$(1))"
	@echo "================================================================"
	@echo ""
	cd runs/$(1) && \
	    CONFIG=$(1) TOP=$(TOP_$(1)) \
	    LIBS_DIR=$(LIBS_DIR) SYN_DIR=$(SYN_DIR) \
	    EFFORT=$(EFFORT) STAGE=$(STAGE) \
	    innovus -cpus $(THREADS) \
	    -files ../../pnr.tcl \
	    -log logs/run_$(STAGE).log \
	    -overwrite

endef

$(foreach cfg,$(CONFIGS),$(eval $(call PNR_RULES,$(cfg))))

#-----------------------------------------------------------------------
# Hard-macro extraction (bottom-up hierarchical flow)
#
# Loads the finalised single-core database (runs/single-core/checkpoints
# /after_12_outputs) and runs the three export sub-stages that live in
# scripts/export_{prep,lef,etm}.tcl via the extract_macro.tcl wrapper:
#
#     1. export_prep : snap I/O pins, assert M9/M10 are empty (reserved
#                      for full-vortex over-the-macro routing).
#     2. export_lef  : write_lef_macromodel → $(EXPORT_DIR)/$(MACRO_CELL).lef
#     3. export_etm  : do_extract_model     → $(EXPORT_DIR)/$(MACRO_CELL).lib
#
# "export-blackbox" is an alias kept for discoverability — the two
# names refer to the same artefact set.
#
# Usage:
#   make extract-macro                             # default view, default cell
#   make extract-macro VIEW_NAME=setup_view        # pick a specific MMMC view
#   make extract-macro MACRO_CELL=VX_socket_top_v2 # version the emitted cell
#-----------------------------------------------------------------------
export-blackbox: extract-macro

extract-macro:
	@if [ ! -f runs/single-core/checkpoints/after_12_outputs ]; then \
	    echo ""; \
	    echo "ERROR: runs/single-core/checkpoints/after_12_outputs is missing."; \
	    echo "       Run 'make single-core STAGE=12' first (and make sure"; \
	    echo "       STAGE=9 was re-run after the M8 routing cap was added"; \
	    echo "       to 09_routing.tcl, otherwise export_prep will reject"; \
	    echo "       the database for having routes on M9/M10)."; \
	    echo ""; \
	    exit 1; \
	fi
	mkdir -p $(EXPORT_DIR) runs/single-core/logs
	@echo ""
	@echo "================================================================"
	@echo " Hard-macro extraction: single-core → $(MACRO_CELL).{lef,lib}"
	@echo " Export dir : $(EXPORT_DIR)"
	@echo " View(s)    : $(VIEW_NAME)"
	@echo "================================================================"
	@echo ""
	cd runs/single-core && \
	    CONFIG=single-core TOP=$(MACRO_CELL) \
	    MACRO_CELL=$(MACRO_CELL) VIEW_NAME="$(VIEW_NAME)" \
	    LIBS_DIR=$(LIBS_DIR) SYN_DIR=$(SYN_DIR) \
	    innovus -cpus $(THREADS) \
	    -files ../../extract_macro.tcl \
	    -log logs/extract_macro.log \
	    -overwrite
	@echo ""
	@echo "Exported:"
	@echo "  $(EXPORT_DIR)/$(MACRO_CELL).lef"
	@echo "  $(EXPORT_DIR)/$(MACRO_CELL).lib"
	@echo ""
	@echo "The full-vortex config (01_init_design.tcl, mmmc.tcl) will"
	@echo "pick these up automatically on its next run."

#-----------------------------------------------------------------------
# clean
#-----------------------------------------------------------------------
clean:
	rm -rf runs/
	rm -rf export/
	rm -f  *.cmd* *.log* innovus.lck
