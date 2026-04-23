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
# Vortex GPGPU — Synopsys DC Makefile
# Target: NanGate45 stdcells + bsg_fakeram SRAM macros
#
# Two configurations are synthesized:
#
#   single-core    Synthesizes the single-core "block" in isolation:
#                  VX_socket_top = 1 VX_core + its private I$, D$, lmem
#                  and register file. Defines force SOCKET_SIZE=1 so the
#                  socket holds exactly one core. The netlist produced
#                  here is what is handed to PnR and, later, turned into
#                  a .lef/.lib macro for reuse in the full-vortex run.
#
#   full-vortex    Synthesizes the complete Vortex GPGPU: 1 cluster,
#                  4 cores (NUM_CORES=4, SOCKET_SIZE=1, so each core has
#                  its own private L1s, lmem and regfile), with a shared
#                  L2 across all four cores. Each core block is still a
#                  VX_socket_top instance — the same module synthesized
#                  in single-core — which is exactly the boundary used
#                  later for blackboxing.
#
# Targets:
#   make all                  — synthesize both configurations
#   make libs                 — copy .db files into libs/
#   make <config>             — synthesize one configuration
#   make elab_<config>        — elaboration-only check for one configuration
#   make flist_<config>       — generate filelist only (no DC)
#   make socket-db            — compile VX_socket_top.lib → libs/VX_socket_top.db
#   make clean                — remove generated filelists, logs, results, reports
#
# Bottom-up hierarchical synthesis
#   `make full-vortex` runs DC in bottom-up mode: VX_socket_top is
#   treated as a pre-characterized hard-macro blackbox linked from
#   libs/VX_socket_top.db, NOT re-synthesized from RTL. The .db is
#   auto-built from the socket's Liberty (.lib) as a prerequisite.
#
#   Prerequisite (run once after you have a routed single-core):
#       cd ../../pnr/cadence && make single-core STAGE=12
#       cd ../../pnr/cadence && make extract-macro
#
#   After that, this directory's `make full-vortex` is self-contained:
#   it finds the exported .lib, compiles it to .db on demand, and
#   synthesizes the top-level wrapper around the blackboxed sockets.
#
#   Override: to force a flat re-synth of the whole GPGPU (e.g. to
#   compare QoR), set `BLACKBOX_SOCKET_full-vortex=0` on the command
#   line:
#       make full-vortex BLACKBOX_SOCKET_full-vortex=0
#=======================================================================

SHELL       := /bin/bash
.SHELLFLAGS := -e -o pipefail -c
.DELETE_ON_ERROR:

SCRIPT_DIR  := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
GPGPU_ROOT  := $(abspath $(SCRIPT_DIR)../../../..)
# SRAM_DIR and STDCELLS source paths on this machine installation.
# libs/ ships pre-populated with the repo; these are only used by 'make libs'
# to refresh the copies from source.
SRAM_DIR    := $(GPGPU_ROOT)/../project/synth/synth_vortex/sram_output
LIBS_DIR    := $(SCRIPT_DIR)libs

CONFIGS := single-core full-vortex

#-----------------------------------------------------------------------
# Per-configuration preprocessor defines and synthesis top module.
# DEFINES_* are forwarded to gen_dc_sources.sh as -D flags; TOP_* tells
# DC which module to elaborate.
#
# SOCKET_SIZE=1 is forced in both cases so that a "socket" is always
# exactly one VX_core with its own private I$/D$. Without this override
# VX_config.vh would pick SOCKET_SIZE=MIN(4, NUM_CORES), which in the
# full-vortex case would collapse all four cores into a single socket
# sharing one I$ and one D$ — breaking the blackbox boundary.
#-----------------------------------------------------------------------

# single-core: VX_socket_top wrapping one VX_core + its L1 caches, lmem
# and register file. No L2. This is the unit we PnR and later package
# as a .lef/.lib macro.
DEFINES_single-core := -DNUM_CLUSTERS=1 -DNUM_CORES=1 -DNUM_WARPS=4 -DNUM_THREADS=4 \
                       -DSOCKET_SIZE=1
TOP_single-core     := VX_socket_top

# full-vortex: full Vortex GPGPU with 4 cores (4 VX_socket_top instances
# inside one VX_cluster) sharing an L2 cache. VX_cluster instantiates
# VX_socket_top directly under ASIC_SYNTHESIS, which is the boundary
# that will later be replaced by the single-core .lib/.lef macro.
DEFINES_full-vortex := -DNUM_CLUSTERS=1 -DNUM_CORES=4 -DNUM_WARPS=4 -DNUM_THREADS=4 \
                       -DSOCKET_SIZE=1 \
                       -DL2_ENABLE -DL2_CACHE_SIZE=131072
TOP_full-vortex     := Vortex

#-----------------------------------------------------------------------
# SRAM macros — names only (files already pre-copied into libs/)
#-----------------------------------------------------------------------
SRAM_MACROS := \
    sram_64x512_1rw \
    sram_256x512_1rw \
    sram_1024x32_1rw \
    sram_64x24_1r1w \
    sram_256x24_1r1w \
    sram_64x128_1r1w \
    sram_128x128_1r1w

SRAM_DBS    := $(addprefix $(LIBS_DIR)/,$(addsuffix .db,$(SRAM_MACROS)))
SRAM_VS     := $(addprefix $(LIBS_DIR)/,$(addsuffix .v, $(SRAM_MACROS)))
STDCELLS_DB := $(LIBS_DIR)/stdcells.db

#-----------------------------------------------------------------------
# VX_socket_top hard-macro artifacts (bottom-up flow)
#
# The canonical .lib is the one Innovus's `make extract-macro` drops
# into pnr/cadence/export/single-core/. Historically users also hand-
# copied the file into libs/, so we keep that as a fallback. Whichever
# path exists is used to compile VX_socket_top.db on demand.
#
# BLACKBOX_SOCKET_<config> controls whether dc_syn.tcl / dc_elab.tcl
# should blackbox VX_socket_top for that config. Default: only the
# full-vortex config does so; single-core must synthesize the socket
# from RTL (it's the block the .lib is *built from*).
#-----------------------------------------------------------------------
PNR_EXPORT_DIR            := $(abspath $(SCRIPT_DIR)../../pnr/cadence/export/single-core)
VX_SOCKET_LIB_CANDIDATES  := $(PNR_EXPORT_DIR)/VX_socket_top.lib $(LIBS_DIR)/VX_socket_top.lib
VX_SOCKET_LIB             := $(firstword $(wildcard $(VX_SOCKET_LIB_CANDIDATES)))
VX_SOCKET_DB              := $(LIBS_DIR)/VX_socket_top.db

BLACKBOX_SOCKET_single-core ?= 0
BLACKBOX_SOCKET_full-vortex ?= 1

#-----------------------------------------------------------------------
# Top-level targets
#-----------------------------------------------------------------------
.PHONY: all libs clean socket-db $(CONFIGS) \
        $(addprefix elab_,$(CONFIGS)) \
        $(addprefix flist_,$(CONFIGS))

all: $(CONFIGS)

#-----------------------------------------------------------------------
# libs — copy .db and .v files into libs/ (idempotent; files ship with
# the repo but this target lets you refresh them from the source dirs)
#-----------------------------------------------------------------------
libs: $(STDCELLS_DB) $(SRAM_DBS) $(SRAM_VS)

$(STDCELLS_DB): $(GPGPU_ROOT)/../project/stdcells.db
	mkdir -p $(LIBS_DIR)
	cp $< $@

$(LIBS_DIR)/%.db: $(SRAM_DIR)/%/%.db
	mkdir -p $(LIBS_DIR)
	cp $< $@

$(LIBS_DIR)/%.v: $(SRAM_DIR)/%/%.v
	mkdir -p $(LIBS_DIR)
	cp $< $@

#-----------------------------------------------------------------------
# socket-db — compile VX_socket_top.lib into VX_socket_top.db
#
# The rule uses $(VX_SOCKET_LIB), which resolves to the first .lib
# found in $(VX_SOCKET_LIB_CANDIDATES). If no .lib exists anywhere, we
# fail the build with a clear message pointing at the extraction step
# the user needs to run first (rather than letting dc_shell attempt a
# read_lib on an empty path).
#-----------------------------------------------------------------------
socket-db: $(VX_SOCKET_DB)

$(VX_SOCKET_DB): $(VX_SOCKET_LIB) compile_lib_to_db.tcl
	@if [ -z "$(VX_SOCKET_LIB)" ]; then \
	    echo ""; \
	    echo "ERROR: VX_socket_top.lib not found in either:"; \
	    echo "  $(PNR_EXPORT_DIR)/VX_socket_top.lib   (canonical)"; \
	    echo "  $(LIBS_DIR)/VX_socket_top.lib        (hand-copied fallback)"; \
	    echo ""; \
	    echo "Produce it by routing the single-core and running the"; \
	    echo "macro-extraction flow:"; \
	    echo "  cd ../../pnr/cadence"; \
	    echo "  make single-core STAGE=12    # finish PnR if not done"; \
	    echo "  make extract-macro           # emits .lef/.lib"; \
	    echo ""; \
	    exit 1; \
	fi
	@mkdir -p $(LIBS_DIR) logs
	@echo ""
	@echo "================================================================"
	@echo " Compile Liberty → DB"
	@echo "   src: $(VX_SOCKET_LIB)"
	@echo "   out: $(VX_SOCKET_DB)"
	@echo "================================================================"
	@echo ""
	LIB_SRC="$(VX_SOCKET_LIB)" LIB_DB_OUT="$(VX_SOCKET_DB)" \
	LIB_NAME="VX_socket_top" \
	    dc_shell -f $(SCRIPT_DIR)compile_lib_to_db.tcl \
	    2>&1 | tee logs/compile_lib_to_db.log

#-----------------------------------------------------------------------
# Per-config pattern: synthesize, elaborate, and generate filelist
#
# Each config target depends on libs (all .db files present) and its
# filelist.  DC is invoked with CONFIG set via -x so dc_syn.tcl /
# dc_elab.tcl can write to the correct reports/ and results/ subdir.
#-----------------------------------------------------------------------
#
# BLACKBOX_DEPS_<cfg> expands to the list of prerequisites that a
# full-vortex-style (blackbox=1) run needs before dc_shell starts —
# i.e. the compiled VX_socket_top.db. For configs that don't blackbox,
# the list is empty.
define CONFIG_RULES

.PHONY: $(1) elab_$(1) flist_$(1)

BLACKBOX_DEPS_$(1) := $$(if $$(filter 1,$$(BLACKBOX_SOCKET_$(1))),$$(VX_SOCKET_DB),)

flists/dc_flist_$(1).f:
	mkdir -p logs flists
	./gen_dc_sources.sh $(DEFINES_$(1)) -T $(TOP_$(1)) -O flists/dc_flist_$(1).f

flist_$(1): flists/dc_flist_$(1).f

$(1): flists/dc_flist_$(1).f $$(BLACKBOX_DEPS_$(1))
	mkdir -p runs/$(1)/logs
	cd runs/$(1) && dc_shell -x "set CONFIG $(1); set TOP $(TOP_$(1)); set BLACKBOX_SOCKET $(BLACKBOX_SOCKET_$(1))" -f ../../dc_syn.tcl 2>&1 | tee logs/$(1)_syn.log

elab_$(1): flists/dc_flist_$(1).f $$(BLACKBOX_DEPS_$(1))
	mkdir -p runs/$(1)/logs
	cd runs/$(1) && dc_shell -x "set CONFIG $(1); set TOP $(TOP_$(1)); set BLACKBOX_SOCKET $(BLACKBOX_SOCKET_$(1))" -f ../../dc_elab.tcl 2>&1 | tee logs/$(1)_elab.log

endef

$(foreach cfg,$(CONFIGS),$(eval $(call CONFIG_RULES,$(cfg))))

#-----------------------------------------------------------------------
# clean — remove all generated artifacts (libs/ kept intentionally,
# EXCEPT the auto-built VX_socket_top.db, which is derived and should
# be regenerated from the current .lib on the next full-vortex run.
# Override with KEEP_SOCKET_DB=1 if you want to preserve the binary.
#-----------------------------------------------------------------------
KEEP_SOCKET_DB ?= 0

clean:
	rm -f  dc_flist_*.f
	rm -rf logs/
	rm -rf results/
	rm -rf reports/
	rm -rf work/
	rm -rf runs/
	rm -rf flists/
	rm -rf alib-52/
	rm -f  *.svf *.key *.log *.ddc
	rm -f  *.pvl *.syn *.mr *.pvk
	rm -rf cksum_dir/
	rm -f  moduleparamid
	@if [ "$(KEEP_SOCKET_DB)" != "1" ]; then \
	    rm -f $(VX_SOCKET_DB); \
	fi
