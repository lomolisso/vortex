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
# Targets:
#   make all                  — synthesize all 5 configurations
#   make libs                 — copy .db files into libs/
#   make <config>             — synthesize one configuration
#   make elab_<config>        — elaboration-only check for one configuration
#   make flist_<config>       — generate filelist only (no DC)
#   make clean                — remove generated filelists, logs, results, reports
#=======================================================================

SCRIPT_DIR  := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
GPGPU_ROOT  := $(abspath $(SCRIPT_DIR)../../../..)
# SRAM_DIR and STDCELLS source paths on this machine installation.
# libs/ ships pre-populated with the repo; these are only used by 'make libs'
# to refresh the copies from source.
SRAM_DIR    := $(GPGPU_ROOT)/../project/synth/synth_vortex/sram_output
LIBS_DIR    := $(SCRIPT_DIR)libs

CONFIGS := 1c1n4w4t 1c2n4w4t 2c2n4w4t_l2 4c2n4w4t_l2l3 4c4n8w4t_l2l3

#-----------------------------------------------------------------------
# Per-configuration preprocessor defines
# These are forwarded to gen_dc_sources.sh as -D flags and become
# +define+ entries in the generated filelist consumed by DC.
#-----------------------------------------------------------------------
DEFINES_1c1n4w4t      := -DNUM_CLUSTERS=1 -DNUM_CORES=1 -DNUM_WARPS=4 -DNUM_THREADS=4

DEFINES_1c2n4w4t      := -DNUM_CLUSTERS=1 -DNUM_CORES=2 -DNUM_WARPS=4 -DNUM_THREADS=4

DEFINES_2c2n4w4t_l2   := -DNUM_CLUSTERS=2 -DNUM_CORES=2 -DNUM_WARPS=4 -DNUM_THREADS=4 \
                          -DL2_ENABLE -DL2_CACHE_SIZE=131072

DEFINES_4c2n4w4t_l2l3 := -DNUM_CLUSTERS=4 -DNUM_CORES=2 -DNUM_WARPS=4 -DNUM_THREADS=4 \
                          -DL2_ENABLE -DL3_ENABLE \
                          -DL2_CACHE_SIZE=131072 -DL3_CACHE_SIZE=524288

DEFINES_4c4n8w4t_l2l3 := -DNUM_CLUSTERS=4 -DNUM_CORES=4 -DNUM_WARPS=8 -DNUM_THREADS=4 \
                          -DL2_ENABLE -DL3_ENABLE \
                          -DL2_CACHE_SIZE=262144 -DL3_CACHE_SIZE=524288

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
# Top-level targets
#-----------------------------------------------------------------------
.PHONY: all libs clean $(CONFIGS) \
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
# Per-config pattern: synthesize, elaborate, and generate filelist
#
# Each config target depends on libs (all .db files present) and its
# filelist.  DC is invoked with CONFIG set via -x so dc_syn.tcl /
# dc_elab.tcl can write to the correct reports/ and results/ subdir.
#-----------------------------------------------------------------------
define CONFIG_RULES

.PHONY: $(1) elab_$(1) flist_$(1)

FLIST_$(1) := flists/dc_flist_$(1).f

$(FLIST_$(1)):
	mkdir -p logs
	./gen_dc_sources.sh $(DEFINES_$(1)) -O $(FLIST_$(1))

flist_$(1): $(FLIST_$(1))

$(1): $(FLIST_$(1))
	mkdir -p runs/$(1)/logs
	cd runs/$(1) && dc_shell -x "set CONFIG $(1)" -f ../../dc_syn.tcl 2>&1 | tee logs/$(1)_syn.log

elab_$(1): $(FLIST_$(1))
	mkdir -p runs/$(1)/logs
	cd runs/$(1) && dc_shell -x "set CONFIG $(1)" -f ../../dc_elab.tcl 2>&1 | tee logs/$(1)_elab.log

endef

$(foreach cfg,$(CONFIGS),$(eval $(call CONFIG_RULES,$(cfg))))

#-----------------------------------------------------------------------
# clean — remove all generated artifacts (libs/ kept intentionally)
#-----------------------------------------------------------------------
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
