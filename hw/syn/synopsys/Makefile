# Vortex GPGPU — Synopsys DC synthesis.
# Targets: <cfg> | elab_<cfg> | flist_<cfg> | libs | all | clean

SHELL       := /bin/bash
.SHELLFLAGS := -e -o pipefail -c
.DELETE_ON_ERROR:

# GNU Make 3.80 lacks $(abspath); use pwd -P.
SCRIPT_DIR  := $(shell pwd -P)/
LIBS_DIR    := $(SCRIPT_DIR)libs

CONFIGS := 2d-single-core 2d-1c8s4w4t 2d-1c4s4w4t fpu tcu \
           lmem gpr l1_icache_bank l1_dcache_bank l2_bank

# Clock constraints sourced from hw/pnr/cadence/config/<cfg>.toml.
ICACHE_RESIZE := -DICACHE_SIZE=4096 -DICACHE_NUM_WAYS=2

PNR_CFG_DIR := $(SCRIPT_DIR)../../pnr/cadence/config
TOML_GET    := python3 $(SCRIPT_DIR)../../scripts/toml_get.py
toml_get = $(shell $(TOML_GET) $(PNR_CFG_DIR)/$(1).toml $(2))

TARGET_FREQ_MHZ_2d-single-core := $(call toml_get,2d-single-core,clock.target_freq_mhz)
MAX_FANOUT_2d-single-core      := $(call toml_get,2d-single-core,clock.max_fanout)
TARGET_FREQ_MHZ_2d-1c8s4w4t    := $(call toml_get,2d-1c8s4w4t,clock.target_freq_mhz)
MAX_FANOUT_2d-1c8s4w4t         := $(call toml_get,2d-1c8s4w4t,clock.max_fanout)
TARGET_FREQ_MHZ_2d-1c4s4w4t    := $(call toml_get,2d-1c4s4w4t,clock.target_freq_mhz)
MAX_FANOUT_2d-1c4s4w4t         := $(call toml_get,2d-1c4s4w4t,clock.max_fanout)
TARGET_FREQ_MHZ_fpu            := $(call toml_get,fpu,clock.target_freq_mhz)
MAX_FANOUT_fpu                 := $(call toml_get,fpu,clock.max_fanout)
TARGET_FREQ_MHZ_tcu            := $(call toml_get,tcu,clock.target_freq_mhz)
MAX_FANOUT_tcu                 := $(call toml_get,tcu,clock.max_fanout)
TARGET_FREQ_MHZ_lmem           := $(call toml_get,lmem,clock.target_freq_mhz)
MAX_FANOUT_lmem                := $(call toml_get,lmem,clock.max_fanout)
TARGET_FREQ_MHZ_gpr            := $(call toml_get,gpr,clock.target_freq_mhz)
MAX_FANOUT_gpr                 := $(call toml_get,gpr,clock.max_fanout)
TARGET_FREQ_MHZ_l1_icache_bank := $(call toml_get,l1_icache_bank,clock.target_freq_mhz)
MAX_FANOUT_l1_icache_bank      := $(call toml_get,l1_icache_bank,clock.max_fanout)
TARGET_FREQ_MHZ_l1_dcache_bank := $(call toml_get,l1_dcache_bank,clock.target_freq_mhz)
MAX_FANOUT_l1_dcache_bank      := $(call toml_get,l1_dcache_bank,clock.max_fanout)
TARGET_FREQ_MHZ_l2_bank        := $(call toml_get,l2_bank,clock.target_freq_mhz)
MAX_FANOUT_l2_bank             := $(call toml_get,l2_bank,clock.max_fanout)

DEFINES_2d-single-core := -DNUM_CLUSTERS=1 -DNUM_CORES=1 -DNUM_WARPS=4 -DNUM_THREADS=4 \
                          -DSOCKET_SIZE=1 $(ICACHE_RESIZE)
TOP_2d-single-core         := VX_socket_top
SRAM_MACROS_2d-single-core :=
BLACKBOX_CORES_2d-single-core := VX_fpu_top@fpu VX_tcu_top@tcu \
                                 VX_local_mem_top@lmem VX_gpr_top@gpr \
                                 VX_l1_icache_bank_top@l1_icache_bank \
                                 VX_l1_dcache_bank_top@l1_dcache_bank

DEFINES_2d-1c8s4w4t := -DNUM_CLUSTERS=1 -DNUM_CORES=8 -DNUM_WARPS=4 -DNUM_THREADS=4 \
                      -DSOCKET_SIZE=1 -DL2_ENABLE -DL2_CACHE_SIZE=1048576 $(ICACHE_RESIZE)
TOP_2d-1c8s4w4t         := VX_cluster_top
SRAM_MACROS_2d-1c8s4w4t :=
BLACKBOX_CORES_2d-1c8s4w4t := VX_socket_top@2d-single-core \
                              VX_l2_bank_top@l2_bank

DEFINES_2d-1c4s4w4t := -DNUM_CLUSTERS=1 -DNUM_CORES=4 -DNUM_WARPS=4 -DNUM_THREADS=4 \
                      -DSOCKET_SIZE=1 -DL2_ENABLE -DL2_CACHE_SIZE=524288 $(ICACHE_RESIZE)
TOP_2d-1c4s4w4t         := VX_cluster_top
SRAM_MACROS_2d-1c4s4w4t :=
BLACKBOX_CORES_2d-1c4s4w4t := VX_socket_top@2d-single-core \
                              VX_l2_bank_top@l2_bank

DEFINES_fpu := -DNUM_CLUSTERS=1 -DNUM_CORES=1 -DNUM_WARPS=4 -DNUM_THREADS=4 -DSOCKET_SIZE=1
TOP_fpu     := VX_fpu_top
SRAM_MACROS_fpu    :=
BLACKBOX_CORES_fpu :=

DEFINES_tcu := -DNUM_CLUSTERS=1 -DNUM_CORES=1 -DNUM_WARPS=4 -DNUM_THREADS=4 -DSOCKET_SIZE=1
TOP_tcu     := VX_tcu_top
SRAM_MACROS_tcu    :=
BLACKBOX_CORES_tcu :=

# Each leaf's DEFINES must match the parent that consumes it so derived widths line up at link.
DEFINES_lmem := $(DEFINES_2d-single-core)
TOP_lmem         := VX_local_mem_top
SRAM_MACROS_lmem := sram_1024x32_1rw
BLACKBOX_CORES_lmem :=

DEFINES_gpr := $(DEFINES_2d-single-core)
TOP_gpr         := VX_gpr_top
SRAM_MACROS_gpr := sram_64x128_1r1w
BLACKBOX_CORES_gpr :=

DEFINES_l1_icache_bank := $(DEFINES_2d-single-core)
TOP_l1_icache_bank         := VX_l1_icache_bank_top
SRAM_MACROS_l1_icache_bank := sram_32x512_1rw sram_32x24_1r1w
BLACKBOX_CORES_l1_icache_bank :=

DEFINES_l1_dcache_bank := $(DEFINES_2d-single-core)
TOP_l1_dcache_bank         := VX_l1_dcache_bank_top
SRAM_MACROS_l1_dcache_bank := sram_64x512_1rw sram_64x24_1r1w
BLACKBOX_CORES_l1_dcache_bank :=

DEFINES_l2_bank := $(DEFINES_2d-1c8s4w4t)
TOP_l2_bank         := VX_l2_bank_top
SRAM_MACROS_l2_bank := sram_256x512_1rw sram_256x24_1r1w
BLACKBOX_CORES_l2_bank :=

sram_db_path  = $(LIBS_DIR)/sram/$(1)/$(1).db
sram_lib_path = $(LIBS_DIR)/sram/$(1)/$(1).lib
sram_bb_path  = $(LIBS_DIR)/sram/$(1)/$(1).bb.v
sram_dir_path = $(LIBS_DIR)/sram/$(1)

ALL_SRAM_DIRS := $(wildcard $(LIBS_DIR)/sram/*/)
ALL_SRAM_NAMES := $(notdir $(patsubst %/,%,$(ALL_SRAM_DIRS)))
ALL_SRAM_LIBS  := $(foreach m,$(ALL_SRAM_NAMES),$(call sram_lib_path,$(m)))
ALL_SRAM_BBS   := $(foreach m,$(ALL_SRAM_NAMES),$(call sram_bb_path,$(m)))

# BLACKBOX_CORES entries are "module@libdir" → libs/core/<libdir>/<module>.db.
bb_module   = $(word 1,$(subst @, ,$(1)))
bb_libdir   = $(word 2,$(subst @, ,$(1)))
bb_db_path  = $(LIBS_DIR)/core/$(call bb_libdir,$(1))/$(call bb_module,$(1)).db
bb_pair     = $(call bb_module,$(1)):$(call bb_db_path,$(1))

$(foreach c,$(CONFIGS),$(eval SRAM_DBS_$(c)  := $(foreach m,$(SRAM_MACROS_$(c)),$(call sram_db_path,$(m)))))
$(foreach c,$(CONFIGS),$(eval SRAM_BBS_$(c)  := $(foreach m,$(SRAM_MACROS_$(c)),$(call sram_bb_path,$(m)))))
$(foreach c,$(CONFIGS),$(eval SRAM_DIRS_$(c) := $(foreach m,$(SRAM_MACROS_$(c)),$(call sram_dir_path,$(m)))))
$(foreach c,$(CONFIGS),$(eval CORE_DBS_$(c)  := $(foreach e,$(BLACKBOX_CORES_$(c)),$(call bb_db_path,$(e)))))
$(foreach c,$(CONFIGS),$(eval BB_DBS_$(c)    := $(foreach e,$(BLACKBOX_CORES_$(c)),$(call bb_pair,$(e)))))

ALL_SRAM_DBS := $(sort $(foreach c,$(CONFIGS),$(SRAM_DBS_$(c))))
ALL_CORE_DBS := $(sort $(foreach c,$(CONFIGS),$(CORE_DBS_$(c))))

# LD_PRELOAD: RHEL 8.10 libkrb5 ABI mismatch in lc_shell.
LC_PRELOAD := /lib64/libcrypto.so.1.1 /lib64/libk5crypto.so.3 /lib64/libtinfo.so.5
define COMPILE_LIB_TO_DB
	@mkdir -p logs
	@echo "=== Compile Liberty → DB: $< → $@ ==="
	LIB_SRC="$<" LIB_DB_OUT="$@" LD_PRELOAD="$(LC_PRELOAD)" \
	    lc_shell -batch -f $(SCRIPT_DIR)compile_lib_to_db.tcl \
	    2>&1 | tee logs/compile_lib_to_db_$(notdir $*).log
endef

$(LIBS_DIR)/sram/%.db: $(LIBS_DIR)/sram/%.lib compile_lib_to_db.tcl
	$(COMPILE_LIB_TO_DB)

$(LIBS_DIR)/core/%.db: $(LIBS_DIR)/core/%.lib compile_lib_to_db.tcl
	$(COMPILE_LIB_TO_DB)

# .lib and .bb.v are shipped by bsg_fakeram via ../../../../sram/build_macros.sh.

.PHONY: all libs lib-prep clean $(CONFIGS) \
        $(addprefix elab_,$(CONFIGS)) \
        $(addprefix flist_,$(CONFIGS))

all:  $(CONFIGS)

lib-prep: $(ALL_SRAM_LIBS) $(ALL_SRAM_BBS)

libs: lib-prep $(ALL_SRAM_DBS) $(ALL_CORE_DBS)

define CONFIG_RULES

.PHONY: $(1) elab_$(1) flist_$(1)

SRAM_S_ARGS_$(1) := $$(foreach d,$$(SRAM_DIRS_$(1)),-S $$(d))
DC_VARS_$(1)     := set CONFIG $(1); set TOP $(TOP_$(1)); set TARGET_FREQ_MHZ $(TARGET_FREQ_MHZ_$(1)); set MAX_FANOUT $(MAX_FANOUT_$(1)); set SRAM_DBS \"$$(SRAM_DBS_$(1))\"; set BLACKBOX_DBS \"$$(BB_DBS_$(1))\"

flists/dc_flist_$(1).f: gen_dc_sources.sh $(SCRIPT_DIR)../../scripts/gen_sources.sh $$(SRAM_BBS_$(1))
	mkdir -p logs flists
	./gen_dc_sources.sh $(DEFINES_$(1)) -T $(TOP_$(1)) -O flists/dc_flist_$(1).f $$(SRAM_S_ARGS_$(1))

flist_$(1): flists/dc_flist_$(1).f

$(1): flists/dc_flist_$(1).f $$(SRAM_DBS_$(1)) $$(CORE_DBS_$(1))
	mkdir -p runs/$(1)/logs
	cd runs/$(1) && dc_shell -x "$$(DC_VARS_$(1))" -f ../../dc_syn.tcl 2>&1 | tee logs/$(1)_syn.log

elab_$(1): flists/dc_flist_$(1).f $$(SRAM_DBS_$(1)) $$(CORE_DBS_$(1))
	mkdir -p runs/$(1)/logs
	cd runs/$(1) && dc_shell -x "$$(DC_VARS_$(1))" -f ../../dc_elab.tcl 2>&1 | tee logs/$(1)_elab.log

endef

$(foreach cfg,$(CONFIGS),$(eval $(call CONFIG_RULES,$(cfg))))

KEEP_DBS ?= 0

clean:
	rm -f  dc_flist_*.f
	rm -rf logs/ results/ reports/ work/ runs/ flists/ alib-52/ cksum_dir/
	rm -f  *.svf *.key *.log *.ddc *.pvl *.syn *.mr *.pvk moduleparamid
	@if [ "$(KEEP_DBS)" != "1" ]; then \
	    find $(LIBS_DIR)/sram -mindepth 2 -maxdepth 2 -name '*.db' -type f -delete 2>/dev/null || true; \
	    find $(LIBS_DIR)/core -mindepth 2 -maxdepth 2 -name '*.db'  -type f -delete 2>/dev/null || true; \
	fi
