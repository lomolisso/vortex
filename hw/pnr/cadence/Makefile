# Vortex GPGPU — Cadence Innovus PnR.
# Usage:
#   make <cfg>                       full flow (stages 1..12)
#   make <cfg> STAGE=N               run only stage N (loads ckpt N-1)
#   make <cfg> STAGE=N CONTINUE=1    run N..12 in one session
# Vars: STAGE, CONTINUE, EFFORT (low|medium|high), THREADS, BATCH (1 = no GUI).

# GNU Make 3.80 lacks $(abspath); use pwd -P.
SCRIPT_DIR  := $(shell pwd -P)/
SYN_DIR     := $(shell cd "$(SCRIPT_DIR)../../syn/synopsys" 2>/dev/null && pwd -P)
LIBS_DIR    := $(SYN_DIR)/libs
SCRIPTS_DIR := $(SCRIPT_DIR)../../scripts
CONFIG_DIR  := $(SCRIPT_DIR)config

EFFORT  ?= low
THREADS ?= 8
STAGE   ?=
CONTINUE ?=
BATCH   ?= 0
INNOVUS_BATCH_FLAG := $(if $(filter 1,$(BATCH)),-batch,)

CONFIGS := $(notdir $(basename $(wildcard $(CONFIG_DIR)/*.toml)))

# Suppressed when STAGE or CONTINUE is set: partial runs would otherwise
# relaunch Innovus from scratch on dep configs every invocation.
ifeq ($(strip $(STAGE)$(CONTINUE)),)
DEPS_2d-single-core := fpu tcu lmem gpr l1_icache_bank l1_dcache_bank
DEPS_2d-1c4s4w4t    := 2d-single-core l2_bank
DEPS_2d-1c8s4w4t    := 2d-single-core l2_bank
endif

.PHONY: clean $(CONFIGS)

define PNR_RULES

.PHONY: $(1)

$(1): $$(DEPS_$(1))
	mkdir -p runs/$(1)/logs runs/$(1)/reports runs/$(1)/results
	$(MAKE) -C $(SYN_DIR) lib-prep
	python3 $(SCRIPTS_DIR)/gen_floorplan_config.py \
	    --config       $(1) \
	    --config_dir   $(CONFIG_DIR) \
	    --libs_dir     $(LIBS_DIR) \
	    --syn_runs_dir $(SYN_DIR)/runs
	@echo "=== Config $(1)  STAGE='$(STAGE)' CONTINUE='$(CONTINUE)' ==="
	cd runs/$(1) && { \
	    CONFIG=$(1) LIBS_DIR=$(LIBS_DIR) SYN_DIR=$(SYN_DIR) \
	    EFFORT=$(EFFORT) STAGE=$(STAGE) CONTINUE=$(CONTINUE) \
	    SKIP_POWER=$(SKIP_POWER) BATCH=$(BATCH) \
	    innovus $(INNOVUS_BATCH_FLAG) -cpus $(THREADS) \
	        -files ../../pnr.tcl -log logs/run.log -overwrite ; \
	    rc=$$$$? ; \
	    "$(SCRIPT_DIR)split_run_log.sh" . >/dev/null || true ; \
	    exit $$$$rc ; }

endef

$(foreach cfg,$(CONFIGS),$(eval $(call PNR_RULES,$(cfg))))

clean:
	rm -rf runs/
	rm -f  *.cmd* *.log* innovus.lck
