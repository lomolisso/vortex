ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := cpa_unit

SIMX_DIR := $(ROOT_DIR)/sim/simx
DESTDIR  := $(SIMX_DIR)/cpa_unit

# The unit at the default shape, then with 8 threads over 4 LSU lanes (two
# packets per warp) and a deeper pending table. Each builds in its own directory.
CASES := base pids
MBAR_CONFIGS  := -DVX_CFG_EXT_DLMEM_ENABLE -DVX_CFG_EXT_A_ENABLE -DVX_CFG_EXT_MBAR_ENABLE
base_CONFIGS  := $(MBAR_CONFIGS)
pids_CONFIGS  := $(MBAR_CONFIGS) -DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_LSU_LANES=4 -DVX_CFG_LSU_PENDING_SIZE=16

case_args = CONFIGS="$(CONFIGS) $($(1)_CONFIGS)" DESTDIR=$(DESTDIR)/$(1)

.PHONY: all run clean

all:
	$(foreach c,$(CASES),$(MAKE) -C $(SIMX_DIR) $(DESTDIR)/$(c)/cpa_unit_test $(call case_args,$(c)) &&) true

run:
	$(foreach c,$(CASES),$(MAKE) -C $(SIMX_DIR) test-cpa-unit $(call case_args,$(c)) CPA_UNIT_OPTS="$(OPTS)" &&) true

clean:
	rm -rf $(DESTDIR)
