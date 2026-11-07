ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := bg_unit

SIMX_DIR := $(ROOT_DIR)/sim/simx
DESTDIR  := $(SIMX_DIR)/bg_unit

# The unit at the default shape, then with 16 warps. Each builds in its own directory.
CASES := base w16
DXA_MBAR_CONFIGS := -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_EXT_DLMEM_ENABLE -DVX_CFG_EXT_A_ENABLE -DVX_CFG_EXT_MBAR_ENABLE
base_CONFIGS := $(DXA_MBAR_CONFIGS)
w16_CONFIGS  := $(DXA_MBAR_CONFIGS) -DVX_CFG_NUM_WARPS=16

case_args = CONFIGS="$(CONFIGS) $($(1)_CONFIGS)" DESTDIR=$(DESTDIR)/$(1)

.PHONY: all run clean

all:
	$(foreach c,$(CASES),$(MAKE) -C $(SIMX_DIR) $(DESTDIR)/$(c)/bg_unit_test $(call case_args,$(c)) &&) true

run:
	$(foreach c,$(CASES),$(MAKE) -C $(SIMX_DIR) test-bg-unit $(call case_args,$(c)) BG_UNIT_OPTS="$(OPTS)" &&) true

clean:
	rm -rf $(DESTDIR)
