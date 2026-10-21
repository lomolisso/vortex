ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := lmem_switch

SIMX_DIR := $(ROOT_DIR)/sim/simx

# Only a DLMEM build has the remote leg and the three-input response merge.
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DLMEM_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DLMEM_ENABLE)

# A second build adds the bank atomics and mbarriers. It has its own directory,
# so the two builds do not rebuild each other's objects.
MBAR_CONFIGS := $(CONFIGS) $(filter-out $(CONFIGS),-DVX_CFG_EXT_A_ENABLE -DVX_CFG_EXT_MBAR_ENABLE)
MBAR_DIR := $(SIMX_DIR)/lmem_switch_mbar

.PHONY: all run clean

all:
	$(MAKE) -C $(SIMX_DIR) $(SIMX_DIR)/lmem_switch_test CONFIGS="$(CONFIGS)"
	$(MAKE) -C $(SIMX_DIR) $(MBAR_DIR)/lmem_switch_test CONFIGS="$(MBAR_CONFIGS)" DESTDIR=$(MBAR_DIR)

run:
	$(MAKE) -C $(SIMX_DIR) test-lmem-switch CONFIGS="$(CONFIGS)" LMEM_SWITCH_OPTS="$(OPTS)"
	$(MAKE) -C $(SIMX_DIR) test-lmem-switch CONFIGS="$(MBAR_CONFIGS)" DESTDIR=$(MBAR_DIR) LMEM_SWITCH_OPTS="$(OPTS)"

clean:
	rm -f $(SIMX_DIR)/lmem_switch_test $(SIMX_DIR)/obj/tests/lmem_switch_test.o $(SIMX_DIR)/obj/tests/lmem_switch_test.d
	rm -rf $(MBAR_DIR)
