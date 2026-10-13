ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := lmem_switch

SIMX_DIR := $(ROOT_DIR)/sim/simx

# Only a DLMEM build has the remote leg and the three-input response merge.
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DLMEM_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DLMEM_ENABLE)

.PHONY: all run clean

all:
	$(MAKE) -C $(SIMX_DIR) $(SIMX_DIR)/lmem_switch_test CONFIGS="$(CONFIGS)"

run:
	$(MAKE) -C $(SIMX_DIR) test-lmem-switch CONFIGS="$(CONFIGS)" LMEM_SWITCH_OPTS="$(OPTS)"

clean:
	rm -f $(SIMX_DIR)/lmem_switch_test $(SIMX_DIR)/obj/tests/lmem_switch_test.o $(SIMX_DIR)/obj/tests/lmem_switch_test.d
