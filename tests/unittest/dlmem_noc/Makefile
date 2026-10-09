ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := dlmem_noc

SIMX_DIR := $(ROOT_DIR)/sim/simx

CONFIGS := $(if $(findstring -DVX_CFG_EXT_DLMEM_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DLMEM_ENABLE)

.PHONY: all run clean

all:
	$(MAKE) -C $(SIMX_DIR) $(SIMX_DIR)/dlmem_noc_test CONFIGS="$(CONFIGS)"

run:
	$(MAKE) -C $(SIMX_DIR) test-dlmem-noc CONFIGS="$(CONFIGS)" DLMEM_NOC_OPTS="$(OPTS)"

clean:
	rm -f $(SIMX_DIR)/dlmem_noc_test $(SIMX_DIR)/obj/tests/dlmem_noc_test.o $(SIMX_DIR)/obj/tests/dlmem_noc_test.d
	rm -f $(SIMX_DIR)/obj/dlmem/*.o $(SIMX_DIR)/obj/dlmem/*.d
