ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := dlmem_xlate

SIMX_DIR := $(ROOT_DIR)/sim/simx

CONFIGS := $(if $(findstring -DVX_CFG_EXT_DLMEM_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DLMEM_ENABLE)

.PHONY: all run clean

all:
	$(MAKE) -C $(SIMX_DIR) $(SIMX_DIR)/dlmem_xlate_test CONFIGS="$(CONFIGS)"

run:
	$(MAKE) -C $(SIMX_DIR) test-dlmem-xlate CONFIGS="$(CONFIGS)" DLMEM_XLATE_OPTS="$(OPTS)"

clean:
	rm -f $(SIMX_DIR)/dlmem_xlate_test $(SIMX_DIR)/obj/tests/dlmem_xlate_test.o $(SIMX_DIR)/obj/tests/dlmem_xlate_test.d
	rm -f $(SIMX_DIR)/obj/dlmem/dlmem_xlate.o $(SIMX_DIR)/obj/dlmem/dlmem_xlate.d
