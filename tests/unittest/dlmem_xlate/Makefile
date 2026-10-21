ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := dlmem_xlate

SIMX_DIR := $(ROOT_DIR)/sim/simx

CONFIGS := $(if $(findstring -DVX_CFG_EXT_DLMEM_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DLMEM_ENABLE)

# A second build adds the mbarrier lane checks, in its own directory.
MBAR_CONFIGS := $(CONFIGS) $(filter-out $(CONFIGS),-DVX_CFG_EXT_A_ENABLE -DVX_CFG_EXT_MBAR_ENABLE)
MBAR_DIR := $(SIMX_DIR)/dlmem_xlate_mbar

.PHONY: all run clean

all:
	$(MAKE) -C $(SIMX_DIR) $(SIMX_DIR)/dlmem_xlate_test CONFIGS="$(CONFIGS)"
	$(MAKE) -C $(SIMX_DIR) $(MBAR_DIR)/dlmem_xlate_test CONFIGS="$(MBAR_CONFIGS)" DESTDIR=$(MBAR_DIR)

run:
	$(MAKE) -C $(SIMX_DIR) test-dlmem-xlate CONFIGS="$(CONFIGS)" DLMEM_XLATE_OPTS="$(OPTS)"
	$(MAKE) -C $(SIMX_DIR) test-dlmem-xlate CONFIGS="$(MBAR_CONFIGS)" DESTDIR=$(MBAR_DIR) DLMEM_XLATE_OPTS="$(OPTS)"

clean:
	rm -f $(SIMX_DIR)/dlmem_xlate_test $(SIMX_DIR)/obj/tests/dlmem_xlate_test.o $(SIMX_DIR)/obj/tests/dlmem_xlate_test.d
	rm -f $(SIMX_DIR)/obj/dlmem/dlmem_xlate.o $(SIMX_DIR)/obj/dlmem/dlmem_xlate.d
	rm -rf $(MBAR_DIR)
