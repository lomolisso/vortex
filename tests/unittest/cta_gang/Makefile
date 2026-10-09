ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := cta_gang

SIMX_DIR := $(ROOT_DIR)/sim/simx

CONFIGS := $(if $(findstring -DVX_CFG_EXT_DLMEM_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DLMEM_ENABLE)

.PHONY: all run clean

all:
	$(MAKE) -C $(SIMX_DIR) $(SIMX_DIR)/cta_gang_test CONFIGS="$(CONFIGS)"

run:
	$(MAKE) -C $(SIMX_DIR) test-cta-gang CONFIGS="$(CONFIGS)" CTA_GANG_OPTS="$(OPTS)"

clean:
	rm -f $(SIMX_DIR)/cta_gang_test $(SIMX_DIR)/obj/tests/cta_gang_test.o $(SIMX_DIR)/obj/tests/cta_gang_test.d
