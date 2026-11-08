ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := dxa_tmap_core

SIMX_DIR := $(ROOT_DIR)/sim/simx
DESTDIR  := $(SIMX_DIR)/dxa_tmap_core

# The descriptor cache's map entries with one DXA worker, then with two, whose
# arbiter tags the port's requests. Each builds in its own directory.
CASES := base dxa2
DXA_MBAR_CONFIGS := -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_EXT_DLMEM_ENABLE -DVX_CFG_EXT_A_ENABLE -DVX_CFG_EXT_MBAR_ENABLE
base_CONFIGS := $(DXA_MBAR_CONFIGS)
dxa2_CONFIGS := $(DXA_MBAR_CONFIGS) -DVX_CFG_NUM_DXA_CORES=2

case_args = CONFIGS="$(CONFIGS) $($(1)_CONFIGS)" DESTDIR=$(DESTDIR)/$(1)

.PHONY: all run clean

all:
	$(foreach c,$(CASES),$(MAKE) -C $(SIMX_DIR) $(DESTDIR)/$(c)/dxa_tmap_core_test $(call case_args,$(c)) &&) true

run:
	$(foreach c,$(CASES),$(MAKE) -C $(SIMX_DIR) test-dxa-tmap-core $(call case_args,$(c)) DXA_TMAP_CORE_OPTS="$(OPTS)" &&) true

clean:
	rm -rf $(DESTDIR)
