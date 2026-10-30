ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := dxa_compl

SIMX_DIR := $(ROOT_DIR)/sim/simx
DESTDIR  := $(SIMX_DIR)/dxa_compl

# The unit with mbarriers (DLMEM, A), at the default banks and at 16, whose
# 128-byte row a DMA beat writes as two blocks; then a DXA build without them,
# at both bank counts. Each builds in its own directory, so they do not
# rebuild each other.
CASES := mbar mbar16 legacy legacy16
MBAR_CONFIGS    := -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_EXT_DLMEM_ENABLE -DVX_CFG_EXT_A_ENABLE -DVX_CFG_EXT_MBAR_ENABLE
mbar_CONFIGS    := $(MBAR_CONFIGS)
mbar16_CONFIGS  := $(MBAR_CONFIGS) -DVX_CFG_LMEM_NUM_BANKS=16
legacy_CONFIGS  := -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_EXT_A_ENABLE
legacy16_CONFIGS := $(legacy_CONFIGS) -DVX_CFG_LMEM_NUM_BANKS=16

case_args = CONFIGS="$(CONFIGS) $($(1)_CONFIGS)" DESTDIR=$(DESTDIR)/$(1)

.PHONY: all run clean

all:
	$(foreach c,$(CASES),$(MAKE) -C $(SIMX_DIR) $(DESTDIR)/$(c)/dxa_compl_test $(call case_args,$(c)) &&) true

run:
	$(foreach c,$(CASES),$(MAKE) -C $(SIMX_DIR) test-dxa-compl $(call case_args,$(c)) DXA_COMPL_OPTS="$(OPTS)" &&) true

clean:
	rm -rf $(DESTDIR)
