ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := dxa_bulk

SIMX_DIR := $(ROOT_DIR)/sim/simx
DESTDIR  := $(SIMX_DIR)/dxa_bulk

# The engine with mbarriers (DLMEM, A): a 32-byte local-memory word, half a
# line, at the default 4 banks; one line at 8; two blocks at 16. Each builds
# in its own directory, so they do not rebuild each other.
CASES := banks4 banks8 banks16
BULK_CONFIGS   := -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_EXT_DLMEM_ENABLE -DVX_CFG_EXT_A_ENABLE -DVX_CFG_EXT_MBAR_ENABLE
banks4_CONFIGS  := $(BULK_CONFIGS)
banks8_CONFIGS  := $(BULK_CONFIGS) -DVX_CFG_LMEM_NUM_BANKS=8
banks16_CONFIGS := $(BULK_CONFIGS) -DVX_CFG_LMEM_NUM_BANKS=16

case_args = CONFIGS="$(CONFIGS) $($(1)_CONFIGS)" DESTDIR=$(DESTDIR)/$(1)

.PHONY: all run clean

all:
	$(foreach c,$(CASES),$(MAKE) -C $(SIMX_DIR) $(DESTDIR)/$(c)/dxa_bulk_test $(call case_args,$(c)) &&) true

run:
	$(foreach c,$(CASES),$(MAKE) -C $(SIMX_DIR) test-dxa-bulk $(call case_args,$(c)) DXA_BULK_OPTS="$(OPTS)" &&) true

clean:
	rm -rf $(DESTDIR)
