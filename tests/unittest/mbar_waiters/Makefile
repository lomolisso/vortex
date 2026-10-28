ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := mbar_waiters

SIMX_DIR := $(ROOT_DIR)/sim/simx

# The table exists only in mbarrier builds, which need DLMEM and the A extension.
CONFIGS := $(CONFIGS) $(filter-out $(CONFIGS),-DVX_CFG_EXT_DLMEM_ENABLE -DVX_CFG_EXT_A_ENABLE -DVX_CFG_EXT_MBAR_ENABLE)
DESTDIR := $(SIMX_DIR)/mbar_waiters

# The configurations and seeds of the RTL testbench (hw/unittest/Makefile,
# run-mbar-waiters). Each builds in its own directory, so the four do not
# rebuild each other's objects.
CASES := base s0 s24 s24w8
base_CONFIGS  :=
base_OPTS     := --seed=1 --cycles=5000
s0_CONFIGS    := -DVX_CFG_MBAR_SUSPEND_CYCLES=0 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_LMEM_NUM_BANKS=8
s0_OPTS       := --seed=2
s24_CONFIGS   := -DVX_CFG_MBAR_SUSPEND_CYCLES=24
s24_OPTS      := --seed=3
s24w8_CONFIGS := -DVX_CFG_MBAR_SUSPEND_CYCLES=24 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_ISSUE_WIDTH=2 -DVX_CFG_NUM_LSU_BLOCKS=2
s24w8_OPTS    := --seed=4

case_args = CONFIGS="$(CONFIGS) $($(1)_CONFIGS)" DESTDIR=$(DESTDIR)/$(1)

.PHONY: all run clean

all:
	$(foreach c,$(CASES),$(MAKE) -C $(SIMX_DIR) $(DESTDIR)/$(c)/mbar_waiters_test $(call case_args,$(c)) &&) true

run:
	$(foreach c,$(CASES),$(MAKE) -C $(SIMX_DIR) test-mbar-waiters $(call case_args,$(c)) MBAR_WAITERS_OPTS="$($(c)_OPTS) $(OPTS)" &&) true

clean:
	rm -rf $(DESTDIR)
