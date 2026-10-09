ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := dlmem_ep

SIMX_DIR := $(ROOT_DIR)/sim/simx

# The endpoints belong to the DLMEM extension and carry AMOs: always build with both.
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DLMEM_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DLMEM_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_A_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_A_ENABLE)

# The RTL bench's shapes (sockets x socket size x LSU blocks x LSU lanes). Each
# builds into its own SimX output directory, so the shapes never share objects
# with each other or with the simulator build. Multi-socket shapes also build
# with worker threads (SIMX_MT), whose run must match the serial one.
SHAPE_s1214  := -DVX_CFG_NUM_CORES=2 -DVX_CFG_SOCKET_SIZE=2 -DVX_CFG_NUM_LSU_LANES=4
SHAPE_s1422  := -DVX_CFG_NUM_CORES=4 -DVX_CFG_SOCKET_SIZE=4 -DVX_CFG_ISSUE_WIDTH=2 -DVX_CFG_NUM_LSU_BLOCKS=2 -DVX_CFG_NUM_LSU_LANES=2
SHAPE_s2114  := -DVX_CFG_NUM_CORES=2 -DVX_CFG_SOCKET_SIZE=1 -DVX_CFG_NUM_LSU_LANES=4
SHAPE_s2224  := -DVX_CFG_NUM_CORES=4 -DVX_CFG_SOCKET_SIZE=2 -DVX_CFG_ISSUE_WIDTH=2 -DVX_CFG_NUM_LSU_BLOCKS=2 -DVX_CFG_NUM_LSU_LANES=4
SHAPE_s4112  := -DVX_CFG_NUM_CORES=4 -DVX_CFG_SOCKET_SIZE=1 -DVX_CFG_NUM_LSU_LANES=2
SHAPE_s4214  := -DVX_CFG_NUM_CORES=8 -DVX_CFG_SOCKET_SIZE=2 -DVX_CFG_NUM_LSU_LANES=4
SHAPE_s2214z := -DVX_CFG_NUM_CORES=4 -DVX_CFG_SOCKET_SIZE=2 -DVX_CFG_NUM_LSU_LANES=4 -DVX_CFG_EXT_ZACAS_ENABLE

SHAPES    ?= s1214 s1422 s2114 s2224 s4112 s4214 s2214z
MT_SHAPES := $(filter s2114 s2224 s4112 s4214 s2214z,$(SHAPES))
MT        ?= 4
BUILDS    := $(SHAPES) $(addsuffix _mt,$(MT_SHAPES))

cfg_of = $(CONFIGS) $(SHAPE_$(patsubst %_mt,%,$(1))) $(if $(filter %_mt,$(1)),-DSIMX_MT=$(MT))

.PHONY: all run clean

all:
	$(foreach b,$(BUILDS),$(MAKE) -C $(SIMX_DIR) DESTDIR=$(CURDIR)/simx_$(b) $(CURDIR)/simx_$(b)/dlmem_ep_test CONFIGS="$(call cfg_of,$(b))" &&) true

# With the assert on, a remote LR/SC must abort, as it does in the core.
LRSC_BUILD := simx_$(firstword $(SHAPES))

run:
	$(foreach b,$(BUILDS),$(MAKE) -C $(SIMX_DIR) DESTDIR=$(CURDIR)/simx_$(b) test-dlmem-ep CONFIGS="$(call cfg_of,$(b))" DLMEM_EP_OPTS="$(OPTS)" &&) true
	! $(CURDIR)/$(LRSC_BUILD)/dlmem_ep_test --mode=lrsc --lrsc-assert > $(LRSC_BUILD)/lrsc_assert.log 2>&1
	grep -q "remote LR/SC is not supported" $(LRSC_BUILD)/lrsc_assert.log

clean:
	rm -rf simx_*
