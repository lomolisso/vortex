ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := dlmem_ep

SIMX_DIR := $(ROOT_DIR)/sim/simx

# The endpoints belong to the DLMEM extension and carry AMOs, mbarrier ops
# among them: always build with all three, also under a command-line CONFIGS
# (a DXA build adds the row flits of remote copies, --mode=rows*).
override CONFIGS := $(if $(findstring -DVX_CFG_EXT_DLMEM_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DLMEM_ENABLE)
override CONFIGS := $(if $(findstring -DVX_CFG_EXT_A_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_A_ENABLE)
override CONFIGS := $(if $(findstring -DVX_CFG_EXT_MBAR_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_MBAR_ENABLE)

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

# With the assert on, a remote LR/SC must abort, as it does in the core. Every
# build also runs random st.async / red.async lanes (mbarriers are always on).
LRSC_BUILD := simx_$(firstword $(SHAPES))

# Without DXA in CONFIGS, run also adds a DXA build per ROWS_SHAPES for the
# row flits of remote copies: the fixed schedule, random copies under stalls,
# and a flit into a free CTA slot, which must stop the run.
ROWS_SHAPES ?= s2114 s2224
ROWS_RUN := $(if $(findstring -DVX_CFG_EXT_DXA_ENABLE,$(CONFIGS)),,$(ROWS_SHAPES))
ROWS_BUILD := simx_rows_$(firstword $(ROWS_SHAPES))
rows_cfg = $(CONFIGS) -DVX_CFG_EXT_DXA_ENABLE $(SHAPE_$(1))

run:
	$(foreach b,$(BUILDS),$(MAKE) -C $(SIMX_DIR) DESTDIR=$(CURDIR)/simx_$(b) test-dlmem-ep CONFIGS="$(call cfg_of,$(b))" DLMEM_EP_OPTS="$(OPTS)" &&) true
	! $(CURDIR)/$(LRSC_BUILD)/dlmem_ep_test --mode=lrsc --lrsc-assert > $(LRSC_BUILD)/lrsc_assert.log 2>&1
	grep -q "remote LR/SC is not supported" $(LRSC_BUILD)/lrsc_assert.log
	$(foreach b,$(BUILDS),$(CURDIR)/simx_$(b)/dlmem_ep_test --mode=async-random &&) true
ifneq ($(ROWS_RUN),)
	$(foreach s,$(ROWS_RUN),$(MAKE) -C $(SIMX_DIR) DESTDIR=$(CURDIR)/simx_rows_$(s) test-dlmem-ep CONFIGS="$(call rows_cfg,$(s))" DLMEM_EP_OPTS="--mode=rows" && \
		$(MAKE) -C $(SIMX_DIR) DESTDIR=$(CURDIR)/simx_rows_$(s) test-dlmem-ep CONFIGS="$(call rows_cfg,$(s))" DLMEM_EP_OPTS="--mode=rows-random" &&) true
	! $(CURDIR)/$(ROWS_BUILD)/dlmem_ep_test --mode=rows-retired > $(ROWS_BUILD)/rows_retired.log 2>&1
	grep -q "DXA write into a retired CTA slot" $(ROWS_BUILD)/rows_retired.log
endif

clean:
	rm -rf simx_*
