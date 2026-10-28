ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

# Distributed local memory units: the feature is RV64 only. A unit is listed
# once its directory exists, so its test can land without touching the list.
DLMEM_UNIT_DIRS := dlmem_noc cta_gang dlmem_xlate dlmem_ep lmem_switch mbar_waiters
DLMEM_UNITS := $(if $(filter 64,$(XLEN)),$(patsubst %/Makefile,%,$(wildcard $(addsuffix /Makefile,$(DLMEM_UNIT_DIRS)))))

PROJECTS := vx_malloc dxa_ooo rvc_decompress gfx_binsort gfx_tex_sw gfx_rast_sw gfx_msaa gfx_om_mrt launch_tree $(DLMEM_UNITS)

all:
	$(foreach p,$(PROJECTS),$(MAKE) -C $(p) &&) true

run:
	$(foreach p,$(PROJECTS),$(MAKE) -C $(p) run &&) true

clean:
	$(foreach p,$(PROJECTS),$(MAKE) -C $(p) clean &&) true
