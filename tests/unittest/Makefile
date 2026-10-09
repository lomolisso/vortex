ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

# Distributed local memory units: the feature is RV64 only.
DLMEM_UNITS := $(if $(filter 64,$(XLEN)),dlmem_noc)

PROJECTS := vx_malloc dxa_ooo rvc_decompress gfx_binsort gfx_tex_sw gfx_rast_sw gfx_msaa gfx_om_mrt $(DLMEM_UNITS)

all:
	$(foreach p,$(PROJECTS),$(MAKE) -C $(p) &&) true

run:
	$(foreach p,$(PROJECTS),$(MAKE) -C $(p) run &&) true

clean:
	$(foreach p,$(PROJECTS),$(MAKE) -C $(p) clean &&) true
