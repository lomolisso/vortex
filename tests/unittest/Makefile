ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

# Distributed local memory units: the feature is RV64 only. A unit is listed
# once its directory exists, so its test can land without touching the list.
DLMEM_UNIT_DIRS := dlmem_noc cta_gang cluster_dispatch dlmem_xlate dlmem_ep lmem_switch mbar_waiters dxa_compl dxa_bulk cpa_unit bg_unit dxa_tmap dxa_tmap_core
DLMEM_UNITS := $(if $(filter 64,$(XLEN)),$(patsubst %/Makefile,%,$(wildcard $(addsuffix /Makefile,$(DLMEM_UNIT_DIRS)))))
# The H100 SM path's SimX twins (RV64), listed once their directory exists.
H100_SM_UNIT_DIRS := smem_bank l1_sector sm_port
H100_SM_UNITS := $(if $(filter 64,$(XLEN)),$(patsubst %/Makefile,%,$(wildcard $(addsuffix /Makefile,$(H100_SM_UNIT_DIRS)))))

# The DRAM path: the golden map at both XLENs; DramSim's H100 model needs XLEN 64.
DRAM_UNITS := dram_map $(if $(filter 64,$(XLEN)),dram_sim)

PROJECTS := vx_malloc dxa_ooo rvc_decompress gfx_binsort gfx_tex_sw gfx_rast_sw gfx_msaa gfx_om_mrt launch_tree amo_ref mbar_ref dxa_ref cuda_fp tcu_desc $(DRAM_UNITS) $(DLMEM_UNITS) $(H100_SM_UNITS)
PROJECTS += sram_port_ref smem_carveout l1_victim

all:
	$(foreach p,$(PROJECTS),$(MAKE) -C $(p) &&) true

# The gang twin also at one GPC of an H100 build (2 CPCs of 2 TPCs: least-occupied
# placement on distinct SMs); its policy and caps are build-time, so it builds apart.
CTA_GANG_H100 := -DVX_CFG_EXT_DLMEM_ENABLE -DVX_CFG_H100_ORG_ENABLE -DVX_CFG_CPCS_PER_GPC=2 -DVX_CFG_TPCS_PER_CPC=2 -DVX_CFG_NUM_THREADS=16

run:
	$(foreach p,$(PROJECTS),$(MAKE) -C $(p) run &&) true
ifeq ($(XLEN),64)
	$(MAKE) -C $(ROOT_DIR)/sim/simx DESTDIR=$(CURDIR)/cta_gang/simx_h100 $(CURDIR)/cta_gang/simx_h100/cta_gang_test \
		CONFIGS="$(CTA_GANG_H100)"
	cta_gang/simx_h100/cta_gang_test
	rm -rf cta_gang/simx_h100
endif

clean:
	$(foreach p,$(PROJECTS),$(MAKE) -C $(p) clean &&) true

# The SimX admission alone over the cluster-dispatch shapes (RV64 only), against
# dlmem_ref::place_chip; hw/unittest run-cluster-dispatch adds the RTL and the CSV parity.
include $(VORTEX_HOME)/tests/unittest/cluster_dispatch/shapes.mk
run-cluster-dispatch:
ifeq ($(XLEN),64)
	$(foreach s,$(CD_SHAPES),$(MAKE) -C $(ROOT_DIR)/sim/simx DESTDIR=$(CURDIR)/cluster_dispatch/simx_$(s) \
		$(CURDIR)/cluster_dispatch/simx_$(s)/cluster_dispatch_test CONFIGS="-DVX_CFG_EXT_DLMEM_ENABLE $(CD_$(s))" && \
		cluster_dispatch/simx_$(s)/cluster_dispatch_test --csv=cluster_dispatch/simx_$(s).csv &&) true
	rm -rf cluster_dispatch/simx_*
else
	@echo "run-cluster-dispatch: cluster dispatch is RV64 only; nothing to run at XLEN $(XLEN)"
endif

# The front end's SimX twins against their goldens (any XLEN; they build with the switch on).
run-warp-ops:
	$(MAKE) -C warp_ops clean
	$(MAKE) -C warp_ops run

run-uniform:
	$(MAKE) -C uniform clean
	$(MAKE) -C uniform run
