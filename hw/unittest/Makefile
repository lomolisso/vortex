ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

# Distributed local memory units: the feature is RV64 only. A unit is listed
# once its directory exists, so its bench can land without touching the lists.
DLMEM_UNIT_DIRS := dlmem_noc cta_gang bar_unit dlmem_xlate dlmem_ep
DLMEM_UNITS := $(if $(filter 64,$(XLEN)),$(patsubst %/Makefile,%,$(wildcard $(addsuffix /Makefile,$(DLMEM_UNIT_DIRS)))))

all:
	$(MAKE) -C generic_queue
	$(MAKE) -C mem_scheduler
	$(MAKE) -C cache
	$(MAKE) -C core
	$(MAKE) -C issue
	$(MAKE) -C local_mem
	$(MAKE) -C lmem_switch
	$(MAKE) -C mem_unit
	$(MAKE) -C tcu_fedp
	$(MAKE) -C cta_dispatcher
	$(MAKE) -C kmu
	$(MAKE) -C dxa_core
	$(MAKE) -C tcu_unit
	$(MAKE) -C cp_arbiter
	$(MAKE) -C cp_engine
	$(MAKE) -C cp_launch
	$(MAKE) -C cp_dcr_proxy
	$(MAKE) -C cp_unpack
	$(MAKE) -C cp_axil_regfile
	$(MAKE) -C afu_axil_demux
	$(MAKE) -C afu_reset_seq
	$(MAKE) -C cp_axi_path
	$(MAKE) -C cp_dma
	$(MAKE) -C cp_core
	$(MAKE) -C fma_unit
	$(MAKE) -C fdiv_unit
	$(MAKE) -C fsqrt_unit
	$(MAKE) -C fcvt_unit
	$(MAKE) -C fdivsqrt_unit
	$(foreach u,$(DLMEM_UNITS),$(MAKE) -C $(u) &&) true

run:
	$(MAKE) -C generic_queue run
	$(MAKE) -C mem_scheduler run
	$(MAKE) -C cache run
	$(MAKE) -C core run
	$(MAKE) -C issue run
	$(MAKE) -C local_mem run
	$(MAKE) -C lmem_switch run
	$(MAKE) -C mem_unit run
	$(MAKE) -C tcu_fedp run
	$(MAKE) -C cta_dispatcher run
	$(MAKE) -C kmu run
	$(MAKE) -C dxa_core run
	$(MAKE) -C tcu_unit run
	$(MAKE) -C cp_arbiter run
	$(MAKE) -C cp_engine run
	$(MAKE) -C cp_launch run
	$(MAKE) -C cp_dcr_proxy run
	$(MAKE) -C cp_unpack run
	$(MAKE) -C cp_axil_regfile run
	$(MAKE) -C afu_axil_demux run
	$(MAKE) -C afu_reset_seq run
	$(MAKE) -C cp_axi_path run
	$(MAKE) -C cp_dma run
	$(MAKE) -C cp_core run
	$(MAKE) -C fma_unit run
	$(MAKE) -C fdiv_unit run
	$(MAKE) -C fsqrt_unit run
	$(MAKE) -C fcvt_unit run
	$(MAKE) -C fdivsqrt_unit run
	$(foreach u,$(DLMEM_UNITS),$(MAKE) -C $(u) run &&) true

# FPU arithmetic units only — executed (not just built) so CI gates correctness.
run-fpu:
	$(MAKE) -C fma_unit run
	$(MAKE) -C fdiv_unit run
	$(MAKE) -C fsqrt_unit run
	$(MAKE) -C fcvt_unit run
	$(MAKE) -C fdivsqrt_unit run

# Local memory — LSU atomics racing the DMA row port, executed with the A
# extension, again with Zacas so the compare-and-swap operand path is covered,
# and once with atomics off so the load/store/DMA path runs with the atomic
# hardware compiled out.
# Each config rebuilds: CONFIGS is not a make dependency of the testbench.
LMEM_BASE := -DVX_CFG_EXT_A_ENABLE
define lmem_case
	$(MAKE) -C local_mem clean
	$(MAKE) -C local_mem CONFIGS="$(1)"
	$(MAKE) -C local_mem run CONFIGS="$(1)"
endef
# The LSU-side global/local switch, at each LSU lane count: mixed-lane
# requests under independent port back-pressure (one lane cannot mix, so it
# covers the single-port paths), and the widest attribute with Zacas.
define lmem_switch_case
	$(MAKE) -C lmem_switch clean
	$(MAKE) -C lmem_switch CONFIGS="$(1)"
	$(MAKE) -C lmem_switch run CONFIGS="$(1)"
endef
# With DLMEM (RV64 only) the switch gains its remote leg and round-robin
# response merge, the local memory a store-acknowledging ingress input and
# per-bank commit reports of every LSU-input access, and the memory unit is
# linted with all of them wired. With mbarriers the banks also run the mbarrier
# ops against mbar_ref and report their waiter events, and each erroneous op
# must stop the last build with its assert; the final clean keeps a plain `run`
# from reusing it. Then the mbarrier waiter table (run-mbar-waiters).
DLMEM_LMEM := -DVX_CFG_EXT_DLMEM_ENABLE
MBAR_LMEM  := $(DLMEM_LMEM) $(LMEM_BASE) -DVX_CFG_EXT_MBAR_ENABLE
run-lmem:
	$(call lmem_case,$(LMEM_BASE))
	$(call lmem_case,$(LMEM_BASE) -DVX_CFG_EXT_ZACAS_ENABLE)
	$(call lmem_case,)
	$(call lmem_switch_case,-DVX_CFG_NUM_LSU_LANES=1)
	$(call lmem_switch_case,-DVX_CFG_NUM_LSU_LANES=2)
	$(call lmem_switch_case,-DVX_CFG_NUM_LSU_LANES=4 $(LMEM_BASE) -DVX_CFG_EXT_ZACAS_ENABLE)
ifeq ($(XLEN),64)
	$(call lmem_case,$(DLMEM_LMEM) $(LMEM_BASE))
	$(call lmem_case,$(DLMEM_LMEM) $(LMEM_BASE) -DVX_CFG_EXT_ZACAS_ENABLE)
	$(call lmem_case,$(DLMEM_LMEM))
	$(call lmem_case,$(MBAR_LMEM))
	! local_mem/local_mem --neg=underflow > local_mem/neg_underflow.log 2>&1
	grep -q "mbarrier pending underflow" local_mem/neg_underflow.log
	! local_mem/local_mem --neg=txrange > local_mem/neg_txrange.log 2>&1
	grep -q "mbarrier tx out of range" local_mem/neg_txrange.log
	! local_mem/local_mem --neg=align > local_mem/neg_align.log 2>&1
	grep -q "mbarrier op is not a naturally aligned doubleword" local_mem/neg_align.log
	$(MAKE) -C local_mem clean
	$(MAKE) run-mbar-waiters
	$(call lmem_switch_case,$(DLMEM_LMEM) -DVX_CFG_NUM_LSU_LANES=1)
	$(call lmem_switch_case,$(DLMEM_LMEM) -DVX_CFG_NUM_LSU_LANES=2)
	$(call lmem_switch_case,$(DLMEM_LMEM) -DVX_CFG_NUM_LSU_LANES=4 $(LMEM_BASE) -DVX_CFG_EXT_ZACAS_ENABLE)
	$(MAKE) -C mem_unit lint CONFIGS="$(DLMEM_LMEM) $(LMEM_BASE)"
	$(MAKE) -C mem_unit lint CONFIGS="$(DLMEM_LMEM) $(LMEM_BASE) -DVX_CFG_ISSUE_WIDTH=2"
	$(MAKE) -C mem_unit lint CONFIGS="$(MBAR_LMEM)"
endif

# The mbarrier waiter table (RV64 only): directed release timings (wake on a
# flip, the time limit, every warp on one object and on many, a retire racing a
# flip, ...) and random try_wait lifecycles checked cycle by cycle against a
# model of the table. At the default time limit; with no suspension at all, 16
# warps and 8 banks; at a short limit, so the random runs reach it often, also
# with 8 warps and two LSU blocks. An insert for a retired warp must stop the last
# build with its assert; the final clean keeps a plain `run` from reusing it.
# $(1)=CONFIGS $(2)=OPTS
define mbar_waiters_case
	$(MAKE) -C mbar_waiters clean
	$(MAKE) -C mbar_waiters CONFIGS="$(MBAR_LMEM) $(1)"
	$(MAKE) -C mbar_waiters run CONFIGS="$(MBAR_LMEM) $(1)" OPTS="$(2)"
endef
run-mbar-waiters:
ifeq ($(XLEN),64)
	$(call mbar_waiters_case,,--seed=1 --cycles=5000)
	$(call mbar_waiters_case,-DVX_CFG_MBAR_SUSPEND_CYCLES=0 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_LMEM_NUM_BANKS=8,--seed=2)
	$(call mbar_waiters_case,-DVX_CFG_MBAR_SUSPEND_CYCLES=24,--seed=3)
	$(call mbar_waiters_case,-DVX_CFG_MBAR_SUSPEND_CYCLES=24 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_ISSUE_WIDTH=2 \
		-DVX_CFG_NUM_LSU_BLOCKS=2,--seed=4)
	! mbar_waiters/mbar_waiters --neg=insert > mbar_waiters/neg_insert.log 2>&1
	grep -q "mbarrier waiter insert after retire" mbar_waiters/neg_insert.log
	$(MAKE) -C mbar_waiters clean
else
	@echo "run-mbar-waiters: mbarriers are RV64 only; nothing to run at XLEN $(XLEN)"
endif

# The DXA completion unit (RV64 only, DXA with mbarriers): directed release and
# complete_tx timings, bursts of last beats, the two modes behind a stalled head,
# a full queue holding the DMA port's last beat, then random traffic against a
# model of the queue; at the default shape, with 8 warps, 8 banks and a 256 KB
# local memory, and with 2 warps (the shallowest queue). A beat or complete_tx
# for a freed or re-admitted slot, and an mbarrier outside the issuing block,
# must stop the last build with their asserts; the final clean keeps a plain
# `run` from reusing it.
# $(1)=CONFIGS $(2)=OPTS
define dxa_completion_case
	$(MAKE) -C dxa_completion clean
	$(MAKE) -C dxa_completion CONFIGS="$(MBAR_LMEM) -DVX_CFG_EXT_DXA_ENABLE $(1)"
	$(MAKE) -C dxa_completion run CONFIGS="$(MBAR_LMEM) -DVX_CFG_EXT_DXA_ENABLE $(1)" OPTS="$(2)"
endef
run-dxa-completion:
ifeq ($(XLEN),64)
	$(call dxa_completion_case,,--seed=1)
	$(call dxa_completion_case,-DVX_CFG_NUM_WARPS=8 -DVX_CFG_LMEM_NUM_BANKS=8 -DVX_CFG_LMEM_LOG_SIZE=18,--seed=2)
	$(call dxa_completion_case,-DVX_CFG_NUM_WARPS=2,--seed=3)
	! dxa_completion/dxa_completion --neg=retired > dxa_completion/neg_retired.log 2>&1
	grep -q "DXA write into a retired CTA slot" dxa_completion/neg_retired.log
	! dxa_completion/dxa_completion --neg=gen > dxa_completion/neg_gen.log 2>&1
	grep -q "DXA write into a retired CTA slot" dxa_completion/neg_gen.log
	! dxa_completion/dxa_completion --neg=compl > dxa_completion/neg_compl.log 2>&1
	grep -q "DXA write into a retired CTA slot" dxa_completion/neg_compl.log
	! dxa_completion/dxa_completion --neg=bounds > dxa_completion/neg_bounds.log 2>&1
	grep -q "mbarrier offset outside the block" dxa_completion/neg_bounds.log
	$(MAKE) -C dxa_completion clean
else
	@echo "run-dxa-completion: mbarriers are RV64 only; nothing to run at XLEN $(XLEN)"
endif

# CTA launch path: the KMU's cluster walk under launch-bus back-pressure, and
# the dispatcher's slot placement across launches checked against SimX's
# per-launch pointer reset, with 4 and 8 CTA slots. With DLMEM (RV64 only) both
# also carry the cluster rank to the CSR read-back, up to the largest cluster
# (8 cores) and with a 256 KB local memory; the KMU carries the sticky cluster
# policy, and the dispatcher takes pinned blocks into their forced slots, writes
# every block's peer row, and reports each pinned block's CTA_DONE event once,
# holding its slot until the event leaves.
define cta_case
	$(MAKE) -C cta_dispatcher clean
	$(MAKE) -C cta_dispatcher CONFIGS="$(1)"
	$(MAKE) -C cta_dispatcher run CONFIGS="$(1)"
	$(MAKE) -C kmu clean
	$(MAKE) -C kmu CONFIGS="$(1)"
	$(MAKE) -C kmu run CONFIGS="$(1)"
endef
run-cta:
	$(call cta_case,)
	$(call cta_case,-DVX_CFG_NUM_WARPS=8)
ifeq ($(XLEN),64)
	$(call cta_case,-DVX_CFG_EXT_DLMEM_ENABLE)
	$(call cta_case,-DVX_CFG_EXT_DLMEM_ENABLE -DVX_CFG_NUM_CORES=8)
	$(call cta_case,-DVX_CFG_EXT_DLMEM_ENABLE -DVX_CFG_NUM_WARPS=8 -DVX_CFG_LMEM_LOG_SIZE=18)
endif

# Distributed local memory units, each over its parameter sets (RV64 only).
run-dlmem:
ifeq ($(XLEN),64)
	$(MAKE) run-dlmem-noc
	$(MAKE) run-dlmem-gang
	$(MAKE) run-dlmem-xlate
	$(MAKE) run-dlmem-ep
else
	@echo "run-dlmem: distributed local memory is RV64 only; nothing to run at XLEN $(XLEN)"
endif

# The network at every cluster shape (sockets x socket size) and two packet
# widths: random traffic, then the fixed latency schedule, whose CSV must match
# the SimX model's run of the same schedule packet for packet (fabric parity).
# The width doubles as the random seed, so the two builds of a shape see
# different traffic. 16x1 and 2x16 also have stages above 12 ports (a 16-port
# crossbar, a 17-port switch), where the arbiters must stay flat.
# $(1)=NUM_CORES $(2)=SOCKET_SIZE $(3)=DATAW
DLMEM_NOC_SIMX := $(ROOT_DIR)/sim/simx/dlmem_noc_test
define dlmem_noc_case
	$(MAKE) -C dlmem_noc clean
	$(MAKE) -C dlmem_noc CONFIGS="-DVX_CFG_EXT_DLMEM_ENABLE -DVX_CFG_NUM_CORES=$(1) -DVX_CFG_SOCKET_SIZE=$(2)" DATAW=$(3)
	$(MAKE) -C dlmem_noc run CONFIGS="-DVX_CFG_EXT_DLMEM_ENABLE -DVX_CFG_NUM_CORES=$(1) -DVX_CFG_SOCKET_SIZE=$(2)" DATAW=$(3) \
		OPTS="--seed=$(3)"
	$(MAKE) -C dlmem_noc run CONFIGS="-DVX_CFG_EXT_DLMEM_ENABLE -DVX_CFG_NUM_CORES=$(1) -DVX_CFG_SOCKET_SIZE=$(2)" DATAW=$(3) \
		OPTS="--mode=latency --csv=latency_c$(1)_s$(2)_w$(3).csv"
	$(DLMEM_NOC_SIMX) --shape=$(shell expr $(1) / $(2))x$(2) --mode=latency --dataw=$(3) \
		--csv=dlmem_noc/simx_latency_c$(1)_s$(2)_w$(3).csv
	python3 $(VORTEX_HOME)/hw/unittest/dlmem_noc/compare_latency.py --socket-size=$(2) \
		dlmem_noc/latency_c$(1)_s$(2)_w$(3).csv dlmem_noc/simx_latency_c$(1)_s$(2)_w$(3).csv
endef
run-dlmem-noc:
ifeq ($(XLEN),64)
	$(MAKE) -C $(ROOT_DIR)/tests/unittest/dlmem_noc
	$(call dlmem_noc_case,1,1,160)
	$(call dlmem_noc_case,1,1,48)
	$(call dlmem_noc_case,2,2,160)
	$(call dlmem_noc_case,2,2,48)
	$(call dlmem_noc_case,4,4,160)
	$(call dlmem_noc_case,4,4,48)
	$(call dlmem_noc_case,2,1,160)
	$(call dlmem_noc_case,2,1,48)
	$(call dlmem_noc_case,4,2,160)
	$(call dlmem_noc_case,4,2,48)
	$(call dlmem_noc_case,4,1,160)
	$(call dlmem_noc_case,4,1,48)
	$(call dlmem_noc_case,8,2,160)
	$(call dlmem_noc_case,8,2,48)
	$(call dlmem_noc_case,16,1,160)
	$(call dlmem_noc_case,32,16,48)
else
	@echo "run-dlmem: distributed local memory is RV64 only; nothing to run at XLEN $(XLEN)"
endif

# Gang admission. The gang unit at every placement shape, also as a second
# cluster, over 6 cores (3 sockets, MAX_CLUSTER 3) and with 16 warps: directed
# scenarios and random traffic against dlmem_ref, then the deterministic
# schedule, whose placement CSV must match the SimX model's run of the same
# schedule row for row, commit cycles included (placement parity). The same
# event port carries the gangs' cluster-barrier arrivals: directed release
# scenarios, random phases against dlmem_ref, and the deterministic barrier
# schedule's release table (cbar_*.csv); a gang that ends inside a barrier
# phase must stop the last build with its assert. Then the barrier units' shared
# request queue: CTA_DONE events against global barrier rounds under response
# stalls, cluster barriers (gangs, in-core clusters, split arrive/wait, stale
# releases, all mixed) and both latencies, at 1 core, 2x2 and 4x2.
# $(1)=NUM_CORES $(2)=SOCKET_SIZE $(3)=NUM_WARPS $(4)=DLMEM_MAX_CLUSTER
# $(5)=LMEM_LOG_SIZE $(6)=CLUSTER_ID $(7)=seed $(8)=other CONFIGS
CTA_GANG_SIMX := cta_gang/simx/cta_gang_test
define cta_gang_case
	$(MAKE) -C cta_gang clean
	$(MAKE) -C cta_gang CLUSTER_ID=$(6) CONFIGS="-DVX_CFG_EXT_DLMEM_ENABLE -DVX_CFG_NUM_CORES=$(1) \
		-DVX_CFG_SOCKET_SIZE=$(2) -DVX_CFG_NUM_WARPS=$(3) -DVX_CFG_DLMEM_MAX_CLUSTER=$(4) -DVX_CFG_LMEM_LOG_SIZE=$(5) $(8)"
	$(MAKE) -C cta_gang run CLUSTER_ID=$(6) CONFIGS="-DVX_CFG_EXT_DLMEM_ENABLE -DVX_CFG_NUM_CORES=$(1) \
		-DVX_CFG_SOCKET_SIZE=$(2) -DVX_CFG_NUM_WARPS=$(3) -DVX_CFG_DLMEM_MAX_CLUSTER=$(4) -DVX_CFG_LMEM_LOG_SIZE=$(5) $(8)" \
		OPTS="--seed=$(7) --csv=placement_c$(1)s$(2)w$(3)m$(4)k$(6).csv --cbar-csv=cbar_c$(1)s$(2)w$(3)m$(4)k$(6).csv"
	$(CTA_GANG_SIMX) --shape=$(1),$(2),$(3),$(4),$(5) --csv=cta_gang/simx_placement_c$(1)s$(2)w$(3)m$(4)k$(6).csv \
		--cbar-csv=cta_gang/simx_cbar_c$(1)s$(2)w$(3)m$(4)k$(6).csv
	python3 $(VORTEX_HOME)/hw/unittest/cta_gang/compare_placement.py \
		cta_gang/placement_c$(1)s$(2)w$(3)m$(4)k$(6).csv cta_gang/simx_placement_c$(1)s$(2)w$(3)m$(4)k$(6).csv
	python3 $(VORTEX_HOME)/hw/unittest/cta_gang/compare_cbar.py \
		cta_gang/cbar_c$(1)s$(2)w$(3)m$(4)k$(6).csv cta_gang/simx_cbar_c$(1)s$(2)w$(3)m$(4)k$(6).csv
endef
# $(1)=CONFIGS $(2)=seed
define bar_unit_case
	$(MAKE) -C bar_unit clean
	$(MAKE) -C bar_unit CONFIGS="-DVX_CFG_EXT_DLMEM_ENABLE $(1)"
	$(MAKE) -C bar_unit run CONFIGS="-DVX_CFG_EXT_DLMEM_ENABLE $(1)" OPTS="--seed=$(2)"
endef
run-dlmem-gang:
ifeq ($(XLEN),64)
	$(MAKE) -C $(ROOT_DIR)/sim/simx DESTDIR=$(CURDIR)/cta_gang/simx $(CURDIR)/$(CTA_GANG_SIMX) \
		CONFIGS="-DVX_CFG_EXT_DLMEM_ENABLE"
	$(call cta_gang_case,1,1,4,1,14,0,1,)
	$(call cta_gang_case,2,1,4,2,14,0,2,)
	$(call cta_gang_case,2,2,8,2,14,0,3,)
	$(call cta_gang_case,4,2,4,2,14,0,4,)
	$(call cta_gang_case,4,4,8,4,14,0,5,)
	$(call cta_gang_case,8,2,4,8,14,0,6,)
	$(call cta_gang_case,8,4,8,4,14,0,7,)
	$(call cta_gang_case,8,1,8,8,14,0,8,)
	$(call cta_gang_case,4,2,4,4,14,1,9,-DVX_CFG_NUM_CLUSTERS=2)
	$(call cta_gang_case,6,2,4,3,14,0,10,)
	$(call cta_gang_case,4,2,16,4,17,0,11,-DVX_CFG_NUM_THREADS=8)
	! cta_gang/cta_gang --mode=midphase > cta_gang/midphase.log 2>&1
	grep -q "gang ended inside a cluster barrier phase" cta_gang/midphase.log
	$(call bar_unit_case,-DVX_CFG_NUM_CORES=1 -DVX_CFG_SOCKET_SIZE=1,1)
	$(call bar_unit_case,-DVX_CFG_NUM_CORES=4 -DVX_CFG_SOCKET_SIZE=2,2)
	$(call bar_unit_case,-DVX_CFG_NUM_CORES=8 -DVX_CFG_SOCKET_SIZE=2 -DVX_CFG_NUM_WARPS=8,3)
	$(MAKE) -C cta_gang clean
	$(MAKE) -C bar_unit clean
endif

# Aperture translation: one core's peer table and per-block stages, checked
# against dlmem_ref::translate on the shared vectors, then on random traffic.
# The shapes span every window size (LMEM_LOG 14..18), MAX_CLUSTER 1, 2, 3, 4
# and 8, core fields wider than the rank field (NCC 6, 8 and 16 over MAX 3, 4
# and 8), second-cluster core ids, 1 to 8 lanes, 1 and 2 blocks, and Zacas.
# NCC 6 has no vectors, so it runs random traffic only. The full runs also feed
# a model of the LSU scheduler's queue, stressing back-pressure and handoffs,
# and every run models the core's remote-access and local-access trackers, so
# fences and release AMOs wait for their warp's remote accesses and local
# accesses. The bound-assert build must abort on an out-of-bounds lane; a build
# with mbarriers counts their ops on local lanes, holds an arrive like a
# release AMO, sends arrive and tx ops on remote lanes as remote AMOs, and must
# abort on an op on a global lane and on a wait on a remote lane; the last
# build sends a remote lane, holds a fence until its store is acknowledged and
# a local access's fence until its bank commit, and holds a release AMO the
# same way; the final clean keeps a plain `run` from reusing it.
# Then the core elaborates and lints with the stages and the DLMEM endpoints in
# place, at 1 core, at 8 cores with two LSU blocks and two cores per socket,
# and at 4 cores in one socket.
# $(1)=CONFIGS $(2)=CORE_ID $(3)=OOB_ASSERT $(4)=OPTS $(5),$(6)=OPTS of extra runs
define dlmem_xlate_case
	$(MAKE) -C dlmem_xlate clean
	$(MAKE) -C dlmem_xlate CONFIGS="-DVX_CFG_EXT_DLMEM_ENABLE $(1)" CORE_ID=$(2) OOB_ASSERT=$(3)
	$(if $(4),$(MAKE) -C dlmem_xlate run CONFIGS="-DVX_CFG_EXT_DLMEM_ENABLE $(1)" CORE_ID=$(2) OOB_ASSERT=$(3) \
		OPTS="$(4)")
	$(if $(5),$(MAKE) -C dlmem_xlate run CONFIGS="-DVX_CFG_EXT_DLMEM_ENABLE $(1)" CORE_ID=$(2) OOB_ASSERT=$(3) \
		OPTS="$(5)")
	$(if $(6),$(MAKE) -C dlmem_xlate run CONFIGS="-DVX_CFG_EXT_DLMEM_ENABLE $(1)" CORE_ID=$(2) OOB_ASSERT=$(3) \
		OPTS="$(6)")
endef
run-dlmem-xlate:
ifeq ($(XLEN),64)
	$(call dlmem_xlate_case,-DVX_CFG_NUM_CORES=1 -DVX_CFG_SOCKET_SIZE=1 -DVX_CFG_NUM_WARPS=4 \
		-DVX_CFG_LMEM_LOG_SIZE=14,0,0,--seed=1)
	$(call dlmem_xlate_case,-DVX_CFG_NUM_CLUSTERS=2 -DVX_CFG_NUM_CORES=2 -DVX_CFG_SOCKET_SIZE=1 \
		-DVX_CFG_NUM_WARPS=4 -DVX_CFG_LMEM_LOG_SIZE=14,3,0,--seed=2)
	$(call dlmem_xlate_case,-DVX_CFG_NUM_CORES=4 -DVX_CFG_SOCKET_SIZE=2 -DVX_CFG_NUM_WARPS=8 \
		-DVX_CFG_LMEM_LOG_SIZE=16 -DVX_CFG_ISSUE_WIDTH=2,2,0,--seed=3,--mode=remote)
	$(call dlmem_xlate_case,-DVX_CFG_NUM_CORES=8 -DVX_CFG_SOCKET_SIZE=2 -DVX_CFG_NUM_WARPS=8 \
		-DVX_CFG_NUM_THREADS=8 -DVX_CFG_LMEM_LOG_SIZE=18,5,0,--seed=4)
	$(call dlmem_xlate_case,-DVX_CFG_NUM_CORES=8 -DVX_CFG_SOCKET_SIZE=4 -DVX_CFG_NUM_WARPS=16 \
		-DVX_CFG_DLMEM_MAX_CLUSTER=4 -DVX_CFG_LMEM_LOG_SIZE=17 -DVX_CFG_ISSUE_WIDTH=2 \
		-DVX_CFG_NUM_LSU_LANES=2,6,0,--seed=5)
	$(call dlmem_xlate_case,-DVX_CFG_NUM_CLUSTERS=2 -DVX_CFG_NUM_CORES=16 -DVX_CFG_SOCKET_SIZE=4 \
		-DVX_CFG_NUM_WARPS=16 -DVX_CFG_LMEM_LOG_SIZE=18 -DVX_CFG_EXT_A_ENABLE \
		-DVX_CFG_EXT_ZACAS_ENABLE,25,0,--seed=6)
	$(call dlmem_xlate_case,-DVX_CFG_NUM_CLUSTERS=2 -DVX_CFG_NUM_CORES=6 -DVX_CFG_SOCKET_SIZE=2 \
		-DVX_CFG_NUM_WARPS=4 -DVX_CFG_DLMEM_MAX_CLUSTER=3 -DVX_CFG_LMEM_LOG_SIZE=15 \
		-DVX_CFG_NUM_LSU_LANES=1,10,0,--mode=random --seed=7)
	$(call dlmem_xlate_case,-DVX_CFG_NUM_CORES=2 -DVX_CFG_SOCKET_SIZE=1 -DVX_CFG_NUM_WARPS=4 \
		-DVX_CFG_LMEM_LOG_SIZE=14,1,1,--mode=random --no-oob --seed=8)
	! dlmem_xlate/dlmem_xlate --mode=oob > dlmem_xlate/oob.log 2>&1
	grep -q "DLMEM access out of bounds" dlmem_xlate/oob.log
	$(call dlmem_xlate_case,-DVX_CFG_NUM_CORES=2 -DVX_CFG_SOCKET_SIZE=1 -DVX_CFG_NUM_WARPS=4 \
		-DVX_CFG_LMEM_LOG_SIZE=14 -DVX_CFG_EXT_A_ENABLE -DVX_CFG_EXT_MBAR_ENABLE,1,0,--seed=9,--mode=mbar,--mode=mbar-remote)
	! dlmem_xlate/dlmem_xlate --mode=mbar-global > dlmem_xlate/mbar_global.log 2>&1
	grep -q "mbarrier op on a global address" dlmem_xlate/mbar_global.log
	! dlmem_xlate/dlmem_xlate --mode=mbar-remote-wait > dlmem_xlate/mbar_remote_wait.log 2>&1
	grep -q "remote mbarrier wait is not supported" dlmem_xlate/mbar_remote_wait.log
	! dlmem_xlate/dlmem_xlate --mode=mbar-remote-trywait > dlmem_xlate/mbar_remote_trywait.log 2>&1
	grep -q "remote mbarrier wait is not supported" dlmem_xlate/mbar_remote_trywait.log
	$(call dlmem_xlate_case,-DVX_CFG_NUM_CORES=2 -DVX_CFG_SOCKET_SIZE=1 -DVX_CFG_NUM_WARPS=4 \
		-DVX_CFG_LMEM_LOG_SIZE=14,0,0,--mode=remote,--mode=fence,--mode=rl)
	$(MAKE) -C dlmem_xlate clean
	$(MAKE) -C core lint CONFIGS="-DVX_CFG_EXT_DLMEM_ENABLE -DVX_CFG_NUM_CORES=1"
	$(MAKE) -C core lint CONFIGS="-DVX_CFG_EXT_DLMEM_ENABLE -DVX_CFG_NUM_CORES=8 -DVX_CFG_SOCKET_SIZE=2 \
		-DVX_CFG_ISSUE_WIDTH=2"
	$(MAKE) -C core lint CONFIGS="-DVX_CFG_EXT_DLMEM_ENABLE -DVX_CFG_NUM_CORES=4 -DVX_CFG_SOCKET_SIZE=4"
endif

# Network endpoints: every core's egress, ingress and store tracker of one
# cluster, joined by a request and a response network, at each shape (sockets,
# socket size, LSU blocks, LSU lanes) and once with Zacas. Every traffic mode
# runs under random stalls, remote LR/SC as the defined hardware behaviour;
# then the fixed latency schedule, whose CSV must match the SimX endpoints' run
# of the same schedule packet for packet. The last build has the remote LR/SC
# assert on and must abort; the final clean keeps a plain `run` from reusing it.
# $(1)=shape tag of the SimX test (sockets, socket size, LSU blocks, LSU lanes)
# $(2)=that shape's CONFIGS $(3)=seed
DLMEM_EP_CFG  := -DVX_CFG_EXT_DLMEM_ENABLE -DVX_CFG_EXT_A_ENABLE
DLMEM_EP_SIMX := $(ROOT_DIR)/tests/unittest/dlmem_ep
define dlmem_ep_case
	$(MAKE) -C dlmem_ep clean
	$(MAKE) -C dlmem_ep CONFIGS="$(DLMEM_EP_CFG) $(2)"
	$(MAKE) -C dlmem_ep run CONFIGS="$(DLMEM_EP_CFG) $(2)" OPTS="--mode=random --seed=$(3)"
	$(MAKE) -C dlmem_ep run CONFIGS="$(DLMEM_EP_CFG) $(2)" OPTS="--mode=hotspot --seed=$(3)"
	$(MAKE) -C dlmem_ep run CONFIGS="$(DLMEM_EP_CFG) $(2)" OPTS="--mode=storeflood --seed=$(3)"
	$(MAKE) -C dlmem_ep run CONFIGS="$(DLMEM_EP_CFG) $(2)" OPTS="--mode=loadflood --seed=$(3)"
	$(MAKE) -C dlmem_ep run CONFIGS="$(DLMEM_EP_CFG) $(2)" OPTS="--mode=backpressure --seed=$(3)"
	$(MAKE) -C dlmem_ep run CONFIGS="$(DLMEM_EP_CFG) $(2)" OPTS="--mode=lrsc --seed=$(3)"
	$(MAKE) -C dlmem_ep run CONFIGS="$(DLMEM_EP_CFG) $(2)" OPTS="--mode=latency --csv=latency_$(1).csv"
	$(MAKE) -C $(DLMEM_EP_SIMX) run SHAPES=$(1) OPTS="--mode=latency --csv=$(CURDIR)/dlmem_ep/simx_latency_$(1).csv"
	python3 $(VORTEX_HOME)/hw/unittest/dlmem_ep/compare_latency.py \
		dlmem_ep/latency_$(1).csv dlmem_ep/simx_latency_$(1).csv
endef
run-dlmem-ep:
ifeq ($(XLEN),64)
	$(call dlmem_ep_case,s1214,-DVX_CFG_NUM_CORES=2 -DVX_CFG_SOCKET_SIZE=2 -DVX_CFG_NUM_LSU_LANES=4,1)
	$(call dlmem_ep_case,s1422,-DVX_CFG_NUM_CORES=4 -DVX_CFG_SOCKET_SIZE=4 -DVX_CFG_ISSUE_WIDTH=2 \
		-DVX_CFG_NUM_LSU_BLOCKS=2 -DVX_CFG_NUM_LSU_LANES=2,2)
	$(call dlmem_ep_case,s2114,-DVX_CFG_NUM_CORES=2 -DVX_CFG_SOCKET_SIZE=1 -DVX_CFG_NUM_LSU_LANES=4,3)
	$(call dlmem_ep_case,s2224,-DVX_CFG_NUM_CORES=4 -DVX_CFG_SOCKET_SIZE=2 -DVX_CFG_ISSUE_WIDTH=2 \
		-DVX_CFG_NUM_LSU_BLOCKS=2 -DVX_CFG_NUM_LSU_LANES=4,4)
	$(call dlmem_ep_case,s4112,-DVX_CFG_NUM_CORES=4 -DVX_CFG_SOCKET_SIZE=1 -DVX_CFG_NUM_LSU_LANES=2,5)
	$(call dlmem_ep_case,s4214,-DVX_CFG_NUM_CORES=8 -DVX_CFG_SOCKET_SIZE=2 -DVX_CFG_NUM_LSU_LANES=4,6)
	$(call dlmem_ep_case,s2214z,-DVX_CFG_NUM_CORES=4 -DVX_CFG_SOCKET_SIZE=2 -DVX_CFG_NUM_LSU_LANES=4 \
		-DVX_CFG_EXT_ZACAS_ENABLE,7)
	$(MAKE) -C dlmem_ep clean
	$(MAKE) -C dlmem_ep CONFIGS="$(DLMEM_EP_CFG) -DVX_CFG_NUM_CORES=2 -DVX_CFG_SOCKET_SIZE=2" LRSC_ASSERT=1
	! dlmem_ep/dlmem_ep --mode=lrsc > dlmem_ep/lrsc_assert.log 2>&1
	grep -q "remote LR/SC is not supported" dlmem_ep/lrsc_assert.log
	$(MAKE) -C dlmem_ep clean
endif

# Cache bank in the last-level role: directed fill/replay races around the
# atomic engine and the MSHR chains plus randomized traffic, checked against a
# reference memory. Ten sets: the synthesis-gate DUT as configured; the L1
# dcache LLC as built (2 ports and 2 banks, behind an unbuffered request
# crossbar) with Zacas; one LSU channel into one bank; a deferred data stage
# (LATENCY=3); write-through, also with two deferred stages (LATENCY=4); the
# L2/L3 shape (word = sector, 2-sector lines, 2 memory ports); and atomics
# compiled out, write-back and write-through, as in the default build, whose
# load/store ordering is the same bank logic. Those nine are the device's
# shared level, where a fence drains; the tenth, write-through above that
# level (IS_LLC=0, so no atomics), is where a fence walks and invalidates.
# Each set rebuilds: CONFIGS and CACHE_PARAMS are not make dependencies, so an
# extra run goes in the call that builds its set.
CACHE_BASE := -DVX_CFG_EXT_A_ENABLE
CACHE_L2   := WORD_SIZE=64 LINE_SIZE=128 SECTOR_SIZE=64 NUM_WAYS=8 LATENCY=4 NUM_BANKS=2 MEM_PORTS=2 CACHE_SIZE=32768
# $(1)=CONFIGS $(2)=CACHE_PARAMS $(3)=OPTS of an extra run on the same build
define cache_case
	$(MAKE) -C cache clean
	$(MAKE) -C cache CONFIGS="$(1)" CACHE_PARAMS="$(2)"
	$(MAKE) -C cache run CONFIGS="$(1)" CACHE_PARAMS="$(2)"
	$(if $(3),$(MAKE) -C cache run CONFIGS="$(1)" CACHE_PARAMS="$(2)" OPTS="$(3)")
endef
# Long stress, memory accepting 30% of the time: it shifts when fills and
# replays arrive, and response stalls (85% ready) hold the pipe across an
# atomic's settling writeback, the case the settle sweep directs.
CACHE_STALL := --tests=stress --seed=2 --stress-seeds=16 --stress-cycles=40000 --mem-ready=30
# Randomized traffic with fences mixed in, each draining or walking per set.
CACHE_FENCE := --tests=stress --seed=3 --stress-seeds=4 --stress-fence=4
# A CTA-scope fence (DLMEM, RV64 only) drains the first cache level although it
# is not the shared level: write-through and write-back L1s, with CTA fences
# also mixed into the randomized traffic.
CACHE_CTA  := AMO_ENABLE=0 IS_LLC=0 NUM_REQS=2 CTA_FENCE=1
run-cache:
	$(call cache_case,$(CACHE_BASE),,$(CACHE_FENCE))
	$(call cache_case,$(CACHE_BASE) -DVX_CFG_EXT_ZACAS_ENABLE,NUM_REQS=2 NUM_BANKS=2,$(CACHE_STALL))
	$(call cache_case,$(CACHE_BASE),NUM_REQS=1 NUM_BANKS=1)
	$(call cache_case,$(CACHE_BASE) -DVX_CFG_EXT_ZACAS_ENABLE,NUM_BANKS=2 LATENCY=3)
	$(call cache_case,$(CACHE_BASE),WRITEBACK=0 NUM_REQS=2,$(CACHE_FENCE))
	$(call cache_case,$(CACHE_BASE),WRITEBACK=0 NUM_REQS=2 LATENCY=4)
	$(call cache_case,$(CACHE_BASE) -DVX_CFG_EXT_ZACAS_ENABLE,$(CACHE_L2))
	$(call cache_case,,AMO_ENABLE=0)
	$(call cache_case,,AMO_ENABLE=0 WRITEBACK=0 NUM_REQS=2)
	$(call cache_case,,AMO_ENABLE=0 IS_LLC=0 WRITEBACK=0 NUM_REQS=2,$(CACHE_FENCE))
ifeq ($(XLEN),64)
	$(call cache_case,-DVX_CFG_EXT_DLMEM_ENABLE,$(CACHE_CTA) WRITEBACK=0,$(CACHE_FENCE))
	$(call cache_case,-DVX_CFG_EXT_DLMEM_ENABLE,$(CACHE_CTA) WRITEBACK=1,$(CACHE_FENCE))
endif

# TCU FEDP datapath — executed per format so a tcu/tfr change is gated by CI.
# Each format must enable its own RTL datapath (FP16-only is the default). Covers
# the native TFR and the DPI reference. Float formats compare against the windowed
# FEDP reference (USE_FEDP) with the datapath window --W=25 and the per-format
# NUM_REGS; DPI compares discrete mul/add (--no-fused). Integer formats use the
# exact reference (no window). Switches mirror ci/regression.sh's tensor() suite.
TCU_BASE := -DSIMULATION -DSV_DPI -DVX_CFG_EXT_TCU_ENABLE
# USE_DSP=1 maps the native-TFR FEDP multipliers onto DSP48 (vs the default
# LUT/Wallace path). Forwarded to tcu_fedp; ignored by the non-TFR datapaths.
USE_DSP ?= 0
# $(1)=NUM_REGS $(2)=extra CONFIGS $(3)=run OPTS
define tcu_case
	$(MAKE) -C tcu_fedp clean
	$(MAKE) -C tcu_fedp NUM_REGS=$(1) USE_DSP=$(USE_DSP) CONFIGS="$(TCU_BASE) $(2)"
	$(MAKE) -C tcu_fedp run NUM_REGS=$(1) USE_DSP=$(USE_DSP) CONFIGS="$(TCU_BASE) $(2)" OPTS="$(3)"
endef
# integer pair on one build (int8/uint8 share INT8; int4/uint4 share INT4)
# $(1)=type $(2)=enable $(3)=fmtA $(4)=fmtB
define tcu_int_pair
	$(MAKE) -C tcu_fedp clean
	$(MAKE) -C tcu_fedp USE_DSP=$(USE_DSP) CONFIGS="$(TCU_BASE) -DVX_CFG_TCU_TYPE_$(1) -DVX_CFG_TCU_$(2)_ENABLE"
	$(MAKE) -C tcu_fedp run USE_DSP=$(USE_DSP) CONFIGS="$(TCU_BASE) -DVX_CFG_TCU_TYPE_$(1) -DVX_CFG_TCU_$(2)_ENABLE" OPTS="--fmt=$(3)"
	$(MAKE) -C tcu_fedp run USE_DSP=$(USE_DSP) CONFIGS="$(TCU_BASE) -DVX_CFG_TCU_TYPE_$(1) -DVX_CFG_TCU_$(2)_ENABLE" OPTS="--fmt=$(4)"
endef
# Native-TFR float + integer formats; USE_DSP comes from the caller.
# The testbench latency derives from the configured VX_CFG_TCU_LATENCY; the
# last case pins the 2-stage multiply used for FPGA synthesis
# (VX_CFG_TCU_LATENCY=5).
define tcu_tfr_suite
	$(call tcu_case,2,-DVX_CFG_TCU_TYPE_TFR -DUSE_FEDP -DVX_CFG_TCU_TF32_ENABLE,--fmt=1 --W=25)
	$(call tcu_case,2,-DVX_CFG_TCU_TYPE_TFR -DUSE_FEDP,--fmt=2 --W=25)
	$(call tcu_case,4,-DVX_CFG_TCU_TYPE_TFR -DUSE_FEDP,--fmt=3 --W=25)
	$(call tcu_case,4,-DVX_CFG_TCU_TYPE_TFR -DUSE_FEDP -DVX_CFG_TCU_FP8_ENABLE,--fmt=4 --W=25)
	$(call tcu_case,2,-DVX_CFG_TCU_TYPE_TFR -DUSE_FEDP -DVX_CFG_TCU_FP8_ENABLE,--fmt=5 --W=25)
	$(call tcu_case,2,-DVX_CFG_TCU_TYPE_TFR -DUSE_FEDP -DVX_CFG_TCU_LATENCY=5,--fmt=2 --W=25)
	$(call tcu_int_pair,TFR,INT8,17,18)
	$(call tcu_int_pair,TFR,INT4,19,20)
endef
run-tcu:
	# native TFR — LUT/Wallace multipliers (USE_DSP=0)
	$(call tcu_tfr_suite)
	# DPI reference — float (discrete mul/add, --no-fused)
	$(call tcu_case,1,-DVX_CFG_TCU_TYPE_DPI -DVX_CFG_TCU_TF32_ENABLE,--fmt=1 --no-fused)
	$(call tcu_case,1,-DVX_CFG_TCU_TYPE_DPI,--fmt=2 --no-fused)
	$(call tcu_case,1,-DVX_CFG_TCU_TYPE_DPI,--fmt=3 --no-fused)
	$(call tcu_case,1,-DVX_CFG_TCU_TYPE_DPI -DVX_CFG_TCU_FP8_ENABLE,--fmt=4 --no-fused)
	$(call tcu_case,1,-DVX_CFG_TCU_TYPE_DPI -DVX_CFG_TCU_FP8_ENABLE,--fmt=5 --no-fused)
	# DPI reference — integer
	$(call tcu_int_pair,DPI,INT8,17,18)
	$(call tcu_int_pair,DPI,INT4,19,20)

# Same native-TFR formats as run-tcu, but with the multipliers mapped onto DSP48
# (USE_DSP=1). USE_DSP is bit-identical to the LUT path by contract, so the same
# references apply — a mismatch here is a DSP-packing bug.
run-tcu-dsp: USE_DSP := 1
run-tcu-dsp:
	$(call tcu_tfr_suite)

clean:
	$(MAKE) -C generic_queue clean
	$(MAKE) -C mem_scheduler clean
	$(MAKE) -C cache clean
	$(MAKE) -C core clean
	$(MAKE) -C issue clean
	$(MAKE) -C local_mem clean
	$(MAKE) -C lmem_switch clean
	$(MAKE) -C mem_unit clean
	$(MAKE) -C mbar_waiters clean
	$(MAKE) -C tcu_fedp clean
	$(MAKE) -C cta_dispatcher clean
	$(MAKE) -C kmu clean
	$(MAKE) -C dxa_core clean
	$(MAKE) -C tcu_unit clean
	$(MAKE) -C cp_arbiter clean
	$(MAKE) -C cp_engine clean
	$(MAKE) -C cp_launch clean
	$(MAKE) -C cp_dcr_proxy clean
	$(MAKE) -C cp_unpack clean
	$(MAKE) -C cp_axil_regfile clean
	$(MAKE) -C cp_axi_path clean
	$(MAKE) -C cp_dma clean
	$(MAKE) -C cp_core clean
	$(MAKE) -C fma_unit clean
	$(MAKE) -C fdiv_unit clean
	$(MAKE) -C fsqrt_unit clean
	$(MAKE) -C fcvt_unit clean
	$(MAKE) -C fdivsqrt_unit clean
	$(foreach u,$(DLMEM_UNITS),$(MAKE) -C $(u) clean &&) true
