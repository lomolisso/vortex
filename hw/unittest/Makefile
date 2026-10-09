ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

# Distributed local memory units: the feature is RV64 only.
DLMEM_UNITS := $(if $(filter 64,$(XLEN)),dlmem_noc)

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
run-lmem:
	$(call lmem_case,$(LMEM_BASE))
	$(call lmem_case,$(LMEM_BASE) -DVX_CFG_EXT_ZACAS_ENABLE)
	$(call lmem_case,)
	$(call lmem_switch_case,-DVX_CFG_NUM_LSU_LANES=1)
	$(call lmem_switch_case,-DVX_CFG_NUM_LSU_LANES=2)
	$(call lmem_switch_case,-DVX_CFG_NUM_LSU_LANES=4 $(LMEM_BASE) -DVX_CFG_EXT_ZACAS_ENABLE)

# CTA launch path: the KMU's cluster walk under launch-bus back-pressure, and
# the dispatcher's slot placement across launches checked against SimX's
# per-launch pointer reset, with 4 and 8 CTA slots. With DLMEM (RV64 only) both
# also carry the cluster rank to the CSR read-back, up to the largest cluster
# (8 cores) and with a 256 KB local memory.
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
run-dlmem:
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

# Cache bank in the last-level role: directed fill/replay races around the
# atomic engine and the MSHR chains plus randomized traffic, checked against a
# reference memory. Nine sets: the synthesis-gate DUT as configured; the L1
# dcache LLC as built (2 ports and 2 banks, behind an unbuffered request
# crossbar) with Zacas; one LSU channel into one bank; a deferred data stage
# (LATENCY=3); write-through, also with two deferred stages (LATENCY=4); the
# L2/L3 shape (word = sector, 2-sector lines, 2 memory ports); and atomics
# compiled out, write-back and write-through, as in the default build, whose
# load/store ordering is the same bank logic.
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
run-cache:
	$(call cache_case,$(CACHE_BASE),)
	$(call cache_case,$(CACHE_BASE) -DVX_CFG_EXT_ZACAS_ENABLE,NUM_REQS=2 NUM_BANKS=2,$(CACHE_STALL))
	$(call cache_case,$(CACHE_BASE),NUM_REQS=1 NUM_BANKS=1)
	$(call cache_case,$(CACHE_BASE) -DVX_CFG_EXT_ZACAS_ENABLE,NUM_BANKS=2 LATENCY=3)
	$(call cache_case,$(CACHE_BASE),WRITEBACK=0 NUM_REQS=2)
	$(call cache_case,$(CACHE_BASE),WRITEBACK=0 NUM_REQS=2 LATENCY=4)
	$(call cache_case,$(CACHE_BASE) -DVX_CFG_EXT_ZACAS_ENABLE,$(CACHE_L2))
	$(call cache_case,,AMO_ENABLE=0)
	$(call cache_case,,AMO_ENABLE=0 WRITEBACK=0 NUM_REQS=2)

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
