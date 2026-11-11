ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# ClusterFusion's fused decoder kernels (Llama-2 attention block, DeepSeek-V2-Lite
# MLA), at the ports' shape: a DLMEM device with the A extension, Zacas,
# mbarriers, the DXA and FPAMO (RV64).
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DLMEM_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DLMEM_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_A_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_A_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_ZACAS_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_ZACAS_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_MBAR_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_MBAR_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DXA_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DXA_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_FPAMO_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_FPAMO_ENABLE)

PROJECT := cf_decoder

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

# The problem sizes the H100 build fixes in config.h, reduced: the Llama head
# count (HIDDEN_DIM = heads x 128) and the MLA head count, hidden size and
# sequence length. They are not CONFIGS (the simulator's build key);
# config.stamp records the flags, so a new value rebuilds the app.
CF_LLAMA_HEADS ?= 4
CF_MLA_HEADS   ?= 2
CF_MLA_HIDDEN  ?= 1024
CF_MLA_SEQ     ?= 128
CF_FLAGS := -DCF_LLAMA_HEADS=$(CF_LLAMA_HEADS) -DCF_MLA_HEADS=$(CF_MLA_HEADS) \
            -DCF_MLA_HIDDEN=$(CF_MLA_HIDDEN) -DCF_MLA_SEQ=$(CF_MLA_SEQ)

APP_HDRS := $(SRC_DIR)/common.h $(SRC_DIR)/cf_dsm.h

SRCS := $(SRC_DIR)/main.cpp
HDRS := $(APP_HDRS) $(SW_COMMON_DIR)/vx_cuda_fp.h $(SW_COMMON_DIR)/dxa_tmap.h $(SW_COMMON_DIR)/dxa_meta.h \
        $(VORTEX_HOME)/sw/runtime/include/vx_cuda_runtime.h $(VORTEX_HOME)/sw/runtime/include/vortex2.h \
        $(VORTEX_HOME)/sw/runtime/include/vortex.h $(VORTEX_HOME)/sw/runtime/include/dxa.h

VX_SRCS := $(SRC_DIR)/kernel.cpp
VX_HDRS := $(APP_HDRS) $(SW_COMMON_DIR)/vx_cuda_fp.h $(SW_COMMON_DIR)/dxa_tmap.h $(SW_COMMON_DIR)/dxa_meta.h \
           $(VORTEX_HOME)/sw/kernel/include/vx_cuda.h $(VORTEX_HOME)/sw/kernel/include/vx_mbar.h \
           $(VORTEX_HOME)/sw/kernel/include/vx_dxa.h $(VORTEX_HOME)/sw/kernel/include/vx_dlmem.h \
           $(VORTEX_HOME)/sw/kernel/include/vx_cp_async.h $(VORTEX_HOME)/sw/kernel/include/vx_atomic.h \
           $(VORTEX_HOME)/sw/kernel/include/vx_spawn2.h $(VORTEX_HOME)/sw/kernel/include/vx_intrinsics.h \
           $(VORTEX_HOME)/sw/kernel/include/vx_print.h $(VORTEX_HOME)/sw/kernel/include/vx_barrier.h

VX_CFLAGS += $(CF_FLAGS)
CXXFLAGS += $(CF_FLAGS)

# The kernels move fp16 data through uint4 and uint64_t lvalues, as the CUDA
# originals do; nvcc accepts that, clang's type-based alias analysis would not.
VX_CFLAGS += -fno-strict-aliasing

# vx_cuda.h builds the DXA operands in x30/x31 (t5/t6): the wgather that
# broadcasts them also writes the parked lanes' destination register.
VX_CFLAGS += -ffixed-x30 -ffixed-x31 -DVX_CUDA_FIXED_TEMPS

# Barriers, shuffles and votes are inline asm. Unless every call is convergent,
# the compiler may copy one into both arms of a divergent branch (seen: a
# cluster.sync between two `if (threadIdx.x < n)` blocks), and a split warp then
# arrives twice in one barrier phase. CUDA compiles device code this way.
VX_CFLAGS += -fconvergent-functions

# vortex2 = the KMU kernel runtime, providing _start and __vx_cta_entry.
KERNEL_LIB := vortex2

include ../common.mk
