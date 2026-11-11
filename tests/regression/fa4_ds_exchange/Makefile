ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# The FlashAttention-4 two-CTA dS exchange at the ports' shape: a DLMEM device
# with the A extension, Zacas, mbarriers, the DXA and FPAMO (RV64).
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DLMEM_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DLMEM_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_A_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_A_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_ZACAS_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_ZACAS_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_MBAR_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_MBAR_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DXA_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DXA_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_FPAMO_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_FPAMO_ENABLE)

PROJECT := fa4_ds_exchange

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp
HDRS := $(SRC_DIR)/common.h $(SW_COMMON_DIR)/vx_cuda_fp.h $(SW_COMMON_DIR)/dxa_tmap.h $(SW_COMMON_DIR)/dxa_meta.h \
        $(VORTEX_HOME)/sw/runtime/include/vx_cuda_runtime.h $(VORTEX_HOME)/sw/runtime/include/vortex2.h \
        $(VORTEX_HOME)/sw/runtime/include/vortex.h $(VORTEX_HOME)/sw/runtime/include/dxa.h

VX_SRCS := $(SRC_DIR)/kernel.cpp
VX_HDRS := $(SRC_DIR)/common.h $(SW_COMMON_DIR)/vx_cuda_fp.h $(SW_COMMON_DIR)/dxa_tmap.h $(SW_COMMON_DIR)/dxa_meta.h \
           $(VORTEX_HOME)/sw/kernel/include/vx_cuda.h $(VORTEX_HOME)/sw/kernel/include/vx_mbar.h \
           $(VORTEX_HOME)/sw/kernel/include/vx_dxa.h $(VORTEX_HOME)/sw/kernel/include/vx_dlmem.h \
           $(VORTEX_HOME)/sw/kernel/include/vx_cp_async.h $(VORTEX_HOME)/sw/kernel/include/vx_atomic.h \
           $(VORTEX_HOME)/sw/kernel/include/vx_spawn2.h $(VORTEX_HOME)/sw/kernel/include/vx_intrinsics.h \
           $(VORTEX_HOME)/sw/kernel/include/vx_print.h $(VORTEX_HOME)/sw/kernel/include/vx_barrier.h

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
