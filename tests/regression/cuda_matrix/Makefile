ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# The CUDA porting layer at the ports' shape (RV64, 32 threads, a DLMEM
# device with the A extension, Zacas, mbarriers, the DXA and FPAMO), 8 warps,
# 16 named barriers and 256 KB of local memory.
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DLMEM_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DLMEM_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_A_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_A_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_ZACAS_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_ZACAS_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_MBAR_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_MBAR_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DXA_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DXA_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_FPAMO_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_FPAMO_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_NUM_THREADS=,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_NUM_THREADS=32)
CONFIGS := $(if $(findstring -DVX_CFG_NUM_WARPS=,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_NUM_WARPS=8)
CONFIGS := $(if $(findstring -DVX_CFG_NUM_BARRIERS=,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_NUM_BARRIERS=16)
CONFIGS := $(if $(findstring -DVX_CFG_LMEM_LOG_SIZE=,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_LMEM_LOG_SIZE=18)

PROJECT := cuda_matrix

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp
HDRS := $(SRC_DIR)/common.h $(VORTEX_HOME)/sw/runtime/include/vx_cuda_runtime.h

VX_SRCS := $(SRC_DIR)/kernel.cpp
VX_HDRS := $(SRC_DIR)/common.h $(VORTEX_HOME)/sw/kernel/include/vx_cuda.h \
           $(VORTEX_HOME)/sw/kernel/include/vx_intrinsics.h $(VORTEX_HOME)/sw/kernel/include/vx_spawn2.h

# vx_cuda.h builds the DXA operands in x30/x31 (t5/t6): the wgather that
# broadcasts them also writes the parked lanes' destination register.
VX_CFLAGS += -ffixed-x30 -ffixed-x31 -DVX_CUDA_FIXED_TEMPS

# Barriers and shuffles are inline asm; every call is convergent so the
# compiler never copies one into both arms of a divergent branch.
VX_CFLAGS += -fconvergent-functions

# vortex2 = the KMU kernel runtime, providing _start and __vx_cta_entry.
KERNEL_LIB := vortex2

include ../common.mk
