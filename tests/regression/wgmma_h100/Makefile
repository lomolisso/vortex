ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# The H100 WGMMA test shape: RV64, 32 threads, one warpgroup per issue width
# of 4, a DLMEM device with the A extension, Zacas, mbarriers, the DXA, FPAMO
# and the tensor unit with WGMMA, FP8 and the H100 mode.
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DLMEM_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DLMEM_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_A_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_A_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_ZACAS_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_ZACAS_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_MBAR_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_MBAR_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DXA_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DXA_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_FPAMO_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_FPAMO_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_TCU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_TCU_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_TCU_WGMMA_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_TCU_WGMMA_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_TCU_FP8_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_TCU_FP8_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_TCU_H100_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_TCU_H100_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_NUM_THREADS=,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_NUM_THREADS=32)
CONFIGS := $(if $(findstring -DVX_CFG_ISSUE_WIDTH=,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_ISSUE_WIDTH=4)
CONFIGS := $(if $(findstring -DVX_CFG_LMEM_LOG_SIZE=,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_LMEM_LOG_SIZE=18)
# Both operands come from shared memory: keep common.mk's register-A default off.
CONFIGS := $(if $(findstring -DWGMMA_SS,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DWGMMA_SS)

PROJECT := wgmma_h100

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp
HDRS := $(SRC_DIR)/common.h $(SW_COMMON_DIR)/tcu_desc_ref.h $(SW_COMMON_DIR)/tcu_fedp_ref.h \
        $(SW_COMMON_DIR)/dxa_ref.h $(SW_COMMON_DIR)/dxa_tmap.h $(SW_COMMON_DIR)/dxa_meta.h \
        $(VORTEX_HOME)/sw/runtime/include/vx_cuda_runtime.h $(VORTEX_HOME)/sw/runtime/include/dxa.h

VX_SRCS := $(SRC_DIR)/kernel.cpp
VX_HDRS := $(SRC_DIR)/common.h $(SW_COMMON_DIR)/dxa_tmap.h $(SW_COMMON_DIR)/dxa_meta.h \
           $(VORTEX_HOME)/sw/kernel/include/vx_cuda.h $(VORTEX_HOME)/sw/kernel/include/vx_tensor.h \
           $(VORTEX_HOME)/sw/kernel/include/vx_mbar.h $(VORTEX_HOME)/sw/kernel/include/vx_dxa.h \
           $(VORTEX_HOME)/sw/kernel/include/vx_dlmem.h $(VORTEX_HOME)/sw/kernel/include/vx_intrinsics.h

# vx_cuda.h builds the DXA operands in x30/x31 (t5/t6): the wgather that
# broadcasts them also writes the parked lanes' destination register.
VX_CFLAGS += -ffixed-x30 -ffixed-x31 -DVX_CUDA_FIXED_TEMPS

# Barriers, shuffles and votes are inline asm; every call is convergent so the
# compiler never copies one into both arms of a divergent branch.
VX_CFLAGS += -fconvergent-functions

# vortex2 = the KMU kernel runtime, providing _start and __vx_cta_entry.
KERNEL_LIB := vortex2

include ../common.mk
