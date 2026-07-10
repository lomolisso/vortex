ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DVX_CFG_EXT_TCU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_TCU_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DXA_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DXA_ENABLE)

PROJECT := sgemm_tcu_wg_dxa_mcast

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp $(SW_COMMON_DIR)/rvfloats.cpp $(SW_COMMON_DIR)/softfloat_ext.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

# NUM_WARPS=8 => mc_group_size = 8/4 = 2 co-resident 4-warp CTAs. At the
# pocket-H100 8 KB scratchpad, each CTA's A+B tiles need 2560 B, and the
# cluster admission reserves the whole mc_group span contiguously — 2 CTAs
# (5120 B) fit; 4 (10240 B) can never be admitted.
CONFIGS := $(if $(findstring -DVX_CFG_NUM_WARPS=,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_NUM_WARPS=8)
CONFIGS := $(if $(findstring -DVX_CFG_ISSUE_WIDTH=,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_ISSUE_WIDTH=4)
CONFIGS += -DVX_CFG_TCU_WGMMA_ENABLE -DWGMMA_RS


CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include

LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

KERNEL_LIB := vortex2

include ../common.mk
