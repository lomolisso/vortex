ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# The local-access ordering guarded here is a DLMEM-device property, and DXA
# copies are what keep the banks busy.
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DLMEM_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DLMEM_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DXA_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DXA_ENABLE)

PROJECT := lmem_bar_order

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp
HDRS := $(SRC_DIR)/common.h

VX_SRCS := $(SRC_DIR)/kernel.cpp
VX_HDRS := $(SRC_DIR)/common.h $(VORTEX_HOME)/sw/kernel/include/vx_dxa.h $(VORTEX_HOME)/sw/kernel/include/vx_barrier.h

# vortex2 = the KMU kernel runtime, providing _start and __vx_cta_entry.
KERNEL_LIB := vortex2

include ../common.mk
