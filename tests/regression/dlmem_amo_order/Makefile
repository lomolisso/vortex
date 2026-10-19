ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# The window exists only on a DLMEM device, the probes are atomics, and DXA
# copies are one of the ways to keep the target's banks busy.
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DLMEM_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DLMEM_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_A_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_A_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DXA_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DXA_ENABLE)

PROJECT := dlmem_amo_order

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp
HDRS := $(SRC_DIR)/common.h

VX_SRCS := $(SRC_DIR)/kernel.cpp
VX_HDRS := $(SRC_DIR)/common.h $(VORTEX_HOME)/sw/kernel/include/vx_dlmem.h $(VORTEX_HOME)/sw/kernel/include/vx_dxa.h

# vortex2 = the KMU kernel runtime, providing _start and __vx_cta_entry.
KERNEL_LIB := vortex2

include ../common.mk
