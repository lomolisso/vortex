ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# Tensor-map cache levels: capacity, replacement, invalidate, prefetch, param maps, launches, sharing.
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DLMEM_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DLMEM_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_A_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_A_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_MBAR_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_MBAR_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DXA_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DXA_ENABLE)

PROJECT := tmap_cache

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp
HDRS := $(SRC_DIR)/common.h 

VX_SRCS := $(SRC_DIR)/kernel.cpp
VX_HDRS := $(SRC_DIR)/common.h $(SW_COMMON_DIR)/dxa_meta.h $(SW_COMMON_DIR)/dxa_tmap.h $(VORTEX_HOME)/sw/runtime/include/dxa.h \
           $(VORTEX_HOME)/sw/kernel/include/vx_mbar.h $(VORTEX_HOME)/sw/kernel/include/vx_dxa.h \
           $(VORTEX_HOME)/sw/kernel/include/vx_dlmem.h

KERNEL_LIB := vortex2

include ../common.mk
