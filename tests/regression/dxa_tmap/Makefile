ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# PTX tensor maps in memory: a DLMEM device with the A extension, mbarriers and the DXA.
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DLMEM_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DLMEM_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_A_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_A_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_MBAR_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_MBAR_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DXA_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DXA_ENABLE)

PROJECT := dxa_tmap

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp
HDRS := $(SRC_DIR)/common.h $(SW_COMMON_DIR)/dxa_ref.h $(SW_COMMON_DIR)/dxa_meta.h $(SW_COMMON_DIR)/dxa_tmap.h \
        $(SW_COMMON_DIR)/amo_ref.h $(VORTEX_HOME)/sw/runtime/include/dxa.h

VX_SRCS := $(SRC_DIR)/kernel.cpp
VX_HDRS := $(SRC_DIR)/common.h $(SW_COMMON_DIR)/dxa_meta.h $(SW_COMMON_DIR)/dxa_tmap.h \
           $(VORTEX_HOME)/sw/kernel/include/vx_mbar.h $(VORTEX_HOME)/sw/kernel/include/vx_dxa.h \
           $(VORTEX_HOME)/sw/kernel/include/vx_dlmem.h

# The reduction model takes its float rules from amo_ref.h (softfloat).
CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
LDFLAGS  += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

# vortex2 = the KMU kernel runtime, providing _start and __vx_cta_entry.
KERNEL_LIB := vortex2

include ../common.mk
