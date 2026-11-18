ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# ELECT, MATCH and REDUX exist only with the H100 front end.
CONFIGS := $(if $(findstring -DVX_CFG_H100_FE_ENABLE,$(CONFIGS))$(findstring -DVX_CFG_H100_ORG_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_H100_FE_ENABLE)

PROJECT := warp_ops

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp
HDRS := $(SRC_DIR)/common.h $(SW_COMMON_DIR)/warp_ops_ref.h $(VORTEX_HOME)/tests/unittest/warp_ops/stream.h

VX_SRCS := $(SRC_DIR)/kernel.cpp
VX_HDRS := $(SRC_DIR)/common.h $(SW_COMMON_DIR)/warp_ops_ref.h $(SW_COMMON_DIR)/vx_fe_isa.h \
           $(VORTEX_HOME)/sw/kernel/include/vx_warp_ops.h

# vortex2 = the KMU kernel runtime, providing _start and __vx_cta_entry.
KERNEL_LIB := vortex2

include ../common.mk
