ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# Uniform registers exist only with the H100 front end.
CONFIGS := $(if $(findstring -DVX_CFG_H100_FE_ENABLE,$(CONFIGS))$(findstring -DVX_CFG_H100_ORG_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_H100_FE_ENABLE)

PROJECT := uniform_regs

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp
HDRS := $(SRC_DIR)/common.h $(SW_COMMON_DIR)/uniform_ref.h $(SW_COMMON_DIR)/vx_fe_isa.h

VX_SRCS := $(SRC_DIR)/kernel.cpp
VX_HDRS := $(SRC_DIR)/common.h $(SW_COMMON_DIR)/vx_fe_isa.h $(VORTEX_HOME)/sw/kernel/include/vx_uniform.h

# vortex2 = the KMU kernel runtime, providing _start and __vx_cta_entry.
KERNEL_LIB := vortex2

include ../common.mk
