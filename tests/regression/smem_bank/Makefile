ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# The atomic modes need A; amocas.w needs Zacas.
CONFIGS := $(if $(findstring -DVX_CFG_EXT_A_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_A_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_ZACAS_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_ZACAS_ENABLE)

PROJECT := smem_bank

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

VX_HDRS := $(SRC_DIR)/common.h $(VORTEX_HOME)/sw/common/smem_bank_ref.h
HDRS    := $(SRC_DIR)/common.h $(VORTEX_HOME)/sw/common/smem_bank_ref.h

OPTS ?=

KERNEL_LIB := vortex2

include ../common.mk
