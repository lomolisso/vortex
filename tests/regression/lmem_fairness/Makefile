ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# The pollers spin on a local-memory atomic; force-enable A if not already set.
CONFIGS := $(if $(findstring -DVX_CFG_EXT_A_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_A_ENABLE)

PROJECT := lmem_fairness

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

VX_HDRS := $(SRC_DIR)/common.h
HDRS    := $(SRC_DIR)/common.h

OPTS ?=

KERNEL_LIB := vortex2

include ../common.mk
