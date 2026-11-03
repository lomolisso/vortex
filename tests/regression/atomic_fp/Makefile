ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# The FPAMO atomics and the loops' compare-and-swap.
CONFIGS := $(if $(findstring -DVX_CFG_EXT_A_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_A_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_ZACAS_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_ZACAS_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_FPAMO_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_FPAMO_ENABLE)

PROJECT := atomic_fp

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp
HDRS := $(SRC_DIR)/common.h $(SRC_DIR)/probe_csv.h $(VORTEX_HOME)/sw/common/amo_ref.h

VX_SRCS := $(SRC_DIR)/kernel.cpp
VX_HDRS := $(SRC_DIR)/common.h $(VORTEX_HOME)/sw/kernel/include/vx_atomic.h

PROBE_CSV ?= $(VORTEX_HOME)/tests/unittest/amo_ref/h100_atomics.csv.gz

CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include -DPROBE_CSV='"$(PROBE_CSV)"'
LDFLAGS  += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

KERNEL_LIB := vortex2

include ../common.mk
