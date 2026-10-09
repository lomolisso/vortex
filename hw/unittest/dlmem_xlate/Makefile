ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := dlmem_xlate

RTL_DIR := $(VORTEX_HOME)/hw/rtl
DPI_DIR := $(VORTEX_HOME)/hw/dpi

SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)

# The translation belongs to the DLMEM extension: always build it with the knob on.
ifeq ($(filter -DVX_CFG_EXT_DLMEM_ENABLE,$(CONFIGS)),)
override CONFIGS += -DVX_CFG_EXT_DLMEM_ENABLE
endif

# The stage's global core id and its out-of-bounds assert. Neither is a make
# dependency, so rebuild after changing them.
CORE_ID    ?= 0
OOB_ASSERT ?= 0

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SW_COMMON_DIR)
CXXFLAGS += -I$(VORTEX_HOME)/tests/unittest/dlmem_xlate
CXXFLAGS += -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw

SRCS := $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

RTL_PKGS := $(RTL_DIR)/VX_gpu_pkg.sv $(RTL_DIR)/dlmem/VX_dlmem_pkg.sv

RTL_INCLUDE := -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(DPI_DIR) -I$(RTL_DIR)/libs
RTL_INCLUDE += -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/dlmem -I$(SRC_DIR)

TOP := VX_dlmem_xlate_top

include ../common.mk

VL_FLAGS += -GCORE_ID=$(CORE_ID) -GOOB_ASSERT=$(OOB_ASSERT)
