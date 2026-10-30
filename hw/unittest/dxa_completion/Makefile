ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := dxa_completion

RTL_DIR := $(VORTEX_HOME)/hw/rtl
DPI_DIR := $(VORTEX_HOME)/hw/dpi

SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)

# mbarrier completion exists only in DXA builds with mbarriers: always build with the knobs on.
ifeq ($(filter -DVX_CFG_EXT_MBAR_ENABLE,$(CONFIGS)),)
override CONFIGS += -DVX_CFG_EXT_DLMEM_ENABLE -DVX_CFG_EXT_A_ENABLE -DVX_CFG_EXT_MBAR_ENABLE
endif
ifeq ($(filter -DVX_CFG_EXT_DXA_ENABLE,$(CONFIGS)),)
override CONFIGS += -DVX_CFG_EXT_DXA_ENABLE
endif

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SW_COMMON_DIR)
CXXFLAGS += -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw

SRCS := $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

RTL_PKGS := $(RTL_DIR)/VX_gpu_pkg.sv $(RTL_DIR)/dlmem/VX_dlmem_pkg.sv $(RTL_DIR)/dxa/VX_dxa_pkg.sv

RTL_INCLUDE := -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(DPI_DIR) -I$(RTL_DIR)/libs
RTL_INCLUDE += -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/mem -I$(RTL_DIR)/dxa -I$(RTL_DIR)/dlmem -I$(SRC_DIR)

TOP := VX_dxa_completion_top

include ../common.mk

# The unit imports the DXA package for none of its constants (after -Wall).
VL_FLAGS += -Wno-UNUSEDPARAM

# Elaborate and lint the completion unit without building the model.
lint:
	$(VERILATOR_PATH)/bin/verilator --lint-only $(VL_FLAGS)
