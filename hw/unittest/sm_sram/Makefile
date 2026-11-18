ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := sm_sram

RTL_DIR := $(VORTEX_HOME)/hw/rtl
DPI_DIR := $(VORTEX_HOME)/hw/dpi

SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)

# The shared-memory port exists only on the H100 SM path.
CONFIGS := $(if $(findstring -DVX_CFG_H100_SM_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_H100_SM_ENABLE)

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SW_COMMON_DIR)
CXXFLAGS += -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(VORTEX_HOME)/hw/unittest/l1h100

SRCS += $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

RTL_PKGS := $(RTL_DIR)/VX_gpu_pkg.sv

RTL_INCLUDE := -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(DPI_DIR) -I$(RTL_DIR)/libs
RTL_INCLUDE += -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/mem -I$(RTL_DIR)/cache -I$(SRC_DIR)

TOP := VX_sm_sram_top

include ../common.mk
