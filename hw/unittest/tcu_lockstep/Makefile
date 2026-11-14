ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := tcu_lockstep

RTL_DIR := $(VORTEX_HOME)/hw/rtl
DPI_DIR := $(VORTEX_HOME)/hw/dpi

SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SW_COMMON_DIR)
CXXFLAGS += -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw

SRCS += $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

# The gate of every WGMMA build; H100=1 builds it with the H100 knob's shape.
CONFIGS += -DVX_CFG_NUM_WARPS=8 -DVX_CFG_ISSUE_WIDTH=4 -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE
ifeq ($(H100),1)
CONFIGS += -DVX_CFG_NUM_THREADS=32 -DVX_CFG_LMEM_LOG_SIZE=18 \
	-DVX_CFG_EXT_A_ENABLE -DVX_CFG_EXT_ZACAS_ENABLE -DVX_CFG_EXT_DLMEM_ENABLE \
	-DVX_CFG_EXT_MBAR_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_TCU_FP8_ENABLE \
	-DVX_CFG_TCU_H100_ENABLE
endif

RTL_PKGS := $(RTL_DIR)/VX_gpu_pkg.sv $(RTL_DIR)/tcu/VX_tcu_pkg.sv

RTL_INCLUDE := -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(DPI_DIR) -I$(RTL_DIR)/libs
RTL_INCLUDE += -I$(RTL_DIR)/tcu -I$(RTL_DIR)/interfaces -I$(SRC_DIR)

TOP := VX_tcu_lockstep_top

include ../common.mk
