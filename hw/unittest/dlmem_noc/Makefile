ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := dlmem_noc

RTL_DIR := $(VORTEX_HOME)/hw/rtl
DPI_DIR := $(VORTEX_HOME)/hw/dpi

SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)

# Packet payload width. The RTL parameter and the testbench constant come from
# this one variable; DATAW is not a make dependency, so rebuild after changing it.
DATAW ?= 160

# The fabric belongs to the DLMEM extension: always build it with the knob on.
ifeq ($(filter -DVX_CFG_EXT_DLMEM_ENABLE,$(CONFIGS)),)
override CONFIGS += -DVX_CFG_EXT_DLMEM_ENABLE
endif

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SW_COMMON_DIR)
CXXFLAGS += -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw
CXXFLAGS += -DNOC_DATAW=$(DATAW)

SRCS := $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

RTL_PKGS := $(RTL_DIR)/VX_gpu_pkg.sv $(RTL_DIR)/dlmem/VX_dlmem_pkg.sv

RTL_INCLUDE := -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(DPI_DIR) -I$(RTL_DIR)/libs
RTL_INCLUDE += -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/dlmem -I$(SRC_DIR)

TOP := VX_dlmem_noc_top

include ../common.mk

VL_FLAGS += -GDATAW=$(DATAW)
