ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := dlmem_ep

RTL_DIR := $(VORTEX_HOME)/hw/rtl
DPI_DIR := $(VORTEX_HOME)/hw/dpi

SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)
NOC_DIR := $(VORTEX_HOME)/hw/unittest/dlmem_noc

# 1 makes a remote LR/SC an assertion failure, as in the core; the random and
# directed modes run with 0 to cover the defined hardware behaviour. It is not a
# make dependency, so rebuild after changing it.
LRSC_ASSERT ?= 0

# The endpoints belong to the DLMEM extension and carry AMOs: always build with both.
ifeq ($(filter -DVX_CFG_EXT_DLMEM_ENABLE,$(CONFIGS)),)
override CONFIGS += -DVX_CFG_EXT_DLMEM_ENABLE
endif
ifeq ($(filter -DVX_CFG_EXT_A_ENABLE,$(CONFIGS)),)
override CONFIGS += -DVX_CFG_EXT_A_ENABLE
endif
# A remote access needs a second core: without a shape, build one 2-core socket.
ifeq ($(filter -DVX_CFG_NUM_CORES=% -DVX_CFG_SOCKET_SIZE=%,$(CONFIGS)),)
override CONFIGS += -DVX_CFG_NUM_CORES=2 -DVX_CFG_SOCKET_SIZE=2
endif

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SW_COMMON_DIR)
CXXFLAGS += -I$(VORTEX_HOME)/tests/unittest/dlmem_ep
CXXFLAGS += -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw
CXXFLAGS += -DLRSC_ASSERT=$(LRSC_ASSERT)

SRCS := $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

RTL_PKGS := $(RTL_DIR)/VX_gpu_pkg.sv $(RTL_DIR)/dlmem/VX_dlmem_pkg.sv

RTL_INCLUDE := -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(DPI_DIR) -I$(RTL_DIR)/libs
RTL_INCLUDE += -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/mem -I$(RTL_DIR)/dlmem -I$(NOC_DIR) -I$(SRC_DIR)

TOP := VX_dlmem_ep_top

include ../common.mk

VL_FLAGS += -GLRSC_ASSERT=$(LRSC_ASSERT)
