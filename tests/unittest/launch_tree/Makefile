ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := launch_tree

SRC_DIR := $(VORTEX_HOME)/tests/unittest/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

CXXFLAGS += -DVX_CFG_XLEN=$(XLEN) -DVX_CFG_XLEN_$(XLEN)
CXXFLAGS += -I$(VORTEX_HOME)/sim/simx -I$(VORTEX_HOME)/sim/simx/kmu -I$(SIM_COMMON_DIR)
CXXFLAGS += -I$(THIRD_PARTY_DIR) -I$(THIRD_PARTY_DIR)/softfloat/source/include

include ../common.mk

# types.h has anonymous structs
CXXFLAGS += -Wno-pedantic
