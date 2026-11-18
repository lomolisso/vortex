ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := warp_ops

SRC_DIR := $(VORTEX_HOME)/tests/unittest/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

CXXFLAGS += -I$(SRC_DIR)

CONFIGS := $(if $(findstring -DVX_CFG_H100_FE_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_H100_FE_ENABLE)

XCONFIGS := $(shell python3 $(ROOT_DIR)/ci/gen_config.py --config=$(VORTEX_HOME)/VX_config.toml --cflags='$(CONFIGS) -DVX_CFG_XLEN=$(XLEN)')
CXXFLAGS += $(XCONFIGS)

include ../common.mk
