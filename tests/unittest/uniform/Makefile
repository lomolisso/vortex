ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := uniform

SRC_DIR := $(VORTEX_HOME)/tests/unittest/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

CXXFLAGS += -I$(SRC_DIR)

ifeq ($(filter -DVX_CFG_H100_FE_ENABLE,$(CONFIGS)),)
override CONFIGS += -DVX_CFG_H100_FE_ENABLE
endif

XCONFIGS := $(shell python3 $(ROOT_DIR)/ci/gen_config.py --config=$(VORTEX_HOME)/VX_config.toml --cflags='$(CONFIGS) -DVX_CFG_XLEN=$(XLEN)')
CXXFLAGS += $(XCONFIGS)

include ../common.mk
