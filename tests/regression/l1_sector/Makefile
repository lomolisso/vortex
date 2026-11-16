ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := l1_sector

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

VX_HDRS := $(SRC_DIR)/common.h $(VORTEX_HOME)/sw/common/l1_ref.h
HDRS    := $(SRC_DIR)/common.h $(VORTEX_HOME)/sw/common/l1_ref.h

OPTS ?=

KERNEL_LIB := vortex2

include ../common.mk
