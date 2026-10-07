ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := cache_order

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

VX_HDRS := $(SRC_DIR)/common.h
HDRS    := $(SRC_DIR)/common.h

OPTS ?=

KERNEL_LIB := vortex2

include ../common.mk
