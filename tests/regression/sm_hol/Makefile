ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# A cluster of 2 on a DLMEM build of the H100 SM path.

PROJECT := sm_hol

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

VX_HDRS := $(SRC_DIR)/common.h $(VORTEX_HOME)/sw/kernel/include/vx_dlmem.h
HDRS    := $(SRC_DIR)/common.h

OPTS ?=

KERNEL_LIB := vortex2

include ../common.mk
