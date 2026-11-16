ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# The SM-to-SM modes run on DLMEM builds and need A for the remote atomics; the
# global-store modes run on any build of the H100 SM path.

PROJECT := sm_port

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

VX_HDRS := $(SRC_DIR)/common.h $(VORTEX_HOME)/sw/kernel/include/vx_dlmem.h
HDRS    := $(SRC_DIR)/common.h $(VORTEX_HOME)/sw/common/sm_port_ref.h

OPTS ?=

KERNEL_LIB := vortex2

include ../common.mk
