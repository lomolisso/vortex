ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# DLMEM is not forced: on a device without it the app checks that a SPREAD
# launch is refused and runs only same-core clusters.
PROJECT := cta_gang

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp
HDRS := $(SRC_DIR)/common.h

VX_SRCS := $(SRC_DIR)/kernel.cpp
VX_HDRS := $(SRC_DIR)/common.h

# vortex2 = the KMU kernel runtime, providing _start and __vx_cta_entry.
KERNEL_LIB := vortex2

include ../common.mk
