ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DVX_CFG_EXT_DXA_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DXA_ENABLE)

PROJECT := dxa_copy_mcast

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

# Pocket-H100 shape: tile_cols must equal NUM_THREADS (32), and a 1-warp CTA's
# scratchpad budget at 8 KB LMEM / 16 warps is 512 B => tile 4x32 floats.
OPTS ?= -r4 -c32 -R4 -C32

KERNEL_LIB := vortex2

include ../common.mk
