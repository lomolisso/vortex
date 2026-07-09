ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DVX_CFG_CLUSTER_MEM_MODE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_CLUSTER_MEM_MODE_DSMEM)

PROJECT := dsmem_pingpong

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

OPTS ?= -n2

KERNEL_LIB := vortex2

include ../common.mk
