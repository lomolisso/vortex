ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := dxa_tmap

SRC_DIR := $(VORTEX_HOME)/tests/unittest/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

# The host encoder lives in the runtime's public header dxa.h.
CXXFLAGS += -I$(VORTEX_HOME)/sw/runtime/include

include ../common.mk
