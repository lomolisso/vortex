ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := cuda_fp

SRC_DIR := $(VORTEX_HOME)/tests/unittest/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
# The library is built with 64-bit fast paths; its f128 API is declared only then.
CXXFLAGS += -DSOFTFLOAT_FAST_INT64
LDFLAGS  += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

include ../common.mk
