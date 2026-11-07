ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := dxa_ref

SRC_DIR := $(VORTEX_HOME)/tests/unittest/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

# The reduction model takes its float rules from amo_ref.h (softfloat).
CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
LDFLAGS  += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

include ../common.mk
