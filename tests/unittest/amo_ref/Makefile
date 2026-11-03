ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := amo_ref

SRC_DIR := $(VORTEX_HOME)/tests/unittest/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

PROBE_CSV ?= $(VORTEX_HOME)/tests/unittest/amo_ref/h100_atomics.csv.gz

CXXFLAGS += -I$(VORTEX_HOME)/tests/regression/atomic_fp
CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
CXXFLAGS += -DPROBE_CSV='"$(PROBE_CSV)"'
LDFLAGS  += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

include ../common.mk

