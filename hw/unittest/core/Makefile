ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := core

RTL_DIR := $(VORTEX_HOME)/hw/rtl
DPI_DIR := $(VORTEX_HOME)/hw/dpi

SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)

LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SW_COMMON_DIR)
CXXFLAGS += -I/$(THIRD_PARTY_DIR)/softfloat/source/include
CXXFLAGS += -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw

VL_FLAGS += -DSV_DPI

SRCS := $(DPI_DIR)/dpi_util.cpp
SRCS += $(DPI_DIR)/dpi_float.cpp
SRCS += $(SW_COMMON_DIR)/softfloat_ext.cpp
SRCS += $(SW_COMMON_DIR)/rvfloats.cpp
SRCS += $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS := -DDBG_TRACE_CACHE

RTL_PKGS := $(RTL_DIR)/VX_gpu_pkg.sv $(RTL_DIR)/fpu/VX_fpu_pkg.sv

RTL_INCLUDE := -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(DPI_DIR) -I$(RTL_DIR)/libs
RTL_INCLUDE += -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm -I$(RTL_DIR)/cache -I$(RTL_DIR)/fpu -I$(RTL_DIR)/core -I$(SRC_DIR)

# DLMEM builds compile the hw/rtl/dlmem units and their package.
ifneq (,$(filter -DVX_CFG_EXT_DLMEM_ENABLE,$(CONFIGS)))
RTL_PKGS += $(RTL_DIR)/dlmem/VX_dlmem_pkg.sv
RTL_INCLUDE += -I$(RTL_DIR)/dlmem
endif

# DXA builds import the bulk-copy package.
ifneq (,$(filter -DVX_CFG_EXT_DXA_ENABLE,$(CONFIGS)))
RTL_PKGS += $(RTL_DIR)/dxa/VX_dxa_pkg.sv
RTL_INCLUDE += -I$(RTL_DIR)/dxa
endif

# TCU builds import the tensor package and the selected FEDP backend.
ifneq (,$(filter -DVX_CFG_EXT_TCU_ENABLE,$(CONFIGS)))
XCONFIGS := $(shell python3 $(ROOT_DIR)/ci/gen_config.py --config=$(VORTEX_HOME)/VX_config.toml --cflags='$(CONFIGS) -DVX_CFG_XLEN=$(XLEN)')
RTL_PKGS += $(RTL_DIR)/tcu/VX_tcu_pkg.sv
RTL_INCLUDE += -I$(RTL_DIR)/tcu
ifneq (,$(filter -DVX_CFG_TCU_H100_ENABLE,$(XCONFIGS)))
SRCS += $(DPI_DIR)/dpi_tcu.cpp
endif
ifneq (,$(filter -DVX_CFG_TCU_TYPE_DPI,$(XCONFIGS)))
RTL_INCLUDE += -I$(RTL_DIR)/tcu/dpi
else ifneq (,$(filter -DVX_CFG_TCU_TYPE_DSP,$(XCONFIGS)))
RTL_INCLUDE += -I$(RTL_DIR)/tcu/dsp
else ifneq (,$(filter -DVX_CFG_TCU_TYPE_BHF,$(XCONFIGS)))
RTL_INCLUDE += -I$(RTL_DIR)/tcu/bhf
RTL_PKGS += $(THIRD_PARTY_DIR)/hardfloat/source/fNToRecFN.v
RTL_PKGS += $(THIRD_PARTY_DIR)/hardfloat/source/addRecFN.v
RTL_PKGS += $(THIRD_PARTY_DIR)/hardfloat/source/bsg_hardfloat_pkg.sv
RTL_PKGS += $(THIRD_PARTY_DIR)/hardfloat/source/HardFloat_primitives.v
RTL_PKGS += $(THIRD_PARTY_DIR)/hardfloat/source/HardFloat_rawFN.v
RTL_PKGS += $(THIRD_PARTY_DIR)/hardfloat/source/isSigNaNRecFN.v
RTL_PKGS += $(THIRD_PARTY_DIR)/hardfloat/source/mulRecFN.v
RTL_PKGS += $(THIRD_PARTY_DIR)/hardfloat/source/recFNToFN.v
RTL_PKGS += $(THIRD_PARTY_DIR)/hardfloat/source/recFNToRecFN.v
RTL_INCLUDE += -I$(THIRD_PARTY_DIR)/hardfloat/source/RISCV -I$(THIRD_PARTY_DIR)/hardfloat/source
else ifneq (,$(filter -DVX_CFG_TCU_TYPE_FPNEW,$(XCONFIGS)))
RTL_INCLUDE += -I$(RTL_DIR)/tcu/fpnew
RTL_INCLUDE += -I$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/include -I$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/src -I$(THIRD_PARTY_DIR)/cvfpu/src
else
RTL_INCLUDE += -I$(RTL_DIR)/tcu/tfr
endif
endif

TOP := VX_core_top

include ../common.mk

# Elaborate and lint the core without building the model.
lint:
	$(VERILATOR_PATH)/bin/verilator --lint-only $(VL_FLAGS)
