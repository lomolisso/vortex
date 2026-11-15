ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := dram_sim

SRC_DIR := $(VORTEX_HOME)/tests/unittest/$(PROJECT)

# The DRAM knobs are compile-time, so each memory shape is its own binary.
# Memory side of the micro/nano/pocket/h100 presets, with the SM counts that set
# the per-SM share; the legacy binary is the default shape on the HBM2 model.
H100_MEM := -DVX_CFG_DRAM_MODEL=H100 -DVX_CFG_NUM_PARTITIONS=2
SHAPE_legacy :=
SHAPE_micro  := $(H100_MEM) -DVX_CFG_SMS_PER_TPC=2 -DVX_CFG_LTCS_PER_MP=2 -DVX_CFG_SLICES_PER_LTC=2
SHAPE_nano   := $(H100_MEM) -DVX_CFG_GPCS_PER_PARTITION=4 -DVX_CFG_SMS_PER_TPC=2 -DVX_CFG_LTCS_PER_MP=2 -DVX_CFG_SLICES_PER_LTC=4
SHAPE_pocket := $(H100_MEM) -DVX_CFG_GPCS_PER_PARTITION=4 -DVX_CFG_CPCS_PER_GPC=2 -DVX_CFG_SMS_PER_TPC=2 \
                -DVX_CFG_MPS_PER_PARTITION=2 -DVX_CFG_LTCS_PER_MP=2 -DVX_CFG_SLICES_PER_LTC=4
SHAPE_h100   := $(H100_MEM) -DVX_CFG_GPCS_PER_PARTITION=4 -DVX_CFG_CPCS_PER_GPC=2 -DVX_CFG_TPCS_PER_CPC=4 -DVX_CFG_SMS_PER_TPC=2 \
                -DVX_CFG_MPS_PER_PARTITION=8 -DVX_CFG_LTCS_PER_MP=2 -DVX_CFG_SLICES_PER_LTC=4

# The h100 binary only measures wall time.
ARGS_h100 := SPEED

SHAPES ?= legacy micro nano pocket h100

ifeq ($(SHAPE),)

.PHONY: all run clean

all:
	$(foreach s,$(SHAPES),$(MAKE) SHAPE=$(s) shape &&) true

run:
	$(foreach s,$(SHAPES),$(MAKE) SHAPE=$(s) run-shape &&) true

clean:
	rm -rf $(SHAPES)

else

DESTDIR := $(CURDIR)/$(SHAPE)
CONFIG_FILE := $(DESTDIR)/config.stamp

SHAPE_CONFIGS := $(CONFIGS) $(SHAPE_$(SHAPE))
XCONFIGS := $(shell python3 $(ROOT_DIR)/ci/gen_config.py --config=$(VORTEX_HOME)/VX_config.toml --cflags='$(SHAPE_CONFIGS) -DVX_CFG_XLEN=$(XLEN)')
ifeq ($(XCONFIGS),)
$(error gen_config.py rejected the $(SHAPE) shape)
endif

ifdef DEBUG
CXXFLAGS += -g -O0
else
CXXFLAGS += -O2
endif
CXXFLAGS += -I$(SIM_COMMON_DIR) -I$(SW_COMMON_DIR) -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw
CXXFLAGS += $(XCONFIGS)

include $(SIM_COMMON_DIR)/dram/ramulator.mk

TEST_CXXFLAGS := -std=c++17 -Wall -Wextra -Wfatal-errors -Werror -I$(dir $(VXDRAM_CONFIG_H)) $(CXXFLAGS)

.PHONY: shape run-shape force

shape: $(DESTDIR)/$(PROJECT)

$(CONFIG_FILE): force
	@mkdir -p $(@D)
	@printf '%s\n' "$(CXXFLAGS)" > $@.tmp
	@if cmp -s $@.tmp $@; then rm $@.tmp; else mv $@.tmp $@; fi

$(DESTDIR)/$(PROJECT): $(SRC_DIR)/main.cpp $(VXDRAM_LIB) $(VXDRAM_CONFIG_H) $(SW_COMMON_DIR)/vx_dram_map.h $(CONFIG_FILE)
	$(CXX) $(TEST_CXXFLAGS) $(SRC_DIR)/main.cpp $(VXDRAM_LDFLAGS) -o $@

run-shape: $(DESTDIR)/$(PROJECT)
	@echo "== dram_sim $(SHAPE)"
	$(DESTDIR)/$(PROJECT) $(or $(ARGS),$(ARGS_$(SHAPE)))

endif
