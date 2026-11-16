ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := smem_bank

# SimX twin of the shared-memory port bench (hw/unittest/smem_gate): the same
# cases (smem_gate_cases.h) against SmemGate, linked with the simulator's
# objects, which build in this directory.
SIMX_DIR := $(ROOT_DIR)/sim/simx
SRC_DIR  := $(VORTEX_HOME)/tests/unittest/$(PROJECT)
OUT      := $(CURDIR)/simx

CONFIGS := $(if $(findstring -DVX_CFG_H100_SM_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_H100_SM_ENABLE)
# The switch needs the LSU lanes to cover an L1 line (SM_LANES_SPAN_LINE).
CONFIGS := $(if $(findstring -DVX_CFG_NUM_THREADS,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_NUM_THREADS=32)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_A_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_A_ENABLE)

SIMX_SRC := $(VORTEX_HOME)/sim/simx
CXXFLAGS := -std=c++17 -Wall -Wextra -Wfatal-errors -Werror -O2 -DNDEBUG
CXXFLAGS += -I$(SIMX_SRC) -I$(SIMX_SRC)/dxa -I$(SIMX_SRC)/tex -I$(SIMX_SRC)/om -I$(SIMX_SRC)/raster -I$(SIMX_SRC)/rtu -I$(SIMX_SRC)/kmu -I$(SIMX_SRC)/mem -I$(SIMX_SRC)/dtm -I$(SIMX_SRC)/amo
CXXFLAGS += -I$(SW_COMMON_DIR) -I$(SIM_COMMON_DIR) -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(THIRD_PARTY_DIR)
CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include -I$(THIRD_PARTY_DIR)/ramulator/ext/spdlog/include
CXXFLAGS += -I$(THIRD_PARTY_DIR)/ramulator/ext/yaml-cpp/include -I$(THIRD_PARTY_DIR)/ramulator/src
CXXFLAGS += -I$(VORTEX_HOME)/hw/unittest/smem_gate
CXXFLAGS += -DVX_CFG_XLEN=$(XLEN) -DVX_CFG_XLEN_$(XLEN) $(CONFIGS)
LDFLAGS  := $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a
# SimX's DRAM model is its own archive (libvxdram.a), built by the SimX make under $(OUT).
LDFLAGS  += -Wl,--whole-archive $$(find $(OUT) -name libvxdram.a | head -1) -Wl,--no-whole-archive
LDFLAGS  += -Wl,-rpath,$(THIRD_PARTY_DIR)/ramulator -L$(THIRD_PARTY_DIR)/ramulator -lramulator

.PHONY: all run clean obj

all: $(PROJECT)

obj: $(OUT)/$(PROJECT)_test.o

$(OUT)/$(PROJECT)_test.o: $(SRC_DIR)/main.cpp $(VORTEX_HOME)/hw/unittest/smem_gate/smem_gate_cases.h
	mkdir -p $(OUT)
	$(CXX) $(CXXFLAGS) -c $(SRC_DIR)/main.cpp -o $@

$(PROJECT): $(OUT)/$(PROJECT)_test.o
	$(MAKE) -C $(SIMX_DIR) DESTDIR=$(OUT) CONFIGS="$(CONFIGS)" $(OUT)/simx
	$(CXX) $(CXXFLAGS) $$(find $(OUT)/obj -name '*.o' ! -name main.o ! -path '$(OUT)/obj/tests/*') $< $(LDFLAGS) -o $@

run: $(PROJECT)
	./$(PROJECT) $(OPTS)

clean:
	rm -rf $(PROJECT) $(OUT)
