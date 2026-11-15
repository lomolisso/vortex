# DramSim and the Vortex Ramulator 2.1 extensions, built once per simulator build directory.
#
# Ramulator 2.1 is C++20 and ships fmt 10; every other simulator TU stays C++17 and never sees a
# Ramulator header. So dram_sim.cpp and sim/common/dram/*.cpp go into libvxdram.a with their own
# rule (outside Verilator's generated make, which has one CFLAGS set), and every simulator links it
# whole (the extension TUs are reached only through their static registration) with libramulator.so.
#
# Include after DESTDIR, CXXFLAGS (its -I/-D/-O/-g flags are reused), XCONFIGS and CONFIG_FILE are set.
# Provides: VXDRAM_LIB, VXDRAM_CONFIG_H, VXDRAM_LDFLAGS.
#
# Simulators share one DESTDIR (build/runtime) with different configurations, so the outputs are keyed
# on the including simulator's config stamp.

VXDRAM_DIR       := $(SIM_COMMON_DIR)/dram
RAMULATOR_DIR    := $(THIRD_PARTY_DIR)/ramulator
VXDRAM_OUT       ?= $(DESTDIR)/vxdram-$(basename $(notdir $(CONFIG_FILE)))
VXDRAM_OBJ_DIR   := $(VXDRAM_OUT)/obj
VXDRAM_LIB       := $(VXDRAM_OUT)/libvxdram.a
VXDRAM_CONFIG_H  := $(VXDRAM_OUT)/vx_dram_config.h
VXDRAM_FLAGS     := $(VXDRAM_OUT)/flags.stamp
VXDRAM_SRCS      := $(SIM_COMMON_DIR)/dram_sim.cpp $(wildcard $(VXDRAM_DIR)/*.cpp)
VXDRAM_OBJS      := $(patsubst %.cpp,$(VXDRAM_OBJ_DIR)/%.o,$(notdir $(VXDRAM_SRCS)))
VXDRAM_PY_SRCS   := $(shell find $(RAMULATOR_DIR)/python -name '*.py' 2>/dev/null)

VXDRAM_CXXFLAGS   = -std=c++20 -fPIC -Wall -Wextra -Wfatal-errors -Werror
VXDRAM_CXXFLAGS  += -isystem $(RAMULATOR_DIR)/src -isystem $(RAMULATOR_DIR)/ext/fmt/include -isystem $(RAMULATOR_DIR)/ext/yaml-cpp/include
VXDRAM_CXXFLAGS  += -I$(VXDRAM_OUT) -I$(VXDRAM_DIR) -I$(SIM_COMMON_DIR) -I$(SW_COMMON_DIR)
VXDRAM_CXXFLAGS  += $(filter -I% -D% -O% -g,$(CXXFLAGS))

VXDRAM_LDFLAGS   := -Wl,--whole-archive $(VXDRAM_LIB) -Wl,--no-whole-archive
VXDRAM_LDFLAGS   += -Wl,-rpath,$(RAMULATOR_DIR) -L$(RAMULATOR_DIR) -lramulator

# Ramulator's exporter needs Python >= 3.10.
VXDRAM_PYTHON    ?= python3

# The stamp changes only when the expanded configuration or the compile flags do, e.g. after a
# VX_config.toml edit that leaves CONFIGS and the sim's own stamp untouched.
$(VXDRAM_FLAGS): vxdram-force
	@mkdir -p $(@D)
	@echo '$(XCONFIGS) $(VXDRAM_CXXFLAGS)' > $@.tmp
	@if cmp -s $@.tmp $@; then rm $@.tmp; else mv $@.tmp $@; fi

$(VXDRAM_CONFIG_H): $(VXDRAM_DIR)/gen_dram_config.py $(VXDRAM_PY_SRCS) $(VXDRAM_FLAGS)
	@mkdir -p $(@D)
	PYTHONPATH=$(RAMULATOR_DIR)/python $(VXDRAM_PYTHON) $(VXDRAM_DIR)/gen_dram_config.py --cflags '$(XCONFIGS)' -o $@.tmp
	@mv $@.tmp $@

# -MD, not -MMD: the Ramulator headers come in through -isystem.
$(VXDRAM_OBJ_DIR)/dram_sim.o: $(SIM_COMMON_DIR)/dram_sim.cpp $(VXDRAM_CONFIG_H) $(VXDRAM_FLAGS)
	@mkdir -p $(@D)
	$(CXX) $(VXDRAM_CXXFLAGS) -MD -MP -MF $(@:.o=.d) -c $< -o $@

$(VXDRAM_OBJ_DIR)/%.o: $(VXDRAM_DIR)/%.cpp $(VXDRAM_CONFIG_H) $(VXDRAM_FLAGS)
	@mkdir -p $(@D)
	$(CXX) $(VXDRAM_CXXFLAGS) -MD -MP -MF $(@:.o=.d) -c $< -o $@

-include $(VXDRAM_OBJS:.o=.d)

$(VXDRAM_LIB): $(VXDRAM_OBJS)
	rm -f $@
	$(AR) rcs $@ $^

.PHONY: vxdram clean-vxdram vxdram-force
vxdram: $(VXDRAM_LIB)

clean-vxdram:
	rm -rf $(VXDRAM_OUT)
