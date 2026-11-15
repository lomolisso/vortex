ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := dram_map

SRC_DIR := $(VORTEX_HOME)/tests/unittest/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

CXXFLAGS += -DVECTORS_DIR='"$(SRC_DIR)"'

include ../common.mk

# The header is also included from C (kernels): keep it C99-clean.
$(PROJECT): | c99.ok

c99.ok: $(SW_COMMON_DIR)/vx_dram_map.h
	echo '#include <vx_dram_map.h>' | $(CC) -std=c99 -Wall -Wextra -pedantic -Werror -I$(SW_COMMON_DIR) -fsyntax-only -x c -
	touch $@

clean: clean-c99

clean-c99:
	rm -f c99.ok
