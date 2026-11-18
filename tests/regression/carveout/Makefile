ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := carveout

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

VX_HDRS := $(SRC_DIR)/common.h $(SRC_DIR)/chase.h $(VORTEX_HOME)/sw/common/smem_carveout.h
HDRS    := $(SRC_DIR)/common.h $(VORTEX_HOME)/sw/common/smem_carveout.h

OPTS ?=

KERNEL_LIB := vortex2

include ../common.mk

# The legacy vx_start path programs no KMU kernel entry, so -m 4 runs the chase from a legacy image
# (libvortex, main()) of its own.
LEGACY_SRC := $(SRC_DIR)/legacy.cpp
LEGACY_LDFLAGS = $(subst lib$(KERNEL_LIB).a,libvortex.a,$(VX_LDFLAGS))

all: legacy.vxbin
run-simx run-rtlsim run-opae run-xrt run-firesim run-aved: legacy.vxbin

$(VORTEX_KN_PATH)/libvortex.a:
	$(MAKE) -C $(VORTEX_KN_PATH)

legacy.o: $(LEGACY_SRC) $(VX_HDRS) $(CONFIG_STAMP)
	$(VX_CXX) $(VX_CFLAGS) -c $(LEGACY_SRC) -o $@

vx_start_legacy.o: legacy.o $(VORTEX_KN_PATH)/libvortex.a
	$(VX_CXX) $(VX_CFLAGS) -DNEED_GP -DNEED_TLS -DNEED_INITFINI -c $(VX_STARTUP_SRC) -o $@
	$(VX_CXX) $(VX_CFLAGS) $@ legacy.o $(LEGACY_LDFLAGS) -o $@.elf
	$(VX_CXX) $(VX_CFLAGS) $$($(KERNEL_STARTUP) $(VX_DP) $@.elf) -c $(VX_STARTUP_SRC) -o $@ && rm -f $@.elf

legacy.elf: vx_start_legacy.o legacy.o
	$(VX_CXX) $(VX_CFLAGS) vx_start_legacy.o legacy.o $(LEGACY_LDFLAGS) -o $@

legacy.vxbin: legacy.elf
	OBJCOPY=$(VX_CP) $(VORTEX_HOME)/sw/kernel/scripts/vxbin.py $< $@
