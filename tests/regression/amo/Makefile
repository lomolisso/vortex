ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := amo

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

VX_HDRS := $(SRC_DIR)/common.h
HDRS    := $(SRC_DIR)/common.h $(SRC_DIR)/testcases.h

# Per-hart iters — small enough to run quickly under simx, big enough
# to actually exercise contention.
OPTS ?= -n32

KERNEL_LIB := vortex2

include ../common.mk

# The opt-in cases (cas-global, cas-local) run from a kernel of their own, so
# that adding them leaves kernel.vxbin, and every default case, unchanged.
CAS_SRC := $(SRC_DIR)/kernel_cas.cpp

all: kernel_cas.vxbin
run-simx run-rtlsim run-opae run-xrt run-firesim run-aved: kernel_cas.vxbin

kernel_cas.o: $(CAS_SRC) $(VX_HDRS) $(CONFIG_STAMP)
	$(VX_CXX) $(VX_CFLAGS) -c $(CAS_SRC) -o $@

vx_start_cas.o: kernel_cas.o $(VORTEX_KN_PATH)/lib$(KERNEL_LIB).a
	$(VX_CXX) $(VX_CFLAGS) -DNEED_GP -DNEED_TLS -DNEED_INITFINI $(VX_KMU_FLAG) -c $(VX_STARTUP_SRC) -o $@
	$(VX_CXX) $(VX_CFLAGS) $@ kernel_cas.o $(VX_LDFLAGS) -o $@.elf
	$(VX_CXX) $(VX_CFLAGS) $$($(KERNEL_STARTUP) $(VX_DP) $@.elf) $(VX_KMU_FLAG) -c $(VX_STARTUP_SRC) -o $@ && rm -f $@.elf

kernel_cas.elf: vx_start_cas.o kernel_cas.o
	$(VX_CXX) $(VX_CFLAGS) vx_start_cas.o kernel_cas.o $(VX_LDFLAGS) -o $@

kernel_cas.vxbin: kernel_cas.elf
	OBJCOPY=$(VX_CP) $(VORTEX_HOME)/sw/kernel/scripts/vxbin.py $< $@
