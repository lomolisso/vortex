# Built and run by the standalone sim/simx and rtlsim mains, which never write VX_DCR_KMU_SMEM_CONFIG.
# tests/kernel/common.mk derives ROOT_DIR from a depth of three, so the build root is passed down.
ifeq ($(origin ROOT_DIR),command line)
include $(ROOT_DIR)/config.mk

PROJECT := carveout_default

SRCS := $(VORTEX_HOME)/tests/regression/carveout/standalone/main.cpp

include $(ROOT_DIR)/tests/kernel/common.mk
else
.DEFAULT_GOAL := all
%:
	$(MAKE) ROOT_DIR=$(realpath ../../../..) $@
endif
