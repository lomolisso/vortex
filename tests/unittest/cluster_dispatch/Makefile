ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := cluster_dispatch

SIMX_DIR := $(ROOT_DIR)/sim/simx

CONFIGS := $(if $(findstring -DVX_CFG_EXT_DLMEM_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DLMEM_ENABLE)

.PHONY: all run clean

all:
	$(MAKE) -C $(SIMX_DIR) $(SIMX_DIR)/cluster_dispatch_test CONFIGS="$(CONFIGS)"

run:
	$(MAKE) -C $(SIMX_DIR) test-cluster-dispatch CONFIGS="$(CONFIGS)" CLUSTER_DISPATCH_OPTS="$(OPTS)"

clean:
	rm -f $(SIMX_DIR)/cluster_dispatch_test $(SIMX_DIR)/obj/tests/cluster_dispatch_test.o $(SIMX_DIR)/obj/tests/cluster_dispatch_test.d
