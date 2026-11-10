ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# The 29 DSMEM collectives of the H100 study, at the ports' shape: a DLMEM
# device with the A extension, Zacas, mbarriers, the DXA and FPAMO (RV64).
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DLMEM_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DLMEM_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_A_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_A_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_ZACAS_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_ZACAS_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_MBAR_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_MBAR_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DXA_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DXA_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_FPAMO_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_FPAMO_ENABLE)

PROJECT := dsmem_collectives

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

# Cluster size N, message elements M and element type T are compile-time, as
# in the H100 build (one binary per N, M, T). They are not CONFIGS: CONFIGS
# is the simulator's build key. config.stamp records both flag sets, so a new
# N, M or T rebuilds the app.
COLL_N ?= 4
COLL_M ?= 1024
COLL_T ?= f32
ifeq ($(COLL_T),f32)
COLL_ELEM := float
else ifeq ($(COLL_T),i32)
COLL_ELEM := int
else
$(error COLL_T must be f32 or i32 (got $(COLL_T)))
endif
COLL_FLAGS := -DCLUSTER_N=$(COLL_N) -DMSG_ELEMS=$(COLL_M) -DELEM_T=$(COLL_ELEM)

COLL_FILES := \
  allgather__bruck__mpich \
  allgather__direct_pull__generic \
  allgather__direct_replicated_push__credit \
  allgather__direct_shift__clusterfusion \
  allgather__gather_bcast_binomial__mpich \
  allgather__recursive_doubling__clusterfusion \
  allgather__ring__nccl \
  allreduce__2d_mesh__generic \
  allreduce__atomic_all_push__generic \
  allreduce__direct_replicated_push__credit \
  allreduce__one_shot_pull__trtllm \
  allreduce__pull_reduce_scatter_push_allgather__hpcops \
  allreduce__rabenseifner__mpich \
  allreduce__recursive_doubling__clusterfusion \
  allreduce__reduce_bcast_binomial__mpich \
  allreduce__ring__nccl \
  allreduce__shift_accumulate_atomic__clusterfusion \
  allreduce__shift_accumulate_bulk__clusterfusion \
  allreduce__two_shot_push__trtllm \
  reducescatter__direct_pull__generic \
  reducescatter__direct_push__generic \
  reducescatter__recursive_halving__mpich \
  reducescatter__ring__nccl \
  alltoall__bruck__mpich \
  alltoall__direct_pull__generic \
  alltoall__direct_push__generic \
  alltoall__pairwise_exchange__fa4 \
  alltoall__pairwise_xor__mpich \
  alltoall__shift__generic
COLL_HDRS := $(SRC_DIR)/common.h $(SRC_DIR)/dsmem_primitives.h $(addprefix $(SRC_DIR)/coll/,$(addsuffix .h,$(COLL_FILES)))

SRCS := $(SRC_DIR)/main.cpp
HDRS := $(COLL_HDRS) $(SW_COMMON_DIR)/dxa_tmap.h $(SW_COMMON_DIR)/dxa_meta.h \
        $(VORTEX_HOME)/sw/runtime/include/vx_cuda_runtime.h $(VORTEX_HOME)/sw/runtime/include/vortex2.h \
        $(VORTEX_HOME)/sw/runtime/include/vortex.h $(VORTEX_HOME)/sw/runtime/include/dxa.h

VX_SRCS := $(SRC_DIR)/kernel.cpp
VX_HDRS := $(COLL_HDRS) $(SW_COMMON_DIR)/vx_cuda_fp.h $(SW_COMMON_DIR)/dxa_tmap.h $(SW_COMMON_DIR)/dxa_meta.h \
           $(VORTEX_HOME)/sw/kernel/include/vx_cuda.h $(VORTEX_HOME)/sw/kernel/include/vx_mbar.h \
           $(VORTEX_HOME)/sw/kernel/include/vx_dxa.h $(VORTEX_HOME)/sw/kernel/include/vx_dlmem.h \
           $(VORTEX_HOME)/sw/kernel/include/vx_cp_async.h $(VORTEX_HOME)/sw/kernel/include/vx_atomic.h \
           $(VORTEX_HOME)/sw/kernel/include/vx_spawn2.h $(VORTEX_HOME)/sw/kernel/include/vx_intrinsics.h \
           $(VORTEX_HOME)/sw/kernel/include/vx_print.h $(VORTEX_HOME)/sw/kernel/include/vx_barrier.h

VX_CFLAGS += $(COLL_FLAGS)
CXXFLAGS += $(COLL_FLAGS)

# vx_cuda.h builds the DXA operands in x30/x31 (t5/t6): the wgather that
# broadcasts them also writes the parked lanes' destination register.
VX_CFLAGS += -ffixed-x30 -ffixed-x31 -DVX_CUDA_FIXED_TEMPS

# vortex2 = the KMU kernel runtime, providing _start and __vx_cta_entry.
KERNEL_LIB := vortex2

include ../common.mk
