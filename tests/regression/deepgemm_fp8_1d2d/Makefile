ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# DeepGEMM's SM90 FP8 1D2D grouped GEMM at the ports' shape: RV64, 32
# threads, a DLMEM device with the A extension, Zacas, mbarriers, the DXA,
# FPAMO and the tensor unit with WGMMA, FP8 and the H100 mode.
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DLMEM_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DLMEM_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_A_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_A_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_ZACAS_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_ZACAS_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_MBAR_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_MBAR_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DXA_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DXA_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_FPAMO_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_FPAMO_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_TCU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_TCU_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_TCU_WGMMA_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_TCU_WGMMA_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_TCU_FP8_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_TCU_FP8_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_TCU_H100_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_TCU_H100_ENABLE)
# Both wgmma operands come from shared memory: keep common.mk's register-A default off.
CONFIGS := $(if $(findstring -DWGMMA_SS,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DWGMMA_SS)

PROJECT := deepgemm_fp8_1d2d

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

# N, K, the group count and the SM count are template constants of the H100
# build (one kernel per shape), reduced here. They are not CONFIGS (the
# simulator's build key); config.stamp records the flags, so a new value
# rebuilds the app. The rows per group stay a run-time option, as on the H100.
DG_SHAPE_N    ?= 384
DG_SHAPE_K    ?= 1024
DG_NUM_GROUPS ?= 2
DG_NUM_SMS    ?= 2
DG_FLAGS := -DDG_SHAPE_N=$(DG_SHAPE_N) -DDG_SHAPE_K=$(DG_SHAPE_K) -DDG_NUM_GROUPS=$(DG_NUM_GROUPS) \
            -DDG_NUM_SMS=$(DG_NUM_SMS)

APP_HDRS := $(SRC_DIR)/common.h

SRCS := $(SRC_DIR)/main.cpp
HDRS := $(APP_HDRS) $(SW_COMMON_DIR)/vx_cuda_fp.h $(SW_COMMON_DIR)/dxa_tmap.h $(SW_COMMON_DIR)/dxa_meta.h \
        $(SW_COMMON_DIR)/tcu_fedp_ref.h \
        $(VORTEX_HOME)/sw/runtime/include/vx_cuda_runtime.h $(VORTEX_HOME)/sw/runtime/include/vortex2.h \
        $(VORTEX_HOME)/sw/runtime/include/vortex.h $(VORTEX_HOME)/sw/runtime/include/dxa.h

VX_SRCS := $(SRC_DIR)/kernel.cpp
VX_HDRS := $(APP_HDRS) $(SRC_DIR)/dg_cutlass.h $(SRC_DIR)/dg_scheduler.h \
           $(SW_COMMON_DIR)/vx_cuda_fp.h $(SW_COMMON_DIR)/dxa_tmap.h $(SW_COMMON_DIR)/dxa_meta.h \
           $(SW_COMMON_DIR)/tensor_cfg.h \
           $(VORTEX_HOME)/sw/kernel/include/vx_cuda.h $(VORTEX_HOME)/sw/kernel/include/vx_tensor.h \
           $(VORTEX_HOME)/sw/kernel/include/vx_mbar.h \
           $(VORTEX_HOME)/sw/kernel/include/vx_dxa.h $(VORTEX_HOME)/sw/kernel/include/vx_dlmem.h \
           $(VORTEX_HOME)/sw/kernel/include/vx_cp_async.h $(VORTEX_HOME)/sw/kernel/include/vx_atomic.h \
           $(VORTEX_HOME)/sw/kernel/include/vx_spawn2.h $(VORTEX_HOME)/sw/kernel/include/vx_intrinsics.h \
           $(VORTEX_HOME)/sw/kernel/include/vx_print.h $(VORTEX_HOME)/sw/kernel/include/vx_barrier.h

VX_CFLAGS += $(DG_FLAGS)
CXXFLAGS += $(DG_FLAGS)

# vx_cuda.h builds the DXA operands in x30/x31 (t5/t6): the wgather that
# broadcasts them also writes the parked lanes' destination register.
VX_CFLAGS += -ffixed-x30 -ffixed-x31 -DVX_CUDA_FIXED_TEMPS

# Barriers, shuffles and votes are inline asm. Unless every call is convergent,
# the compiler may copy one into both arms of a divergent branch, and a split
# warp then arrives twice in one barrier phase. CUDA compiles device code this way.
VX_CFLAGS += -fconvergent-functions

# vortex2 = the KMU kernel runtime, providing _start and __vx_cta_entry.
KERNEL_LIB := vortex2

include ../common.mk

# vxbin.py reads each __vx_kentry_* size as a decimal; readelf prints sizes of
# 100000 bytes or more in hex, and such an entry silently drops out of the
# binary. Fail the build instead.
kernel.vxbin: kentry_size.ok

kentry_size.ok: kernel.elf
	@readelf -s -W $< | awk '$$8 ~ /^__vx_kentry_/ { n++; \
	   if ($$3 !~ /^[0-9]+$$/ || $$3 + 0 >= 100000) { \
	     printf "kernel entry %s is %s bytes: vxbin.py drops entries of 100000 bytes or more\n", $$8, $$3; bad = 1 } \
	   else printf "kernel entry %s is %s bytes\n", $$8, $$3 } \
	   END { if (n == 0) { print "no __vx_kentry_ symbol in kernel.elf"; exit 1 } exit bad }'
	@touch $@

clean: clean-kentry
clean-kentry:
	rm -f kentry_size.ok
