# HardFloat sources of the float atomic unit (hw/rtl/cache/VX_amo_fpu.sv), for
# the Verilator flows (rtlsim, xrtsim, hw/unittest). The synthesis flows take
# the directories from hw/syn/extensions.mk instead. Include after CONFIGS (and
# XCONFIGS, where the flow resolves one) are set; THIRD_PARTY_DIR must be set.
#
# HardFloat files hold several modules each, so Verilator cannot find them by
# module name on a -y path: they are passed as explicit sources.

FPAMO_ENABLED := $(if $(filter -DVX_CFG_EXT_FPAMO_ENABLE -DVX_CFG_EXT_FPAMO_ENABLE=1 -DVX_CFG_EXT_FPAMO_ENABLED=1,$(CONFIGS) $(XCONFIGS)),1,)

HARDFLOAT_DIR     := $(THIRD_PARTY_DIR)/hardfloat/source
HARDFLOAT_SRCS    := $(addprefix $(HARDFLOAT_DIR)/,addRecFN.v HardFloat_primitives.v HardFloat_rawFN.v isSigNaNRecFN.v recFNToFN.v recFNToRecFN.v)
HARDFLOAT_INCLUDE := -I$(HARDFLOAT_DIR)/RISCV -I$(HARDFLOAT_DIR)
