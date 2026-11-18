// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

// Encodings of the H100 front-end instructions (uniform registers, UR operands, warp ops,
// UR-form DXA ops), shared by the kernel headers, SimX, the RTL benches and the host twins.
// All are R-type on the EXT1 (custom-0) opcode. VX_gpu_pkg's fe_* functions decode the same.

// Plain integers for kernel asm ("i" constraints, register names built by stringification).
#define VX_FE_F7_UMOV        6
#define VX_FE_F3_R2UR        0
#define VX_FE_F3_UR2R        1
#define VX_FE_F3_S2UR        2
#define VX_FE_F7_UPFX        5
#define VX_FE_F7_UALU        8
#define VX_FE_F7_UCMP        16
#define VX_FE_F7_REDUX       24
#define VX_FE_REDUX_ADD      0
#define VX_FE_REDUX_MIN_S    1
#define VX_FE_REDUX_MAX_S    2
#define VX_FE_REDUX_MIN_U    3
#define VX_FE_REDUX_MAX_U    4
#define VX_FE_REDUX_AND      5
#define VX_FE_REDUX_OR       6
#define VX_FE_REDUX_XOR      7
#define VX_FE_F7_ELECT       26
#define VX_FE_ELECT_P        0
#define VX_FE_ELECT_L        1
#define VX_FE_F7_MATCH       27
#define VX_FE_MATCH_ANY_B32  0
#define VX_FE_MATCH_ANY_B64  1
#define VX_FE_MATCH_ALL_B32  2
#define VX_FE_MATCH_ALL_B64  3
#define VX_FE_F7_DXA_UR      28
#define VX_FE_UR_DXA         0
#define VX_FE_UR_WG_DESC     8
#define VX_FE_UR_WOP         10

#ifdef __cplusplus

#include <cstdint>

namespace vortex {
namespace fe {

constexpr uint32_t OPCODE_EXT1 = 0x0b;

// funct7 values.
constexpr uint32_t F7_UPFX     = VX_FE_F7_UPFX; // funct3 = replaced source slot
constexpr uint32_t F7_UMOVE    = VX_FE_F7_UMOV; // funct3 = R2UR / UR2R / S2UR
constexpr uint32_t F7_UALU     = VX_FE_F7_UALU; // 0x08-0x0F: funct7[2:0] = {URd[5], URs1[5], URs2[5]}
constexpr uint32_t F7_UCMP     = VX_FE_F7_UCMP; // 0x10-0x17: as UALU
constexpr uint32_t F7_REDUX    = VX_FE_F7_REDUX; // 0x18-0x19: funct7[0] = URd[5]
constexpr uint32_t F7_ELECT    = VX_FE_F7_ELECT;
constexpr uint32_t F7_MATCH    = VX_FE_F7_MATCH;
constexpr uint32_t F7_DXA_UR   = VX_FE_F7_DXA_UR;

// F7_UPFX funct3: the source slot the UR replaces.
constexpr uint32_t UPFX_RS1 = 0, UPFX_RS2 = 1, UPFX_RS3 = 2;

// F7_UMOVE funct3.
constexpr uint32_t UMOVE_R2UR = 0; // URd <- GPR rs1 of the first active lane
constexpr uint32_t UMOVE_UR2R = 1; // GPR rd (every active lane) <- URs
constexpr uint32_t UMOVE_S2UR = 2; // URd <- special register (rs1 = selector)

// F7_UALU funct3.
constexpr uint32_t UALU_ADD = 0, UALU_SUB = 1, UALU_AND = 2, UALU_OR = 3,
                   UALU_XOR = 4, UALU_SLL = 5, UALU_SRL = 6, UALU_SRA = 7;
// F7_UCMP funct3 (3-7 reserved).
constexpr uint32_t UCMP_SLT = 0, UCMP_SLTU = 1, UCMP_SEQ = 2;

// F7_REDUX funct3; 32-bit operations on the low word of every active lane.
constexpr uint32_t REDUX_ADD = 0, REDUX_MIN = 1, REDUX_MAX = 2, REDUX_UMIN = 3,
                   REDUX_UMAX = 4, REDUX_AND = 5, REDUX_OR = 6, REDUX_XOR = 7;

// F7_ELECT funct3.
constexpr uint32_t ELECT_P = 0; // GPR rd = (lane == leader) in every active lane
constexpr uint32_t ELECT_L = 1; // URd = leader lane index

// F7_MATCH funct3.
constexpr uint32_t MATCH_ANY_B32 = 0, MATCH_ANY_B64 = 1, MATCH_ALL_B32 = 2, MATCH_ALL_B64 = 3;

// F7_DXA_UR funct3 0-5: the DXA 0x03 form with the same funct3, operands from two UR quads:
// lanes 0..3 of rs1 = UR[b..b+3], of rs2 = UR[b+4..b+7], b = {rs2[0], rs1}, b % 8 == 0.
// funct3 7: tensormap prefetch/invalidate, rs1 operand = UR[rs1] (UR0..UR31), rs2 = the
// 0x03 funct3 7 op selector. funct3 6 is illegal (bulk-group ops stay on 0x03).
constexpr uint32_t DXA_UR_TMAP = 7;
constexpr uint32_t DXA_UR_RESERVED = 6;
constexpr uint32_t DXA_UR_QUAD_ALIGN = 8;

constexpr uint32_t UREG_BITS = 6;

// UR use in the kernel headers: allocated by hand inside one asm block.
constexpr uint32_t UR_DXA_BASE   = VX_FE_UR_DXA;  // UR0-UR7: DXA operand block
constexpr uint32_t UR_WG_DESC    = VX_FE_UR_WG_DESC;  // UR8-UR9: wgmma descriptors A, B
constexpr uint32_t UR_SCRATCH    = VX_FE_UR_WOP; // UR10-UR15: warp-op and helper scratch
constexpr uint32_t UR_RESERVED   = 16;

// S2UR selector (rs1 field) -> CSR address. VX_gpu_pkg's fe_s2ur_csr() holds the same table.
constexpr uint32_t S2UR_NUM_SELECTORS = 27;
constexpr uint16_t S2UR_CSR[S2UR_NUM_SELECTORS] = {
  0xB00, // 0  clock low   (mcycle)
  0xB80, // 1  clock high  (mcycle_h)
  0xCC2, // 2  smid        (core id)
  0xFC2, // 3  nsmid       (cores)
  0xCC1, // 4  warp id
  0xCD0, // 5  cta id (local)
  0xCD1, // 6  warp rank in the cta
  0xCD2, // 7  warps in the cta
  0xCD6, // 8  block id x
  0xCD7, // 9  block id y
  0xCD8, // 10 block id z
  0xCD9, // 11 block dim x
  0xCDA, // 12 block dim y
  0xCDB, // 13 block dim z
  0xCDC, // 14 grid dim x
  0xCDD, // 15 grid dim y
  0xCDE, // 16 grid dim z
  0xCDF, // 17 cta shared-memory address
  0xCE0, // 18 cluster size
  0xCE4, // 19 block rank in the cluster
  0xCE5, // 20 cluster dim x
  0xCE6, // 21 cluster dim y
  0xCE7, // 22 cluster dim z
  0xCE8, // 23 cta shared-memory size
  0xFC0, // 24 threads per warp
  0xFC1, // 25 warps per core
  0xFC3, // 26 local memory base
};

constexpr uint32_t ur_lo(uint32_t ur) { return ur & 0x1f; }
constexpr uint32_t ur_hi(uint32_t ur) { return (ur >> 5) & 1; }

constexpr uint32_t rtype(uint32_t funct7, uint32_t rs2, uint32_t rs1, uint32_t funct3, uint32_t rd) {
  return (funct7 << 25) | ((rs2 & 0x1f) << 20) | ((rs1 & 0x1f) << 15) | ((funct3 & 7) << 12)
       | ((rd & 0x1f) << 7) | OPCODE_EXT1;
}

constexpr uint32_t enc_upfx(uint32_t slot, uint32_t ur) {
  return rtype(F7_UPFX, ur_hi(ur), ur_lo(ur), slot, 0);
}
constexpr uint32_t enc_r2ur(uint32_t urd, uint32_t rs1) {
  return rtype(F7_UMOVE, ur_hi(urd), rs1, UMOVE_R2UR, ur_lo(urd));
}
constexpr uint32_t enc_ur2r(uint32_t rd, uint32_t urs) {
  return rtype(F7_UMOVE, ur_hi(urs), ur_lo(urs), UMOVE_UR2R, rd);
}
constexpr uint32_t enc_s2ur(uint32_t urd, uint32_t sel) {
  return rtype(F7_UMOVE, ur_hi(urd), sel, UMOVE_S2UR, ur_lo(urd));
}
constexpr uint32_t enc_ualu(uint32_t base, uint32_t op, uint32_t urd, uint32_t urs1, uint32_t urs2) {
  return rtype(base | (ur_hi(urd) << 2) | (ur_hi(urs1) << 1) | ur_hi(urs2),
               ur_lo(urs2), ur_lo(urs1), op, ur_lo(urd));
}
constexpr uint32_t enc_redux(uint32_t op, uint32_t urd, uint32_t rs1) {
  return rtype(F7_REDUX | ur_hi(urd), 0, rs1, op, ur_lo(urd));
}
constexpr uint32_t enc_elect_p(uint32_t rd, uint32_t rs1) {
  return rtype(F7_ELECT, 0, rs1, ELECT_P, rd);
}
constexpr uint32_t enc_elect_l(uint32_t urd, uint32_t rs1) {
  return rtype(F7_ELECT, ur_hi(urd), rs1, ELECT_L, ur_lo(urd));
}
constexpr uint32_t enc_match(uint32_t op, uint32_t rd, uint32_t rs1, uint32_t rs2) {
  return rtype(F7_MATCH, rs2, rs1, op, rd);
}
constexpr uint32_t enc_dxa_ur(uint32_t form, uint32_t ur_base) {
  return rtype(F7_DXA_UR, ur_hi(ur_base), ur_lo(ur_base), form, 0);
}
constexpr uint32_t enc_dxa_ur_tmap(uint32_t ur, uint32_t op) {
  return rtype(F7_DXA_UR, op, ur, DXA_UR_TMAP, 0);
}

} // namespace fe
} // namespace vortex

#endif // __cplusplus
