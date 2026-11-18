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

// Uniform registers from kernel code. URs are allocated by hand and live only inside one
// asm block: the VX_UASM_* macros give the .insn lines (UR indices are literal numbers
// 0..31, register names of GPRs are strings), the vx_u* functions wrap common sequences.
// UR0-UR7: DXA operands, UR8-UR9: wgmma descriptors, UR10-UR15: warp-op and helper
// scratch (these functions use UR12-UR15). Encodings: sw/common/vx_fe_isa.h.

#ifndef __VX_UNIFORM_H__
#define __VX_UNIFORM_H__

#include <stdint.h>
#include <vx_fe_isa.h>

#define VX_UASM_STR_(x) #x
#define VX_UASM_STR(x)  VX_UASM_STR_(x)

// URd <- GPR rs of the first active lane.
#define VX_UASM_R2UR(urd, rs) \
  ".insn r 0x0b, " VX_UASM_STR(VX_FE_F3_R2UR) ", " VX_UASM_STR(VX_FE_F7_UMOV) ", x" #urd ", " rs ", x0\n\t"
// GPR rd <- URs in every active lane.
#define VX_UASM_UR2R(rd, urs) \
  ".insn r 0x0b, " VX_UASM_STR(VX_FE_F3_UR2R) ", " VX_UASM_STR(VX_FE_F7_UMOV) ", " rd ", x" #urs ", x0\n\t"
// URd <- special register `sel` (vortex::fe::S2UR_CSR).
#define VX_UASM_S2UR(urd, sel) \
  ".insn r 0x0b, " VX_UASM_STR(VX_FE_F3_S2UR) ", " VX_UASM_STR(VX_FE_F7_UMOV) ", x" #urd ", x" #sel ", x0\n\t"
// URd <- URs1 op URs2; op 0 add, 1 sub, 2 and, 3 or, 4 xor, 5 sll, 6 srl, 7 sra.
#define VX_UASM_UALU(op, urd, urs1, urs2) \
  ".insn r 0x0b, " #op ", " VX_UASM_STR(VX_FE_F7_UALU) ", x" #urd ", x" #urs1 ", x" #urs2 "\n\t"
// URd <- URs1 cmp URs2; cmp 0 slt, 1 sltu, 2 seq.
#define VX_UASM_UCMP(cmp, urd, urs1, urs2) \
  ".insn r 0x0b, " #cmp ", " VX_UASM_STR(VX_FE_F7_UCMP) ", x" #urd ", x" #urs1 ", x" #urs2 "\n\t"
// The next instruction of the warp (an ALU or FPU op) reads source slot `slot`
// (0 rs1, 1 rs2, 2 rs3) from URs instead of its register field.
#define VX_UASM_UPFX(slot, urs) \
  ".insn r 0x0b, " #slot ", " VX_UASM_STR(VX_FE_F7_UPFX) ", x0, x" #urs ", x0\n\t"

#define VX_UOP_ADD 0
#define VX_UOP_SUB 1
#define VX_UOP_AND 2
#define VX_UOP_OR  3
#define VX_UOP_XOR 4
#define VX_UOP_SLL 5
#define VX_UOP_SRL 6
#define VX_UOP_SRA 7
#define VX_UCMP_SLT  0
#define VX_UCMP_SLTU 1
#define VX_UCMP_SEQ  2

// The first active lane's value, in every active lane (R2UR + UR2R).
inline uintptr_t vx_ubroadcast(uintptr_t a) {
  uintptr_t r;
  __asm__ volatile (VX_UASM_R2UR(12, "%1") VX_UASM_UR2R("%0", 12) : "=r"(r) : "r"(a));
  return r;
}

// A special register through S2UR (selector: vortex::fe::S2UR_CSR index).
#define vx_us2r(sel) __extension__ ({                                  \
  uintptr_t __r;                                                       \
  __asm__ volatile (VX_UASM_S2UR(12, sel) VX_UASM_UR2R("%0", 12) : "=r"(__r)); \
  __r; })

// A uniform ALU op on the first active lane's a and b.
#define vx_uop(op, a, b) __extension__ ({                              \
  uintptr_t __r;                                                       \
  __asm__ volatile (VX_UASM_R2UR(12, "%1") VX_UASM_R2UR(13, "%2")      \
                    VX_UASM_UALU(op, 14, 12, 13) VX_UASM_UR2R("%0", 14) \
                    : "=r"(__r) : "r"((uintptr_t)(a)), "r"((uintptr_t)(b))); \
  __r; })

#define vx_ucmp(cmp, a, b) __extension__ ({                            \
  uintptr_t __r;                                                       \
  __asm__ volatile (VX_UASM_R2UR(12, "%1") VX_UASM_R2UR(13, "%2")      \
                    VX_UASM_UCMP(cmp, 14, 12, 13) VX_UASM_UR2R("%0", 14) \
                    : "=r"(__r) : "r"((uintptr_t)(a)), "r"((uintptr_t)(b))); \
  __r; })

// x + u with u read from a UR (the first active lane's u) through a prefixed add.
inline uintptr_t vx_uadd_operand(uintptr_t x, uintptr_t u) {
  uintptr_t r;
  __asm__ volatile (VX_UASM_R2UR(15, "%2") VX_UASM_UPFX(1, 15) "add %0, %1, x0\n\t"
                    : "=r"(r) : "r"(x), "r"(u));
  return r;
}

#endif // __VX_UNIFORM_H__
