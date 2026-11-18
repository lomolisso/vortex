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

// ELECT, MATCH and REDUX as single warp instructions over the active lanes.
// REDUX and ELECT.L write a uniform register; the value is copied to a GPR in
// the same asm block, so the UR is never live across blocks.

#include <stdint.h>
#include <vx_fe_isa.h>
#include <vx_intrinsics.h>

#if !VX_CFG_H100_FE_ENABLED
#error "vx_warp_ops.h needs the H100 front end (VX_CFG_H100_FE_ENABLE)"
#endif

#define __VX_WOP_STR2(x) #x
#define __VX_WOP_STR(x) __VX_WOP_STR2(x)
#define __VX_WOP_UR "x" __VX_WOP_STR(VX_FE_UR_WOP)

#ifdef __cplusplus
extern "C" {
#endif

// 1 in the lowest active lane whose own bit of `mask` is set, 0 in the others.
inline int vx_elect_p(uint32_t mask) {
  int ret;
  __asm__ volatile (".insn r %1, %2, %3, %0, %4, x0"
                    : "=r"(ret)
                    : "i"(RISCV_CUSTOM0), "i"(VX_FE_ELECT_P), "i"(VX_FE_F7_ELECT), "r"(mask));
  return ret;
}

// The leader's lane index (as vx_elect_p picks it); all ones when no member lane is active.
inline unsigned long vx_elect_lane(uint32_t mask) {
  unsigned long ret;
  __asm__ volatile (".insn r %1, %2, %3, " __VX_WOP_UR ", %4, x0\n\t"
                    ".insn r %1, %5, %6, %0, " __VX_WOP_UR ", x0"
                    : "=r"(ret)
                    : "i"(RISCV_CUSTOM0), "i"(VX_FE_ELECT_L), "i"(VX_FE_F7_ELECT), "r"(mask),
                      "i"(VX_FE_F3_UR2R), "i"(VX_FE_F7_UMOV));
  return ret;
}

// Mask of the active lanes holding this lane's value.
inline uint32_t vx_match_any_b32(uint32_t value) {
  uint32_t ret;
  __asm__ volatile (".insn r %1, %2, %3, %0, %4, x0"
                    : "=r"(ret)
                    : "i"(RISCV_CUSTOM0), "i"(VX_FE_MATCH_ANY_B32), "i"(VX_FE_F7_MATCH), "r"(value));
  return ret;
}

// The active mask when every active lane holds the same value, else 0.
inline uint32_t vx_match_all_b32(uint32_t value) {
  uint32_t ret;
  __asm__ volatile (".insn r %1, %2, %3, %0, %4, x0"
                    : "=r"(ret)
                    : "i"(RISCV_CUSTOM0), "i"(VX_FE_MATCH_ALL_B32), "i"(VX_FE_F7_MATCH), "r"(value));
  return ret;
}

#if (VX_CFG_XLEN == 64)
#define __VX_WOP_MATCH64(name, sub)                                                       \
  inline uint32_t name(uint64_t value) {                                                  \
    uint32_t ret;                                                                         \
    __asm__ volatile (".insn r %1, %2, %3, %0, %4, x0"                                    \
                      : "=r"(ret)                                                         \
                      : "i"(RISCV_CUSTOM0), "i"(sub), "i"(VX_FE_F7_MATCH), "r"(value));   \
    return ret;                                                                           \
  }
#else
#define __VX_WOP_MATCH64(name, sub)                                                       \
  inline uint32_t name(uint64_t value) {                                                  \
    uint32_t ret;                                                                         \
    __asm__ volatile (".insn r %1, %2, %3, %0, %4, %5"                                    \
                      : "=r"(ret)                                                         \
                      : "i"(RISCV_CUSTOM0), "i"(sub), "i"(VX_FE_F7_MATCH),                \
                        "r"((uint32_t)value), "r"((uint32_t)(value >> 32)));              \
    return ret;                                                                           \
  }
#endif
__VX_WOP_MATCH64(vx_match_any_b64, VX_FE_MATCH_ANY_B64)
__VX_WOP_MATCH64(vx_match_all_b64, VX_FE_MATCH_ALL_B64)
#undef __VX_WOP_MATCH64

// The 32-bit reduction over the active lanes, the same value in every lane.
#define __VX_WOP_REDUX(name, T, sub)                                                      \
  inline T name(T value) {                                                                \
    long ret;                                                                             \
    __asm__ volatile (".insn r %1, %2, %3, " __VX_WOP_UR ", %4, x0\n\t"                   \
                      ".insn r %1, %5, %6, %0, " __VX_WOP_UR ", x0"                       \
                      : "=r"(ret)                                                         \
                      : "i"(RISCV_CUSTOM0), "i"(sub), "i"(VX_FE_F7_REDUX), "r"(value),    \
                        "i"(VX_FE_F3_UR2R), "i"(VX_FE_F7_UMOV));                          \
    return (T)ret;                                                                        \
  }
__VX_WOP_REDUX(vx_redux_add,   uint32_t, VX_FE_REDUX_ADD)
__VX_WOP_REDUX(vx_redux_min_s, int32_t,  VX_FE_REDUX_MIN_S)
__VX_WOP_REDUX(vx_redux_max_s, int32_t,  VX_FE_REDUX_MAX_S)
__VX_WOP_REDUX(vx_redux_min_u, uint32_t, VX_FE_REDUX_MIN_U)
__VX_WOP_REDUX(vx_redux_max_u, uint32_t, VX_FE_REDUX_MAX_U)
__VX_WOP_REDUX(vx_redux_and,   uint32_t, VX_FE_REDUX_AND)
__VX_WOP_REDUX(vx_redux_or,    uint32_t, VX_FE_REDUX_OR)
__VX_WOP_REDUX(vx_redux_xor,   uint32_t, VX_FE_REDUX_XOR)
#undef __VX_WOP_REDUX

#ifdef __cplusplus
}
#endif
