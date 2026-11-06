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

#ifndef __VX_CP_ASYNC_H__
#define __VX_CP_ASYNC_H__

// cp.async: asynchronous copies from global memory into the block's own local
// memory, like PTX cp.async (sm_80, LDGSTS on the H100).
//
// - A copy of 4, 8 or 16 bytes per lane retires at issue; its global read and
//   local write complete later. `dst` must be in the block's own local memory
//   and `src` in global memory, both aligned to the copy size. Bytes
//   [0, src_size) of the source are copied, bytes [src_size, size) of the
//   destination are written 0, and nothing past it is written. A lane with
//   src_size 0 does not read its source.
// - No fence or barrier waits for a copy. A warp waits for its own copies
//   with commit_group / wait_group (per warp, like the H100's per-thread
//   groups: other warps' copies need a block barrier after the wait), or has
//   them complete on an mbarrier with vx_cp_async_mbar_arrive{,_noinc}().
// - .ca and .cg are the same instruction: the L1 is not bypassed.
// - A warp has at most 8 copies (a 16-byte copy counts as 2), 8 committed
//   groups and one pending mbarrier arrive in flight; issue stalls beyond that.

#if !defined(__riscv_xlen) || __riscv_xlen != 64
#error "cp.async requires XLEN 64"
#endif

#if !VX_CFG_EXT_MBAR_ENABLED
#error "vx_cp_async.h needs a device built with VX_CFG_EXT_MBAR_ENABLE"
#endif

#include <vx_intrinsics.h>
#include <vx_mbar.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// custom-2 R-type with funct7 3: funct3 2/3/4 copy 4/8/16 bytes, with rs1 the
// source, rs2 the source size and rd the destination, read as a source.
#define __VX_CP_ASYNC(f3, dst, src, ss)                                  \
    __asm__ volatile (".insn r %0, %1, 3, %2, %3, %4"                    \
                      :: "i"(RISCV_CUSTOM2), "i"(f3), "r"(dst),          \
                         "r"(src), "r"((uint64_t)(ss))                   \
                      : "memory")

static inline __attribute__((always_inline)) void vx_cp_async_4(void* dst, const void* src, uint32_t src_size) {
  __VX_CP_ASYNC(2, dst, src, src_size);
}

static inline __attribute__((always_inline)) void vx_cp_async_8(void* dst, const void* src, uint32_t src_size) {
  __VX_CP_ASYNC(3, dst, src, src_size);
}

// One instruction, performed as two 8-byte halves.
static inline __attribute__((always_inline)) void vx_cp_async_16(void* dst, const void* src, uint32_t src_size) {
  __VX_CP_ASYNC(4, dst, src, src_size);
}

static inline __attribute__((always_inline)) void vx_cp_async_ca_4(void* dst, const void* src, uint32_t src_size) {
  vx_cp_async_4(dst, src, src_size);
}

static inline __attribute__((always_inline)) void vx_cp_async_ca_8(void* dst, const void* src, uint32_t src_size) {
  vx_cp_async_8(dst, src, src_size);
}

static inline __attribute__((always_inline)) void vx_cp_async_ca_16(void* dst, const void* src, uint32_t src_size) {
  vx_cp_async_16(dst, src, src_size);
}

static inline __attribute__((always_inline)) void vx_cp_async_cg_16(void* dst, const void* src, uint32_t src_size) {
  vx_cp_async_16(dst, src, src_size);
}

#undef __VX_CP_ASYNC

// PTX cp.async.commit_group: the warp's copies since the last commit form a group.
static inline __attribute__((always_inline)) void vx_cp_async_commit_group(void) {
  __asm__ volatile (".insn r %0, 0, 3, x0, x0, x0" :: "i"(RISCV_CUSTOM2) : "memory");
}

// PTX cp.async.wait_group N (a constant 0..31, in the rs2 field): stalls the
// warp until at most N of its committed groups have copies outstanding.
#define __VX_CP_ASYNC_WAIT_GROUP(N)                                      \
    __asm__ volatile (".insn r %0, 1, 3, x0, x0, x%1"                    \
                      :: "i"(RISCV_CUSTOM2), "i"(N) : "memory")

// PTX cp.async.wait_all.
static inline __attribute__((always_inline)) void vx_cp_async_wait_all(void) {
  vx_cp_async_commit_group();
  __VX_CP_ASYNC_WAIT_GROUP(0);
}

// PTX cp.async.mbarrier.arrive.noinc: one arrival on `mbar`, the block's own
// object, once every copy the warp issued before it has completed.
static inline __attribute__((always_inline)) void vx_cp_async_mbar_arrive_noinc(uint64_t* mbar) {
  __asm__ volatile (".insn r %0, 6, 3, x0, %1, x0" :: "i"(RISCV_CUSTOM2), "r"(mbar) : "memory");
}

// PTX cp.async.mbarrier.arrive: no arrival. As on the H100, tx += 1 now and
// tx -= 1 once the warp's earlier copies have completed, so the phase cannot
// complete before them.
static inline __attribute__((always_inline)) void vx_cp_async_mbar_arrive(uint64_t* mbar) {
  vx_mbar_expect_tx(mbar, 1);
  __asm__ volatile (".insn r %0, 5, 3, x0, %1, x0" :: "i"(RISCV_CUSTOM2), "r"(mbar) : "memory");
}

#ifdef __cplusplus
}

// The full copy, and PTX's ignore-src predicate: src_size 0 when true.
static inline __attribute__((always_inline)) void vx_cp_async_4(void* dst, const void* src) {
  vx_cp_async_4(dst, src, 4);
}

static inline __attribute__((always_inline)) void vx_cp_async_8(void* dst, const void* src) {
  vx_cp_async_8(dst, src, 8);
}

static inline __attribute__((always_inline)) void vx_cp_async_16(void* dst, const void* src) {
  vx_cp_async_16(dst, src, 16);
}

static inline __attribute__((always_inline)) void vx_cp_async_4(void* dst, const void* src, uint32_t src_size,
                                                                bool ignore_src) {
  vx_cp_async_4(dst, src, ignore_src ? 0 : src_size);
}

static inline __attribute__((always_inline)) void vx_cp_async_8(void* dst, const void* src, uint32_t src_size,
                                                                bool ignore_src) {
  vx_cp_async_8(dst, src, ignore_src ? 0 : src_size);
}

static inline __attribute__((always_inline)) void vx_cp_async_16(void* dst, const void* src, uint32_t src_size,
                                                                 bool ignore_src) {
  vx_cp_async_16(dst, src, ignore_src ? 0 : src_size);
}

static inline __attribute__((always_inline)) void vx_cp_async_ca_4(void* dst, const void* src) {
  vx_cp_async_4(dst, src, 4);
}

static inline __attribute__((always_inline)) void vx_cp_async_ca_8(void* dst, const void* src) {
  vx_cp_async_8(dst, src, 8);
}

static inline __attribute__((always_inline)) void vx_cp_async_ca_16(void* dst, const void* src) {
  vx_cp_async_16(dst, src, 16);
}

static inline __attribute__((always_inline)) void vx_cp_async_cg_16(void* dst, const void* src) {
  vx_cp_async_16(dst, src, 16);
}

static inline __attribute__((always_inline)) void vx_cp_async_ca_4(void* dst, const void* src, uint32_t src_size,
                                                                   bool ignore_src) {
  vx_cp_async_4(dst, src, ignore_src ? 0 : src_size);
}

static inline __attribute__((always_inline)) void vx_cp_async_ca_8(void* dst, const void* src, uint32_t src_size,
                                                                   bool ignore_src) {
  vx_cp_async_8(dst, src, ignore_src ? 0 : src_size);
}

static inline __attribute__((always_inline)) void vx_cp_async_ca_16(void* dst, const void* src, uint32_t src_size,
                                                                    bool ignore_src) {
  vx_cp_async_16(dst, src, ignore_src ? 0 : src_size);
}

static inline __attribute__((always_inline)) void vx_cp_async_cg_16(void* dst, const void* src, uint32_t src_size,
                                                                    bool ignore_src) {
  vx_cp_async_16(dst, src, ignore_src ? 0 : src_size);
}

template <int N>
static inline __attribute__((always_inline)) void vx_cp_async_wait_group() {
  static_assert(N >= 0 && N <= 31, "cp.async wait_group takes 0 to 31");
  __VX_CP_ASYNC_WAIT_GROUP(N);
}

namespace vortex {

// cuda_pipeline.h's primitives: __pipeline_memcpy_async copies `size` (4, 8
// or 16) bytes, the last `zfill` of them as zeros.
static inline __attribute__((always_inline)) void pipeline_memcpy_async(void* dst, const void* src, size_t size,
                                                                        size_t zfill = 0) {
  const uint32_t ss = (uint32_t)(size - zfill);
  switch (size) {
  case 4:  vx_cp_async_4(dst, src, ss); break;
  case 8:  vx_cp_async_8(dst, src, ss); break;
  case 16: vx_cp_async_16(dst, src, ss); break;
  default: __builtin_trap();
  }
}

static inline __attribute__((always_inline)) void pipeline_commit() {
  vx_cp_async_commit_group();
}

template <int N>
static inline __attribute__((always_inline)) void pipeline_wait_prior() {
  vx_cp_async_wait_group<N>();
}

} // namespace vortex

#endif // __cplusplus

#undef __VX_CP_ASYNC_WAIT_GROUP

#endif // __VX_CP_ASYNC_H__
