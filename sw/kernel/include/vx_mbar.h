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

#ifndef __VX_MBAR_H__
#define __VX_MBAR_H__

// mbarriers: 64-bit synchronization objects in the block's local memory,
// updated by the local memory banks. Layout: pending[19:0] expected[39:20]
// tx[60:40] (signed bytes) phase[63]. A phase completes when pending and tx
// both reach zero; pending then reloads from expected and the phase bit flips.
//
// Usage rules:
// - vx_mbar_init() is a plain store: follow it with __syncthreads() (or a
//   cluster sync) before any thread uses the object.
// - An arrive is a CTA-scope release, like PTX mbarrier.arrive: the warp's
//   earlier local, window (DLMEM) and global accesses, and its tensor-core
//   metadata loads, are performed before it. Stack accesses are not: the
//   stack holds thread-private local memory, as in CUDA, so stack data shared
//   through a pointer needs vx_fence() before the arrive.
// - An arrive does not wait for asynchronous wgmma operations, which read
//   their operands from local memory after they issue: call
//   vx_wgmma_wait_group() (vx_tensor.h) before arriving on a barrier that
//   guards a buffer a wgmma reads, as warpgroup_wait<N>() comes before the
//   arrive on the H100.
// - Every lane of a warp is a separate update of the object, serialized at
//   its bank: one elected lane arriving with a count is the efficient form.
// - The object must be an 8-byte aligned word of the caller's own local memory.

#if !defined(__riscv_xlen) || __riscv_xlen != 64
#error "mbarriers require XLEN 64"
#endif

#if !VX_CFG_EXT_MBAR_ENABLED
#error "vx_mbar.h needs a device built with VX_CFG_EXT_MBAR_ENABLE"
#endif

#include <vx_intrinsics.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// custom-2 R-type: funct3 selects the op, rs1 the object, rs2 the operand.
#define __VX_MBAR_OP(f3, p, v) ({                                       \
    uint64_t __r;                                                       \
    __asm__ volatile (".insn r %1, %2, 0, %0, %3, %4"                   \
                      : "=r"(__r)                                       \
                      : "i"(RISCV_CUSTOM2), "i"(f3), "r"(p), "r"(v)     \
                      : "memory");                                      \
    __r; })

// count must be in [1, 2^20 - 1]: it fills the 20-bit pending and expected fields.
static inline __attribute__((always_inline)) void vx_mbar_init(uint64_t* p, uint32_t count) {
  *(volatile uint64_t*)p = (uint64_t)count | ((uint64_t)count << 20);
}

static inline __attribute__((always_inline)) void vx_mbar_inval(uint64_t* p) {
  *(volatile uint64_t*)p = 0;
}

// The arrive and tx ops return the object's word before the update: its phase
// bit is the token for vx_mbar_test_wait().
static inline __attribute__((always_inline)) uint64_t vx_mbar_arrive(uint64_t* p, uint32_t count) {
  return __VX_MBAR_OP(0, p, (uint64_t)count);
}

// tx += bytes, then one arrival.
static inline __attribute__((always_inline)) uint64_t vx_mbar_arrive_tx(uint64_t* p, uint32_t bytes) {
  return __VX_MBAR_OP(1, p, (uint64_t)bytes);
}

// Not a release, like vx_mbar_complete_tx().
static inline __attribute__((always_inline)) uint64_t vx_mbar_expect_tx(uint64_t* p, uint32_t bytes) {
  return __VX_MBAR_OP(2, p, (uint64_t)bytes);
}

// May come before the matching expect: tx goes negative meanwhile.
static inline __attribute__((always_inline)) uint64_t vx_mbar_complete_tx(uint64_t* p, uint32_t bytes) {
  return __VX_MBAR_OP(3, p, (uint64_t)bytes);
}

// True once the phase `token` was taken in has completed. The object is not
// written. Branching on the result is what orders the caller's later accesses.
static inline __attribute__((always_inline)) bool vx_mbar_test_wait(uint64_t* p, uint64_t token) {
  return __VX_MBAR_OP(4, p, token) != 0;
}

// For a waiter that did not arrive: true once the phase with parity `parity` has completed.
static inline __attribute__((always_inline)) bool vx_mbar_test_wait_parity(uint64_t* p, uint32_t parity) {
  return vx_mbar_test_wait(p, (uint64_t)(parity & 1) << 63);
}

#ifdef __cplusplus
}

namespace vortex {

// A handle on an mbarrier object, after cuda::barrier.
class mbarrier {
public:
  explicit mbarrier(uint64_t* obj) : obj_(obj) {}

  void init(uint32_t count) const { vx_mbar_init(obj_, count); }

  uint64_t arrive(uint32_t count = 1) const { return vx_mbar_arrive(obj_, count); }

  uint64_t arrive_tx(uint32_t bytes) const { return vx_mbar_arrive_tx(obj_, bytes); }

  uint64_t expect_tx(uint32_t bytes) const { return vx_mbar_expect_tx(obj_, bytes); }

  uint64_t complete_tx(uint32_t bytes) const { return vx_mbar_complete_tx(obj_, bytes); }

  bool test_wait(uint64_t token) const { return vx_mbar_test_wait(obj_, token); }

  bool test_wait_parity(uint32_t parity) const { return vx_mbar_test_wait_parity(obj_, parity); }

  void wait(uint64_t token) const {
    while (!test_wait(token)) {}
  }

  void wait_parity(uint32_t parity) const {
    while (!test_wait_parity(parity)) {}
  }

  void arrive_and_wait(uint32_t count = 1) const { wait(arrive(count)); }

  uint64_t* native_handle() const { return obj_; }

private:
  uint64_t* obj_;
};

} // namespace vortex

#endif // __cplusplus

#endif // __VX_MBAR_H__
