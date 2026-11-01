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
//   cluster sync) before any thread uses the object, and with
//   vx_cluster_sync() before a peer block does.
// - An arrive is a CTA-scope release, like PTX mbarrier.arrive: the warp's
//   earlier local, window (DLMEM) and global accesses, and its tensor-core
//   metadata loads, are performed before it. Stack accesses are not: the
//   stack holds thread-private local memory, as in CUDA, so stack data shared
//   through a pointer needs vx_fence() before the arrive.
// - That release covers the block: the L1s are not coherent across cores, so
//   a waiter on another core sees the arriver's global data only through the
//   _cluster forms (vx_mbar_arrive_release_cluster(),
//   vx_mbar_try_wait_acquire_cluster()), whose fences drain the arriver's
//   stores to the shared cache and drop the waiter's stale L1 lines, like
//   vx_cluster_arrive() and vx_cluster_wait(). Local and window data need no
//   more than the plain forms.
// - A relaxed arrive (vx_mbar_arrive_relaxed(), vx_mbar_arrive_tx_relaxed())
//   orders nothing: it may complete the phase before the warp's earlier
//   accesses are performed.
// - An arrive does not wait for asynchronous wgmma operations, which read
//   their operands from local memory after they issue: call
//   vx_wgmma_wait_group() (vx_tensor.h) before arriving on a barrier that
//   guards a buffer a wgmma reads, as warpgroup_wait<N>() comes before the
//   arrive on the H100.
// - Every lane of a warp is a separate update of the object, serialized at
//   its bank: one elected lane arriving with a count is the efficient form.
// - The object is an 8-byte aligned word of local memory. The arrive and tx
//   ops may also update a peer block's object, through the address
//   vx_cluster_map_shared_rank() (vx_dlmem.h) gives, as PTX allows on
//   shared::cluster; the word such an op returns is the peer's old word, which
//   is no token. Waits are allowed only on the caller's own object, as in
//   PTX: one on a peer's object on another core stops the simulation.
// - vx_mbar_try_wait() returns true once the phase has completed, like PTX
//   mbarrier.try_wait: while it is incomplete the warp suspends until it
//   completes or a time limit of VX_CFG_MBAR_SUSPEND_CYCLES cycles passes, and
//   false means the limit expired first, so a caller that must wait loops on
//   it. A warp waits on one object at a time: when its lanes wait on different
//   objects, it is woken when the object read by its last waiting lane
//   completes, or at the time limit, so the other lanes may return false
//   early and poll again.
// - vx_mbar_test_wait() never suspends: it reads the phase and returns.

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

// custom-2 R-type: funct3 selects the op, funct7 1 makes an arrive relaxed,
// rs1 is the object, rs2 the operand.
#define __VX_MBAR_OP(f3, f7, p, v) ({                                   \
    uint64_t __r;                                                       \
    __asm__ volatile (".insn r %1, %2, %3, %0, %4, %5"                  \
                      : "=r"(__r)                                       \
                      : "i"(RISCV_CUSTOM2), "i"(f3), "i"(f7), "r"(p), "r"(v) \
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
  return __VX_MBAR_OP(0, 0, p, (uint64_t)count);
}

// tx += bytes, then one arrival.
static inline __attribute__((always_inline)) uint64_t vx_mbar_arrive_tx(uint64_t* p, uint32_t bytes) {
  return __VX_MBAR_OP(1, 0, p, (uint64_t)bytes);
}

// PTX mbarrier.arrive.relaxed: not held for the warp's earlier accesses.
static inline __attribute__((always_inline)) uint64_t vx_mbar_arrive_relaxed(uint64_t* p, uint32_t count) {
  return __VX_MBAR_OP(0, 1, p, (uint64_t)count);
}

static inline __attribute__((always_inline)) uint64_t vx_mbar_arrive_tx_relaxed(uint64_t* p, uint32_t bytes) {
  return __VX_MBAR_OP(1, 1, p, (uint64_t)bytes);
}

// PTX mbarrier.arrive.release.cluster: the warp's earlier global stores are
// also visible to waiters on other cores once the phase completes.
static inline __attribute__((always_inline)) uint64_t vx_mbar_arrive_release_cluster(uint64_t* p, uint32_t count) {
  vx_fence();
  return vx_mbar_arrive(p, count);
}

// Not a release, like vx_mbar_complete_tx().
static inline __attribute__((always_inline)) uint64_t vx_mbar_expect_tx(uint64_t* p, uint32_t bytes) {
  return __VX_MBAR_OP(2, 0, p, (uint64_t)bytes);
}

// May come before the matching expect: tx goes negative meanwhile.
static inline __attribute__((always_inline)) uint64_t vx_mbar_complete_tx(uint64_t* p, uint32_t bytes) {
  return __VX_MBAR_OP(3, 0, p, (uint64_t)bytes);
}

// True once the phase `token` was taken in has completed. The object is not
// written. Branching on the result is what orders the caller's later accesses.
static inline __attribute__((always_inline)) bool vx_mbar_test_wait(uint64_t* p, uint64_t token) {
  return __VX_MBAR_OP(4, 0, p, token) != 0;
}

// For a waiter that did not arrive: true once the phase with parity `parity` has completed.
static inline __attribute__((always_inline)) bool vx_mbar_test_wait_parity(uint64_t* p, uint32_t parity) {
  return vx_mbar_test_wait(p, (uint64_t)(parity & 1) << 63);
}

// vx_mbar_test_wait() that suspends the warp while the phase is incomplete;
// false once the time limit expires first.
static inline __attribute__((always_inline)) bool vx_mbar_try_wait(uint64_t* p, uint64_t token) {
  // The op answers with the phase it read before suspending the warp: after a
  // wake-up, one test_wait tells a completed phase from an expired time limit.
  return __VX_MBAR_OP(5, 0, p, token) != 0 || vx_mbar_test_wait(p, token);
}

static inline __attribute__((always_inline)) bool vx_mbar_try_wait_parity(uint64_t* p, uint32_t parity) {
  return vx_mbar_try_wait(p, (uint64_t)(parity & 1) << 63);
}

// PTX mbarrier.try_wait.acquire.cluster: once true, the caller's later global
// loads see what the arrivers released with vx_mbar_arrive_release_cluster().
static inline __attribute__((always_inline)) bool vx_mbar_try_wait_acquire_cluster(uint64_t* p, uint64_t token) {
  if (!vx_mbar_try_wait(p, token))
    return false;
  vx_fence();
  return true;
}

// st.async and red.async, like PTX st.async / red.async .shared::cluster
// .mbarrier::complete_tx::bytes: `addr` and `mbar` are window addresses
// (vx_cluster_map_shared_rank()) in one block of the caller's cluster, which
// may be the caller's own. Once the store or the add has been performed in
// that block, complete_tx(4 or 8 bytes) is applied to `mbar`, per lane. They
// are asynchronous and weak: no fence, barrier or exit waits for them, and
// the consumer learns of the data only through its mbarrier. The add returns
// nothing.
#define __VX_ASYNC_OP(f2, p, v, m)                                       \
    __asm__ volatile (".insn r4 %0, 6, %1, x0, %2, %3, %4"               \
                      :: "i"(RISCV_CUSTOM2), "i"(f2), "r"(p), "r"(v), "r"(m) \
                      : "memory")

static inline __attribute__((always_inline)) void vx_st_async_u32(void* addr, uint32_t v, uint64_t* mbar) {
  __VX_ASYNC_OP(0, addr, (uint64_t)v, mbar);
}

static inline __attribute__((always_inline)) void vx_st_async_u64(void* addr, uint64_t v, uint64_t* mbar) {
  __VX_ASYNC_OP(1, addr, v, mbar);
}

static inline __attribute__((always_inline)) void vx_red_async_add_u32(void* addr, uint32_t v, uint64_t* mbar) {
  __VX_ASYNC_OP(2, addr, (uint64_t)v, mbar);
}

static inline __attribute__((always_inline)) void vx_red_async_add_u64(void* addr, uint64_t v, uint64_t* mbar) {
  __VX_ASYNC_OP(3, addr, v, mbar);
}

#undef __VX_ASYNC_OP

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

  uint64_t arrive_relaxed(uint32_t count = 1) const { return vx_mbar_arrive_relaxed(obj_, count); }

  uint64_t arrive_release_cluster(uint32_t count = 1) const {
    return vx_mbar_arrive_release_cluster(obj_, count);
  }

  uint64_t expect_tx(uint32_t bytes) const { return vx_mbar_expect_tx(obj_, bytes); }

  uint64_t complete_tx(uint32_t bytes) const { return vx_mbar_complete_tx(obj_, bytes); }

  bool test_wait(uint64_t token) const { return vx_mbar_test_wait(obj_, token); }

  bool test_wait_parity(uint32_t parity) const { return vx_mbar_test_wait_parity(obj_, parity); }

  bool try_wait(uint64_t token) const { return vx_mbar_try_wait(obj_, token); }

  bool try_wait_parity(uint32_t parity) const { return vx_mbar_try_wait_parity(obj_, parity); }

  bool try_wait_acquire_cluster(uint64_t token) const { return vx_mbar_try_wait_acquire_cluster(obj_, token); }

  // Sleeps between polls instead of spinning, like cuda::barrier::wait().
  void wait(uint64_t token) const {
    while (!try_wait(token)) {}
  }

  void wait_parity(uint32_t parity) const {
    while (!try_wait_parity(parity)) {}
  }

  void wait_acquire_cluster(uint64_t token) const {
    while (!try_wait_acquire_cluster(token)) {}
  }

  void arrive_and_wait(uint32_t count = 1) const { wait(arrive(count)); }

  uint64_t* native_handle() const { return obj_; }

private:
  uint64_t* obj_;
};

} // namespace vortex

#endif // __cplusplus

#endif // __VX_MBAR_H__
