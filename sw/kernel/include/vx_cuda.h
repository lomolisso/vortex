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

#ifndef __VX_CUDA_H__
#define __VX_CUDA_H__

// CUDA names over the Vortex kernel headers, so that a port reads line by line
// like its CUDA original. Each name is a thin call or a short exact sequence.
//
// Porting rules:
// - __shared__ is a compile error. A port places its shared variables in the
//   block's local memory, `T* p = (T*)((char*)__local_mem() + off)`, and
//   launches with that many bytes of dynamic local memory.
// - __global__ declares a non-template kernel, launched by its own name. C++
//   gives a template no C linkage, so a templated kernel is declared
//   __device__ and each instantiation is exported under a name with
//   VX_CUDA_KERNEL_EXPORT(name, kernel<args...>). Kernels take one argument.
// - A kernel that calls __syncthreads_count/_and/_or requests
//   VX_CUDA_SYNC_RESERVE bytes of dynamic local memory beyond its own data and
//   leaves them alone: two counter words sit in the last 8 bytes of the
//   request (rounded down to a multiple of 4). A request below
//   VX_CUDA_SYNC_RESERVE traps.
// - The DXA operations (cp_async_bulk*, cp_reduce_async_bulk*, the tensor
//   prefetches, prefetch_tensormap and the tensormap acquire) issue from lane
//   0 of the warp: lane 0 must be active, or the call traps. Each active lane
//   still issues its own operation, one after the other.
// - Bulk async-groups are tracked per warp, not per thread.

#ifndef __cplusplus
#error "vx_cuda.h is C++ only"
#endif
#if !defined(__riscv_xlen) || __riscv_xlen != 64
#error "vx_cuda.h needs XLEN 64"
#endif
#if !VX_CFG_EXT_DLMEM_ENABLED
#error "vx_cuda.h needs VX_CFG_EXT_DLMEM_ENABLE"
#endif
#if !VX_CFG_EXT_A_ENABLED
#error "vx_cuda.h needs VX_CFG_EXT_A_ENABLE"
#endif
#if !VX_CFG_EXT_ZACAS_ENABLED
#error "vx_cuda.h needs VX_CFG_EXT_ZACAS_ENABLE"
#endif
#if !VX_CFG_EXT_MBAR_ENABLED
#error "vx_cuda.h needs VX_CFG_EXT_MBAR_ENABLE"
#endif
#if !VX_CFG_EXT_DXA_ENABLED
#error "vx_cuda.h needs VX_CFG_EXT_DXA_ENABLE"
#endif
#if !VX_CFG_EXT_FPAMO_ENABLED
#error "vx_cuda.h needs VX_CFG_EXT_FPAMO_ENABLE"
#endif
// The DXA operands are packed in t5/t6 (see __vxc_issue1); the build keeps
// them out of register allocation.
#ifndef VX_CUDA_FIXED_TEMPS
#error "vx_cuda.h: build the kernel with -ffixed-x30 -ffixed-x31 -DVX_CUDA_FIXED_TEMPS"
#endif

#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_dlmem.h>
#include <vx_mbar.h>
#include <vx_cp_async.h>
#include <vx_dxa.h>
#include <vx_atomic.h>
#include <vx_cuda_fp.h>
#include <stddef.h>
#include <stdint.h>

#define __VX_CUDA_INL inline __attribute__((always_inline))

// The trap instruction of __builtin_trap() (unimp), but not noreturn: a
// noreturn call leaves the divergent branches around it without a
// post-dominator, and the compiler then emits no split/join for them. Its
// condition is made warp-uniform (a vote, or a uniform value): under a
// divergent condition the trap fired with the condition false (simx).
__VX_CUDA_INL void __vx_cuda_trap() { __asm__ volatile ("unimp" ::: "memory"); }

__VX_CUDA_INL void __vx_cuda_trap_if_any(bool bad) {
  const uint32_t __UNIFORM__ any = (uint32_t)vx_vote_any(bad);
  if (any)
    __vx_cuda_trap();
}

#ifdef NDEBUG
#define __VX_CUDA_ASSERT(cond) ((void)0)
#else
#define __VX_CUDA_ASSERT(cond) __vx_cuda_trap_if_any(!(cond))
#endif

// The warp functions are templates so that the 32-lane assert fires only
// where one is used: other code of a port still builds at another width.
#define __VX_WARP template <int __vx_w = 0> __VX_CUDA_INL
#define __VX_WARP_ASSERT()                                                              \
  static_assert(__vx_w != 0 || (VX_CFG_NUM_THREADS == 32 && VX_CFG_NUM_ALU_LANES == 32), \
                "warp intrinsics need 32-thread warps and a 32-lane ALU group")

///////////////////////////////////////////////////////////////////////////////
// Attributes and no-ops

#ifndef __global__
#define __global__ __kernel
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __forceinline__
#define __forceinline__ inline __attribute__((always_inline))
#endif
#ifndef __noinline__
#define __noinline__ __attribute__((noinline))
#endif
// A __shared__ variable would silently become per-thread storage.
#ifndef __shared__
#define __shared__ _Pragma("GCC error \"__shared__ is not supported: place shared variables at offsets in __local_mem()\"")
#endif
#ifndef __constant__
#define __constant__
#endif
#ifndef __align__
#define __align__(n) __attribute__((aligned(n)))
#endif
#ifndef __grid_constant__
#define __grid_constant__
#endif
#ifndef __launch_bounds__
#define __launch_bounds__(...)
#endif
#ifndef __cluster_dims__
#define __cluster_dims__(...)
#endif

// The H100 runs the dependent grid early and waits here; Vortex runs the
// launches of a queue one after the other, so the prerequisite grid has
// completed and its writes are visible.
__VX_CUDA_INL void cudaGridDependencySynchronize() {}
__VX_CUDA_INL void cudaTriggerProgrammaticLaunchCompletion() {}

__VX_CUDA_INL void __setmaxnreg(unsigned) {}

template <typename T>
__VX_CUDA_INL T __ldg(const T* p) { return *p; }

// ld.global.cg: Vortex has no cache-global hint, so __ldcg is a plain load.
#define __VX_LDCG(T) __VX_CUDA_INL T __ldcg(const T* p) { return *p; }
__VX_LDCG(char)
__VX_LDCG(signed char)
__VX_LDCG(short)
__VX_LDCG(int)
__VX_LDCG(long)
__VX_LDCG(long long)
__VX_LDCG(unsigned char)
__VX_LDCG(unsigned short)
__VX_LDCG(unsigned int)
__VX_LDCG(unsigned long)
__VX_LDCG(unsigned long long)
__VX_LDCG(float)
__VX_LDCG(double)
__VX_LDCG(char2)
__VX_LDCG(char4)
__VX_LDCG(short2)
__VX_LDCG(short4)
__VX_LDCG(int2)
__VX_LDCG(int4)
__VX_LDCG(longlong2)
__VX_LDCG(uchar2)
__VX_LDCG(uchar4)
__VX_LDCG(ushort2)
__VX_LDCG(ushort4)
__VX_LDCG(uint2)
__VX_LDCG(uint4)
__VX_LDCG(ulonglong2)
__VX_LDCG(float2)
__VX_LDCG(float4)
__VX_LDCG(double2)
__VX_LDCG(__half)
__VX_LDCG(__half2)
__VX_LDCG(__nv_bfloat16)
__VX_LDCG(__nv_bfloat162)
#undef __VX_LDCG

// cuda_fp16.h's alternative names.
typedef __half __nv_half;
typedef __half2 __nv_half2;

// math_constants.h, as constant expressions.
#define CUDART_INF_F __builtin_bit_cast(float, 0x7f800000U)
#define CUDART_NAN_F __builtin_bit_cast(float, 0x7fffffffU)

__VX_CUDA_INL void __prefetch_global_l2(const void*) {}

///////////////////////////////////////////////////////////////////////////////
// __cluster_dims__: VX_CUDA_CLUSTER_DIMS in the port's shared header,
// VX_CUDA_KERNEL_ENTRY as the kernel's first statement.

#ifndef VX_CUDA_CLUSTER_DIMS
#define VX_CUDA_CLUSTER_DIMS(name, X, Y, Z)                                     \
  struct name##_cluster_dims {                                                   \
    static constexpr uint32_t x = (X), y = (Y), z = (Z);                         \
  }
#endif

// Active with NDEBUG: the block barrier lets thread 0 print before any warp
// of the block traps.
#define VX_CUDA_KERNEL_ENTRY(name)                                               \
  do {                                                                           \
    const uint32_t __UNIFORM__ __vx_bad =                                        \
        (vx_cluster_nctaid_x() != name##_cluster_dims::x)                        \
        | (vx_cluster_nctaid_y() != name##_cluster_dims::y)                      \
        | (vx_cluster_nctaid_z() != name##_cluster_dims::z);                     \
    if (__vx_bad) {                                                              \
      if (cooperative_groups::this_thread_block().thread_rank() == 0)            \
        vx_printf("cluster shape does not match __cluster_dims__\n");            \
      __syncthreads();                                                           \
      __vx_cuda_trap();                                                          \
    }                                                                            \
  } while (0)

// VX_CUDA_KERNEL_EXPORT(name, kernel<args...>): the kernel `name`, launched
// by that name, running one instantiation of a templated __device__ kernel.
template <typename F>
struct __vx_cuda_kernel_arg;
template <typename A>
struct __vx_cuda_kernel_arg<void(A)> {
  using type = A;
};

#define VX_CUDA_KERNEL_EXPORT(name, ...)                                         \
  __kernel void name(typename __vx_cuda_kernel_arg<decltype(__VA_ARGS__)>::type  \
                         __UNIFORM__ __vx_arg) {                                 \
    __VA_ARGS__(__vx_arg);                                                       \
  }

///////////////////////////////////////////////////////////////////////////////
// Bit moves for the shuffles and atomics.

template <typename To, typename From>
__VX_CUDA_INL To __vx_cuda_bit_cast(const From& f) {
  static_assert(sizeof(To) == sizeof(From), "bit cast between sizes");
  return __builtin_bit_cast(To, f);
}

template <typename T>
__VX_CUDA_INL size_t __vx_cuda_to_word(T v) {
  static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8, "1-8 byte values");
  if constexpr (sizeof(T) == 8) return (size_t)__vx_cuda_bit_cast<uint64_t>(v);
  else if constexpr (sizeof(T) == 4) return (size_t)__vx_cuda_bit_cast<uint32_t>(v);
  else if constexpr (sizeof(T) == 2) return (size_t)__vx_cuda_bit_cast<uint16_t>(v);
  else return (size_t)__vx_cuda_bit_cast<uint8_t>(v);
}

template <typename T>
__VX_CUDA_INL T __vx_cuda_from_word(size_t w) {
  if constexpr (sizeof(T) == 8) return __vx_cuda_bit_cast<T>((uint64_t)w);
  else if constexpr (sizeof(T) == 4) return __vx_cuda_bit_cast<T>((uint32_t)w);
  else if constexpr (sizeof(T) == 2) return __vx_cuda_bit_cast<T>((uint16_t)w);
  else return __vx_cuda_bit_cast<T>((uint8_t)w);
}

///////////////////////////////////////////////////////////////////////////////
// Warp intrinsics

static constexpr int warpSize = VX_CFG_NUM_THREADS;

__VX_WARP unsigned __activemask() {
  __VX_WARP_ASSERT();
  return (unsigned)vx_active_threads();
}

__VX_WARP unsigned __laneid() {
  __VX_WARP_ASSERT();
  return (unsigned)vx_thread_id();
}

__VX_WARP unsigned __lanemask_eq() { __VX_WARP_ASSERT(); return 1u << __laneid(); }
__VX_WARP unsigned __lanemask_lt() { __VX_WARP_ASSERT(); return (unsigned)((1ull << __laneid()) - 1); }
__VX_WARP unsigned __lanemask_le() { __VX_WARP_ASSERT(); return (unsigned)((2ull << __laneid()) - 1); }
__VX_WARP unsigned __lanemask_gt() { __VX_WARP_ASSERT(); return ~__lanemask_le(); }
__VX_WARP unsigned __lanemask_ge() { __VX_WARP_ASSERT(); return ~__lanemask_lt(); }

// The hardware does not use the mask; a caller outside its own mask (undefined
// in CUDA) traps when NDEBUG is off. Exited lanes in the mask are legal.
#define __VX_CUDA_MASK_CHECK(mask) __VX_CUDA_ASSERT((((mask) >> __laneid()) & 1u) != 0)

// PTX shfl.sync with c = ((32 - width) << 8) | clamp: Vortex takes the same
// segment mask and clamp, and b[4:0], so an out-of-range lane wraps as in PTX.
enum { __VX_SHFL_IDX, __VX_SHFL_UP, __VX_SHFL_DOWN, __VX_SHFL_XOR };

template <int Mode, typename T>
__VX_CUDA_INL T __vx_cuda_shfl(unsigned mask, T var, int b, int width) {
  __VX_CUDA_MASK_CHECK(mask);
  __VX_CUDA_ASSERT(width >= 1 && width <= 32 && (width & (width - 1)) == 0);
  (void)mask;
  const int seg = 32 - width;
  const size_t w = __vx_cuda_to_word(var);
  size_t r;
  if constexpr (Mode == __VX_SHFL_IDX) r = vx_shfl_idx(w, b & 31, 0x1f, seg);
  else if constexpr (Mode == __VX_SHFL_UP) r = vx_shfl_up(w, b & 31, 0, seg);
  else if constexpr (Mode == __VX_SHFL_DOWN) r = vx_shfl_down(w, b & 31, 0x1f, seg);
  else r = vx_shfl_bfly(w, b & 31, 0x1f, seg);
  return __vx_cuda_from_word<T>(r);
}

#define __VX_SHFL_OVERLOADS(T)                                                                    \
  __VX_WARP T __shfl_sync(unsigned mask, T var, int srcLane, int width = warpSize) {              \
    __VX_WARP_ASSERT();                                                                           \
    return __vx_cuda_shfl<__VX_SHFL_IDX>(mask, var, srcLane, width);                              \
  }                                                                                               \
  __VX_WARP T __shfl_up_sync(unsigned mask, T var, unsigned delta, int width = warpSize) {        \
    __VX_WARP_ASSERT();                                                                           \
    return __vx_cuda_shfl<__VX_SHFL_UP>(mask, var, (int)delta, width);                            \
  }                                                                                               \
  __VX_WARP T __shfl_down_sync(unsigned mask, T var, unsigned delta, int width = warpSize) {      \
    __VX_WARP_ASSERT();                                                                           \
    return __vx_cuda_shfl<__VX_SHFL_DOWN>(mask, var, (int)delta, width);                          \
  }                                                                                               \
  __VX_WARP T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width = warpSize) {         \
    __VX_WARP_ASSERT();                                                                           \
    return __vx_cuda_shfl<__VX_SHFL_XOR>(mask, var, laneMask, width);                             \
  }

__VX_SHFL_OVERLOADS(int)
__VX_SHFL_OVERLOADS(unsigned int)
__VX_SHFL_OVERLOADS(long)
__VX_SHFL_OVERLOADS(unsigned long)
__VX_SHFL_OVERLOADS(long long)
__VX_SHFL_OVERLOADS(unsigned long long)
__VX_SHFL_OVERLOADS(float)
__VX_SHFL_OVERLOADS(double)
__VX_SHFL_OVERLOADS(__half)
__VX_SHFL_OVERLOADS(__half2)
__VX_SHFL_OVERLOADS(__nv_bfloat16)
__VX_SHFL_OVERLOADS(__nv_bfloat162)
#undef __VX_SHFL_OVERLOADS

__VX_WARP unsigned __ballot_sync(unsigned mask, int predicate) {
  __VX_WARP_ASSERT();
  __VX_CUDA_MASK_CHECK(mask);
  return (unsigned)vx_vote_ballot(predicate != 0) & mask;
}

__VX_WARP int __any_sync(unsigned mask, int predicate) {
  __VX_WARP_ASSERT();
  return __ballot_sync(mask, predicate) != 0;
}

__VX_WARP int __all_sync(unsigned mask, int predicate) {
  __VX_WARP_ASSERT();
  return __ballot_sync(mask, predicate) == (mask & __activemask());
}

// bar.warp.sync: the lanes of a Vortex warp share one instruction stream, so
// the fence alone orders every lane's earlier accesses before its later ones.
__VX_WARP void __syncwarp(unsigned mask = 0xffffffffu) {
  __VX_WARP_ASSERT();
  __VX_CUDA_MASK_CHECK(mask);
  (void)mask;
  vx_fence_cta();
}

// __reduce_*_sync (redux.sync), over the non-exited lanes of the mask: a
// butterfly over a full, fully active warp; otherwise each lane in ascending
// order. Equal to redux.sync.
template <typename T, typename Op>
__VX_CUDA_INL T __vx_cuda_reduce(unsigned mask, T v, Op op) {
  mask &= __activemask();
  if (mask == 0xffffffffu) {
    for (int d = 16; d != 0; d >>= 1)
      v = op(v, __shfl_xor_sync(mask, v, d));
    return v;
  }
  if (mask == 0)
    return v;
  unsigned m = mask;
  T acc = __shfl_sync(mask, v, __builtin_ctz(m));
  for (m &= m - 1; m != 0; m &= m - 1)
    acc = op(acc, __shfl_sync(mask, v, __builtin_ctz(m)));
  return acc;
}

__VX_WARP unsigned __reduce_add_sync(unsigned mask, unsigned value) {
  __VX_WARP_ASSERT();
  return __vx_cuda_reduce(mask, value, [](unsigned a, unsigned b) { return a + b; });
}
__VX_WARP int __reduce_add_sync(unsigned mask, int value) {
  __VX_WARP_ASSERT();
  return (int)__reduce_add_sync(mask, (unsigned)value);
}
__VX_WARP unsigned __reduce_min_sync(unsigned mask, unsigned value) {
  __VX_WARP_ASSERT();
  return __vx_cuda_reduce(mask, value, [](unsigned a, unsigned b) { return a < b ? a : b; });
}
__VX_WARP int __reduce_min_sync(unsigned mask, int value) {
  __VX_WARP_ASSERT();
  return __vx_cuda_reduce(mask, value, [](int a, int b) { return a < b ? a : b; });
}
__VX_WARP unsigned __reduce_max_sync(unsigned mask, unsigned value) {
  __VX_WARP_ASSERT();
  return __vx_cuda_reduce(mask, value, [](unsigned a, unsigned b) { return a > b ? a : b; });
}
__VX_WARP int __reduce_max_sync(unsigned mask, int value) {
  __VX_WARP_ASSERT();
  return __vx_cuda_reduce(mask, value, [](int a, int b) { return a > b ? a : b; });
}
__VX_WARP unsigned __reduce_and_sync(unsigned mask, unsigned value) {
  __VX_WARP_ASSERT();
  return __vx_cuda_reduce(mask, value, [](unsigned a, unsigned b) { return a & b; });
}
__VX_WARP unsigned __reduce_or_sync(unsigned mask, unsigned value) {
  __VX_WARP_ASSERT();
  return __vx_cuda_reduce(mask, value, [](unsigned a, unsigned b) { return a | b; });
}
__VX_WARP unsigned __reduce_xor_sync(unsigned mask, unsigned value) {
  __VX_WARP_ASSERT();
  return __vx_cuda_reduce(mask, value, [](unsigned a, unsigned b) { return a ^ b; });
}

// match.sync compares bit patterns over the non-exited lanes of the mask; each
// round's leader is such a lane, so it resolves at least itself.
template <typename T>
__VX_CUDA_INL unsigned __vx_cuda_match_any(unsigned mask, T value) {
  mask &= __activemask();
  const size_t v = __vx_cuda_to_word(value);
  unsigned result = 0;
  for (unsigned left = mask; left != 0;) {
    const size_t key = __shfl_sync(mask, (unsigned long)v, __builtin_ctz(left));
    const unsigned same = __ballot_sync(mask, v == key);
    if (v == key)
      result = same;
    left &= ~same;
  }
  return result;
}

template <typename T>
__VX_CUDA_INL unsigned __vx_cuda_match_all(unsigned mask, T value, int* pred) {
  mask &= __activemask();
  const size_t v = __vx_cuda_to_word(value);
  const size_t key = mask ? __shfl_sync(mask, (unsigned long)v, __builtin_ctz(mask)) : v;
  const bool all = __ballot_sync(mask, v == key) == mask;
  *pred = all;
  return all ? mask : 0u;
}

#define __VX_MATCH_OVERLOADS(T)                                                    \
  __VX_WARP unsigned __match_any_sync(unsigned mask, T value) {                     \
    __VX_WARP_ASSERT();                                                             \
    return __vx_cuda_match_any(mask, value);                                        \
  }                                                                                 \
  __VX_WARP unsigned __match_all_sync(unsigned mask, T value, int* pred) {          \
    __VX_WARP_ASSERT();                                                             \
    return __vx_cuda_match_all(mask, value, pred);                                  \
  }

__VX_MATCH_OVERLOADS(int)
__VX_MATCH_OVERLOADS(unsigned int)
__VX_MATCH_OVERLOADS(long)
__VX_MATCH_OVERLOADS(unsigned long)
__VX_MATCH_OVERLOADS(long long)
__VX_MATCH_OVERLOADS(unsigned long long)
__VX_MATCH_OVERLOADS(float)
__VX_MATCH_OVERLOADS(double)
#undef __VX_MATCH_OVERLOADS

///////////////////////////////////////////////////////////////////////////////
// Block reductions: bar.red as barrier, local-memory atomics, barrier.

#define VX_CUDA_SYNC_RESERVE 16u

// [0] true predicates, [1] arriving threads.
__VX_CUDA_INL volatile uint32_t* __vx_cuda_sync_counter() {
  const uint32_t __UNIFORM__ n = vx_dynamic_smem_size();
  if (n < VX_CUDA_SYNC_RESERVE)
    __vx_cuda_trap();
  return (volatile uint32_t*)((char*)__local_mem() + ((n & ~3u) - 8));
}

__VX_CUDA_INL uint32_t __vx_cuda_block_thread_rank() {
  return (uint32_t)threadIdx.x + (uint32_t)blockDim.x * ((uint32_t)threadIdx.y + (uint32_t)blockDim.y * (uint32_t)threadIdx.z);
}

__VX_CUDA_INL uint32_t __vx_cuda_block_threads() {
  return (uint32_t)blockDim.x * (uint32_t)blockDim.y * (uint32_t)blockDim.z;
}

// bar.red over the threads that arrive (exited threads take no part): warp
// 0's lowest active lane clears the words, each warp's lowest active lane adds
// its true predicates and its arrivals. The last barrier keeps the next call
// from clearing before every thread has read. Returns {count, arrivals}.
struct __vx_cuda_red_t {
  uint32_t count, arrivals;
};

__VX_WARP __vx_cuda_red_t __vx_cuda_bar_red(int predicate) {
  __VX_WARP_ASSERT();
  volatile uint32_t* c = __vx_cuda_sync_counter();
  const unsigned act = __activemask();
  const bool leader = __laneid() == (unsigned)__builtin_ctz(act);
  __syncthreads();
  if (leader && __vx_cuda_block_thread_rank() < 32) {
    c[0] = 0;
    c[1] = 0;
  }
  __syncthreads();
  const unsigned votes = __ballot_sync(act, predicate);
  if (leader) {
    __atomic_fetch_add((uint32_t*)&c[0], (uint32_t)__builtin_popcount(votes), __ATOMIC_RELAXED);
    __atomic_fetch_add((uint32_t*)&c[1], (uint32_t)__builtin_popcount(act), __ATOMIC_RELAXED);
  }
  __syncthreads();
  const __vx_cuda_red_t r = {c[0], c[1]};
  __syncthreads();
  return r;
}

__VX_WARP int __syncthreads_count(int predicate) {
  __VX_WARP_ASSERT();
  return (int)__vx_cuda_bar_red(predicate).count;
}

__VX_WARP int __syncthreads_and(int predicate) {
  __VX_WARP_ASSERT();
  const __vx_cuda_red_t r = __vx_cuda_bar_red(predicate);
  return r.count == r.arrivals;
}

__VX_WARP int __syncthreads_or(int predicate) {
  __VX_WARP_ASSERT();
  return __vx_cuda_bar_red(predicate).count != 0;
}

///////////////////////////////////////////////////////////////////////////////
// Atomics: relaxed; integers through RISC-V A and Zacas, the rest through the
// float and 16-bit forms of vx_atomic.h. _block takes the cta-scope form,
// _system the gpu one.

#define __VX_ATOMIC_INT(T)                                                                     \
  __VX_CUDA_INL T atomicAdd(T* p, T v) { return __atomic_fetch_add(p, v, __ATOMIC_RELAXED); }  \
  __VX_CUDA_INL T atomicAnd(T* p, T v) { return __atomic_fetch_and(p, v, __ATOMIC_RELAXED); }  \
  __VX_CUDA_INL T atomicOr(T* p, T v) { return __atomic_fetch_or(p, v, __ATOMIC_RELAXED); }    \
  __VX_CUDA_INL T atomicXor(T* p, T v) { return __atomic_fetch_xor(p, v, __ATOMIC_RELAXED); }  \
  __VX_CUDA_INL T atomicExch(T* p, T v) { return __atomic_exchange_n(p, v, __ATOMIC_RELAXED); } \
  __VX_CUDA_INL T atomicCAS(T* p, T cmp, T v) {                                                \
    __atomic_compare_exchange_n(p, &cmp, v, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);        \
    return cmp;                                                                                \
  }

#define __VX_ATOMIC_MINMAX(T)                                                                  \
  __VX_CUDA_INL T atomicMin(T* p, T v) { return __atomic_fetch_min(p, v, __ATOMIC_RELAXED); }  \
  __VX_CUDA_INL T atomicMax(T* p, T v) { return __atomic_fetch_max(p, v, __ATOMIC_RELAXED); }

__VX_ATOMIC_INT(int)
__VX_ATOMIC_INT(unsigned int)
__VX_ATOMIC_INT(unsigned long long)
__VX_ATOMIC_MINMAX(int)
__VX_ATOMIC_MINMAX(unsigned int)
__VX_ATOMIC_MINMAX(long long)
__VX_ATOMIC_MINMAX(unsigned long long)
#undef __VX_ATOMIC_INT
#undef __VX_ATOMIC_MINMAX

__VX_CUDA_INL int atomicSub(int* p, int v) { return __atomic_fetch_sub(p, v, __ATOMIC_RELAXED); }
__VX_CUDA_INL unsigned atomicSub(unsigned* p, unsigned v) { return __atomic_fetch_sub(p, v, __ATOMIC_RELAXED); }

__VX_CUDA_INL float atomicExch(float* p, float v) {
  return __vx_cuda_bit_cast<float>(
      __atomic_exchange_n((uint32_t*)p, __vx_cuda_bit_cast<uint32_t>(v), __ATOMIC_RELAXED));
}

#define __VX_ATOMIC_FP(SFX, SC)                                                                       \
  __VX_CUDA_INL float atomicAdd##SFX(float* p, float v) { return vx_atom_add_relaxed_##SC##_f32(p, v); } \
  __VX_CUDA_INL double atomicAdd##SFX(double* p, double v) { return vx_atom_add_relaxed_##SC##_f64(p, v); } \
  __VX_CUDA_INL __half atomicAdd##SFX(__half* p, __half v) {                                           \
    return __vx_cuda_bit_cast<__half>(vx_atom_add_relaxed_##SC##_f16((uint16_t*)p, __vx_cuda_bit_cast<uint16_t>(v))); \
  }                                                                                                   \
  __VX_CUDA_INL __nv_bfloat16 atomicAdd##SFX(__nv_bfloat16* p, __nv_bfloat16 v) {                      \
    return __vx_cuda_bit_cast<__nv_bfloat16>(                                                         \
        vx_atom_add_relaxed_##SC##_bf16((uint16_t*)p, __vx_cuda_bit_cast<uint16_t>(v)));              \
  }                                                                                                   \
  __VX_CUDA_INL __half2 atomicAdd##SFX(__half2* p, __half2 v) {                                        \
    return __vx_cuda_bit_cast<__half2>(vx_atom_add_relaxed_##SC##_f16x2((uint32_t*)p, __vx_cuda_bit_cast<uint32_t>(v))); \
  }                                                                                                   \
  __VX_CUDA_INL __nv_bfloat162 atomicAdd##SFX(__nv_bfloat162* p, __nv_bfloat162 v) {                   \
    return __vx_cuda_bit_cast<__nv_bfloat162>(                                                        \
        vx_atom_add_relaxed_##SC##_bf16x2((uint32_t*)p, __vx_cuda_bit_cast<uint32_t>(v)));            \
  }                                                                                                   \
  __VX_CUDA_INL unsigned atomicInc##SFX(unsigned* p, unsigned v) { return vx_atom_inc_relaxed_##SC##_u32(p, v); } \
  __VX_CUDA_INL unsigned atomicDec##SFX(unsigned* p, unsigned v) { return vx_atom_dec_relaxed_##SC##_u32(p, v); } \
  __VX_CUDA_INL unsigned short atomicCAS##SFX(unsigned short* p, unsigned short cmp, unsigned short v) { \
    return vx_atom_cas_relaxed_##SC##_b16(p, cmp, v);                                                  \
  }

__VX_ATOMIC_FP(, gpu)
__VX_ATOMIC_FP(_block, cta)
__VX_ATOMIC_FP(_system, gpu)
#undef __VX_ATOMIC_FP

// The vector float adds of sm_90, on global memory only (as CUDA): one packed
// 64-bit add per pair of elements, each element atomic.
__VX_CUDA_INL float2 atomicAdd(float2* p, float2 v) {
  const float in[2] = {v.x, v.y};
  float old[2];
  vx_atomic_add_v2_f32((float*)p, in, old);
  float2 r;
  r.x = old[0];
  r.y = old[1];
  return r;
}

__VX_CUDA_INL float4 atomicAdd(float4* p, float4 v) {
  const float in[4] = {v.x, v.y, v.z, v.w};
  float old[4];
  vx_atomic_add_v4_f32((float*)p, in, old);
  float4 r;
  r.x = old[0];
  r.y = old[1];
  r.z = old[2];
  r.w = old[3];
  return r;
}

// Integer scopes: a relaxed AMO is the same instruction at every scope.
#define __VX_ATOMIC_SCOPED_INT(SFX)                                                                   \
  template <typename T, typename... A>                                                                \
  __VX_CUDA_INL auto atomicAdd##SFX(T* p, A... a) -> decltype(atomicAdd(p, a...)) { return atomicAdd(p, a...); } \
  template <typename T, typename... A>                                                                \
  __VX_CUDA_INL auto atomicSub##SFX(T* p, A... a) -> decltype(atomicSub(p, a...)) { return atomicSub(p, a...); } \
  template <typename T, typename... A>                                                                \
  __VX_CUDA_INL auto atomicExch##SFX(T* p, A... a) -> decltype(atomicExch(p, a...)) { return atomicExch(p, a...); } \
  template <typename T, typename... A>                                                                \
  __VX_CUDA_INL auto atomicMin##SFX(T* p, A... a) -> decltype(atomicMin(p, a...)) { return atomicMin(p, a...); } \
  template <typename T, typename... A>                                                                \
  __VX_CUDA_INL auto atomicMax##SFX(T* p, A... a) -> decltype(atomicMax(p, a...)) { return atomicMax(p, a...); } \
  template <typename T, typename... A>                                                                \
  __VX_CUDA_INL auto atomicAnd##SFX(T* p, A... a) -> decltype(atomicAnd(p, a...)) { return atomicAnd(p, a...); } \
  template <typename T, typename... A>                                                                \
  __VX_CUDA_INL auto atomicOr##SFX(T* p, A... a) -> decltype(atomicOr(p, a...)) { return atomicOr(p, a...); } \
  template <typename T, typename... A>                                                                \
  __VX_CUDA_INL auto atomicXor##SFX(T* p, A... a) -> decltype(atomicXor(p, a...)) { return atomicXor(p, a...); } \
  template <typename T, typename... A>                                                                \
  __VX_CUDA_INL auto atomicCAS##SFX(T* p, A... a) -> decltype(atomicCAS(p, a...)) { return atomicCAS(p, a...); }

__VX_ATOMIC_SCOPED_INT(_block)
__VX_ATOMIC_SCOPED_INT(_system)
#undef __VX_ATOMIC_SCOPED_INT

///////////////////////////////////////////////////////////////////////////////
// Special registers

__VX_CUDA_INL unsigned __smid() { return (unsigned)vx_core_id(); }
__VX_CUDA_INL unsigned __nsmid() { return (unsigned)vx_num_cores(); }
__VX_CUDA_INL unsigned long long __globaltimer() { return vx_globaltimer(); }

// Device functions of the CUDA math and intrinsics APIs.

#define __VX_MINMAX(T)                                                   \
  __VX_CUDA_INL T max(T a, T b) { return a > b ? a : b; }                 \
  __VX_CUDA_INL T min(T a, T b) { return a < b ? a : b; }
__VX_MINMAX(int)
__VX_MINMAX(unsigned int)
__VX_MINMAX(long)
__VX_MINMAX(unsigned long)
__VX_MINMAX(long long)
__VX_MINMAX(unsigned long long)
#undef __VX_MINMAX
#define __VX_MINMAX_MIXED(S, U)                                                 \
  __VX_CUDA_INL U max(S a, U b) { return max((U)a, b); }                        \
  __VX_CUDA_INL U max(U a, S b) { return max(a, (U)b); }                        \
  __VX_CUDA_INL U min(S a, U b) { return min((U)a, b); }                        \
  __VX_CUDA_INL U min(U a, S b) { return min(a, (U)b); }
__VX_MINMAX_MIXED(int, unsigned int)
__VX_MINMAX_MIXED(long, unsigned long)
__VX_MINMAX_MIXED(long long, unsigned long long)
#undef __VX_MINMAX_MIXED
// fmaxf/fminf: a NaN operand gives the other one.
__VX_CUDA_INL float max(float a, float b) { return __builtin_fmaxf(a, b); }
__VX_CUDA_INL float min(float a, float b) { return __builtin_fminf(a, b); }
__VX_CUDA_INL double max(double a, double b) { return __builtin_fmax(a, b); }
__VX_CUDA_INL double min(double a, double b) { return __builtin_fmin(a, b); }
__VX_CUDA_INL double max(float a, double b) { return __builtin_fmax(a, b); }
__VX_CUDA_INL double max(double a, float b) { return __builtin_fmax(a, b); }
__VX_CUDA_INL double min(float a, double b) { return __builtin_fmin(a, b); }
__VX_CUDA_INL double min(double a, float b) { return __builtin_fmin(a, b); }

__VX_CUDA_INL int __popc(unsigned x) { return __builtin_popcount(x); }
__VX_CUDA_INL int __popcll(unsigned long long x) { return __builtin_popcountll(x); }
__VX_CUDA_INL int __ffs(int x) { return __builtin_ffs(x); }
__VX_CUDA_INL int __ffsll(long long x) { return __builtin_ffsll(x); }
__VX_CUDA_INL int __clz(int x) { return x ? __builtin_clz((unsigned)x) : 32; }
__VX_CUDA_INL int __clzll(long long x) { return x ? __builtin_clzll((unsigned long long)x) : 64; }
__VX_CUDA_INL unsigned __brev(unsigned x) { return __builtin_bitreverse32(x); }
__VX_CUDA_INL unsigned long long __brevll(unsigned long long x) { return __builtin_bitreverse64(x); }

__VX_CUDA_INL long long clock64() { return (long long)vx_rdcycle(); }

// A spin of `ns` cycles: one cycle is one nanosecond of %globaltimer.
__VX_CUDA_INL void __nanosleep(unsigned ns) {
  const uint64_t t0 = vx_globaltimer();
  while (vx_globaltimer() - t0 < ns) {}
}

__VX_CUDA_INL void __threadfence_block() { vx_fence_cta(); }
__VX_CUDA_INL void __threadfence() { vx_fence(); }
__VX_CUDA_INL void __threadfence_system() { vx_fence(); }

// The .shared address of a block's own variable is its .shared::cluster
// address, as in PTX, where shared::cta lies inside shared::cluster.
__VX_CUDA_INL size_t __cvta_generic_to_shared(const void* p) { return vx_cvta_shared(p); }
__VX_CUDA_INL size_t __cvta_generic_to_global(const void* p) { return (size_t)(uintptr_t)p; }
__VX_CUDA_INL void* __cvta_shared_to_generic(size_t a) { return vx_cvta_cluster_to_generic((uint32_t)a); }
__VX_CUDA_INL void* __cvta_global_to_generic(size_t a) { return (void*)(uintptr_t)a; }

// cuda_pipeline.h primitives over vx_cp_async.h; a warp holds at most 8
// committed groups, so a wait for more prior groups waits for nothing.
__VX_CUDA_INL void __pipeline_memcpy_async(void* dst, const void* src, size_t size, size_t zfill = 0) {
  vortex::pipeline_memcpy_async(dst, src, size, zfill);
}
__VX_CUDA_INL void __pipeline_commit() { vortex::pipeline_commit(); }
__VX_CUDA_INL void __pipeline_wait_prior(size_t prior) {
  switch (prior) {
  case 0: vortex::pipeline_wait_prior<0>(); break;
  case 1: vortex::pipeline_wait_prior<1>(); break;
  case 2: vortex::pipeline_wait_prior<2>(); break;
  case 3: vortex::pipeline_wait_prior<3>(); break;
  case 4: vortex::pipeline_wait_prior<4>(); break;
  case 5: vortex::pipeline_wait_prior<5>(); break;
  case 6: vortex::pipeline_wait_prior<6>(); break;
  case 7: vortex::pipeline_wait_prior<7>(); break;
  default: break;
  }
}

typedef vx_tensor_map_t CUtensorMap;

using vortex::dim3;

///////////////////////////////////////////////////////////////////////////////
// Cooperative groups

namespace cooperative_groups {

using cluster_group = vortex::cluster_group;

__VX_CUDA_INL cluster_group this_cluster() { return cluster_group(); }

class thread_block {
public:
  static void sync() { __syncthreads(); }
  static unsigned thread_rank() { return __vx_cuda_block_thread_rank(); }
  static unsigned num_threads() { return __vx_cuda_block_threads(); }
  static unsigned size() { return num_threads(); }
  static dim3 group_index() { return {(uint32_t)blockIdx.x, (uint32_t)blockIdx.y, (uint32_t)blockIdx.z}; }
  static dim3 thread_index() { return {(uint32_t)threadIdx.x, (uint32_t)threadIdx.y, (uint32_t)threadIdx.z}; }
  static dim3 dim_threads() { return {(uint32_t)blockDim.x, (uint32_t)blockDim.y, (uint32_t)blockDim.z}; }
  static dim3 group_dim() { return dim_threads(); }
};

__VX_CUDA_INL thread_block this_thread_block() { return thread_block(); }

class grid_group {
public:
  static unsigned long long block_rank() {
    return (uint32_t)blockIdx.x + (uint64_t)gridDim.x * ((uint32_t)blockIdx.y + (uint64_t)gridDim.y * (uint32_t)blockIdx.z);
  }
  static unsigned long long num_blocks() {
    return (uint64_t)gridDim.x * (uint32_t)gridDim.y * (uint32_t)gridDim.z;
  }
  static unsigned long long thread_rank() {
    return block_rank() * __vx_cuda_block_threads() + __vx_cuda_block_thread_rank();
  }
  static unsigned long long num_threads() { return num_blocks() * __vx_cuda_block_threads(); }
  static unsigned long long size() { return num_threads(); }
  static dim3 dim_blocks() { return {(uint32_t)gridDim.x, (uint32_t)gridDim.y, (uint32_t)gridDim.z}; }
  static dim3 block_index() { return {(uint32_t)blockIdx.x, (uint32_t)blockIdx.y, (uint32_t)blockIdx.z}; }
  static unsigned long long cluster_rank() {
    return vx_clusterid_x() + (uint64_t)vx_nclusterid_x() * (vx_clusterid_y() + (uint64_t)vx_nclusterid_y() * vx_clusterid_z());
  }
  static unsigned long long num_clusters() {
    return (uint64_t)vx_nclusterid_x() * vx_nclusterid_y() * vx_nclusterid_z();
  }
  static dim3 dim_clusters() { return {vx_nclusterid_x(), vx_nclusterid_y(), vx_nclusterid_z()}; }
  static bool is_valid() { return true; }
  template <bool __vx_b = false>
  static void sync() {
    static_assert(__vx_b, "grid_group::sync() is not supported: no grid-wide barrier");
  }
};

__VX_CUDA_INL grid_group this_grid() { return grid_group(); }

// thread_block_tile<32, Parent> is a thread_block_tile<32>, as in CUDA.
template <unsigned Size, typename Parent = void>
class thread_block_tile;

template <unsigned Size>
class thread_block_tile<Size, void> {
  static_assert(Size == 32, "thread_block_tile supports only the 32-thread warp");

public:
  template <int __vx_w = 0>
  unsigned long long thread_rank() const { __VX_WARP_ASSERT(); return __laneid(); }
  static constexpr unsigned long long size() { return Size; }
  static constexpr unsigned long long num_threads() { return Size; }
  unsigned long long meta_group_rank() const { return __vx_cuda_block_thread_rank() / Size; }
  unsigned long long meta_group_size() const { return (__vx_cuda_block_threads() + Size - 1) / Size; }
  void sync() const { __syncwarp(); }
  template <typename T>
  T shfl(T var, int srcRank) const { return __shfl_sync(0xffffffffu, var, srcRank, (int)Size); }
  template <typename T>
  T shfl_up(T var, unsigned delta) const { return __shfl_up_sync(0xffffffffu, var, delta, (int)Size); }
  template <typename T>
  T shfl_down(T var, unsigned delta) const { return __shfl_down_sync(0xffffffffu, var, delta, (int)Size); }
  template <typename T>
  T shfl_xor(T var, int laneMask) const { return __shfl_xor_sync(0xffffffffu, var, laneMask, (int)Size); }
  int any(int predicate) const { return __any_sync(0xffffffffu, predicate); }
  int all(int predicate) const { return __all_sync(0xffffffffu, predicate); }
  unsigned ballot(int predicate) const { return __ballot_sync(0xffffffffu, predicate); }
};

template <unsigned Size, typename Parent>
class thread_block_tile : public thread_block_tile<Size, void> {};

template <unsigned Size, typename Parent>
__VX_CUDA_INL thread_block_tile<Size, Parent> tiled_partition(const Parent&) {
  return thread_block_tile<Size, Parent>();
}

template <typename Group>
__VX_CUDA_INL void sync(const Group& g) { g.sync(); }

} // namespace cooperative_groups

///////////////////////////////////////////////////////////////////////////////
// cuda:: barriers, scopes, sizes

namespace cuda {

enum thread_scope {
  thread_scope_system = 0,
  thread_scope_device = 1,
  thread_scope_block  = 2,
  thread_scope_thread = 3,
};

template <size_t Alignment>
struct aligned_size_t {
  static constexpr size_t align = Alignment;
  size_t value;
  constexpr explicit aligned_size_t(size_t v) : value(v) {}
  constexpr operator size_t() const { return value; }
};

// One 8-byte mbarrier in local memory, as on sm_90. Trivially constructible:
// a port places it in local memory and calls init() from one thread.
template <thread_scope Scope>
class barrier {
  static_assert(Scope == thread_scope_block,
                "cuda::barrier supports only thread_scope_block; cluster scope uses cuda::ptx mbarrier names");

public:
  using arrival_token = uint64_t;

  barrier() = default;
  barrier(const barrier&) = delete;
  barrier& operator=(const barrier&) = delete;

  friend __VX_CUDA_INL void init(barrier* b, ptrdiff_t expected) { vx_mbar_init(&b->obj_, (uint32_t)expected); }

  arrival_token arrive(ptrdiff_t update = 1) { return vx_mbar_arrive(&obj_, (uint32_t)update); }

  void wait(arrival_token&& token) const {
    while (!vx_mbar_try_wait(&obj_, token)) {}
  }

  void wait_parity(bool parity) const {
    while (!vx_mbar_try_wait_parity(&obj_, parity)) {}
  }

  void arrive_and_wait() { wait(arrive()); }

  void arrive_and_drop() { vx_mbar_arrive_drop(&obj_, 1); }

  static constexpr ptrdiff_t max() { return (1 << 20) - 1; }

  uint64_t* __vx_handle() const { return &obj_; }

private:
  mutable uint64_t obj_;
};

static_assert(sizeof(barrier<thread_scope_block>) == 8 && alignof(barrier<thread_scope_block>) == 8,
              "cuda::barrier is one 8-byte mbarrier");

// Each active lane in turn issues a DXA operation with its own operands:
// the DXA reads them from lane 0, so the lane's values are broadcast first.
struct __vx_lane_self {
  template <typename T>
  T operator()(T v) const { return v; }
};

struct __vx_lane_from {
  int lane;
  template <typename T>
  T operator()(T v) const {
    return __vx_cuda_from_word<T>(vx_shfl_idx(__vx_cuda_to_word(v), lane, 0x1f, 0));
  }
};

template <typename F>
__VX_CUDA_INL void __vx_dxa_each_lane(F&& f) {
  const uint32_t __UNIFORM__ m = (uint32_t)vx_active_threads();
  if (!(m & 1u))
    __vx_cuda_trap();
  if (m == 1u) {
    f(__vx_lane_self{});
    return;
  }
  for (uint32_t r = m; r != 0; r &= r - 1)
    f(__vx_lane_from{__builtin_ctz(r)});
}

__VX_CUDA_INL bool __vx_is_global(const void* p) { return __vx_atom_is_global(p) != 0; }

// The DXA reads its operands from lanes 0-3 of rs1/rs2, packed by a wgather,
// and a wgather writes its destination in inactive lanes too. A destination
// the compiler allocated may hold a live value of a lane parked by an
// enclosing branch (an IPDOM token, a predicate), so the packing and the issue
// form one asm block on t5/t6, which the build keeps out of allocation (so
// they are no clobbers: clang rejects reserved registers there).

__VX_CUDA_INL size_t __vxc_pair(int32_t lo, int32_t hi) { return ((size_t)(uint32_t)hi << 32) | (size_t)(uint32_t)lo; }

template <uint32_t F3>
__VX_CUDA_INL void __vxc_issue1(size_t a0, size_t a1, size_t a2, size_t a3) {
  __asm__ volatile ("mv t5, %[a0]\n\t"
                    ".insn r4 %[wg], 0, 0, t5, %[a1], %[a2], %[a3]\n\t"
                    ".insn r %[op], %[f3], %[f7], x0, t5, x0"
                    :: [a0] "r"(a0), [a1] "r"(a1), [a2] "r"(a2), [a3] "r"(a3), [wg] "i"(RISCV_CUSTOM1),
                       [op] "i"(VX_DXA_EXT_OPCODE), [f3] "i"(F3), [f7] "i"(VX_DXA_FUNCT7)
                    : "memory");
}

template <uint32_t F3>
__VX_CUDA_INL void __vxc_issue2(size_t a0, size_t a1, size_t a2, size_t a3, size_t b0, size_t b1, size_t b2,
                                size_t b3) {
  __asm__ volatile ("mv t5, %[a0]\n\t"
                    ".insn r4 %[wg], 0, 0, t5, %[a1], %[a2], %[a3]\n\t"
                    "mv t6, %[b0]\n\t"
                    ".insn r4 %[wg], 0, 0, t6, %[b1], %[b2], %[b3]\n\t"
                    ".insn r %[op], %[f3], %[f7], x0, t5, t6"
                    :: [a0] "r"(a0), [a1] "r"(a1), [a2] "r"(a2), [a3] "r"(a3), [b0] "r"(b0), [b1] "r"(b1),
                       [b2] "r"(b2), [b3] "r"(b3), [wg] "i"(RISCV_CUSTOM1), [op] "i"(VX_DXA_EXT_OPCODE),
                       [f3] "i"(F3), [f7] "i"(VX_DXA_FUNCT7)
                    : "memory");
}

#define __VXC_P(x) ((size_t)(uintptr_t)(x))

// The encodings of vx_dxa.h (funct3 1 bulk copy, 2 tensor load/prefetch,
// 3 bulk load, 4 L2 prefetch, 5 stores and reductions).
__VX_CUDA_INL void __vxc_bulk_copy(void* dst, const void* src, uint32_t size, uint64_t* mbar) {
  __vxc_issue1<1>(__VXC_P(dst), __VXC_P(mbar), __VXC_P(src), size);
}
__VX_CUDA_INL void __vxc_bulk_load(void* dst, const void* gsrc, uint32_t size, uint64_t* mbar) {
  __vxc_issue1<3>(__VXC_P(dst), __VXC_P(mbar), __VXC_P(gsrc), size);
}
__VX_CUDA_INL void __vxc_bulk_load_multicast(void* dst, const void* gsrc, uint32_t size, uint64_t* mbar,
                                             uint16_t cta_mask) {
  __vxc_issue2<3>(__VXC_P(dst), __VXC_P(mbar), __VXC_P(gsrc), size, 0, cta_mask, 0, 0);
}
__VX_CUDA_INL void __vxc_prefetch_l2(const void* gsrc, uint32_t size) {
  __vxc_issue1<4>(0, 0, __VXC_P(gsrc), size);
}
__VX_CUDA_INL void __vxc_store(uint32_t ctl, const void* src, size_t a1, size_t a2, size_t a3, int32_t c4,
                               size_t map) {
  __vxc_issue2<5>(__VXC_P(src), a1, a2, a3, ((size_t)ctl << 32) | (size_t)(uint32_t)c4, map, 0, 0);
}
__VX_CUDA_INL void __vxc_bulk_store(void* gdst, const void* src, uint32_t size) {
  __vxc_store(vx_dxa_st_ctl(DXA_ST_BULK, 0, 0, 0), src, __VXC_P(gdst), size, 0, 0, 0);
}
__VX_CUDA_INL void __vxc_reduce_global(void* gdst, const void* src, uint32_t size, uint32_t op, uint32_t type) {
  __vxc_store(vx_dxa_st_ctl(DXA_ST_RED_GLOBAL, op, type, 0), src, __VXC_P(gdst), size, 0, 0, 0);
}
__VX_CUDA_INL void __vxc_reduce_cluster(void* dst, const void* src, uint32_t size, uint64_t* mbar, uint32_t op,
                                        uint32_t type) {
  __vxc_store(vx_dxa_st_ctl(DXA_ST_RED_CLUSTER, op, type, 0), src, __VXC_P(dst), size, __VXC_P(mbar), 0, 0);
}
// The map forms of the tensor store (op ~0u) and reduction.
__VX_CUDA_INL void __vxc_store_map(uint32_t op, const vx_tensor_map_t* map, const void* src, int32_t c0, int32_t c1,
                                   int32_t c2, int32_t c3, int32_t c4) {
  const uint32_t ctl = (op == ~0u ? vx_dxa_st_ctl(DXA_ST_TENSOR, 0, 0, 0)
                                  : vx_dxa_st_ctl(DXA_ST_RED_TENSOR, op, DXA_RT_FROM_MAP, 0))
                     | (1u << DXA_ST_CTL_MAP_LSB);
  __vxc_store(ctl, src, __vxc_pair(c0, c1), __vxc_pair(c2, c3), 0, c4, __VXC_P(map));
}
// The map forms of the tensor load (ctl 0) and prefetch (ctl VX_DXA_CTL_PREFETCH, no dst or mbar).
__VX_CUDA_INL void __vxc_tload_map(uint32_t ctl, const vx_tensor_map_t* map, void* dst, uint64_t* mbar, int32_t c0,
                                   int32_t c1, int32_t c2, int32_t c3, int32_t c4, uint32_t o1, uint32_t o2,
                                   uint32_t o3, uint32_t cta_mask) {
  __vxc_issue2<2>(__VXC_P(dst), __VXC_P(mbar), __vxc_pair(c0, c1), __vxc_pair(c2, c3),
                  ((size_t)(ctl | (1u << DXA_TL_CTL_MAP_LSB)) << 32) | (size_t)(uint32_t)c4,
                  ((size_t)(o3 & 0xFFFFu) << 48) | ((size_t)(o2 & 0xFFFFu) << 32) | ((size_t)(o1 & 0xFFFFu) << 16)
                      | (size_t)(cta_mask & 0xFFFFu),
                  __VXC_P(map), 0);
}

#undef __VXC_P

// cp.async.bulk.shared::cta.global with complete_tx(size) on `bar`; a lane
// that cannot reach the DXA (lane 0 inactive) never gets here.
__VX_CUDA_INL void __vx_bulk_g2s(void* dst, const void* src, uint32_t size, uint64_t* bar) {
  __vx_dxa_each_lane([&](auto b) { __vxc_bulk_load(b(dst), b(src), b(size), b(bar)); });
}

// The per-thread copy of memcpy_async when the bulk copy does not apply:
// cp.async of the widest unit the alignment allows, tied to the barrier with
// cp.async.mbarrier.arrive (no arrival); a copy cp.async cannot make is done
// synchronously, before the caller's own arrive.
__VX_CUDA_INL void __vx_copy_chunks(char* d, const char* s, size_t n, size_t first, size_t step, uint64_t* bar) {
  const uintptr_t a = (uintptr_t)d | (uintptr_t)s | (uintptr_t)n;
  const bool async = vx_isspacep_shared(d) && __vx_is_global(s);
  if (async && (a & 15) == 0) {
    for (size_t i = first * 16; i < n; i += step * 16) vx_cp_async_16(d + i, s + i, 16);
  } else if (async && (a & 7) == 0) {
    for (size_t i = first * 8; i < n; i += step * 8) vx_cp_async_8(d + i, s + i, 8);
  } else if (async && (a & 3) == 0) {
    for (size_t i = first * 4; i < n; i += step * 4) vx_cp_async_4(d + i, s + i, 4);
  } else {
    for (size_t i = first; i < n; i += step) ((volatile char*)d)[i] = s[i];
  }
  vx_cp_async_mbar_arrive(bar);
}

__VX_CUDA_INL bool __vx_bulk_ok(const void* dst, const void* src, size_t n) {
  return n != 0 && n < (1ull << 32) && vx_isspacep_shared(dst) && __vx_is_global(src)
      && (((uintptr_t)dst | (uintptr_t)src | (uintptr_t)n) & 15) == 0;
}

// The sm_90 path: a shared destination, a global source, 16-byte aligned and
// sized, is one bulk copy whose bytes the barrier expects. memcpy_async never
// arrives: the caller arrives afterwards, as in CUDA.
__VX_CUDA_INL void __vx_memcpy_async(void* dst, const void* src, size_t n, uint64_t* bar) {
  const bool bulk = __vx_bulk_ok(dst, src, n) && (vx_active_threads() & 1u);
  const uint32_t __UNIFORM__ want = (uint32_t)vx_vote_ballot(bulk);
  if (want != 0) {
    if (bulk)
      vx_mbar_expect_tx(bar, (uint32_t)n);
    // Only the lanes that chose the bulk copy issue one.
    __vx_dxa_each_lane([&](auto b) {
      if (b(bulk))
        __vxc_bulk_load(b(dst), b(src), b((uint32_t)n), b(bar));
    });
  }
  if (!bulk)
    __vx_copy_chunks((char*)dst, (const char*)src, n, 0, 1, bar);
}

template <thread_scope Scope>
__VX_CUDA_INL void memcpy_async(void* dst, const void* src, size_t size, barrier<Scope>& bar) {
  __vx_memcpy_async(dst, src, size, bar.__vx_handle());
}

template <size_t Align, thread_scope Scope>
__VX_CUDA_INL void memcpy_async(void* dst, const void* src, aligned_size_t<Align> size, barrier<Scope>& bar) {
  __vx_memcpy_async(dst, src, (size_t)size, bar.__vx_handle());
}

// The group forms: thread rank 0 issues the bulk copy; otherwise the group's
// threads split the units of the copy by rank.
template <typename Group, typename Size, thread_scope Scope>
__VX_CUDA_INL void memcpy_async(const Group& group, void* dst, const void* src, Size size, barrier<Scope>& bar) {
  const size_t n = (size_t)size;
  if (__vx_bulk_ok(dst, src, n)) {
    if (group.thread_rank() == 0) {
      vx_mbar_expect_tx(bar.__vx_handle(), (uint32_t)n);
      __vxc_bulk_load(dst, src, (uint32_t)n, bar.__vx_handle());
    }
    return;
  }
  __vx_copy_chunks((char*)dst, (const char*)src, n, group.thread_rank(), group.num_threads(), bar.__vx_handle());
}

namespace device {

__VX_CUDA_INL uint64_t* barrier_native_handle(barrier<thread_scope_block>& bar) { return bar.__vx_handle(); }

// arrive_count arrivals after tx += transaction_count, as one op for one arrival.
__VX_CUDA_INL barrier<thread_scope_block>::arrival_token barrier_arrive_tx(barrier<thread_scope_block>& bar,
                                                                          ptrdiff_t arrive_count_update,
                                                                          ptrdiff_t transaction_count_update) {
  uint64_t* p = bar.__vx_handle();
  if (arrive_count_update == 1)
    return vx_mbar_arrive_tx(p, (uint32_t)transaction_count_update);
  vx_mbar_expect_tx(p, (uint32_t)transaction_count_update);
  return vx_mbar_arrive(p, (uint32_t)arrive_count_update);
}

__VX_CUDA_INL void barrier_expect_tx(barrier<thread_scope_block>& bar, ptrdiff_t transaction_count_update) {
  vx_mbar_expect_tx(bar.__vx_handle(), (uint32_t)transaction_count_update);
}

// cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes; no arrive.
template <typename T, size_t Align>
__VX_CUDA_INL void memcpy_async_tx(T* dst, const T* src, aligned_size_t<Align> size,
                                   barrier<thread_scope_block>& bar) {
  static_assert(Align >= 16, "memcpy_async_tx needs aligned_size_t<16> or wider");
  __vx_bulk_g2s(dst, src, (uint32_t)(size_t)size, bar.__vx_handle());
}

// The cuda::device::experimental TMA names of the CUDA programming guide.
namespace experimental {

__VX_CUDA_INL void cp_async_bulk_global_to_shared(void* dst, const void* src, uint32_t size,
                                                  barrier<thread_scope_block>& bar) {
  __vx_bulk_g2s(dst, src, size, bar.__vx_handle());
}

__VX_CUDA_INL void cp_async_bulk_shared_to_global(void* dst, const void* src, uint32_t size) {
  __vx_dxa_each_lane([&](auto b) { __vxc_bulk_store(b(dst), b(src), b(size)); });
}

__VX_CUDA_INL void cp_async_bulk_tensor_1d_global_to_shared(void* dst, const CUtensorMap* map, int c0,
                                                            barrier<thread_scope_block>& bar) {
  uint64_t* h = bar.__vx_handle();
  __vx_dxa_each_lane([&](auto b) { __vxc_tload_map(0, b(map), b(dst), b(h), b(c0), 0, 0, 0, 0, 0, 0, 0, 0); });
}
__VX_CUDA_INL void cp_async_bulk_tensor_2d_global_to_shared(void* dst, const CUtensorMap* map, int c0, int c1,
                                                            barrier<thread_scope_block>& bar) {
  uint64_t* h = bar.__vx_handle();
  __vx_dxa_each_lane([&](auto b) { __vxc_tload_map(0, b(map), b(dst), b(h), b(c0), b(c1), 0, 0, 0, 0, 0, 0, 0); });
}
__VX_CUDA_INL void cp_async_bulk_tensor_3d_global_to_shared(void* dst, const CUtensorMap* map, int c0, int c1,
                                                            int c2, barrier<thread_scope_block>& bar) {
  uint64_t* h = bar.__vx_handle();
  __vx_dxa_each_lane([&](auto b) { __vxc_tload_map(0, b(map), b(dst), b(h), b(c0), b(c1), b(c2), 0, 0, 0, 0, 0, 0); });
}
__VX_CUDA_INL void cp_async_bulk_tensor_4d_global_to_shared(void* dst, const CUtensorMap* map, int c0, int c1,
                                                            int c2, int c3, barrier<thread_scope_block>& bar) {
  uint64_t* h = bar.__vx_handle();
  __vx_dxa_each_lane([&](auto b) { __vxc_tload_map(0, b(map), b(dst), b(h), b(c0), b(c1), b(c2), b(c3), 0, 0, 0, 0, 0); });
}
__VX_CUDA_INL void cp_async_bulk_tensor_5d_global_to_shared(void* dst, const CUtensorMap* map, int c0, int c1,
                                                            int c2, int c3, int c4,
                                                            barrier<thread_scope_block>& bar) {
  uint64_t* h = bar.__vx_handle();
  __vx_dxa_each_lane([&](auto b) { __vxc_tload_map(0, b(map), b(dst), b(h), b(c0), b(c1), b(c2), b(c3), b(c4), 0, 0, 0, 0); });
}

__VX_CUDA_INL void cp_async_bulk_tensor_1d_shared_to_global(const CUtensorMap* map, int c0, const void* src) {
  __vx_dxa_each_lane([&](auto b) { __vxc_store_map(~0u, b(map), b(src), b(c0), 0, 0, 0, 0); });
}
__VX_CUDA_INL void cp_async_bulk_tensor_2d_shared_to_global(const CUtensorMap* map, int c0, int c1,
                                                            const void* src) {
  __vx_dxa_each_lane([&](auto b) { __vxc_store_map(~0u, b(map), b(src), b(c0), b(c1), 0, 0, 0); });
}
__VX_CUDA_INL void cp_async_bulk_tensor_3d_shared_to_global(const CUtensorMap* map, int c0, int c1, int c2,
                                                            const void* src) {
  __vx_dxa_each_lane([&](auto b) { __vxc_store_map(~0u, b(map), b(src), b(c0), b(c1), b(c2), 0, 0); });
}
__VX_CUDA_INL void cp_async_bulk_tensor_4d_shared_to_global(const CUtensorMap* map, int c0, int c1, int c2, int c3,
                                                            const void* src) {
  __vx_dxa_each_lane([&](auto b) { __vxc_store_map(~0u, b(map), b(src), b(c0), b(c1), b(c2), b(c3), 0); });
}
__VX_CUDA_INL void cp_async_bulk_tensor_5d_shared_to_global(const CUtensorMap* map, int c0, int c1, int c2, int c3,
                                                            int c4, const void* src) {
  __vx_dxa_each_lane([&](auto b) { __vxc_store_map(~0u, b(map), b(src), b(c0), b(c1), b(c2), b(c3), b(c4)); });
}

__VX_CUDA_INL void fence_proxy_async_shared_cta() { vx_fence_proxy_async(); }

__VX_CUDA_INL void cp_async_bulk_commit_group() { vx_bulk_commit_group(); }

template <int N>
__VX_CUDA_INL void cp_async_bulk_wait_group_read() {
  static_assert(N >= 0 && N < 32, "wait_group N is 0-31");
  vx_bulk_wait_group_read<(unsigned)N>();
}

} // namespace experimental
} // namespace device

///////////////////////////////////////////////////////////////////////////////
// cuda::ptx names (libcu++ spelling and tags)

namespace ptx {

enum class dot_space { global, cluster, shared };
enum class dot_sem { sc, acq_rel, acquire, release, relaxed };
enum class dot_scope { cta, cluster, gpu, sys };
enum class dot_op { add, min, max, inc, dec, and_op, or_op, xor_op };

template <dot_space S> struct space_t { static constexpr dot_space value = S; };
template <dot_sem S> struct sem_t { static constexpr dot_sem value = S; };
template <dot_scope S> struct scope_t { static constexpr dot_scope value = S; };
template <dot_op O> struct op_t { static constexpr dot_op value = O; };
template <int N> struct n32_t { static constexpr int value = N; };

using space_global_t  = space_t<dot_space::global>;
using space_cluster_t = space_t<dot_space::cluster>;
using space_shared_t  = space_t<dot_space::shared>;
using sem_sc_t        = sem_t<dot_sem::sc>;
using sem_acq_rel_t   = sem_t<dot_sem::acq_rel>;
using sem_acquire_t   = sem_t<dot_sem::acquire>;
using sem_release_t   = sem_t<dot_sem::release>;
using sem_relaxed_t   = sem_t<dot_sem::relaxed>;
using scope_cta_t     = scope_t<dot_scope::cta>;
using scope_cluster_t = scope_t<dot_scope::cluster>;
using scope_gpu_t     = scope_t<dot_scope::gpu>;
using scope_sys_t     = scope_t<dot_scope::sys>;
using op_add_t        = op_t<dot_op::add>;
using op_min_t        = op_t<dot_op::min>;
using op_max_t        = op_t<dot_op::max>;
using op_inc_t        = op_t<dot_op::inc>;
using op_dec_t        = op_t<dot_op::dec>;
using op_and_op_t     = op_t<dot_op::and_op>;
using op_or_op_t      = op_t<dot_op::or_op>;
using op_xor_op_t     = op_t<dot_op::xor_op>;

constexpr space_global_t  space_global{};
constexpr space_cluster_t space_cluster{};
constexpr space_shared_t  space_shared{};
constexpr sem_sc_t        sem_sc{};
constexpr sem_acq_rel_t   sem_acq_rel{};
constexpr sem_acquire_t   sem_acquire{};
constexpr sem_release_t   sem_release{};
constexpr sem_relaxed_t   sem_relaxed{};
constexpr scope_cta_t     scope_cta{};
constexpr scope_cluster_t scope_cluster{};
constexpr scope_gpu_t     scope_gpu{};
constexpr scope_sys_t     scope_sys{};
constexpr op_add_t        op_add{};
constexpr op_min_t        op_min{};
constexpr op_max_t        op_max{};
constexpr op_inc_t        op_inc{};
constexpr op_dec_t        op_dec{};
constexpr op_and_op_t     op_and_op{};
constexpr op_or_op_t      op_or_op{};
constexpr op_xor_op_t     op_xor_op{};

#define __VX_PTX_CTA_OR_CLUSTER(Scope) \
  static_assert((Scope) == dot_scope::cta || (Scope) == dot_scope::cluster, "scope is .cta or .cluster")

// mbarrier.init / arrive / expect_tx / test_wait / try_wait

__VX_CUDA_INL void mbarrier_init(uint64_t* addr, const uint32_t& count) { vx_mbar_init(addr, count); }

__VX_CUDA_INL uint64_t mbarrier_arrive(uint64_t* addr) { return vx_mbar_arrive(addr, 1); }
__VX_CUDA_INL uint64_t mbarrier_arrive(uint64_t* addr, const uint32_t& count) { return vx_mbar_arrive(addr, count); }

template <dot_scope Scope>
__VX_CUDA_INL uint64_t mbarrier_arrive(sem_release_t, scope_t<Scope>, space_shared_t, uint64_t* addr,
                                       const uint32_t& count = 1) {
  __VX_PTX_CTA_OR_CLUSTER(Scope);
  if constexpr (Scope == dot_scope::cluster) return vx_mbar_arrive_release_cluster(addr, count);
  else return vx_mbar_arrive(addr, count);
}

__VX_CUDA_INL void mbarrier_arrive(sem_release_t, scope_cluster_t, space_cluster_t, uint64_t* addr,
                                   const uint32_t& count = 1) {
  (void)vx_mbar_arrive_release_cluster(addr, count);
}

template <dot_scope Scope>
__VX_CUDA_INL uint64_t mbarrier_arrive(sem_relaxed_t, scope_t<Scope>, space_shared_t, uint64_t* addr,
                                       const uint32_t& count = 1) {
  __VX_PTX_CTA_OR_CLUSTER(Scope);
  return vx_mbar_arrive_relaxed(addr, count);
}

__VX_CUDA_INL void mbarrier_arrive(sem_relaxed_t, scope_cluster_t, space_cluster_t, uint64_t* addr,
                                   const uint32_t& count = 1) {
  (void)vx_mbar_arrive_relaxed(addr, count);
}

template <dot_scope Scope>
__VX_CUDA_INL uint64_t mbarrier_arrive_expect_tx(sem_release_t, scope_t<Scope>, space_shared_t, uint64_t* addr,
                                                 const uint32_t& tx_count) {
  __VX_PTX_CTA_OR_CLUSTER(Scope);
  if constexpr (Scope == dot_scope::cluster) vx_fence_cluster();
  return vx_mbar_arrive_tx(addr, tx_count);
}

__VX_CUDA_INL void mbarrier_arrive_expect_tx(sem_release_t, scope_cluster_t, space_cluster_t, uint64_t* addr,
                                             const uint32_t& tx_count) {
  vx_fence_cluster();
  (void)vx_mbar_arrive_tx(addr, tx_count);
}

template <dot_scope Scope>
__VX_CUDA_INL uint64_t mbarrier_arrive_expect_tx(sem_relaxed_t, scope_t<Scope>, space_shared_t, uint64_t* addr,
                                                 const uint32_t& tx_count) {
  __VX_PTX_CTA_OR_CLUSTER(Scope);
  return vx_mbar_arrive_tx_relaxed(addr, tx_count);
}

__VX_CUDA_INL void mbarrier_arrive_expect_tx(sem_relaxed_t, scope_cluster_t, space_cluster_t, uint64_t* addr,
                                             const uint32_t& tx_count) {
  (void)vx_mbar_arrive_tx_relaxed(addr, tx_count);
}

template <dot_scope Scope, dot_space Space>
__VX_CUDA_INL void mbarrier_expect_tx(sem_relaxed_t, scope_t<Scope>, space_t<Space>, uint64_t* addr,
                                      uint32_t tx_count) {
  __VX_PTX_CTA_OR_CLUSTER(Scope);
  static_assert(Space == dot_space::shared || Space == dot_space::cluster, "space is .shared::cta or .shared::cluster");
  (void)vx_mbar_expect_tx(addr, tx_count);
}

__VX_CUDA_INL bool mbarrier_test_wait(uint64_t* addr, const uint64_t& state) { return vx_mbar_test_wait(addr, state); }
__VX_CUDA_INL bool mbarrier_test_wait_parity(uint64_t* addr, const uint32_t& parity) {
  return vx_mbar_test_wait_parity(addr, parity);
}
__VX_CUDA_INL bool mbarrier_try_wait(uint64_t* addr, const uint64_t& state) { return vx_mbar_try_wait(addr, state); }
__VX_CUDA_INL bool mbarrier_try_wait(uint64_t* addr, const uint64_t& state, const uint32_t& hint) {
  return vx_mbar_try_wait_hint(addr, state, hint);
}
__VX_CUDA_INL bool mbarrier_try_wait_parity(uint64_t* addr, const uint32_t& parity) {
  return vx_mbar_try_wait_parity(addr, parity);
}
__VX_CUDA_INL bool mbarrier_try_wait_parity(uint64_t* addr, const uint32_t& parity, const uint32_t& hint) {
  return vx_mbar_try_wait_parity_hint(addr, parity, hint);
}

// .acquire.cluster: the plain wait, then the cluster fence once it succeeds.
template <dot_scope Scope>
__VX_CUDA_INL bool __vx_acquire_done(bool done) {
  if constexpr (Scope == dot_scope::cluster) {
    if (done)
      vx_fence_cluster();
  }
  return done;
}

template <dot_scope Scope>
__VX_CUDA_INL bool mbarrier_test_wait(sem_acquire_t, scope_t<Scope>, uint64_t* addr, const uint64_t& state) {
  __VX_PTX_CTA_OR_CLUSTER(Scope);
  return __vx_acquire_done<Scope>(vx_mbar_test_wait(addr, state));
}
template <dot_scope Scope>
__VX_CUDA_INL bool mbarrier_test_wait(sem_relaxed_t, scope_t<Scope>, uint64_t* addr, const uint64_t& state) {
  __VX_PTX_CTA_OR_CLUSTER(Scope);
  return vx_mbar_test_wait_relaxed(addr, state);
}
template <dot_scope Scope>
__VX_CUDA_INL bool mbarrier_test_wait_parity(sem_acquire_t, scope_t<Scope>, uint64_t* addr, const uint32_t& parity) {
  __VX_PTX_CTA_OR_CLUSTER(Scope);
  return __vx_acquire_done<Scope>(vx_mbar_test_wait_parity(addr, parity));
}
template <dot_scope Scope>
__VX_CUDA_INL bool mbarrier_test_wait_parity(sem_relaxed_t, scope_t<Scope>, uint64_t* addr, const uint32_t& parity) {
  __VX_PTX_CTA_OR_CLUSTER(Scope);
  return vx_mbar_test_wait_parity_relaxed(addr, parity);
}
template <dot_scope Scope>
__VX_CUDA_INL bool mbarrier_try_wait(sem_acquire_t, scope_t<Scope>, uint64_t* addr, const uint64_t& state) {
  __VX_PTX_CTA_OR_CLUSTER(Scope);
  return __vx_acquire_done<Scope>(vx_mbar_try_wait(addr, state));
}
template <dot_scope Scope>
__VX_CUDA_INL bool mbarrier_try_wait(sem_acquire_t, scope_t<Scope>, uint64_t* addr, const uint64_t& state,
                                     const uint32_t& hint) {
  __VX_PTX_CTA_OR_CLUSTER(Scope);
  return __vx_acquire_done<Scope>(vx_mbar_try_wait_hint(addr, state, hint));
}
template <dot_scope Scope>
__VX_CUDA_INL bool mbarrier_try_wait(sem_relaxed_t, scope_t<Scope>, uint64_t* addr, const uint64_t& state) {
  __VX_PTX_CTA_OR_CLUSTER(Scope);
  return vx_mbar_try_wait_relaxed(addr, state);
}
template <dot_scope Scope>
__VX_CUDA_INL bool mbarrier_try_wait(sem_relaxed_t, scope_t<Scope>, uint64_t* addr, const uint64_t& state,
                                     const uint32_t& hint) {
  __VX_PTX_CTA_OR_CLUSTER(Scope);
  return vx_mbar_try_wait_hint(addr, state, hint);
}
template <dot_scope Scope>
__VX_CUDA_INL bool mbarrier_try_wait_parity(sem_acquire_t, scope_t<Scope>, uint64_t* addr, const uint32_t& parity) {
  __VX_PTX_CTA_OR_CLUSTER(Scope);
  return __vx_acquire_done<Scope>(vx_mbar_try_wait_parity(addr, parity));
}
template <dot_scope Scope>
__VX_CUDA_INL bool mbarrier_try_wait_parity(sem_acquire_t, scope_t<Scope>, uint64_t* addr, const uint32_t& parity,
                                            const uint32_t& hint) {
  __VX_PTX_CTA_OR_CLUSTER(Scope);
  return __vx_acquire_done<Scope>(vx_mbar_try_wait_parity_hint(addr, parity, hint));
}
template <dot_scope Scope>
__VX_CUDA_INL bool mbarrier_try_wait_parity(sem_relaxed_t, scope_t<Scope>, uint64_t* addr, const uint32_t& parity) {
  __VX_PTX_CTA_OR_CLUSTER(Scope);
  return vx_mbar_try_wait_parity_relaxed(addr, parity);
}
template <dot_scope Scope>
__VX_CUDA_INL bool mbarrier_try_wait_parity(sem_relaxed_t, scope_t<Scope>, uint64_t* addr, const uint32_t& parity,
                                            const uint32_t& hint) {
  __VX_PTX_CTA_OR_CLUSTER(Scope);
  return vx_mbar_try_wait_parity_hint(addr, parity, hint);
}

// Fences

__VX_CUDA_INL void fence_proxy_async() { vx_fence_proxy_async_all(); }

template <dot_space Space>
__VX_CUDA_INL void fence_proxy_async(space_t<Space>) {
  if constexpr (Space == dot_space::global) vx_fence_proxy_async_global();
  else vx_fence_proxy_async();
}

__VX_CUDA_INL void fence_mbarrier_init(sem_release_t, scope_cluster_t) { vx_fence_mbarrier_init_release_cluster(); }

template <dot_scope Scope>
__VX_CUDA_INL void fence_proxy_tensormap_generic(sem_release_t, scope_t<Scope>) {
  vx_fence_proxy_tensormap_release();
}

template <int N32, dot_scope Scope>
__VX_CUDA_INL void fence_proxy_tensormap_generic(sem_acquire_t, scope_t<Scope>, const void* addr, n32_t<N32>) {
  static_assert(N32 == 128, "a tensor map is 128 bytes");
  const vx_tensor_map_t* map = (const vx_tensor_map_t*)addr;
  __vx_dxa_each_lane([&](auto b) { vx_fence_proxy_tensormap_acquire(b(map)); });
}

// Bulk copies (cp.async.bulk)

// .shared::cluster.global and .shared::cta.global with complete_tx.
template <dot_space Dst>
__VX_CUDA_INL void cp_async_bulk(space_t<Dst>, space_global_t, void* dstMem, const void* srcMem,
                                 const uint32_t& size, uint64_t* smem_bar) {
  static_assert(Dst == dot_space::cluster || Dst == dot_space::shared, "destination is shared memory");
  const uint32_t n = size;
  __vx_dxa_each_lane([&](auto b) { __vxc_bulk_load(b(dstMem), b(srcMem), b(n), b(smem_bar)); });
}

__VX_CUDA_INL void cp_async_bulk(space_cluster_t, space_global_t, void* dstMem, const void* srcMem,
                                 const uint32_t& size, uint64_t* smem_bar, const uint16_t& ctaMask) {
  const uint32_t n = size;
  const uint16_t m = ctaMask;
  __vx_dxa_each_lane([&](auto b) { __vxc_bulk_load_multicast(b(dstMem), b(srcMem), b(n), b(smem_bar), b(m)); });
}

__VX_CUDA_INL void cp_async_bulk(space_cluster_t, space_shared_t, void* dstMem, const void* srcMem,
                                 const uint32_t& size, uint64_t* rdsmem_bar) {
  const uint32_t n = size;
  __vx_dxa_each_lane([&](auto b) { __vxc_bulk_copy(b(dstMem), b(srcMem), b(n), b(rdsmem_bar)); });
}

__VX_CUDA_INL void cp_async_bulk(space_global_t, space_shared_t, void* dstMem, const void* srcMem,
                                 const uint32_t& size) {
  const uint32_t n = size;
  __vx_dxa_each_lane([&](auto b) { __vxc_bulk_store(b(dstMem), b(srcMem), b(n)); });
}

// Tensor loads, multicast loads and stores (tile mode; the map's mode rules).

template <int N>
__VX_CUDA_INL void __vx_tload(const void* tmap, void* dst, uint64_t* bar, const int32_t (&c)[N], uint32_t mask,
                              bool mcast) {
  static_assert(N >= 1 && N <= 5, "1d to 5d");
  const vx_tensor_map_t* map = (const vx_tensor_map_t*)tmap;
  const int32_t c0 = c[0], c1 = N > 1 ? c[N > 1 ? 1 : 0] : 0, c2 = N > 2 ? c[N > 2 ? 2 : 0] : 0,
                c3 = N > 3 ? c[N > 3 ? 3 : 0] : 0, c4 = N > 4 ? c[N > 4 ? 4 : 0] : 0;
  const uint16_t m = (uint16_t)mask;
  __vx_dxa_each_lane([&](auto b) {
    if (mcast) {
      if constexpr (N == 1) __vxc_tload_map(0, b(map), b(dst), b(bar), b(c0), 0, 0, 0, 0, 0, 0, 0, b(m));
      else if constexpr (N == 2) __vxc_tload_map(0, b(map), b(dst), b(bar), b(c0), b(c1), 0, 0, 0, 0, 0, 0, b(m));
      else if constexpr (N == 3) __vxc_tload_map(0, b(map), b(dst), b(bar), b(c0), b(c1), b(c2), 0, 0, 0, 0, 0, b(m));
      else if constexpr (N == 4) __vxc_tload_map(0, b(map), b(dst), b(bar), b(c0), b(c1), b(c2), b(c3), 0, 0, 0, 0, b(m));
      else __vxc_tload_map(0, b(map), b(dst), b(bar), b(c0), b(c1), b(c2), b(c3), b(c4), 0, 0, 0, b(m));
    } else {
      if constexpr (N == 1) __vxc_tload_map(0, b(map), b(dst), b(bar), b(c0), 0, 0, 0, 0, 0, 0, 0, 0);
      else if constexpr (N == 2) __vxc_tload_map(0, b(map), b(dst), b(bar), b(c0), b(c1), 0, 0, 0, 0, 0, 0, 0);
      else if constexpr (N == 3) __vxc_tload_map(0, b(map), b(dst), b(bar), b(c0), b(c1), b(c2), 0, 0, 0, 0, 0, 0);
      else if constexpr (N == 4) __vxc_tload_map(0, b(map), b(dst), b(bar), b(c0), b(c1), b(c2), b(c3), 0, 0, 0, 0, 0);
      else __vxc_tload_map(0, b(map), b(dst), b(bar), b(c0), b(c1), b(c2), b(c3), b(c4), 0, 0, 0, 0);
    }
  });
}

template <dot_space Dst, int N>
__VX_CUDA_INL void cp_async_bulk_tensor(space_t<Dst>, space_global_t, void* dstMem, const void* tensorMap,
                                        const int32_t (&tensorCoords)[N], uint64_t* smem_bar) {
  static_assert(Dst == dot_space::cluster || Dst == dot_space::shared, "destination is shared memory");
  __vx_tload<N>(tensorMap, dstMem, smem_bar, tensorCoords, 0, false);
}

template <int N>
__VX_CUDA_INL void cp_async_bulk_tensor(space_cluster_t, space_global_t, void* dstMem, const void* tensorMap,
                                        const int32_t (&tensorCoords)[N], uint64_t* smem_bar,
                                        const uint16_t& ctaMask) {
  __vx_tload<N>(tensorMap, dstMem, smem_bar, tensorCoords, ctaMask, true);
}

template <int N>
__VX_CUDA_INL void cp_async_bulk_tensor(space_global_t, space_shared_t, const void* tensorMap,
                                        const int32_t (&tensorCoords)[N], const void* srcMem) {
  static_assert(N >= 1 && N <= 5, "1d to 5d");
  const vx_tensor_map_t* map = (const vx_tensor_map_t*)tensorMap;
  const int32_t* c = tensorCoords;
  const int32_t c0 = c[0], c1 = N > 1 ? c[N > 1 ? 1 : 0] : 0, c2 = N > 2 ? c[N > 2 ? 2 : 0] : 0,
                c3 = N > 3 ? c[N > 3 ? 3 : 0] : 0, c4 = N > 4 ? c[N > 4 ? 4 : 0] : 0;
  __vx_dxa_each_lane([&](auto b) {
    if constexpr (N == 1) __vxc_store_map(~0u, b(map), b(srcMem), b(c0), 0, 0, 0, 0);
    else if constexpr (N == 2) __vxc_store_map(~0u, b(map), b(srcMem), b(c0), b(c1), 0, 0, 0);
    else if constexpr (N == 3) __vxc_store_map(~0u, b(map), b(srcMem), b(c0), b(c1), b(c2), 0, 0);
    else if constexpr (N == 4) __vxc_store_map(~0u, b(map), b(srcMem), b(c0), b(c1), b(c2), b(c3), 0);
    else __vxc_store_map(~0u, b(map), b(srcMem), b(c0), b(c1), b(c2), b(c3), b(c4));
  });
}

// No libcu++ name: cp.async.bulk.tensor.{3,4,5}d.shared::cluster.global.im2col
// with the PTX im2colOffsets (N - 2 of them), plain and .multicast::cluster.
template <int N>
__VX_CUDA_INL void __vx_tload_im2col(const void* tmap, void* dst, uint64_t* bar, const int32_t (&c)[N],
                                     const uint16_t (&o)[N - 2], uint32_t mask, bool mcast) {
  static_assert(N >= 3 && N <= 5, "im2col is 3d to 5d");
  const vx_tensor_map_t* map = (const vx_tensor_map_t*)tmap;
  const int32_t c0 = c[0], c1 = c[1], c2 = c[2], c3 = N > 3 ? c[N > 3 ? 3 : 0] : 0, c4 = N > 4 ? c[N > 4 ? 4 : 0] : 0;
  const uint16_t o1 = o[0], o2 = N > 3 ? o[N > 3 ? 1 : 0] : 0, o3 = N > 4 ? o[N > 4 ? 2 : 0] : 0;
  const uint16_t m = (uint16_t)mask;
  __vx_dxa_each_lane([&](auto b) {
    if (mcast) {
      if constexpr (N == 3) __vxc_tload_map(0, b(map), b(dst), b(bar), b(c0), b(c1), b(c2), 0, 0, b(o1), 0, 0, b(m));
      else if constexpr (N == 4)
        __vxc_tload_map(0, b(map), b(dst), b(bar), b(c0), b(c1), b(c2), b(c3), 0, b(o1), b(o2), 0, b(m));
      else
        __vxc_tload_map(0, b(map), b(dst), b(bar), b(c0), b(c1), b(c2), b(c3), b(c4), b(o1), b(o2), b(o3), b(m));
    } else {
      if constexpr (N == 3) __vxc_tload_map(0, b(map), b(dst), b(bar), b(c0), b(c1), b(c2), 0, 0, b(o1), 0, 0, 0);
      else if constexpr (N == 4) __vxc_tload_map(0, b(map), b(dst), b(bar), b(c0), b(c1), b(c2), b(c3), 0, b(o1), b(o2), 0, 0);
      else
        __vxc_tload_map(0, b(map), b(dst), b(bar), b(c0), b(c1), b(c2), b(c3), b(c4), b(o1), b(o2), b(o3), 0);
    }
  });
}

template <dot_space Dst, int N>
__VX_CUDA_INL void cp_async_bulk_tensor_im2col(space_t<Dst>, space_global_t, void* dstMem, const void* tensorMap,
                                               const int32_t (&tensorCoords)[N],
                                               const uint16_t (&im2colOffsets)[N - 2], uint64_t* smem_bar) {
  static_assert(Dst == dot_space::cluster || Dst == dot_space::shared, "destination is shared memory");
  __vx_tload_im2col<N>(tensorMap, dstMem, smem_bar, tensorCoords, im2colOffsets, 0, false);
}

template <int N>
__VX_CUDA_INL void cp_async_bulk_tensor_im2col(space_cluster_t, space_global_t, void* dstMem, const void* tensorMap,
                                               const int32_t (&tensorCoords)[N],
                                               const uint16_t (&im2colOffsets)[N - 2], uint64_t* smem_bar,
                                               const uint16_t& ctaMask) {
  __vx_tload_im2col<N>(tensorMap, dstMem, smem_bar, tensorCoords, im2colOffsets, ctaMask, true);
}

// Bulk reductions (cp.reduce.async.bulk)

template <dot_op Op>
constexpr uint32_t __vx_red_op() {
  if constexpr (Op == dot_op::add) return DXA_RED_ADD;
  else if constexpr (Op == dot_op::min) return DXA_RED_MIN;
  else if constexpr (Op == dot_op::max) return DXA_RED_MAX;
  else if constexpr (Op == dot_op::inc) return DXA_RED_INC;
  else if constexpr (Op == dot_op::dec) return DXA_RED_DEC;
  else if constexpr (Op == dot_op::and_op) return DXA_RED_AND;
  else if constexpr (Op == dot_op::or_op) return DXA_RED_OR;
  else return DXA_RED_XOR;
}

template <typename T> struct __vx_is_int { static constexpr bool value = false; };
template <> struct __vx_is_int<int> { static constexpr bool value = true; };
template <> struct __vx_is_int<unsigned> { static constexpr bool value = true; };
template <> struct __vx_is_int<long> { static constexpr bool value = true; };
template <> struct __vx_is_int<unsigned long> { static constexpr bool value = true; };
template <> struct __vx_is_int<long long> { static constexpr bool value = true; };
template <> struct __vx_is_int<unsigned long long> { static constexpr bool value = true; };

template <typename T, typename U> struct __vx_same { static constexpr bool value = false; };
template <typename T> struct __vx_same<T, T> { static constexpr bool value = true; };

template <typename T> struct __vx_is_signed { static constexpr bool value = T(-1) < T(0); };

// The DXA element type of a reduction, or ~0u for a pair PTX does not have.
// s64 add runs as u64 add, as libcu++ emulates it.
template <dot_op Op, typename T, bool Cluster>
constexpr uint32_t __vx_red_type() {
  constexpr bool bitwise = Op == dot_op::and_op || Op == dot_op::or_op || Op == dot_op::xor_op;
  constexpr bool arith = Op == dot_op::add || Op == dot_op::min || Op == dot_op::max;
  constexpr bool incdec = Op == dot_op::inc || Op == dot_op::dec;
  if constexpr (bitwise) {
    if constexpr (sizeof(T) == 4) return DXA_RT_U32;
    else if constexpr (sizeof(T) == 8 && !Cluster) return DXA_RT_U64;
    else return ~0u;
  } else if constexpr (__vx_is_int<T>::value) {
    constexpr bool sg = __vx_is_signed<T>::value;
    if constexpr (sizeof(T) == 4 && incdec) return sg ? ~0u : DXA_RT_U32;
    else if constexpr (sizeof(T) == 4 && arith) return sg ? DXA_RT_S32 : DXA_RT_U32;
    else if constexpr (sizeof(T) == 8 && Op == dot_op::add) return DXA_RT_U64;
    else if constexpr (sizeof(T) == 8 && arith && !Cluster) return sg ? DXA_RT_S64 : DXA_RT_U64;
    else return ~0u;
  } else if constexpr (Cluster) {
    return ~0u;
  } else if constexpr (__vx_same<T, float>::value && Op == dot_op::add) {
    return DXA_RT_F32;
  } else if constexpr (__vx_same<T, double>::value && Op == dot_op::add) {
    return DXA_RT_F64;
  } else if constexpr (arith && sizeof(T) == 2) {
    return __vx_same<T, __half>::value ? DXA_RT_F16 : __vx_same<T, __nv_bfloat16>::value ? DXA_RT_BF16 : ~0u;
  } else {
    return ~0u;
  }
}

template <dot_op Op, typename T>
__VX_CUDA_INL void cp_reduce_async_bulk(space_global_t, space_shared_t, op_t<Op>, T* dstMem, const T* srcMem,
                                        uint32_t size) {
  constexpr uint32_t rt = __vx_red_type<Op, T, false>();
  static_assert(rt != ~0u, "cp.reduce.async.bulk.global has no such op and type");
  __vx_dxa_each_lane([&](auto b) {
    __vxc_reduce_global((void*)b(dstMem), (const void*)b(srcMem), b(size), __vx_red_op<Op>(), rt);
  });
}

template <dot_op Op, typename T>
__VX_CUDA_INL void cp_reduce_async_bulk(space_cluster_t, space_shared_t, op_t<Op>, T* dstMem, const T* srcMem,
                                        uint32_t size, uint64_t* rdsmem_bar) {
  constexpr uint32_t rt = __vx_red_type<Op, T, true>();
  static_assert(rt != ~0u, "cp.reduce.async.bulk.shared::cluster has no such op and type");
  __vx_dxa_each_lane([&](auto b) {
    __vxc_reduce_cluster((void*)b(dstMem), (const void*)b(srcMem), b(size), b(rdsmem_bar), __vx_red_op<Op>(), rt);
  });
}

template <dot_op Op, int N>
__VX_CUDA_INL void cp_reduce_async_bulk_tensor(space_global_t, space_shared_t, op_t<Op>, const void* tensorMap,
                                               const int32_t (&tensorCoords)[N], const void* srcMem) {
  static_assert(N >= 1 && N <= 5, "1d to 5d");
  const vx_tensor_map_t* map = (const vx_tensor_map_t*)tensorMap;
  const int32_t* c = tensorCoords;
  const int32_t c0 = c[0], c1 = N > 1 ? c[N > 1 ? 1 : 0] : 0, c2 = N > 2 ? c[N > 2 ? 2 : 0] : 0,
                c3 = N > 3 ? c[N > 3 ? 3 : 0] : 0, c4 = N > 4 ? c[N > 4 ? 4 : 0] : 0;
  constexpr uint32_t op = __vx_red_op<Op>();
  __vx_dxa_each_lane([&](auto b) {
    if constexpr (N == 1) __vxc_store_map(op, b(map), b(srcMem), b(c0), 0, 0, 0, 0);
    else if constexpr (N == 2) __vxc_store_map(op, b(map), b(srcMem), b(c0), b(c1), 0, 0, 0);
    else if constexpr (N == 3) __vxc_store_map(op, b(map), b(srcMem), b(c0), b(c1), b(c2), 0, 0);
    else if constexpr (N == 4) __vxc_store_map(op, b(map), b(srcMem), b(c0), b(c1), b(c2), b(c3), 0);
    else __vxc_store_map(op, b(map), b(srcMem), b(c0), b(c1), b(c2), b(c3), b(c4));
  });
}

// Bulk async-groups (per warp)

__VX_CUDA_INL void cp_async_bulk_commit_group() { vx_bulk_commit_group(); }

template <int N32>
__VX_CUDA_INL void cp_async_bulk_wait_group(n32_t<N32>) {
  static_assert(N32 >= 0 && N32 < 32, "wait_group N is 0-31");
  vx_bulk_wait_group<(unsigned)N32>();
}

template <int N32>
__VX_CUDA_INL void cp_async_bulk_wait_group_read(n32_t<N32>) {
  static_assert(N32 >= 0 && N32 < 32, "wait_group.read N is 0-31");
  vx_bulk_wait_group_read<(unsigned)N32>();
}

// L2 prefetches; the cache policy is dropped.

__VX_CUDA_INL void cp_async_bulk_prefetch(const void* srcMem, uint32_t size, uint64_t cache_policy = 0) {
  (void)cache_policy;
  __vx_dxa_each_lane([&](auto b) { __vxc_prefetch_l2(b(srcMem), b(size)); });
}

__VX_CUDA_INL void cp_async_bulk_prefetch_L2_evict_last(const void* srcMem, uint32_t size) {
  cp_async_bulk_prefetch(srcMem, size);
}

template <int N>
__VX_CUDA_INL void cp_async_bulk_prefetch_tensor(space_global_t, const void* tensorMap,
                                                 const int32_t (&tensorCoords)[N], uint64_t cache_policy = 0) {
  static_assert(N >= 1 && N <= 5, "1d to 5d");
  (void)cache_policy;
  const vx_tensor_map_t* map = (const vx_tensor_map_t*)tensorMap;
  const int32_t* c = tensorCoords;
  const int32_t c0 = c[0], c1 = N > 1 ? c[N > 1 ? 1 : 0] : 0, c2 = N > 2 ? c[N > 2 ? 2 : 0] : 0,
                c3 = N > 3 ? c[N > 3 ? 3 : 0] : 0, c4 = N > 4 ? c[N > 4 ? 4 : 0] : 0;
  __vx_dxa_each_lane([&](auto b) {
    if constexpr (N == 1) __vxc_tload_map(VX_DXA_CTL_PREFETCH, b(map), nullptr, nullptr, b(c0), 0, 0, 0, 0, 0, 0, 0, 0);
    else if constexpr (N == 2) __vxc_tload_map(VX_DXA_CTL_PREFETCH, b(map), nullptr, nullptr, b(c0), b(c1), 0, 0, 0, 0, 0, 0, 0);
    else if constexpr (N == 3) __vxc_tload_map(VX_DXA_CTL_PREFETCH, b(map), nullptr, nullptr, b(c0), b(c1), b(c2), 0, 0, 0, 0, 0, 0);
    else if constexpr (N == 4) __vxc_tload_map(VX_DXA_CTL_PREFETCH, b(map), nullptr, nullptr, b(c0), b(c1), b(c2), b(c3), 0, 0, 0, 0, 0);
    else __vxc_tload_map(VX_DXA_CTL_PREFETCH, b(map), nullptr, nullptr, b(c0), b(c1), b(c2), b(c3), b(c4), 0, 0, 0, 0);
  });
}

template <int N>
__VX_CUDA_INL void cp_async_bulk_prefetch_tensor_L2_evict_last(space_global_t g, const void* tensorMap,
                                                               const int32_t (&tensorCoords)[N]) {
  cp_async_bulk_prefetch_tensor<N>(g, tensorMap, tensorCoords);
}

// No libcu++ name: cp.async.bulk.prefetch.tensor.{3,4,5}d.L2.global.im2col.
template <int N>
__VX_CUDA_INL void cp_async_bulk_prefetch_tensor_im2col(space_global_t, const void* tensorMap,
                                                        const int32_t (&tensorCoords)[N],
                                                        const uint16_t (&im2colOffsets)[N - 2]) {
  static_assert(N >= 3 && N <= 5, "im2col is 3d to 5d");
  const vx_tensor_map_t* map = (const vx_tensor_map_t*)tensorMap;
  const int32_t* c = tensorCoords;
  const int32_t c0 = c[0], c1 = c[1], c2 = c[2], c3 = N > 3 ? c[N > 3 ? 3 : 0] : 0, c4 = N > 4 ? c[N > 4 ? 4 : 0] : 0;
  const uint16_t o1 = im2colOffsets[0], o2 = N > 3 ? im2colOffsets[N > 3 ? 1 : 0] : 0,
                 o3 = N > 4 ? im2colOffsets[N > 4 ? 2 : 0] : 0;
  __vx_dxa_each_lane([&](auto b) {
    if constexpr (N == 3) __vxc_tload_map(VX_DXA_CTL_PREFETCH, b(map), nullptr, nullptr, b(c0), b(c1), b(c2), 0, 0, b(o1), 0, 0, 0);
    else if constexpr (N == 4) __vxc_tload_map(VX_DXA_CTL_PREFETCH, b(map), nullptr, nullptr, b(c0), b(c1), b(c2), b(c3), 0, b(o1), b(o2), 0, 0);
    else __vxc_tload_map(VX_DXA_CTL_PREFETCH, b(map), nullptr, nullptr, b(c0), b(c1), b(c2), b(c3), b(c4), b(o1), b(o2), b(o3), 0);
  });
}

__VX_CUDA_INL void prefetch_tensormap(const void* addr) {
  const vx_tensor_map_t* map = (const vx_tensor_map_t*)addr;
  __vx_dxa_each_lane([&](auto b) { vx_prefetch_tensormap(b(map)); });
}

// Cluster

__VX_CUDA_INL uint32_t getctarank(space_cluster_t, const void* addr) { return vx_getctarank(addr); }

// st.async .b32/.b64, .v2.b32/.v2.b64 and .v4.b32.
template <typename T>
__VX_CUDA_INL void st_async(T* addr, const T& value, uint64_t* remote_bar) {
  static_assert(sizeof(T) == 4 || sizeof(T) == 8, "st.async .b32 or .b64");
  if constexpr (sizeof(T) == 4) vx_st_async_u32(addr, __vx_cuda_bit_cast<uint32_t>(value), remote_bar);
  else vx_st_async_u64(addr, __vx_cuda_bit_cast<uint64_t>(value), remote_bar);
}

template <typename T>
__VX_CUDA_INL void st_async(T* addr, const T (&value)[2], uint64_t* remote_bar) {
  static_assert(sizeof(T) == 4 || sizeof(T) == 8, "st.async .v2.b32 or .v2.b64");
  if constexpr (sizeof(T) == 4)
    vx_st_async_v2_u32(addr, __vx_cuda_bit_cast<uint32_t>(value[0]), __vx_cuda_bit_cast<uint32_t>(value[1]), remote_bar);
  else
    vx_st_async_v2_u64(addr, __vx_cuda_bit_cast<uint64_t>(value[0]), __vx_cuda_bit_cast<uint64_t>(value[1]), remote_bar);
}

template <typename T>
__VX_CUDA_INL void st_async(T* addr, const T (&value)[4], uint64_t* remote_bar) {
  static_assert(sizeof(T) == 4, "st.async .v4.b32");
  vx_st_async_v4_u32(addr, __vx_cuda_bit_cast<uint32_t>(value[0]), __vx_cuda_bit_cast<uint32_t>(value[1]),
                     __vx_cuda_bit_cast<uint32_t>(value[2]), __vx_cuda_bit_cast<uint32_t>(value[3]), remote_bar);
}

__VX_CUDA_INL void red_async(op_inc_t, uint32_t* dest, const uint32_t& value, uint64_t* remote_bar) {
  vx_red_async_inc_u32(dest, value, remote_bar);
}
__VX_CUDA_INL void red_async(op_dec_t, uint32_t* dest, const uint32_t& value, uint64_t* remote_bar) {
  vx_red_async_dec_u32(dest, value, remote_bar);
}
__VX_CUDA_INL void red_async(op_min_t, uint32_t* dest, const uint32_t& value, uint64_t* remote_bar) {
  vx_red_async_min_u32(dest, value, remote_bar);
}
__VX_CUDA_INL void red_async(op_max_t, uint32_t* dest, const uint32_t& value, uint64_t* remote_bar) {
  vx_red_async_max_u32(dest, value, remote_bar);
}
__VX_CUDA_INL void red_async(op_add_t, uint32_t* dest, const uint32_t& value, uint64_t* remote_bar) {
  vx_red_async_add_u32(dest, value, remote_bar);
}
__VX_CUDA_INL void red_async(op_min_t, int32_t* dest, const int32_t& value, uint64_t* remote_bar) {
  vx_red_async_min_s32(dest, value, remote_bar);
}
__VX_CUDA_INL void red_async(op_max_t, int32_t* dest, const int32_t& value, uint64_t* remote_bar) {
  vx_red_async_max_s32(dest, value, remote_bar);
}
__VX_CUDA_INL void red_async(op_add_t, int32_t* dest, const int32_t& value, uint64_t* remote_bar) {
  vx_red_async_add_s32(dest, value, remote_bar);
}
template <typename B32>
__VX_CUDA_INL void red_async(op_and_op_t, B32* dest, const B32& value, uint64_t* remote_bar) {
  static_assert(sizeof(B32) == 4, "red.async .and.b32");
  vx_red_async_and_b32(dest, __vx_cuda_bit_cast<uint32_t>(value), remote_bar);
}
template <typename B32>
__VX_CUDA_INL void red_async(op_or_op_t, B32* dest, const B32& value, uint64_t* remote_bar) {
  static_assert(sizeof(B32) == 4, "red.async .or.b32");
  vx_red_async_or_b32(dest, __vx_cuda_bit_cast<uint32_t>(value), remote_bar);
}
template <typename B32>
__VX_CUDA_INL void red_async(op_xor_op_t, B32* dest, const B32& value, uint64_t* remote_bar) {
  static_assert(sizeof(B32) == 4, "red.async .xor.b32");
  vx_red_async_xor_b32(dest, __vx_cuda_bit_cast<uint32_t>(value), remote_bar);
}
__VX_CUDA_INL void red_async(op_add_t, uint64_t* dest, const uint64_t& value, uint64_t* remote_bar) {
  vx_red_async_add_u64(dest, value, remote_bar);
}
// libcu++'s s64 form, the u64 instruction; its barrier is spelled int64_t*.
__VX_CUDA_INL void red_async(op_add_t, int64_t* dest, const int64_t& value, int64_t* remote_bar) {
  vx_red_async_add_s64(dest, value, (uint64_t*)remote_bar);
}

// barrier.cluster: the arrive's phase token waits in thread-local storage for
// the wait (the H100 keeps it in the barrier hardware).
inline thread_local uint32_t __vx_cluster_token;

__VX_CUDA_INL void barrier_cluster_arrive() { __vx_cluster_token = vx_cluster_arrive(); }
__VX_CUDA_INL void barrier_cluster_arrive(sem_release_t) { __vx_cluster_token = vx_cluster_arrive(); }
__VX_CUDA_INL void barrier_cluster_arrive(sem_relaxed_t) { __vx_cluster_token = vx_cluster_arrive_relaxed(); }
__VX_CUDA_INL void barrier_cluster_wait() { vx_cluster_wait(__vx_cluster_token); }
__VX_CUDA_INL void barrier_cluster_wait(sem_acquire_t) { vx_cluster_wait(__vx_cluster_token); }

// elect.sync: the lowest active lane of the mask, deterministically.
__VX_WARP bool elect_sync(const uint32_t& membermask) {
  __VX_WARP_ASSERT();
  const uint32_t m = membermask & __activemask();
  return m != 0 && __laneid() == (unsigned)__builtin_ctz(m);
}

// tensormap.replace and tensormap.cp_fenceproxy

template <dot_space Space>
__VX_CUDA_INL void __vx_tmap_space() {
  static_assert(Space == dot_space::global || Space == dot_space::shared, "space is .global or .shared::cta");
}

template <dot_space Space, typename B64>
__VX_CUDA_INL void tensormap_replace_global_address(space_t<Space>, void* tm_addr, B64 new_val) {
  __vx_tmap_space<Space>();
  static_assert(sizeof(B64) == 8, ".b64");
  vx_tensormap_replace_global_address((vx_tensor_map_t*)tm_addr, __vx_cuda_bit_cast<uint64_t>(new_val));
}

template <dot_space Space, typename B32>
__VX_CUDA_INL void tensormap_replace_rank(space_t<Space>, void* tm_addr, B32 new_val) {
  __vx_tmap_space<Space>();
  static_assert(sizeof(B32) == 4, ".b32");
  vx_tensormap_replace_rank((vx_tensor_map_t*)tm_addr, __vx_cuda_bit_cast<uint32_t>(new_val));
}

template <dot_space Space, int N32, typename B32>
__VX_CUDA_INL void tensormap_replace_box_dim(space_t<Space>, void* tm_addr, n32_t<N32>, B32 new_val) {
  __vx_tmap_space<Space>();
  static_assert(sizeof(B32) == 4, ".b32");
  vx_tensormap_replace_box_dim((vx_tensor_map_t*)tm_addr, N32, __vx_cuda_bit_cast<uint32_t>(new_val));
}

template <dot_space Space, int N32, typename B32>
__VX_CUDA_INL void tensormap_replace_global_dim(space_t<Space>, void* tm_addr, n32_t<N32>, B32 new_val) {
  __vx_tmap_space<Space>();
  static_assert(sizeof(B32) == 4, ".b32");
  vx_tensormap_replace_global_dim((vx_tensor_map_t*)tm_addr, N32, __vx_cuda_bit_cast<uint32_t>(new_val));
}

template <dot_space Space, int N32, typename B64>
__VX_CUDA_INL void tensormap_replace_global_stride(space_t<Space>, void* tm_addr, n32_t<N32>, B64 new_val) {
  __vx_tmap_space<Space>();
  static_assert(sizeof(B64) == 8, ".b64");
  vx_tensormap_replace_global_stride((vx_tensor_map_t*)tm_addr, N32, __vx_cuda_bit_cast<uint64_t>(new_val));
}

template <dot_space Space, int N32, typename B32>
__VX_CUDA_INL void tensormap_replace_element_stride(space_t<Space>, void* tm_addr, n32_t<N32>, B32 new_val) {
  __vx_tmap_space<Space>();
  static_assert(sizeof(B32) == 4, ".b32");
  vx_tensormap_replace_element_stride((vx_tensor_map_t*)tm_addr, N32, __vx_cuda_bit_cast<uint32_t>(new_val));
}

template <dot_space Space, int N32>
__VX_CUDA_INL void tensormap_replace_elemtype(space_t<Space>, void* tm_addr, n32_t<N32>) {
  __vx_tmap_space<Space>();
  vx_tensormap_replace_elemtype((vx_tensor_map_t*)tm_addr, N32);
}

template <dot_space Space, int N32>
__VX_CUDA_INL void tensormap_replace_interleave_layout(space_t<Space>, void* tm_addr, n32_t<N32>) {
  __vx_tmap_space<Space>();
  vx_tensormap_replace_interleave_layout((vx_tensor_map_t*)tm_addr, N32);
}

template <dot_space Space, int N32>
__VX_CUDA_INL void tensormap_replace_swizzle_mode(space_t<Space>, void* tm_addr, n32_t<N32>) {
  __vx_tmap_space<Space>();
  vx_tensormap_replace_swizzle_mode((vx_tensor_map_t*)tm_addr, N32);
}

template <dot_space Space, int N32>
__VX_CUDA_INL void tensormap_replace_fill_mode(space_t<Space>, void* tm_addr, n32_t<N32>) {
  __vx_tmap_space<Space>();
  vx_tensormap_replace_fill_mode((vx_tensor_map_t*)tm_addr, N32);
}

template <int N32, dot_scope Scope>
__VX_CUDA_INL void tensormap_cp_fenceproxy(sem_release_t, scope_t<Scope>, void* dst, const void* src, n32_t<N32>) {
  static_assert(N32 == 128, "a tensor map is 128 bytes");
  vx_tensormap_cp_fenceproxy((vx_tensor_map_t*)dst, (const vx_tensor_map_t*)src);
}

// No-ops: the register file is not repartitioned.
template <int N32>
__VX_CUDA_INL void setmaxnreg_inc(n32_t<N32>) {}
template <int N32>
__VX_CUDA_INL void setmaxnreg_dec(n32_t<N32>) {}

// Special registers

__VX_WARP uint32_t get_sreg_laneid() { __VX_WARP_ASSERT(); return __laneid(); }
__VX_WARP uint32_t get_sreg_lanemask_eq() { __VX_WARP_ASSERT(); return __lanemask_eq(); }
__VX_WARP uint32_t get_sreg_lanemask_lt() { __VX_WARP_ASSERT(); return __lanemask_lt(); }
__VX_WARP uint32_t get_sreg_lanemask_le() { __VX_WARP_ASSERT(); return __lanemask_le(); }
__VX_WARP uint32_t get_sreg_lanemask_gt() { __VX_WARP_ASSERT(); return __lanemask_gt(); }
__VX_WARP uint32_t get_sreg_lanemask_ge() { __VX_WARP_ASSERT(); return __lanemask_ge(); }
__VX_CUDA_INL uint32_t get_sreg_smid() { return __smid(); }
__VX_CUDA_INL uint32_t get_sreg_nsmid() { return __nsmid(); }
__VX_CUDA_INL uint32_t get_sreg_clusterid_x() { return vx_clusterid_x(); }
__VX_CUDA_INL uint32_t get_sreg_clusterid_y() { return vx_clusterid_y(); }
__VX_CUDA_INL uint32_t get_sreg_clusterid_z() { return vx_clusterid_z(); }
__VX_CUDA_INL uint32_t get_sreg_nclusterid_x() { return vx_nclusterid_x(); }
__VX_CUDA_INL uint32_t get_sreg_nclusterid_y() { return vx_nclusterid_y(); }
__VX_CUDA_INL uint32_t get_sreg_nclusterid_z() { return vx_nclusterid_z(); }
__VX_CUDA_INL uint32_t get_sreg_cluster_ctaid_x() { return vx_cluster_ctaid_x(); }
__VX_CUDA_INL uint32_t get_sreg_cluster_ctaid_y() { return vx_cluster_ctaid_y(); }
__VX_CUDA_INL uint32_t get_sreg_cluster_ctaid_z() { return vx_cluster_ctaid_z(); }
__VX_CUDA_INL uint32_t get_sreg_cluster_nctaid_x() { return vx_cluster_nctaid_x(); }
__VX_CUDA_INL uint32_t get_sreg_cluster_nctaid_y() { return vx_cluster_nctaid_y(); }
__VX_CUDA_INL uint32_t get_sreg_cluster_nctaid_z() { return vx_cluster_nctaid_z(); }
__VX_CUDA_INL uint32_t get_sreg_cluster_ctarank() { return vx_cluster_block_rank(); }
__VX_CUDA_INL uint32_t get_sreg_cluster_nctarank() { return vx_cluster_num_blocks(); }
__VX_CUDA_INL bool get_sreg_is_explicit_cluster() { return vx_is_explicit_cluster() != 0; }
__VX_CUDA_INL uint64_t get_sreg_globaltimer() { return vx_globaltimer(); }
__VX_CUDA_INL uint32_t get_sreg_globaltimer_lo() { return (uint32_t)vx_globaltimer(); }
__VX_CUDA_INL uint32_t get_sreg_globaltimer_hi() { return (uint32_t)(vx_globaltimer() >> 32); }
__VX_CUDA_INL uint32_t get_sreg_dynamic_smem_size() { return vx_dynamic_smem_size(); }
__VX_CUDA_INL uint32_t get_sreg_total_smem_size() { return vx_total_smem_size(); }
__VX_CUDA_INL uint32_t get_sreg_aggr_smem_size() { return vx_aggr_smem_size(); }

#undef __VX_PTX_CTA_OR_CLUSTER

} // namespace ptx
} // namespace cuda

///////////////////////////////////////////////////////////////////////////////
// invoke_one (CUDA allows any thread; this one is deterministic), cute

namespace cooperative_groups {

template <typename Fn, typename... Args>
__VX_CUDA_INL void invoke_one(const thread_block_tile<32>&, Fn&& fn, Args&&... args) {
  if (cuda::ptx::elect_sync(0xffffffffu))
    fn(static_cast<Args&&>(args)...);
  __syncwarp();
}

template <typename Fn, typename... Args>
__VX_CUDA_INL void invoke_one(const thread_block&, Fn&& fn, Args&&... args) {
  if (__vx_cuda_block_thread_rank() < 32 && cuda::ptx::elect_sync(0xffffffffu))
    fn(static_cast<Args&&>(args)...);
  __syncthreads();
}

} // namespace cooperative_groups

namespace cute {

template <int __vx_w = 0>
__VX_CUDA_INL uint32_t elect_one_sync() {
  __VX_WARP_ASSERT();
  return cuda::ptx::elect_sync(0xffffffffu) ? 1u : 0u;
}

} // namespace cute

#undef __VX_CUDA_MASK_CHECK
#undef __VX_WARP_ASSERT
#undef __VX_WARP
#undef __VX_CUDA_ASSERT
#undef __VX_CUDA_INL

#endif // __VX_CUDA_H__
