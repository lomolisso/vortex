#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_dram_map.h>
#include "common.h"

#define WRITE_MARK 0x5a5a5a5a5a5a5a5aull

// Makes the warp's next instruction wait for v: the cycle read after it sees the load complete.
static inline uint64_t consume(uint64_t v) {
  uint64_t r;
  __asm__ volatile ("mv %0, %1" : "=r"(r) : "r"(v));
  return r;
}

// The cycle once v (the phase's last loads) has arrived.
static inline uint64_t consume_cycle(uint64_t v) {
  consume(v);
  return vx_rdcycle();
}

static inline uint64_t ld(uint64_t a) { return *(volatile uint64_t*)a; }
static inline void st(uint64_t a, uint64_t v) { *(volatile uint64_t*)a = v; }

static void chase(kernel_arg_t* arg) {
  uint32_t* lat = reinterpret_cast<uint32_t*>(__local_mem());
  uint64_t p = arg->list_addr;
  uint32_t n = arg->count;
  for (uint32_t pass = 0; pass < arg->passes; ++pass) {
    for (uint32_t i = 0; i < n; ++i) {
      uint64_t t0 = vx_rdcycle();
      p = consume(ld(p));
      uint64_t t1 = vx_rdcycle();
      lat[pass * n + i] = uint32_t(t1 - t0);
    }
  }
  auto out = reinterpret_cast<uint32_t*>(arg->out_addr);
  for (uint32_t k = 0; k < arg->passes * n; ++k)
    out[k] = lat[k];
  // the chain is a cycle: after whole passes the walk is back at its start
  *reinterpret_cast<uint64_t*>(arg->out_addr + 4ull * arg->passes * n) = p;
}

static void pair(kernel_arg_t* arg) {
  uint32_t* lat = reinterpret_cast<uint32_t*>(__local_mem());
  auto list = reinterpret_cast<const uint64_t*>(arg->list_addr);
  uint32_t lane = threadIdx.x;
  uint64_t sum = 0;
  for (uint32_t i = 0; i < arg->count; ++i) {
    // the list load may miss: t0 is read once the address is in the register
    uint64_t a = consume(list[2 * i + lane]);
    uint64_t t0 = vx_rdcycle();
    uint64_t v = consume(ld(a));
    uint64_t t1 = vx_rdcycle();
    sum += v;
    if (lane == 0)
      lat[i] = uint32_t(t1 - t0);
    while (vx_rdcycle() - t1 < arg->gap)
      ;
  }
  auto out = reinterpret_cast<uint32_t*>(arg->out_addr);
  if (lane == 0) {
    for (uint32_t i = 0; i < arg->count; ++i)
      out[i] = lat[i];
  }
  reinterpret_cast<uint64_t*>(arg->out_addr + 4ull * ((arg->count + 1) & ~1u))[lane] = sum;
}

static inline uint64_t line_addr(const kernel_arg_t* arg, uint32_t place, uint64_t j, uint64_t lines) {
  if (place == PLACE_ONE)
    return arg->buf_addr + vx_dram_interim_addr(0, j << arg->lg_sel, arg->lg_sel, arg->lg_channels);
  return arg->buf_addr + ((lines << arg->lg_channels) << arg->lg_sel) + (j << arg->lg_sel);
}

static void stream(kernel_arg_t* arg, uint64_t gid, uint64_t nthreads) {
  uint64_t lines = uint64_t(arg->count) * nthreads;
  uint32_t nw = arg->param;
  uint64_t sum = 0;
  auto out = reinterpret_cast<uint64_t*>(arg->out_addr) + gid * LINES_OUT_WORDS;
  for (uint32_t place = PLACE_ONE; place <= PLACE_ALL; ++place) {
    if (!(arg->phases & (1u << place)))
      continue;
    out[1 + 2 * place] = vx_rdcycle();
    for (uint32_t i = 0; i < arg->count; i += BENCH_ILP) {
      uint64_t a[BENCH_ILP], v[BENCH_ILP];
      bool w[BENCH_ILP];
      for (uint32_t k = 0; k < BENCH_ILP; ++k) {
        a[k] = line_addr(arg, place, uint64_t(i + k) * nthreads + gid, lines);
        uint32_t slot = (i + k) % 8;
        w[k] = nw && ((slot * nw) % 8) < nw;
      }
      for (uint32_t k = 0; k < BENCH_ILP; ++k) {
        if (w[k]) {
          st(a[k], a[k] ^ WRITE_MARK);
          v[k] = 0;
        } else {
          v[k] = ld(a[k]);
        }
      }
      for (uint32_t k = 0; k < BENCH_ILP; ++k)
        sum += v[k];
    }
    out[2 + 2 * place] = consume_cycle(sum);
  }
  out[0] = sum;
}

static void sparse(kernel_arg_t* arg, uint64_t gid, uint64_t nthreads) {
  uint64_t lines = uint64_t(arg->count) * nthreads;
  uint64_t sum = 0;
  auto out = reinterpret_cast<uint64_t*>(arg->out_addr) + gid * LINES_OUT_WORDS;
  out[1 + 2 * PLACE_ALL] = vx_rdcycle();
  for (uint32_t i = 0; i < arg->count; i += BENCH_ILP) {
    uint64_t a[BENCH_ILP], v[BENCH_ILP] = {0};
    for (uint32_t k = 0; k < BENCH_ILP; ++k)
      a[k] = line_addr(arg, PLACE_ALL, uint64_t(i + k) * nthreads + gid, lines);
    for (uint32_t k = 0; k < BENCH_ILP; ++k) {
      switch (arg->param) {
      case SP_LD64: v[k] = ld(a[k]) + ld(a[k] + 32); break;
      case SP_LD32: v[k] = ld(a[k]); break;
      case SP_ST16:
        for (uint32_t o = 0; o < 16; o += 8) st(a[k] + o, a[k] ^ WRITE_MARK);
        break;
      case SP_ST32:
        for (uint32_t o = 0; o < 32; o += 8) st(a[k] + o, a[k] ^ WRITE_MARK);
        break;
      default:
        for (uint32_t o = 0; o < 64; o += 8) st(a[k] + o, a[k] ^ WRITE_MARK);
        break;
      }
    }
    for (uint32_t k = 0; k < BENCH_ILP; ++k)
      sum += v[k];
  }
  out[2 + 2 * PLACE_ALL] = consume_cycle(sum);
  out[0] = sum;
}

static void top(kernel_arg_t* arg, uint64_t gid, uint64_t nthreads) {
  auto buf = reinterpret_cast<volatile uint64_t*>(arg->buf_addr);
  for (uint64_t k = gid; k < arg->count; k += nthreads)
    buf[k] = (arg->buf_addr + 8 * k) ^ WRITE_MARK;
  uint64_t bad = 0;
  for (uint64_t k = gid; k < arg->count; k += nthreads)
    bad += buf[k] != ((arg->buf_addr + 8 * k) ^ WRITE_MARK);
  reinterpret_cast<uint64_t*>(arg->out_addr)[gid] = bad;
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  uint64_t gid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  uint64_t nthreads = uint64_t(gridDim.x) * blockDim.x;
  switch (arg->mode) {
  case MODE_CHASE:  chase(arg); break;
  case MODE_PAIR:   pair(arg); break;
  case MODE_STREAM: stream(arg, gid, nthreads); break;
  case MODE_SPARSE: sparse(arg, gid, nthreads); break;
  default:          top(arg, gid, nthreads); break;
  }
}
