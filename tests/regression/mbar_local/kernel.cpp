#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_barrier.h>
#include <vx_dlmem.h>
#include <vx_dxa.h>
#include <vx_mbar.h>
#include "common.h"

// One block. Warps [0, S) stream DXA tile copies into their own tiles to keep
// the banks busy; the other warps, the workers, run the mode. Thread 0 sets
// the objects up and __syncthreads() publishes them. At the end every warp
// meets in __syncthreads() and thread 0 copies the objects out for the host,
// which predicts each final word with mbar_ref.

namespace {

#define INL inline __attribute__((always_inline))

constexpr uint32_t kWorkerBar = 1;
constexpr uint32_t kCopyBar0  = 2;

struct Ctx {
  uint32_t tid, lane, nt, warp, S;
  bool     worker, parity;
  uint32_t ww, nww, wt, nwt;   // worker warp and thread indexes, and their counts
  uint64_t* lm;
  volatile uint64_t* data;
  uint32_t errors, first, what, negs;

  INL void fail(uint32_t round, uint32_t code) {
    if (errors++ == 0) {
      first = round + 1;
      what  = code;
    }
  }
  INL void check(bool ok, uint32_t round, uint32_t code) {
    if (!ok)
      fail(round, code);
  }
  INL void check_phase(uint64_t w, uint32_t parity, uint32_t round) {
    check(MB_PHASE(w) == (parity & 1), round, CHK_TOKEN_PHASE);
  }
  INL void check_fields(uint64_t w, uint32_t max_pending, int64_t tx_lo, int64_t tx_hi, uint32_t round) {
    const uint32_t p = MB_PENDING(w);
    const int64_t  t = MB_TX(w);
    check(p >= 1 && p <= max_pending && t >= tx_lo && t <= tx_hi, round, CHK_TOKEN_FIELD);
    negs += (t < 0);
  }
  // Parity runs must not spin: every worker warp waits in the register barrier,
  // after which the phase must be complete.
  INL void meet() {
    vortex::barrier(kWorkerBar, nww).arrive_and_wait();
  }
  INL void wait_token(const vortex::mbarrier& b, uint64_t token, uint32_t round) {
    if (parity) {
      meet();
      check(b.test_wait(token), round, CHK_WAIT);
      return;
    }
    uint32_t spins = 0;
    while (!b.test_wait(token)) {
      if (++spins >= SPIN_LIMIT) {
        fail(round, CHK_WAIT);
        break;
      }
    }
  }
  INL void wait_parity(const vortex::mbarrier& b, uint32_t ph, uint32_t round) {
    if (parity) {
      meet();
      check(b.test_wait_parity(ph), round, CHK_WAIT);
      return;
    }
    uint32_t spins = 0;
    while (!b.test_wait_parity(ph)) {
      if (++spins >= SPIN_LIMIT) {
        fail(round, CHK_WAIT);
        break;
      }
    }
  }
};

INL void idle(uint32_t n) {
  for (uint32_t i = 0; i < n; ++i)
    __asm__ volatile ("");
}

INL void ctx_init(Ctx& x, const mbar_arg_t* arg) {
  x.tid    = threadIdx.x;
  x.nt     = vx_num_threads();
  x.lane   = x.tid % x.nt;
  x.warp   = get_sub_group_id();
  x.S      = arg->streamers;
  x.worker = (x.warp >= x.S);
  x.parity = (arg->parity != 0);
  x.ww     = x.warp - x.S;
  x.nww    = get_num_sub_groups() - x.S;
  x.wt     = x.tid - x.S * x.nt;
  x.nwt    = x.nww * x.nt;
  x.lm     = reinterpret_cast<uint64_t*>(__local_mem());
  x.data   = x.lm + LM_DATA;
  x.errors = x.first = x.what = x.negs = 0;
  if (x.tid == 0) {
    for (uint32_t i = 0; i < NUM_OBJS; ++i)
      vx_mbar_inval(x.lm + OBJ_BASE + i);
    reinterpret_cast<volatile uint64_t*>(x.lm)[LM_DONE] = 0;
  }
}

INL void stream(const Ctx& x, const mbar_arg_t* arg) {
  vortex::barrier copy_bar(kCopyBar0 + x.warp, 1);
  uint32_t* tile = reinterpret_cast<uint32_t*>(x.lm + arg->tile_word) + x.warp * arg->tile_elems;
  volatile uint64_t* done = x.lm + LM_DONE;
  uint32_t copies = 0;
  while (*done == 0) {
    copy_bar.expect_tx(1);
    vx_dxa_issue_1d_wg(0, copy_bar.id(), tile, (copies % arg->num_src_tiles) * arg->tile_elems);
    copy_bar.arrive_and_wait();
    ++copies;
  }
}

// The workers meet, then release the streamers; everyone meets, then thread 0
// copies the objects out.
INL void finish(Ctx& x, const mbar_arg_t* arg) {
  if (x.worker) {
    vortex::barrier(kWorkerBar, x.nww).arrive_and_wait();
    if (x.wt == 0)
      reinterpret_cast<volatile uint64_t*>(x.lm)[LM_DONE] = 1;
  }
  __syncthreads();
  if (x.tid == 0) {
    auto words = reinterpret_cast<uint64_t*>(arg->word_addr);
    for (uint32_t i = 0; i < NUM_OBJS; ++i)
      words[i] = reinterpret_cast<volatile uint64_t*>(x.lm)[OBJ_BASE + i];
  }
  auto err = reinterpret_cast<uint32_t*>(arg->err_addr) + x.tid * ERR_WORDS;
  err[ERR_COUNT] = x.errors;
  err[ERR_FIRST] = x.first;
  err[ERR_WHAT]  = x.what;
  err[ERR_NEG]   = x.negs;
}

// A value no other round or thread stores.
INL uint64_t tag(uint32_t round, uint32_t idx) {
  return ((uint64_t)(round + 1) << 32) | idx;
}

} // namespace

// Every worker thread stores its slot of the phase, arrives with 1 and waits on
// its token; then it must see the slots of three other threads, one of them in
// another warp. Two slots per thread: a slot is reused only two phases later,
// after its readers have arrived again.
__kernel void basic_k(mbar_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  vortex::mbarrier bar(x.lm + OBJ_BAR);
  if (x.tid == 0)
    bar.init(x.nwt);
  __syncthreads();
  if (!x.worker) {
    stream(x, arg);
  } else {
    const uint32_t n = x.nwt;
    for (uint32_t p = 0; p < arg->rounds; ++p) {
      volatile uint64_t* slot = x.data + (p & 1) * n;
      slot[x.wt] = tag(p, x.wt);
      idle(((x.ww + p) % x.nww) * arg->delay);
      const uint64_t tok = bar.arrive(1);
      x.check_phase(tok, p, p);
      x.check_fields(tok, n, 0, 0, p);
      x.wait_token(bar, tok, p);
      const uint32_t j0 = (x.wt + 1) % n;
      const uint32_t j1 = (x.wt + x.nt) % n;
      const uint32_t j2 = (x.wt + n / 2) % n;
      x.check(slot[j0] == tag(p, j0) && slot[j1] == tag(p, j1) && slot[j2] == tag(p, j2), p, CHK_DATA);
    }
  }
  finish(x, arg);
}

// Lane 0 of each worker warp arrives for the whole warp, after every lane has
// stored its slot: the arrive releases the other lanes' stores too. Odd warps
// split the count over two arrives. Every lane waits on the phase parity.
__kernel void count_k(mbar_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  vortex::mbarrier bar(x.lm + OBJ_BAR);
  if (x.tid == 0)
    bar.init(x.nwt);
  __syncthreads();
  if (!x.worker) {
    stream(x, arg);
  } else {
    const uint32_t n = x.nwt;
    const bool split = (x.ww & 1) && x.nt >= 2;
    for (uint32_t p = 0; p < arg->rounds; ++p) {
      volatile uint64_t* slot = x.data + (p & 1) * n;
      slot[x.wt] = tag(p, x.wt);
      idle(((x.ww + p) % x.nww) * arg->delay);
      if (x.lane == 0) {
        uint64_t tok;
        if (split) {
          tok = bar.arrive(x.nt / 2);
          x.check_phase(tok, p, p);
          x.check_fields(tok, n, 0, 0, p);
          tok = bar.arrive(x.nt - x.nt / 2);
        } else {
          tok = bar.arrive(x.nt);
        }
        x.check_phase(tok, p, p);
        x.check_fields(tok, n, 0, 0, p);
      }
      x.wait_parity(bar, p, p);
      const uint32_t j0 = (x.wt + 1) % n;
      const uint32_t j1 = (x.wt + x.nt) % n;
      const uint32_t j2 = (x.wt + n / 2) % n;
      x.check(slot[j0] == tag(p, j0) && slot[j1] == tag(p, j1) && slot[j2] == tag(p, j2), p, CHK_DATA);
    }
  }
  finish(x, arg);
}

// Worker warps [0, P) produce and [P, W) consume, over two buffers. FULL[s]
// expects one arrival per producer thread, EMPTY[s] one per consumer warp,
// from its lane 0 once every lane has read the buffer: that arrive must not
// complete before the warp's loads are performed, or a producer could
// overwrite the buffer under them. Fill r uses buffer r % 2, in its object's
// phase r / 2.
__kernel void pc_k(mbar_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  const uint32_t P  = x.nww / 2;
  const uint32_t C  = x.nww - P;
  const uint32_t pt = P * x.nt;
  const uint32_t ct = C * x.nt;
  if (x.tid == 0) {
    for (uint32_t s = 0; s < 2; ++s) {
      vortex::mbarrier(x.lm + OBJ_FULL + s).init(pt);
      vortex::mbarrier(x.lm + OBJ_EMPTY + s).init(C);
    }
  }
  __syncthreads();
  if (!x.worker) {
    stream(x, arg);
  } else if (x.parity) {
    // A fixed schedule with no spin: the producers fill and arrive on FULL;
    // everyone meets; the consumers find FULL complete, read and arrive on
    // EMPTY; everyone meets; the producers find EMPTY complete.
    const bool prod = (x.ww < P);
    for (uint32_t r = 0; r < arg->rounds; ++r) {
      const uint32_t s = r & 1, k = r >> 1;
      vortex::mbarrier full(x.lm + OBJ_FULL + s);
      vortex::mbarrier empty(x.lm + OBJ_EMPTY + s);
      volatile uint64_t* buf = x.data + s * PC_ELEMS;
      if (prod) {
        for (uint32_t i = x.wt; i < PC_ELEMS; i += pt)
          buf[i] = tag(r, i);
        const uint64_t tok = full.arrive(1);
        x.check_phase(tok, k, r);
        x.check_fields(tok, pt, 0, 0, r);
      }
      x.meet();
      if (!prod) {
        x.check(full.test_wait_parity(k), r, CHK_WAIT);
        bool ok = true;
        for (uint32_t i = x.wt - pt; i < PC_ELEMS; i += ct)
          ok &= (buf[i] == tag(r, i));
        x.check(ok, r, CHK_DATA);
        if (x.lane == 0) {
          const uint64_t tok = empty.arrive(1);
          x.check_phase(tok, k, r);
          x.check_fields(tok, C, 0, 0, r);
        }
      }
      x.meet();
      if (prod)
        x.check(empty.test_wait_parity(k), r, CHK_WAIT);
    }
  } else if (x.ww < P) {
    for (uint32_t r = 0; r < arg->rounds; ++r) {
      const uint32_t s = r & 1, k = r >> 1;
      vortex::mbarrier full(x.lm + OBJ_FULL + s);
      vortex::mbarrier empty(x.lm + OBJ_EMPTY + s);
      volatile uint64_t* buf = x.data + s * PC_ELEMS;
      if (r >= 2)
        x.wait_parity(empty, k - 1, r);
      idle((r % 3 == 0) ? arg->delay : 0);
      for (uint32_t i = x.wt; i < PC_ELEMS; i += pt)
        buf[i] = tag(r, i);
      const uint64_t tok = full.arrive(1);
      x.check_phase(tok, k, r);
      x.check_fields(tok, pt, 0, 0, r);
    }
  } else {
    const uint32_t c = x.wt - pt;
    for (uint32_t r = 0; r < arg->rounds; ++r) {
      const uint32_t s = r & 1, k = r >> 1;
      vortex::mbarrier full(x.lm + OBJ_FULL + s);
      vortex::mbarrier empty(x.lm + OBJ_EMPTY + s);
      volatile uint64_t* buf = x.data + s * PC_ELEMS;
      x.wait_parity(full, k, r);
      bool ok = true;
      for (uint32_t i = c; i < PC_ELEMS; i += ct)
        ok &= (buf[i] == tag(r, i));
      x.check(ok, r, CHK_DATA);
      idle((r % 3 == 1) ? arg->delay : 0);
      if (x.lane == 0) {
        const uint64_t tok = empty.arrive(1);
        x.check_phase(tok, k, r);
        x.check_fields(tok, C, 0, 0, r);
      }
    }
  }
  finish(x, arg);
}

// First the first worker thread runs the host's script on its own object,
// recording each returned word. Then software tx accounting on the shared
// object, which expects one arrival per worker warp: per phase, worker warp
// w > 0 completes TX_CHUNK * w bytes and arrives, and worker warp 0 expects all
// of them with one arrive_tx. On odd phases worker warp 0 comes late, so the
// completes run first and tx goes negative; on even phases the others come
// late. Every returned word must still be in the phase.
__kernel void tx_k(mbar_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  vortex::mbarrier bar(x.lm + OBJ_BAR);
  if (x.tid == 0)
    bar.init(x.nww);
  __syncthreads();
  if (!x.worker) {
    stream(x, arg);
  } else {
    if (x.wt == 0) {
      uint64_t* obj = x.lm + OBJ_SCRIPT;
      auto script = reinterpret_cast<const uint64_t*>(arg->script_addr);
      auto ret = reinterpret_cast<uint64_t*>(arg->word_addr) + NUM_OBJS;
      for (uint32_t i = 0; i < arg->script_len; ++i) {
        const uint32_t op = (uint32_t)script[2 * i];
        const uint64_t v  = script[2 * i + 1];
        uint64_t r = 0;
        switch (op) {
        case 10: r = vx_mbar_arrive(obj, (uint32_t)v); break;
        case 11: r = vx_mbar_arrive_tx(obj, (uint32_t)v); break;
        case 12: r = vx_mbar_expect_tx(obj, (uint32_t)v); break;
        case 13: r = vx_mbar_complete_tx(obj, (uint32_t)v); break;
        case 14: r = vx_mbar_test_wait(obj, v); break;
        default: *reinterpret_cast<volatile uint64_t*>(obj) = v; break;
        }
        ret[i] = r;
      }
    }
    vortex::barrier(kWorkerBar, x.nww).arrive_and_wait();

    const uint32_t n = x.nwt;
    const uint32_t total = TX_CHUNK * (x.nww * (x.nww - 1) / 2);
    for (uint32_t p = 0; p < arg->rounds; ++p) {
      volatile uint64_t* slot = x.data + (p & 1) * n;
      slot[x.wt] = tag(p, x.wt);
      const bool late = (x.ww == 0) == ((p & 1) != 0);
      idle(late ? arg->delay : 0);
      if (x.lane == 0) {
        if (x.ww == 0) {
          const uint64_t tok = bar.arrive_tx(total);
          x.check_phase(tok, p, p);
          x.check_fields(tok, x.nww, -(int64_t)total, 0, p);
        } else {
          const uint32_t bytes = TX_CHUNK * x.ww;
          uint64_t tok = bar.complete_tx(bytes);
          x.check_phase(tok, p, p);
          x.check_fields(tok, x.nww, -(int64_t)total, total, p);
          tok = bar.arrive(1);
          x.check_phase(tok, p, p);
          x.check_fields(tok, x.nww, -(int64_t)total, total, p);
        }
      }
      x.wait_parity(bar, p, p);
      const uint32_t j0 = (x.wt + 1) % n;
      const uint32_t j1 = (x.wt + x.nt) % n;
      x.check(slot[j0] == tag(p, j0) && slot[j1] == tag(p, j1), p, CHK_DATA);
    }
  }
  finish(x, arg);
}

// Each must stop the simulation with its assert before the kernel ends.
__kernel void neg_k(mbar_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  vortex::mbarrier bar(x.lm + OBJ_BAR);
  if (x.tid == 0)
    bar.init(1);
  if (arg->neg == NEG_REMOTE) {
    vortex::this_cluster().sync();
  } else {
    __syncthreads();
  }
  if (x.tid == 0) {
    switch (arg->neg) {
    case NEG_UNDERFLOW:
      bar.arrive(2);
      break;
    case NEG_TXRANGE:
      bar.expect_tx(1u << 20);
      break;
    case NEG_GLOBAL:
      vx_mbar_arrive(reinterpret_cast<uint64_t*>(arg->err_addr), 1);
      break;
    case NEG_REMOTE:
      if (vx_cluster_block_rank() == 0)
        vx_mbar_arrive(static_cast<uint64_t*>(vx_cluster_map_shared_rank(x.lm + OBJ_BAR, 1)), 1);
      break;
    default:
      break;
    }
  }
  if (arg->neg == NEG_REMOTE) {
    vortex::this_cluster().sync();
  } else {
    __syncthreads();
  }
  if (blockIdx.x == 0)
    finish(x, arg);
}
