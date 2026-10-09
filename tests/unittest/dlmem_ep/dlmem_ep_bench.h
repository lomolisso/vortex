// Copyright © 2019-2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Shared bench of the DLMEM endpoint tests: stimulus, behavioural local
// memory, reference image and checks, driving one cluster's endpoints cycle by
// cycle through the flat ports of VX_dlmem_ep_top. It has no simulator
// dependency, so the RTL testbench and the SimX test run the same traffic and
// their event traces and latency rows can be compared line by line.

#pragma once

#include <algorithm>
#include <cinttypes>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

// The including file provides the VX_CFG_* configuration first.
#include <VX_types.h>

namespace ep {

constexpr uint32_t log2up(uint64_t n) {
  uint32_t r = 0;
  while ((uint64_t(1) << r) < n) {
    ++r;
  }
  return r;
}

static_assert(VX_CFG_NUM_CORES % VX_CFG_SOCKET_SIZE == 0, "NUM_CORES must be a multiple of SOCKET_SIZE");

constexpr uint32_t SOCKET_SIZE = VX_CFG_SOCKET_SIZE;
constexpr uint32_t NUM_SOCKETS = VX_CFG_NUM_CORES / VX_CFG_SOCKET_SIZE;
constexpr uint32_t NCC         = NUM_SOCKETS * SOCKET_SIZE;
constexpr uint32_t NB          = VX_CFG_NUM_LSU_BLOCKS;
constexpr uint32_t NL          = VX_CFG_NUM_LSU_LANES;
constexpr uint32_t NW          = VX_CFG_NUM_WARPS;
constexpr uint32_t NT          = VX_CFG_NUM_THREADS;
constexpr uint32_t NP          = NCC * NB;
constexpr uint32_t NW_BITS     = log2up(NW);
constexpr uint32_t NT_BITS     = log2up(NT);
constexpr uint32_t LMEM_LOG    = VX_CFG_LMEM_LOG_SIZE;
constexpr uint32_t LMEM_AW     = LMEM_LOG - 3;
constexpr uint32_t NBANKS      = VX_CFG_LMEM_NUM_BANKS;
constexpr uint32_t CREDITS     = VX_CFG_DLMEM_STORE_CREDITS;
constexpr uint32_t PENDING     = VX_CFG_DLMEM_PENDING_SIZE;
constexpr uint32_t SLOTS       = 4;   // DLMEM_INGRESS_SLOTS
constexpr uint32_t TAGS        = VX_CFG_LSU_PENDING_SIZE;
constexpr uint32_t WT_MAX      = CREDITS + (VX_CFG_LSU_QUEUE_IN_SIZE + 3) * NL;
constexpr uint32_t WT_W        = log2up(WT_MAX + 1);
constexpr uint64_t DLMEM_BASE  = VX_MEM_DLMEM_BASE_ADDR;
constexpr uint64_t LMEM_BASE   = VX_MEM_LMEM_BASE_ADDR;
#ifdef VX_CFG_EXT_ZACAS_ENABLE
constexpr bool ZACAS = true;
#else
constexpr bool ZACAS = false;
#endif

static_assert(NCC >= 2, "the endpoints need a second core to talk to");
static_assert(NL <= 32 && NP <= 64 && TAGS <= 256, "bench containers too small");

// RTL amo_op_e codes.
enum : uint8_t {
  OP_LR = 0, OP_SC = 1, OP_ADD = 2, OP_SWAP = 3, OP_XOR = 4,
  OP_OR = 5, OP_AND = 6, OP_MIN = 7, OP_MAX = 8, OP_CAS = 9
};

enum Kind : uint8_t { K_LD, K_ST, K_AMO, K_LR, K_SC };

inline const char* kind_name(uint8_t k) {
  switch (k) {
  case K_LD:  return "LD";
  case K_ST:  return "ST";
  case K_AMO: return "AMO";
  case K_LR:  return "LR";
  case K_SC:  return "SC";
  }
  return "?";
}

inline uint64_t mix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

inline uint64_t byte_mask(uint8_t byteen) {
  uint64_t m = 0;
  for (uint32_t i = 0; i < 8; ++i) {
    if (byteen >> i & 1) {
      m |= 0xffull << (8 * i);
    }
  }
  return m;
}

inline uint32_t make_hart(uint32_t core, uint32_t wid, uint32_t tid) {
  return (core << (NW_BITS + NT_BITS)) | (wid << NT_BITS) | tid;
}

// One active lane of an LSU request, as the RTL top's per-lane ports carry it.
struct LaneView {
  uint32_t dst = 0;
  uint32_t waddr = 0;     // word address in the target's local memory
  uint32_t hart = 0;
  uint32_t tid = 0;
  uint64_t data = 0;
  uint64_t cmp = 0;
  uint8_t  byteen = 0;
  uint8_t  amo_op = 0;
  bool     amo = false;
  bool     uns = false;
};

struct ReqView {
  uint8_t  kind = K_LD;
  bool     rw = false;
  uint32_t mask = 0;
  uint32_t tag = 0;
  uint32_t wid = 0;
  uint64_t id = 0;
  std::vector<LaneView> lanes;
};

struct LmemReqV {
  bool     fire = false;
  bool     rw = false;
  bool     local = false;
  bool     amo = false;
  bool     uns = false;
  uint8_t  amo_op = 0;
  uint8_t  byteen = 0;
  uint32_t waddr = 0;
  uint32_t hart = 0;
  uint32_t tag = 0;
  uint64_t data = 0;
  uint64_t cmp = 0;
};

struct RspV {
  bool fire = false;
  uint32_t mask = 0;
  uint32_t tag = 0;
  std::vector<uint64_t> data;
};

// What the bench drives in one cycle, before the clock edge.
struct In {
  std::vector<const ReqView*> req = std::vector<const ReqView*>(NP, nullptr);
  std::vector<uint8_t>  rsp_ready = std::vector<uint8_t>(NP, 0);
  std::vector<uint8_t>  lmem_ready = std::vector<uint8_t>(NCC, 0);
  std::vector<uint8_t>  lrsp_valid = std::vector<uint8_t>(NCC, 0);
  std::vector<uint32_t> lrsp_tag = std::vector<uint32_t>(NCC, 0);
  std::vector<uint64_t> lrsp_data = std::vector<uint64_t>(NCC, 0);
  std::vector<uint8_t>  inc_valid = std::vector<uint8_t>(NP, 0);
  std::vector<uint32_t> inc_wid = std::vector<uint32_t>(NP, 0);
  std::vector<uint32_t> inc_cnt = std::vector<uint32_t>(NP, 0);
};

// What the bench observes: handshakes of the cycle, then state after the edge.
struct Out {
  std::vector<uint8_t>  req_fire = std::vector<uint8_t>(NP, 0);
  std::vector<RspV>     rsp = std::vector<RspV>(NP);
  std::vector<LmemReqV> lmem = std::vector<LmemReqV>(NCC);
  std::vector<uint8_t>  ack_valid = std::vector<uint8_t>(NCC, 0);
  std::vector<uint32_t> ack_wid = std::vector<uint32_t>(NCC, 0);
  std::vector<uint8_t>  sinks_ready = std::vector<uint8_t>(NCC, 1);
  std::vector<uint32_t> drained = std::vector<uint32_t>(NCC, 0);
  std::vector<uint8_t>  eg_empty = std::vector<uint8_t>(NCC, 0);
  std::vector<uint8_t>  in_empty = std::vector<uint8_t>(NCC, 0);
  std::vector<uint8_t>  wt_empty = std::vector<uint8_t>(NCC, 0);
  // Outstanding stores, loads/AMOs and ingress requests as the DUT counts them.
  std::vector<uint32_t> st_out = std::vector<uint32_t>(NCC, 0);
  std::vector<uint32_t> ld_out = std::vector<uint32_t>(NCC, 0);
  std::vector<uint32_t> slot_out = std::vector<uint32_t>(NCC, 0);
};

class Dut {
public:
  virtual ~Dut() = default;
  // One clock cycle: apply `in`, report this cycle's handshakes and the
  // state after the edge in `out`.
  virtual void cycle(const In& in, Out& out) = 0;
  // Bound on slot_out: the RTL counts requests until their response enters
  // the network, the slot frees when it enters the ingress output buffer.
  virtual uint32_t slot_bound() const = 0;
};

struct Options {
  std::string mode = "random";
  uint32_t seed = 1;
  uint32_t reqs = 0;
  uint32_t watchdog = 4000;
};

struct LatRow {
  uint32_t src, dst;
  uint8_t  op;
  uint32_t seq;
  uint64_t issue, done;
};

class Bench {
public:
  Bench(Dut& dut, const Options& opts, std::ostream* trace)
    : dut_(dut), opts_(opts), trace_(trace), rng_(opts.seed) {
    ports_.resize(NP);
    for (auto& p : ports_) {
      for (uint32_t t = 0; t < TAGS; ++t) {
        p.free_tags.push_back(t);
      }
    }
    image_.assign(NCC, std::vector<uint64_t>(1u << LMEM_AW, 0));
    for (uint32_t c = 0; c < NCC; ++c) {
      for (uint32_t a = 0; a < (1u << LMEM_AW); ++a) {
        image_[c][a] = init_word(c, a);
      }
    }
    banks_.assign(NCC, std::vector<std::deque<LmemOp>>(NBANKS));
    bank_last_.assign(NCC, std::vector<uint64_t>(NBANKS, 0));
    rspq_.resize(NCC);
    lmem_out_.assign(NCC, 0);
    wt_cnt_.assign(NCC, std::vector<int64_t>(NW, 0));
    committed_.assign(NCC, std::vector<uint64_t>(NW, 0));
    acked_.assign(NCC, std::vector<uint64_t>(NW, 0));
    st_fifo_.assign(NCC, std::vector<std::deque<int>>(NW));
    drained_.assign(NCC, (NW >= 32) ? ~0u : ((1u << NW) - 1));
    empty_.assign(NCC, 7);
  }

  // Runs the mode to completion; returns the number of failed checks.
  uint32_t run() {
    generate();
    const uint64_t limit = 2000000;
    while (outstanding() && cycle_ < limit && failures_ == 0) {
      clock();
      if (cycle_ - last_progress_ > opts_.watchdog) {
        fail(strf("watchdog: no progress for %u cycles", opts_.watchdog));
      }
    }
    if (cycle_ >= limit) {
      fail("cycle limit reached");
    }
    done_cycle_ = cycle_;
    for (int i = 0; i < 32 && failures_ == 0; ++i) {
      clock();
    }
    if (failures_ == 0) {
      final_check();
    }
    return failures_;
  }

  uint64_t digest() const { return digest_; }
  uint64_t cycles() const { return done_cycle_; }
  size_t lane_ops() const { return ops_.size(); }
  uint64_t lmem_ops() const { return n_lmem_; }
  uint32_t max_st() const { return max_st_; }
  uint32_t max_ld() const { return max_ld_; }
  uint32_t max_slot() const { return max_slot_; }
  const std::string& first_failure() const { return first_failure_; }

  std::vector<LatRow> latency() const {
    std::vector<LatRow> rows;
    for (auto& op : ops_) {
      rows.push_back({op.src, op.dst, op.kind, op.seq, op.issue, op.done});
    }
    std::sort(rows.begin(), rows.end(), [](const LatRow& a, const LatRow& b) {
      return std::tie(a.src, a.seq) < std::tie(b.src, b.seq);
    });
    return rows;
  }

  // issue_cycle: first cycle the lane's request is offered; done_cycle: its LSU
  // response, or for a store an ack of its warp (acks carry only the warp, so
  // they are matched to the warp's stores in issue order).
  static void write_csv(const std::string& path, const std::vector<LatRow>& rows) {
    std::ofstream ofs(path);
    ofs << "src,dst,op,seq,issue_cycle,done_cycle\n";
    for (auto& r : rows) {
      ofs << r.src << "," << r.dst << "," << kind_name(r.op) << "," << r.seq << ","
          << r.issue << "," << r.done << "\n";
    }
  }

  static uint64_t init_word(uint32_t core, uint32_t waddr) {
    return mix(uint64_t(core) * 7919u + waddr);
  }

private:
  struct LaneOp {
    uint32_t src, blk, lane, wid, tid, dst, waddr, hart, seq, tag;
    uint8_t  kind;
    uint8_t  byteen, amo_op;
    bool     uns;
    uint64_t data, cmp;
    uint64_t result = 0;
    bool     executed = false;
    bool     responded = false;
    uint64_t issue = 0;
    uint64_t done = 0;
  };

  struct Req {
    uint8_t  kind;
    uint32_t wid;
    uint32_t mask;
    uint64_t release;
    std::vector<int> lanes;
  };

  struct Port {
    std::deque<Req> queue;
    bool offered = false;
    ReqView view;
    std::vector<int> view_ops;
    std::deque<uint32_t> free_tags;
    std::map<uint32_t, std::vector<int>> waiting;   // tag -> lane ops awaiting data
  };

  struct LmemOp {
    LmemReqV req;
    int op;
    uint64_t ready;
  };

  static std::string strf(const char* fmt, ...) {
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    return buf;
  }

  void fail(const std::string& msg) {
    if (failures_ == 0) {
      first_failure_ = strf("cycle %" PRIu64 ": ", cycle_) + msg;
    }
    ++failures_;
  }

  uint32_t rnd(uint32_t n) { return uint32_t(rng_() % n); }
  bool chance(uint32_t pct) { return rnd(100) < pct; }
  uint64_t r64() { return (uint64_t(rng_()) << 32) ^ rng_(); }

  // Canonical event record: the trace line and the run digest.
  void event(const std::string& line) {
    for (char ch : line) {
      digest_ ^= uint8_t(ch);
      digest_ *= 0x100000001b3ull;
    }
    digest_ ^= '\n';
    digest_ *= 0x100000001b3ull;
    if (trace_) {
      *trace_ << line << "\n";
    }
  }

  // ---- stimulus ---------------------------------------------------------

  uint32_t pick_dst(uint32_t src) {
    uint32_t d = rnd(NCC - 1);
    return (d >= src) ? d + 1 : d;
  }

  uint32_t rand_mask() {
    uint32_t m = 0;
    while (m == 0) {
      m = rnd(1u << NL);
    }
    return m;
  }

  uint8_t rand_amo() {
    static const uint8_t ops[] = {OP_ADD, OP_SWAP, OP_XOR, OP_OR, OP_AND, OP_MIN, OP_MAX, OP_CAS};
    return ops[rnd(ZACAS ? 8 : 7)];
  }

  // size: log2 bytes, -1 for random; dst/addr: fixed when >= 0.
  void add_req(uint32_t c, uint32_t b, uint8_t kind, uint32_t mask, uint8_t amo_op,
               uint32_t addr_range, int dst, int addr, int size, uint64_t release, int wid = -1) {
    Req r;
    r.kind = kind;
    uint32_t nwb = (NW + NB - 1) / NB;
    r.wid = (wid >= 0) ? uint32_t(wid) : b + NB * rnd(nwb);
    if (r.wid >= NW) {
      r.wid = b;
    }
    r.mask = mask;
    r.release = release;
    r.lanes.assign(NL, -1);
    bool uns = chance(50);
    int sz = size;
    if (sz < 0) {
      sz = (kind == K_AMO || kind == K_LR || kind == K_SC) ? (chance(50) ? 2 : 3) : int(rnd(4));
    }
    for (uint32_t l = 0; l < NL; ++l) {
      if (!(mask >> l & 1)) {
        continue;
      }
      LaneOp op{};
      op.src = c;
      op.blk = b;
      op.lane = l;
      op.wid = r.wid;
      op.tid = l % NT;
      op.kind = kind;
      op.dst = (dst >= 0) ? uint32_t(dst) : pick_dst(c);
      op.waddr = (addr >= 0) ? uint32_t(addr) : rnd(addr_range);
      op.hart = make_hart(c, r.wid, op.tid);
      uint32_t bytes = 1u << sz;
      uint32_t off = rnd(8 / bytes) * bytes;
      op.byteen = uint8_t(((1u << bytes) - 1) << off);
      op.data = r64();
      // Half the compare-and-swaps expect the word's initial value, so some succeed.
      op.cmp = chance(50) ? init_word(op.dst, op.waddr) : r64();
      op.amo_op = (kind == K_LR) ? uint8_t(OP_LR) : (kind == K_SC) ? uint8_t(OP_SC) : amo_op;
      op.uns = (kind == K_AMO) && uns;
      op.seq = seq_[c]++;
      ops_.push_back(op);
      ++n_open_;
      r.lanes[l] = int(ops_.size()) - 1;
    }
    ports_[c * NB + b].queue.push_back(r);
  }

  // The latency schedule: fixed releases, always-ready sinks, fixed bank
  // latency. Phase A sends one isolated lane per (src, block, dst, op);
  // B a contended full-mask burst per port; C a store flood; D a load flood;
  // E an AMO hot spot on core 0. Lanes and addresses are fixed functions of
  // the indices; only the data words are hashed.
  void gen_latency() {
    uint64_t t = 16;
    uint32_t k = 0;
    const uint32_t full = (NL >= 32) ? ~0u : ((1u << NL) - 1);
    for (uint32_t c = 0; c < NCC; ++c) {
      for (uint32_t b = 0; b < NB; ++b) {
        for (uint32_t d = 0; d < NCC; ++d) {
          if (d == c) {
            continue;
          }
          for (uint8_t kind : {K_LD, K_ST, K_AMO}) {
            uint32_t lane = (c + d + kind) % NL;
            add_fixed(c, b, kind, 1u << lane, d, (c * 13 + d * 5 + kind) % 64, t + 40ull * k);
            ++k;
          }
        }
      }
    }
    uint64_t tb = t + 40ull * k + 64;
    for (uint32_t c = 0; c < NCC; ++c) {
      for (uint32_t b = 0; b < NB; ++b) {
        add_fixed(c, b, K_LD, full, (c + 1) % NCC, 64 + b * 8, tb);
        add_fixed(c, b, K_ST, full, (c + 1) % NCC, 80 + b * 8, tb);
        add_fixed(c, b, K_AMO, full, (c + NCC - 1) % NCC, 96 + b * 8, tb);
        add_fixed(c, b, K_LD, full, (c == 0) ? 1 : 0, 112 + b * 8, tb);
      }
    }
    uint64_t tc = tb + 400;
    for (uint32_t c = 0; c < NCC; ++c) {
      for (uint32_t b = 0; b < NB; ++b) {
        for (uint32_t j = 0; j < 3 * CREDITS / NL + 2; ++j) {
          add_fixed(c, b, K_ST, full, (c + 1 + j % (NCC - 1)) % NCC, 128 + (j % 8) * 4, tc);
        }
      }
    }
    uint64_t td = tc + 600;
    for (uint32_t c = 0; c < NCC; ++c) {
      for (uint32_t b = 0; b < NB; ++b) {
        for (uint32_t j = 0; j < 3 * PENDING / NL + 2; ++j) {
          add_fixed(c, b, K_LD, full, (c + 1 + j % (NCC - 1)) % NCC, 192 + (j % 8) * 4, td);
        }
      }
    }
    uint64_t te = td + 600;
    for (uint32_t c = 1; c < NCC; ++c) {
      for (uint32_t b = 0; b < NB; ++b) {
        for (uint32_t j = 0; j < 4; ++j) {
          add_fixed(c, b, K_AMO, full, 0, 0, te, true);
        }
      }
    }
  }

  // A deterministic request of full words: lane l addresses word `base + l`,
  // or every lane word `base` when `one_word`.
  void add_fixed(uint32_t c, uint32_t b, uint8_t kind, uint32_t mask, uint32_t dst,
                 uint32_t base, uint64_t release, bool one_word = false) {
    Req r;
    r.kind = kind;
    r.wid = b;
    r.mask = mask;
    r.release = release;
    r.lanes.assign(NL, -1);
    for (uint32_t l = 0; l < NL; ++l) {
      if (!(mask >> l & 1)) {
        continue;
      }
      LaneOp op{};
      op.src = c;
      op.blk = b;
      op.lane = l;
      op.wid = r.wid;
      op.tid = l % NT;
      op.kind = kind;
      op.dst = dst;
      op.waddr = (base + (one_word ? 0 : l)) % (1u << LMEM_AW);
      op.hart = make_hart(c, r.wid, op.tid);
      op.byteen = 0xff;
      op.seq = seq_[c]++;
      op.data = mix((uint64_t(c) << 40) ^ (uint64_t(op.seq) << 8) ^ l);
      op.cmp = 0;
      op.amo_op = (kind == K_AMO) ? uint8_t(OP_ADD) : uint8_t(0);
      op.uns = false;
      ops_.push_back(op);
      ++n_open_;
      r.lanes[l] = int(ops_.size()) - 1;
    }
    ports_[c * NB + b].queue.push_back(r);
  }

  void generate() {
    const std::string& m = opts_.mode;
    seq_.assign(NCC, 0);
    uint32_t reqs = opts_.reqs ? opts_.reqs : 300;
    if (m == "latency") {
      gen_latency();
      lsu_pct_ = 100;
      lmem_pct_ = 100;
      lat_fixed_ = true;
      return;
    }
    for (uint32_t c = 0; c < NCC; ++c) {
      for (uint32_t b = 0; b < NB; ++b) {
        for (uint32_t i = 0; i < reqs; ++i) {
          if (m == "random" || m == "backpressure") {
            uint32_t k = rnd(100);
            uint8_t kind = (k < 35) ? K_ST : (k < 65) ? K_LD : K_AMO;
            add_req(c, b, kind, rand_mask(), rand_amo(), 8, -1, -1, -1, 0);
          } else if (m == "hotspot") {
            if (c == 0) {
              continue;
            }
            add_req(c, b, K_AMO, rand_mask(), OP_ADD, 1, 0, 0, 3, 0);
          } else if (m == "storeflood") {
            add_req(c, b, K_ST, (1u << NL) - 1, 0, 64, -1, -1, -1, 0);
          } else if (m == "loadflood") {
            add_req(c, b, K_LD, (1u << NL) - 1, 0, 64, -1, -1, -1, 0);
          } else if (m == "lrsc") {
            uint32_t k = rnd(100);
            uint8_t kind = (k < 25) ? K_LR : (k < 50) ? K_SC : (k < 75) ? K_ST : K_LD;
            add_req(c, b, kind, rand_mask(), 0, 4, -1, -1, -1, 0);
          } else {
            std::fprintf(stderr, "unknown mode %s\n", m.c_str());
            std::exit(2);
          }
        }
      }
    }
    if (m == "hotspot") {
      hot_expect_ = image_[0][0];
      for (auto& op : ops_) {
        hot_expect_ += op.data;
      }
    }
    lsu_pct_ = (m == "backpressure") ? 10 : 70;
    lmem_pct_ = (m == "storeflood" || m == "loadflood") ? 100 : 70;
    lat_max_ = (m == "storeflood" || m == "loadflood") ? 24 : 6;
  }

  // ---- behavioural local memory -----------------------------------------

  static uint64_t amo_apply(uint64_t old, uint64_t rhs, uint8_t op, bool uns, uint32_t bits, uint64_t cmp) {
    uint64_t mask = (bits == 64) ? ~0ull : 0xffffffffull;
    old &= mask;
    rhs &= mask;
    cmp &= mask;
    auto sx = [&](uint64_t v) -> int64_t {
      return (bits == 64) ? int64_t(v) : int64_t(int32_t(uint32_t(v)));
    };
    switch (op) {
    case OP_ADD:  return (old + rhs) & mask;
    case OP_SWAP: return rhs;
    case OP_XOR:  return old ^ rhs;
    case OP_OR:   return old | rhs;
    case OP_AND:  return old & rhs;
    case OP_MIN:  return uns ? std::min(old, rhs) : ((sx(old) < sx(rhs)) ? old : rhs);
    case OP_MAX:  return uns ? std::max(old, rhs) : ((sx(old) > sx(rhs)) ? old : rhs);
    case OP_CAS:  return (old == cmp) ? rhs : old;
    default:      return old;
    }
  }

  // Executes a local-memory request on the image; returns the old word.
  uint64_t execute(uint32_t c, const LmemReqV& q) {
    uint64_t& w = image_[c][q.waddr];
    uint64_t old = w;
    if (q.rw) {
      uint64_t m = byte_mask(q.byteen);
      w = (w & ~m) | (q.data & m);
    } else if (q.amo) {
      if (q.byteen == 0xff) {
        w = amo_apply(old, q.data, q.amo_op, q.uns, 64, q.cmp);
      } else {
        uint32_t sh = (q.byteen == 0xf0) ? 32 : 0;
        uint64_t nv = amo_apply(old >> sh, q.data >> sh, q.amo_op, q.uns, 32, q.cmp >> sh);
        w = (w & ~(0xffffffffull << sh)) | ((nv & 0xffffffffull) << sh);
      }
    }
    return old;
  }

  static uint64_t key(uint32_t s, uint32_t b, uint32_t d, uint32_t a) {
    return (uint64_t(s) << 56) | (uint64_t(b) << 48) | (uint64_t(d) << 40) | a;
  }

  // ---- one clock cycle ----------------------------------------------------

  void clock() {
    const uint64_t t = cycle_;
    bool progress = false;

    for (uint32_t p = 0; p < NP; ++p) {
      in_.rsp_ready[p] = (lsu_pct_ >= 100) ? 1 : chance(lsu_pct_);
    }
    for (uint32_t c = 0; c < NCC; ++c) {
      in_.lmem_ready[c] = (lmem_pct_ >= 100) ? 1 : chance(lmem_pct_);
    }

    // Each bank completes its oldest due request; one response per core per cycle.
    for (uint32_t c = 0; c < NCC; ++c) {
      for (uint32_t k = 0; k < NBANKS; ++k) {
        auto& q = banks_[c][k];
        if (!q.empty() && q.front().ready <= t) {
          LmemOp lo = q.front();
          q.pop_front();
          uint64_t old = execute(c, lo.req);
          auto& op = ops_.at(lo.op);
          op.executed = true;
          op.result = old;
          rspq_[c].push_back({lo.req.tag, old, lo.op});
        }
      }
      in_.lrsp_valid[c] = !rspq_[c].empty();
      in_.lrsp_tag[c] = rspq_[c].empty() ? 0 : rspq_[c].front().tag;
      in_.lrsp_data[c] = rspq_[c].empty() ? 0 : rspq_[c].front().data;
    }

    // Offer each free port its next request, counted by the tracker as it issues.
    for (uint32_t p = 0; p < NP; ++p) {
      auto& port = ports_[p];
      in_.inc_valid[p] = 0;
      if (!port.offered && !port.queue.empty() && port.queue.front().release <= t) {
        Req& r = port.queue.front();
        bool needs_tag = (r.kind != K_ST);
        if (!needs_tag || !port.free_tags.empty()) {
          uint32_t tag = 0;
          if (needs_tag) {
            tag = port.free_tags.front();
            port.free_tags.pop_front();
          }
          ReqView& v = port.view;
          v.kind = r.kind;
          v.rw = (r.kind == K_ST);
          v.mask = r.mask;
          v.tag = tag;
          v.wid = r.wid;
          v.id = ++req_ids_;
          v.lanes.assign(NL, LaneView());
          port.view_ops = r.lanes;
          for (uint32_t l = 0; l < NL; ++l) {
            int id = r.lanes[l];
            if (id < 0) {
              continue;
            }
            auto& op = ops_[id];
            op.tag = tag;
            op.issue = t;
            auto& lv = v.lanes[l];
            lv.dst = op.dst;
            lv.waddr = op.waddr;
            lv.hart = op.hart;
            lv.tid = op.tid;
            lv.data = op.data;
            lv.cmp = op.cmp;
            lv.byteen = op.byteen;
            lv.amo_op = op.amo_op;
            lv.amo = (op.kind == K_AMO || op.kind == K_LR || op.kind == K_SC);
            lv.uns = op.uns;
            if (needs_tag) {
              port.waiting[tag].push_back(id);
            }
            if (op.kind != K_SC) {
              fifo_[key(op.src, op.blk, op.dst, op.waddr)].push_back(id);
            }
            if (op.kind == K_ST) {
              st_fifo_[op.src][op.wid].push_back(id);
            }
          }
          if (r.kind == K_ST) {
            uint32_t cnt = __builtin_popcount(r.mask);
            in_.inc_valid[p] = 1;
            in_.inc_wid[p] = r.wid;
            in_.inc_cnt[p] = cnt;
            wt_cnt_[p / NB][r.wid] += cnt;
          }
          port.offered = true;
          port.queue.pop_front();
        }
      }
      in_.req[p] = port.offered ? &port.view : nullptr;
    }

    dut_.cycle(in_, out_);

    for (uint32_t c = 0; c < NCC; ++c) {
      if (!out_.sinks_ready[c]) {
        fail(strf("core %u: an always-ready response sink refused", c));
      }
    }

    for (uint32_t p = 0; p < NP; ++p) {
      auto& port = ports_[p];
      if (out_.req_fire[p]) {
        if (!port.offered) {
          fail(strf("port %u: request accepted while none was offered", p));
        }
        port.offered = false;
        progress = true;
        event(strf("%" PRIu64 " R %u %" PRIu64, t, p, port.view.id));
      }
      const RspV& rsp = out_.rsp[p];
      if (!rsp.fire) {
        continue;
      }
      progress = true;
      if (__builtin_popcount(rsp.mask) != 1) {
        fail(strf("port %u: response mask 0x%x is not one lane", p, rsp.mask));
        continue;
      }
      uint32_t l = __builtin_ctz(rsp.mask);
      uint64_t data = rsp.data.at(l);
      event(strf("%" PRIu64 " D %u %u %u %016" PRIx64, t, p, rsp.tag, l, data));
      auto it = port.waiting.find(rsp.tag);
      if (it == port.waiting.end()) {
        fail(strf("port %u: response for idle tag %u", p, rsp.tag));
        continue;
      }
      auto& w = it->second;
      auto f = std::find_if(w.begin(), w.end(), [&](int id) { return ops_[id].lane == l; });
      if (f == w.end()) {
        fail(strf("port %u: duplicate or unexpected response tag %u lane %u", p, rsp.tag, l));
        continue;
      }
      auto& op = ops_[*f];
      uint64_t expect;
      if (op.kind == K_SC) {
        expect = (op.byteen & 1) ? 1ull : (1ull << 32);
      } else {
        if (!op.executed) {
          fail(strf("port %u: response tag %u lane %u before execution", p, rsp.tag, l));
        }
        expect = op.result;
      }
      if (data != expect) {
        fail(strf("port %u tag %u lane %u %s: data 0x%016" PRIx64 " expected 0x%016" PRIx64,
                  p, rsp.tag, l, kind_name(op.kind), data, expect));
      }
      op.responded = true;
      op.done = t;
      --n_open_;
      w.erase(f);
      if (w.empty()) {
        port.waiting.erase(it);
        port.free_tags.push_back(rsp.tag);
      }
    }

    for (uint32_t c = 0; c < NCC; ++c) {
      const LmemReqV& q = out_.lmem[c];
      if (q.fire) {
        progress = true;
        ++n_lmem_;
        bool data_used = q.rw || q.amo;
        event(strf("%" PRIu64 " M %u %d %d %u %d %u %02x %016" PRIx64 " %016" PRIx64 " %x %u",
                   t, c, q.rw, q.amo, q.amo ? q.amo_op : 0, q.amo && q.uns, q.waddr, q.byteen,
                   data_used ? (q.data & byte_mask(q.byteen)) : 0ull,
                   (q.amo && ZACAS) ? q.cmp : 0ull, q.hart, q.tag));
        if (!q.local) {
          fail(strf("core %u: local-memory request without is_addr_local", c));
        }
        uint32_t src = q.hart >> (NW_BITS + NT_BITS);
        uint32_t wid = (q.hart >> NT_BITS) & ((1u << NW_BITS) - 1);
        uint32_t blk = wid % NB;
        auto fit = fifo_.find(key(src, blk, c, q.waddr));
        if (src >= NCC || fit == fifo_.end() || fit->second.empty()) {
          fail(strf("core %u: unexpected local-memory request hart 0x%x addr %u", c, q.hart, q.waddr));
        } else {
          int id = fit->second.front();
          fit->second.pop_front();
          auto& op = ops_[id];
          bool exp_rw = (op.kind == K_ST);
          bool exp_amo = (op.kind == K_AMO);
          bool ok = (q.rw == exp_rw) && (q.amo == exp_amo) && (q.byteen == op.byteen) && (q.hart == op.hart);
          if (exp_rw || exp_amo) {
            uint64_t m = byte_mask(op.byteen);
            ok = ok && ((q.data & m) == (op.data & m));
          }
          if (exp_amo) {
            ok = ok && (q.amo_op == op.amo_op) && (q.uns == op.uns) && (!ZACAS || q.cmp == op.cmp);
          }
          if (!ok) {
            fail(strf("core %u: local-memory request out of order or corrupted: src %u blk %u addr %u %s",
                      c, src, blk, q.waddr, kind_name(op.kind)));
          }
          uint32_t bank = q.waddr % NBANKS;
          uint64_t lat = lat_fixed_ ? 1 : rnd(lat_max_);
          uint64_t ready = std::max(bank_last_[c][bank], t + 1 + lat);
          bank_last_[c][bank] = ready;
          banks_[c][bank].push_back({q, id, ready});
        }
        ++lmem_out_[c];
        if (lmem_out_[c] > SLOTS) {
          fail(strf("core %u: %u local-memory requests outstanding", c, lmem_out_[c]));
        }
      }
      if (in_.lrsp_valid[c]) {
        auto r = rspq_[c].front();
        rspq_[c].pop_front();
        auto& op = ops_.at(r.op);
        if (op.kind == K_ST) {
          ++committed_[op.src][op.wid];
        }
        --lmem_out_[c];
        progress = true;
      }
      if (out_.ack_valid[c]) {
        uint32_t w = out_.ack_wid[c];
        progress = true;
        event(strf("%" PRIu64 " A %u %u", t, c, w));
        if (w >= NW) {
          fail(strf("core %u: ack for warp %u", c, w));
        } else {
          --wt_cnt_[c][w];
          ++acked_[c][w];
          if (wt_cnt_[c][w] < 0) {
            fail(strf("core %u warp %u: ack without a counted store", c, w));
          }
          if (acked_[c][w] > committed_[c][w]) {
            fail(strf("core %u warp %u: ack before the store committed", c, w));
          }
          auto& sf = st_fifo_[c][w];
          if (sf.empty()) {
            fail(strf("core %u warp %u: ack without an issued store", c, w));
          } else {
            ops_[sf.front()].done = t;
            ops_[sf.front()].responded = true;
            --n_open_;
            sf.pop_front();
          }
        }
      }
      max_st_ = std::max(max_st_, out_.st_out[c]);
      max_ld_ = std::max(max_ld_, out_.ld_out[c]);
      max_slot_ = std::max(max_slot_, out_.slot_out[c]);
      if (out_.st_out[c] > CREDITS) {
        fail(strf("core %u: %u stores outstanding", c, out_.st_out[c]));
      }
      if (out_.ld_out[c] > PENDING) {
        fail(strf("core %u: %u loads/AMOs outstanding", c, out_.ld_out[c]));
      }
      if (out_.slot_out[c] > dut_.slot_bound()) {
        fail(strf("core %u: %u ingress requests outstanding", c, out_.slot_out[c]));
      }
    }

    ++cycle_;
    if (progress) {
      last_progress_ = cycle_;
    }

    // After the edge, drained[w] must say the warp has no unacknowledged store.
    for (uint32_t c = 0; c < NCC; ++c) {
      uint32_t expect = 0;
      for (uint32_t w = 0; w < NW; ++w) {
        expect |= (wt_cnt_[c][w] == 0) ? (1u << w) : 0;
      }
      if (out_.drained[c] != expect) {
        fail(strf("core %u: drained=0x%x, expected 0x%x", c, out_.drained[c], expect));
      }
      if (out_.drained[c] != drained_[c]) {
        drained_[c] = out_.drained[c];
        event(strf("%" PRIu64 " W %u %x", t, c, drained_[c]));
      }
      uint32_t e = (out_.eg_empty[c] ? 4 : 0) | (out_.in_empty[c] ? 2 : 0) | (out_.wt_empty[c] ? 1 : 0);
      if (e != empty_[c]) {
        empty_[c] = e;
        event(strf("%" PRIu64 " E %u %x", t, c, e));
      }
    }
  }

  bool outstanding() const {
    for (auto& p : ports_) {
      if (p.offered || !p.queue.empty() || !p.waiting.empty()) {
        return true;
      }
    }
    if (n_open_ != 0) {
      return true;
    }
    for (uint32_t c = 0; c < NCC; ++c) {
      if (!rspq_[c].empty()) {
        return true;
      }
      for (uint32_t w = 0; w < NW; ++w) {
        if (wt_cnt_[c][w] != 0) {
          return true;
        }
      }
    }
    return false;
  }

  void final_check() {
    for (auto& op : ops_) {
      if (op.kind != K_SC && !op.executed) {
        fail(strf("core %u lane %u seq %u never executed", op.src, op.lane, op.seq));
        return;
      }
      if (!op.responded) {
        fail(strf("core %u lane %u seq %u never completed", op.src, op.lane, op.seq));
        return;
      }
    }
    for (auto& kv : fifo_) {
      if (!kv.second.empty()) {
        fail("an issued lane never reached the local memory");
        return;
      }
    }
    for (uint32_t c = 0; c < NCC; ++c) {
      if (!out_.eg_empty[c]) {
        fail(strf("core %u: egress not empty at the end", c));
      }
      if (!out_.in_empty[c]) {
        fail(strf("core %u: ingress not empty at the end", c));
      }
      if (!out_.wt_empty[c]) {
        fail(strf("core %u: store tracker not empty at the end", c));
      }
      for (uint32_t w = 0; w < NW; ++w) {
        if (acked_[c][w] != committed_[c][w]) {
          fail(strf("core %u warp %u: %" PRIu64 " acks for %" PRIu64 " stores", c, w, acked_[c][w], committed_[c][w]));
        }
      }
    }
    if (opts_.mode == "hotspot" && image_[0][0] != hot_expect_) {
      fail(strf("hot spot: word 0x%016" PRIx64 ", expected 0x%016" PRIx64, image_[0][0], hot_expect_));
    }
  }

  struct LmemRsp {
    uint32_t tag;
    uint64_t data;
    int op;
  };

  Dut& dut_;
  Options opts_;
  std::ostream* trace_;
  std::mt19937 rng_;
  In in_;
  Out out_;
  uint64_t cycle_ = 0;
  uint64_t last_progress_ = 0;
  uint64_t done_cycle_ = 0;
  uint32_t failures_ = 0;
  std::string first_failure_;
  uint64_t digest_ = 0xcbf29ce484222325ull;
  uint64_t req_ids_ = 0;
  uint64_t n_open_ = 0;                                // lane ops not yet completed
  std::vector<Port> ports_;
  std::vector<LaneOp> ops_;
  std::vector<uint32_t> seq_;
  std::map<uint64_t, std::deque<int>> fifo_;            // (src, blk, dst, addr) -> lanes in issue order
  std::vector<std::vector<uint64_t>> image_;
  std::vector<std::vector<std::deque<LmemOp>>> banks_;
  std::vector<std::vector<uint64_t>> bank_last_;
  std::vector<std::deque<LmemRsp>> rspq_;
  std::vector<uint32_t> lmem_out_;
  std::vector<std::vector<int64_t>> wt_cnt_;
  std::vector<std::vector<uint64_t>> committed_;
  std::vector<std::vector<uint64_t>> acked_;
  std::vector<std::vector<std::deque<int>>> st_fifo_;  // (src, wid) -> stores in issue order
  std::vector<uint32_t> drained_;
  std::vector<uint32_t> empty_;                        // egress, ingress, tracker empty bits
  uint32_t lsu_pct_ = 70;
  uint32_t lmem_pct_ = 70;
  uint32_t lat_max_ = 6;
  bool lat_fixed_ = false;
  uint64_t hot_expect_ = 0;
  uint64_t n_lmem_ = 0;
  uint32_t max_st_ = 0;
  uint32_t max_ld_ = 0;
  uint32_t max_slot_ = 0;
};

inline const std::vector<std::string>& all_modes() {
  static const std::vector<std::string> modes = {
    "random", "hotspot", "storeflood", "loadflood", "backpressure", "lrsc", "latency"};
  return modes;
}

} // namespace ep
