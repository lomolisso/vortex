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

// The cases and checks of the SM array's shared-memory bench, run unchanged by
// the RTL testbench (VX_sm_sram) and its SimX twin (SmSram) through a driver
// with the same cycle contract. Every expectation comes from smem_bank_ref.h at
// the build's knobs. The array adds a constant to every read latency (its share
// of the pipeline overhead), so read latencies are checked as golden + K, K
// measured on the first case; occupancy, order and data are checked exactly.
// Data is the array's own: the window is first filled with word_value(), every
// store writes that value back and every atomic adds 0 or compares with 0, so
// each read returns the pattern; the CAS swap case restores it. mbarrier ops run on words above the
// window, against mbar_ref.h. Each run writes a CSV of every request's
// present/accept/response/commit cycle, which must be identical between the
// two models.

#include <smem_bank_ref.h>
#include <l1_slot_ref.h>
#include <mbar_ref.h>
#include "l1h100_cases.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace sm_sram_cases {

using namespace vortex::smem_bank_ref;

constexpr uint32_t MAX_LANES = 32;
constexpr uint32_t pow2_ceil(uint32_t n) { return n <= 1 ? 1 : 2 * pow2_ceil((n + 1) / 2); }
// Tags per block: the bench's 6-bit field, capped at the LSU's pending entries so a tag survives the LSU tag width.
constexpr uint32_t NUM_TAGS  = std::min<uint32_t>(64, pow2_ceil(VX_CFG_LSU_PENDING_SIZE));
constexpr uint32_t ROW_TAGS  = 256;     // the bench's row-read tag: 8 bits
constexpr uint64_t WINDOW    = 1 << 16; // byte window the cases address

static_assert(WINDOW <= VX_CFG_SMEM_PER_SM_MAX, "the cases' window must fit the reset carve-out");

enum Kind : uint32_t { K_LD = 0, K_ST = 1, K_ADD = 2, K_CAS = 3, K_MBAR = 4 };

struct Req {
  std::string name;
  uint32_t kind  = K_LD;
  uint32_t mask  = 0;
  uint32_t bytes = 4;
  bool     lacc  = false;
  uint64_t addr[MAX_LANES] = {};
  uint32_t mop   = 0;      // K_MBAR: the mbar_ref op
  bool     own   = false;  // lanes send val and expect exp instead of the pattern
  uint64_t val[MAX_LANES] = {};
  uint64_t exp[MAX_LANES] = {};
  uint64_t cmp[MAX_LANES] = {};   // own K_CAS: the comparands of the second beat, placed like the data
};

enum EvType : uint32_t {
  EV_ACCEPT = 0, EV_RSP = 1, EV_COMMIT = 2, EV_ROW_GNT = 3, EV_ROW_RSP = 4, EV_MBAR = 5,
  EV_WR_GNT = 6, EV_AMO_GNT = 7, EV_AMO_RSP = 8, EV_CPA_ACC = 9, EV_CPA_ACK = 10, EV_DS_ACC = 11,
  EV_DS_BUSY = 12, EV_GL_ACC = 13, EV_GL_RSP = 14, EV_GL_MEMRD = 15, EV_GL_FILL = 16, EV_GL_FLUSHED = 17
};

// The bench's row clients: reads RD_TCU/RD_DXA/RD_DSMEM, the row writes of
// WR_DXA, the DSMEM slots (merged into WR_DSMEM writes), the cp.async lane
// writes of LSU block 0, and two one-lane atomic clients (AMO_DXA, AMO_DSMEM).
constexpr uint32_t ROW_RD_CLIENTS = 3;
constexpr uint32_t AMO1_CLIENTS   = 2;
// VX_gpu_pkg's SM_SRAM_AMO_STAGES: an AMO read grant's write-back cycle.
constexpr uint32_t AMO_WB_DELAY   = 2;

struct Event {
  uint32_t type;
  uint32_t blk;      // COMMIT: the commit kind (0 wavefront, 1 LSU atomic, 2 mbarrier op); MBAR: the bank;
                     // ROW_*, AMO_*: the client
  uint32_t tag;      // ROW_*, WR_GNT, AMO_*, CPA_*, DS_ACC: the client request's tag; MBAR: the word's row
  bool     data_ok;  // RSP, ROW_RSP, AMO_RSP: every lane or byte holds the pattern; CPA_ACK: the mask
  uint32_t cnt = 0;  // COMMIT: counted lanes; MBAR: the phase flipped; DS_BUSY: the merge register's new state;
                     // GL_RSP: the lanes; GL_MEMRD: the sectors (tag: the line); GL_FILL: the array row
};

// The global leg: the L1 in the array's top rows, driven as hw/unittest/l1h100 does.
namespace gl = ::l1h100_cases;
constexpr uint32_t GL_TAGS = std::min<uint32_t>(gl::NUM_TAGS, NUM_TAGS);

// A global lane whose returned bytes were wrong, as the driver saw it.
struct GlBad {
  uint32_t blk, tag, lane;
  uint64_t word;
};

// A global lane's returned bytes against the value fixed at submission.
inline bool gl_lane_ok(const gl::Req& r, uint32_t i, uint64_t word) {
  return r.kind == gl::K_ST || ((word >> (8 * (r.addr[i] % gl::WORD))) & gl::value_mask(r.bytes)) == r.exp[i];
}

inline SmemCfg knob_cfg() {
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  return SmemCfg{VX_CFG_LMEM_NUM_BANKS, VX_CFG_LMEM_BANK_BYTES, smem_pipe_stages(VX_CFG_NUM_OPCS, true)};
#else
  return SmemCfg{VX_CFG_LMEM_NUM_BANKS, VX_CFG_LMEM_BANK_BYTES, smem_pipe_stages(VX_CFG_NUM_OPCS, false)};
#endif
}

// The bench sees a CAS taken with its second (comparand) beat.
constexpr uint32_t CAS_BEATS = SMEM_CAS_BEATS;

// Read and wavefront arbitration of the array's ports; the bench's shadow has
// no AMO client, so the write-back delay plays no part.
inline SramPortCfg knob_port_cfg() {
  SramPortCfg c{};
  c.dsmem_wr_bytes = VX_CFG_SM_SRAM_DSMEM_WR_BYTES;
  c.rd_pri[RD_TCU]   = VX_CFG_SM_SRAM_RD_PRI_TCU;
  c.rd_pri[RD_DXA]   = VX_CFG_SM_SRAM_RD_PRI_DXA;
  c.rd_pri[RD_DSMEM] = VX_CFG_SM_SRAM_RD_PRI_DSMEM;
  c.rd_pri[RD_AMO]   = VX_CFG_SM_SRAM_RD_PRI_AMO;
  c.rd_pri[RD_LSU]   = VX_CFG_SM_SRAM_RD_PRI_LSU;
  c.wr_pri[WR_L1FILL] = VX_CFG_SM_SRAM_WR_PRI_L1FILL;
  c.wr_pri[WR_DXA]    = VX_CFG_SM_SRAM_WR_PRI_DXA;
  c.wr_pri[WR_DSMEM]  = VX_CFG_SM_SRAM_WR_PRI_DSMEM;
  c.wr_pri[WR_CPASYNC] = VX_CFG_SM_SRAM_WR_PRI_CPASYNC;
  c.wr_pri[WR_LSU]    = VX_CFG_SM_SRAM_WR_PRI_LSU;
  c.wf_pri_shared = VX_CFG_SM_SRAM_WF_PRI_SHARED;
  c.wf_pri_global = VX_CFG_SM_SRAM_WF_PRI_GLOBAL;
  c.amo_pri[AMO_DXA]   = VX_CFG_SM_SRAM_AMO_PRI_DXA;
  c.amo_pri[AMO_DSMEM] = VX_CFG_SM_SRAM_AMO_PRI_DSMEM;
  c.amo_pri[AMO_MBAR]  = VX_CFG_SM_SRAM_AMO_PRI_MBAR;
  c.amo_pri[AMO_LSU]   = VX_CFG_SM_SRAM_AMO_PRI_LSU;
  c.bank_pairs = VX_CFG_LMEM_NUM_BANKS / 2;
  c.amo_wb_delay = 0;
  return c;
}

#ifdef VX_CFG_EXT_MBAR_ENABLE
namespace mbr = vortex::mbar_ref;

// mbarrier words: one per bank pair from the end of the pattern window.
constexpr uint64_t MBAR_BASE  = WINDOW;
constexpr uint32_t MBAR_WORDS = VX_CFG_LMEM_NUM_BANKS / 2;
static_assert(MBAR_BASE + 8 * MBAR_WORDS <= VX_CFG_SMEM_PER_SM_MAX, "the mbarrier words must fit the reset carve-out");

#endif

// The array's split at shared-memory size s_bytes (smem_cfg_t).
struct ArrayCfg {
  uint32_t s_rows, l1_lines, l1_protect;
};

inline ArrayCfg array_cfg(uint32_t s_bytes) {
  vortex::l1_slot_ref::L1SlotCfg c{};
  c.s_bytes = s_bytes;
  c.sram_bytes = VX_CFG_SM_SRAM_BYTES;
  c.reserve_bytes = VX_CFG_L1_RESERVE_BYTES;
  c.row_bytes = VX_CFG_SM_DATAPATH_BYTES;
  c.protect_pct = VX_CFG_L1_REPL_PROTECT_PCT;
  c.mshr_lines = VX_CFG_SM_MAX_READ_LINES;
  return {c.s_bytes / c.row_bytes, vortex::l1_slot_ref::l1_lines(c), vortex::l1_slot_ref::l1_protect(c)};
}

// The split at reset (the largest shared-memory size), as both models hold it.
inline ArrayCfg reset_cfg() { return array_cfg(VX_CFG_SMEM_PER_SM_MAX); }

// The pattern at a lane word (window lane-word address).
inline uint64_t word_value(uint64_t word) {
  return (word + 1) * 0x9E3779B97F4A7C15ull;
}

// The data a lane sends: the pattern for stores, 0 for atomics (add 0,
// compare with 0, which never matches the nonzero pattern).
template <uint32_t WORD_SIZE>
uint64_t lane_data(const Req& r, uint32_t i) {
  if (r.own)
    return r.val[i];
  return (r.kind == K_ST) ? word_value(r.addr[i] / WORD_SIZE) : 0;
}

// A CAS lane's comparand (its second beat): 0 unless the request carries its own.
template <uint32_t WORD_SIZE>
uint64_t lane_cmp(const Req& r, uint32_t i) {
  return r.own ? r.cmp[i] : 0;
}

// The word a lane's response returns.
template <uint32_t WORD_SIZE>
uint64_t lane_expect(const Req& r, uint32_t i) {
  return r.own ? r.exp[i] : word_value(r.addr[i] / WORD_SIZE);
}

// The lane word's bytes a request touches.
template <uint32_t WORD_SIZE>
uint64_t lane_bytes(const Req& r, uint32_t i) {
  uint64_t m = (r.bytes >= 8) ? ~0ull : ((1ull << (r.bytes * 8)) - 1);
  return m << ((r.addr[i] % WORD_SIZE) * 8);
}

// Prints the first lanes whose response differs from the pattern.
inline void report_lane(uint32_t blk, uint32_t tag, uint32_t lane, uint64_t addr, uint64_t got, uint64_t exp) {
  static uint32_t n = 0;
  if (n++ < 8)
    std::printf("  lane data: block %u tag %u lane %u addr 0x%llx got 0x%016llx expected 0x%016llx\n", blk, tag, lane,
                (unsigned long long)addr, (unsigned long long)got, (unsigned long long)exp);
}

inline bool is_read(const Req& r) { return r.kind != K_ST; }
inline bool is_amo(const Req& r) { return r.kind == K_ADD || r.kind == K_CAS; }
inline bool is_mbar(const Req& r) { return r.kind == K_MBAR; }

// An mbarrier op leaves the wavefront stage for the op queue in one cycle.
inline uint32_t cost(const Req& r, const SmemCfg& c, uint32_t nlanes) {
  if (is_mbar(r))
    return 1;
  if (is_amo(r))
    return smem_atom_stage_cycles(smem_atom_passes(r.addr, r.mask, nlanes, r.bytes, c), r.kind == K_CAS);
  const uint32_t n = smem_wavefronts(r.addr, r.mask, nlanes, r.bytes, c);
  return n ? n : 1;
}

// Cycles from a request's presentation on an idle port to its acceptance.
inline uint32_t accept_delay(bool cas) { return cas ? CAS_BEATS - 1 : 0; }

// A read's last wavefront grant to its response: that wavefront's bank read, then the response register.
constexpr long RSP_PATH = 1 + (long)SMEM_RSP_STAGES;

// Response time from the request's (last beat's) acceptance.
inline uint32_t latency(const Req& r, const SmemCfg& c, uint32_t nlanes) {
  if (is_mbar(r))
    return smem_atom_latency(1, c) + MBAR_QUEUE_STAGES;
  if (is_amo(r)) {
    return smem_atom_latency(smem_atom_passes(r.addr, r.mask, nlanes, r.bytes, c), c, r.kind == K_CAS) -
           accept_delay(r.kind == K_CAS);
  }
  if (r.kind == K_LD)
    return smem_load_latency(smem_wavefronts(r.addr, r.mask, nlanes, r.bytes, c), c);
  return 0;
}


// lanes from lane 0; lane i at base + (i / group) * stride_bytes.
inline Req strided(const std::string& name, uint32_t kind, uint32_t bytes, uint32_t lanes, uint32_t nlanes,
                   uint32_t stride_bytes, uint32_t group = 1, uint64_t base = 0) {
  Req r;
  r.name = name;
  r.kind = kind;
  r.bytes = bytes;
  for (uint32_t i = 0; i < lanes && i < nlanes; ++i) {
    r.mask |= 1u << i;
    r.addr[i] = (base + (uint64_t)(i / group) * stride_bytes) % WINDOW;
  }
  return r;
}

inline std::vector<Req> directed(uint32_t nlanes, bool lacc_ok) {
  std::vector<Req> v;
  const uint32_t W = nlanes;
  v.push_back(strided("ld32 one lane", K_LD, 4, 1, W, 4));
  for (uint32_t s : {1u, 2u, 3u, 4u, 8u, 16u, 32u, 33u, 64u})
    v.push_back(strided("ld32 stride " + std::to_string(s), K_LD, 4, W, W, s * 4));
  for (uint32_t d : {16u, 8u, 2u, 1u})
    v.push_back(strided("ld32 " + std::to_string(d) + " distinct words", K_LD, 4, W, W, 4, (W + d - 1) / d));
  for (uint32_t n : {1u, 2u, 4u, 8u, 16u, 24u, 32u})
    v.push_back(strided("ld64 " + std::to_string(n) + " lanes", K_LD, 8, n, W, 8));
  for (uint32_t s : {2u, 4u, 8u, 16u})
    v.push_back(strided("ld64 stride " + std::to_string(s), K_LD, 8, W, W, s * 8));
  v.push_back(strided("ld64 one address", K_LD, 8, W, W, 8, W));
  for (uint32_t s : {1u, 2u, 32u})
    v.push_back(strided("st32 stride " + std::to_string(s), K_ST, 4, W, W, s * 4));
  v.push_back(strided("st64 stride 1", K_ST, 8, W, W, 8));
#if VX_CFG_EXT_A_ENABLED
  v.push_back(strided("add one lane", K_ADD, 4, 1, W, 4));
  for (uint32_t k : {1u, 2u, 4u, 8u, 16u, 32u})
    v.push_back(strided("add " + std::to_string(k) + " per address", K_ADD, 4, W, W, 4, k));
  v.push_back(strided("add stride 32 words", K_ADD, 4, W, W, 128));
  v.push_back(strided("add64 2 per address", K_ADD, 8, W, W, 8, 2));
#ifdef VX_CFG_EXT_ZACAS_ENABLE
  v.push_back(strided("cas one lane", K_CAS, 4, 1, W, 4));
  v.push_back(strided("cas 32 per address", K_CAS, 4, W, W, 4, 32));
  v.push_back(strided("cas consecutive", K_CAS, 4, W, W, 4));
#endif
#endif
  if (lacc_ok) {
    Req r = strided("ld32 counted, one address", K_LD, 4, W, W, 4, W);
    r.lacc = true;
    v.push_back(r);
    Req s = strided("st32 counted, stride 1", K_ST, 4, W, W, 4);
    s.lacc = true;
    v.push_back(s);
#if VX_CFG_EXT_A_ENABLED
    Req a = strided("add counted, 2 per address", K_ADD, 4, W, W, 4, 2);
    a.lacc = true;
    v.push_back(a);
#endif
  }
  return v;
}

inline Req random_req(std::mt19937& rng, uint32_t nlanes, bool lacc_ok) {
  Req r;
  uint32_t k = rng() % 16;
#if VX_CFG_EXT_A_ENABLED
  r.kind = (k < 9) ? K_LD : (k < 13) ? K_ST : K_ADD;
#ifdef VX_CFG_EXT_ZACAS_ENABLE
  if (k == 15)
    r.kind = K_CAS;
#endif
#else
  r.kind = (k < 10) ? K_LD : K_ST;
#endif
  r.bytes = (r.kind == K_CAS || rng() % 3) ? 4 : 8;
  r.name = "random";
  r.lacc = lacc_ok && (rng() % 8 == 0);
  const uint32_t span = 4u << (rng() % 9);       // 4 B .. 1 KiB: few to many conflicts
  const uint64_t base = (rng() % (WINDOW / 1024)) * 1024;
  for (uint32_t i = 0; i < nlanes; ++i) {
    if (rng() % 8 == 0)
      continue;
    r.mask |= 1u << i;
    r.addr[i] = (base + (rng() % span)) / r.bytes * r.bytes % WINDOW;
  }
  if (r.mask == 0) {
    r.mask = 1;
    r.addr[0] = base;
  }
  return r;
}

struct Rec {
  uint32_t id = 0, blk = 0, tag = 0, cost = 0, gold = 0, lacc = 0;
  uint64_t present = 0, accept = 0, rsp = 0, commit = 0;
  bool accepted = false, responded = false, committed = false, read = false, amo = false, mbar = false, cas = false,
       cpa = false, logged = true;
  std::string name;
};

// Driver contract (both models): push() queues a request at a block's input,
// held valid from the next cycle() until accepted; push_row() queues a TCU
// row read the same way; cycle() applies the inputs, samples the outputs
// into events, and advances one clock. A request carries its lanes' data
// (lane_data()) and byte enables (lane_bytes()).
template <typename Driver, uint32_t WORD_SIZE>
class Runner {
public:
  Runner(Driver& d, uint32_t nlanes, uint32_t nblocks, FILE* csv, bool lacc_ok)
    : d_(d), nlanes_(nlanes), nblocks_(nblocks), csv_(csv), lacc_ok_(lacc_ok), cfg_(knob_cfg()) {
    dir_.configure(gl::knob_l1_cfg());
    for (uint32_t b = 0; b < nblocks_; ++b) {
      for (uint32_t t = 0; t < GL_TAGS; ++t)
        gl_free_[b].push_back(t);
    }
  }

  // unified: every client of the array's ports against the port model
  // (random_reqs = cycles of the random mix) instead of the shared-path cases.
  int run(uint32_t seed, uint32_t random_reqs, bool unified = false, bool directed = false) {
    this->preload();
    this->isolated();
#if VX_CFG_EXT_A_ENABLED && defined(VX_CFG_EXT_ZACAS_ENABLE)
    this->cas_swap();
#endif
    if (directed) {
      this->global_directed(seed);
      std::printf("sm_sram directed: %u checks, %u errors (%u global, %u fills)\n", checks_, errors_, gl_n_, gl_nfills_);
      return errors_ ? 1 : 0;
    }
    if (unified) {
      this->ports(seed, random_reqs);
      this->amo_together();
      this->cfg_switch();
      this->global_seq(seed, std::max<uint32_t>(64, random_reqs / 8));
      this->global_mix(seed, random_reqs);
      std::printf("sm_sram unified: %u checks, %u errors (K=%d, row read %d, one-lane atomic %d, %u global, %u fills)\n",
                  checks_, errors_, k_, row_lat_, amo_lat_, gl_n_, gl_nfills_);
      return errors_ ? 1 : 0;
    }
    this->occupancy();
    this->row_reads();
    this->ext_client();
    if (nblocks_ > 1)
      this->blocks();
    this->random(seed, random_reqs);
#ifdef VX_CFG_EXT_MBAR_ENABLE
    this->mbar_isolated();
    this->mbar_streams();
    this->mbar_riders();
    this->mbar_cpa_arrive();
#endif
    for (const auto& r : recs_) {
      if (r.lacc)
        this->check(r.committed, "%s: %ld counted lanes never committed (id %ld)", r.name, r.lacc, r.id);
    }
    std::printf("sm_sram: %u checks, %u errors (K=%d, row read %d)\n", checks_, errors_, k_, row_lat_);
    return errors_ ? 1 : 0;
  }

private:
  void check(bool ok, const char* fmt, const std::string& a, long x, long y) {
    ++checks_;
    if (!ok) {
      ++errors_;
      if (errors_ <= 40) {
        std::printf("  MISMATCH ");
        std::printf(fmt, a.c_str(), x, y);
        std::printf("\n");
      }
    }
    if (errors_ >= 200) {
      std::printf("sm_sram: stopping after %u errors\nFAILED!\n", errors_);
      std::exit(1);
    }
  }

  uint32_t submit(uint32_t blk, const Req& r) {
    Rec rec;
    rec.id = (uint32_t)recs_.size();
    rec.blk = blk;
    rec.tag = next_tag_[blk];
    next_tag_[blk] = (next_tag_[blk] + 1) % NUM_TAGS;
    rec.cost = cost(r, cfg_, nlanes_);
    rec.gold = latency(r, cfg_, nlanes_);
    rec.read = is_read(r);
    rec.amo = is_amo(r);
    rec.mbar = is_mbar(r);
    rec.cas = (r.kind == K_CAS);
    rec.lacc = r.lacc ? __builtin_popcount(r.mask) : 0;
    rec.present = d_.now();
    rec.name = r.name;
    d_.push(blk, r, rec.tag);
    live_[blk][rec.tag] = rec.id;
    recs_.push_back(rec);
    return rec.id;
  }

  void push_row(uint32_t row, uint32_t client = RD_TCU) {
    uint32_t tag = next_row_tag_;
    next_row_tag_ = (next_row_tag_ + 1) % ROW_TAGS;
    rows_[tag] = RowRec{0, false};
    d_.push_row(client, row % (WINDOW / VX_CFG_SM_DATAPATH_BYTES), tag);
  }

  // A DXA row write of the row's own pattern.
  void push_wr(uint32_t row) {
    uint32_t tag = next_wr_tag_;
    next_wr_tag_ = (next_wr_tag_ + 1) % ROW_TAGS;
    ++wr_pend_;
    d_.push_wr(row % (WINDOW / VX_CFG_SM_DATAPATH_BYTES), tag);
  }

  // A one-lane 8-B atomic add of 0 at a window lane word (a DXA completion, a DSMEM AMO slot).
  void push_amo1(uint32_t client, uint64_t addr) {
    uint32_t tag = next_amo_tag_;
    next_amo_tag_ = (next_amo_tag_ + 1) % ROW_TAGS;
    amo_live_[tag] = AmoRec{0, false};
    d_.push_amo(client, addr % WINDOW, tag);
  }

  // A cp.async warp write on block 0: r's lane words, the pattern.
  void push_cpa(const Req& r) {
    uint32_t tag = next_cpa_tag_;
    next_cpa_tag_ = (next_cpa_tag_ + 1) % NUM_TAGS;
    cpa_live_.insert(tag);
    d_.push_cpa(r, tag);
  }

  // A DSMEM slot: slot `slot` of the row, the pattern; `last` ends its packet.
  void push_ds(uint32_t row, uint32_t slot, bool last) {
    uint32_t tag = next_ds_tag_;
    next_ds_tag_ = (next_ds_tag_ + 1) % ROW_TAGS;
    ++ds_pend_;
    d_.push_ds(row % (WINDOW / VX_CFG_SM_DATAPATH_BYTES), slot, last, tag);
  }

  void tick() {
    for (auto& e : d_.cycle()) {
      const uint64_t c = d_.now() - 1;
      if (e.type == EV_COMMIT) {
        auto& q = pend_commit_[e.blk];
        if (e.blk > 2 || q.empty()) {
          this->check(false, "%s: %ld lanes committed on kind %ld with nothing pending", "commit", e.cnt, e.blk);
          continue;
        }
        Rec& r = recs_[q.front()];
        q.pop_front();
        r.committed = true;
        r.commit = c;
        this->check(e.cnt == r.lacc, "%s: %ld counted lanes committed, expected %ld", r.name, e.cnt, r.lacc);
        continue;
      }
      if (e.type == EV_MBAR) {
        this->mbar_event(e, c);
        continue;
      }
      if (e.type == EV_ROW_GNT || e.type == EV_ROW_RSP) {
        auto it = rows_.find(e.tag);
        if (it == rows_.end()) {
          this->check(false, "%s: row event for no read (tag %ld, %ld)", "row", e.tag, 0);
          continue;
        }
        if (e.type == EV_ROW_GNT) {
          it->second.gnt = c;
          it->second.granted = true;
          if (e.blk < ROW_RD_CLIENTS)
            row_gnt_c_[e.blk].push_back(c);
        } else {
          this->check(it->second.granted, "%s: row read %ld answered before its grant (%ld)", "row", e.tag, 0);
          this->check(e.data_ok, "%s: row read %ld data wrong (%ld)", "row", e.tag, 0);
          long lat = (long)(c - it->second.gnt);
          if (row_lat_ < 0)
            row_lat_ = (int)lat;
          this->check(lat == row_lat_, "%s: row read answered %ld cycles after its grant, expected %ld", "row", lat,
                      row_lat_);
          row_gnts_.push_back(it->second.gnt);
          rows_.erase(it);
        }
        continue;
      }
      if (e.type == EV_WR_GNT) {
        this->check(wr_pend_ > 0, "%s: write grant %ld with no write pending (%ld)", "row write", e.tag, 0);
        if (wr_pend_)
          --wr_pend_;
        wr_gnts_.push_back(c);
        continue;
      }
      if (e.type == EV_AMO_GNT || e.type == EV_AMO_RSP) {
        auto it = amo_live_.find(e.tag);
        if (it == amo_live_.end() || e.blk >= AMO1_CLIENTS) {
          this->check(false, "%s: event for no one-lane atomic (tag %ld, client %ld)", "amo", e.tag, e.blk);
          continue;
        }
        if (e.type == EV_AMO_GNT) {
          it->second.gnt = c;
          it->second.granted = true;
          amo_gnt_c_[e.blk].push_back(c);
        } else {
          this->check(it->second.granted, "%s: one-lane atomic %ld answered before its grant (%ld)", "amo", e.tag, 0);
          this->check(e.data_ok, "%s: one-lane atomic %ld returned a wrong word (%ld)", "amo", e.tag, 0);
          long lat = (long)(c - it->second.gnt);
          if (amo_lat_ < 0)
            amo_lat_ = (int)lat;
          this->check(lat == amo_lat_, "%s: one-lane atomic answered %ld cycles after its grant, expected %ld", "amo",
                      lat, amo_lat_);
          amo_live_.erase(it);
        }
        continue;
      }
      if (e.type == EV_CPA_ACC) {
        cpa_acc_c_.push_back(c);
        auto it = cpa_rec_.find(e.tag);
        if (it != cpa_rec_.end()) {
          recs_[it->second].accepted = true;
          recs_[it->second].accept = c;
        }
        continue;
      }
      if (e.type == EV_CPA_ACK) {
        const bool known = cpa_live_.erase(e.tag) != 0;
        this->check(known && e.data_ok, "%s: acknowledgement of tag %ld unknown or with a wrong mask (%ld)", "cp.async",
                    e.tag, known);
        cpa_ack_c_.push_back(c);
        auto it = cpa_rec_.find(e.tag);
        if (it != cpa_rec_.end()) {
          recs_[it->second].responded = true;
          recs_[it->second].rsp = c;
          cpa_rec_.erase(it);
        }
        continue;
      }
      if (e.type == EV_DS_ACC) {
        this->check(ds_pend_ > 0, "%s: slot %ld taken with no slot pending (%ld)", "dsmem", e.tag, 0);
        if (ds_pend_)
          --ds_pend_;
        ds_acc_c_.push_back(c);
        continue;
      }
      if (e.type >= EV_GL_ACC) {
        this->gl_event(e);
        continue;
      }
      if (e.type == EV_DS_BUSY) {
        ds_busy_ = e.cnt != 0;
        ds_busy_c_.push_back(c);
        continue;
      }
      auto it = live_[e.blk].find(e.tag);
      if (it == live_[e.blk].end()) {
        this->check(false, "%s: event for no request on block %ld tag %ld", "event", e.blk, e.tag);
        continue;
      }
      Rec& r = recs_[it->second];
      if (e.type == EV_ACCEPT) {
        r.accepted = true;
        r.accept = c;
        if (r.lacc)
          pend_commit_[r.mbar ? 2 : r.amo ? 1 : 0].push_back(r.id);
        if (!r.read)
          live_[e.blk].erase(it);
      } else {
        r.responded = true;
        r.rsp = c;
        this->check(e.data_ok, "%s: response data or mask wrong (id %ld, %ld)", r.name, r.id, 0);
        live_[e.blk].erase(it);
      }
    }
  }

  bool done(const std::vector<uint32_t>& ids) const {
    for (auto id : ids) {
      const Rec& r = recs_[id];
      if (!r.accepted || (r.read && !r.responded) || (r.lacc && !r.committed))
        return false;
    }
    return rows_.empty() && mevt_.empty() && wr_pend_ == 0 && amo_live_.empty() && cpa_live_.empty() &&
           ds_pend_ == 0 && !ds_busy_;
  }

  void drain(const std::vector<uint32_t>& ids) {
    for (uint32_t n = 0; !this->done(ids); ++n) {
      if (n > 20000) {
        this->check(false, "%s: no progress after %ld cycles (%ld)", "drain", n, 0);
        return;
      }
      this->tick();
    }
    for (uint32_t i = 0; i < 4; ++i)
      this->tick();
  }

  void latency_check(const Rec& r) {
    if (!r.read)
      return;
    long lat = (long)(r.rsp - r.accept);
    this->check(lat - (long)r.gold == k_, "%s: latency %ld, expected golden + K = %ld", r.name, lat, (long)r.gold + k_);
  }

  // Overlapping reads: the array's responses leave in the order the
  // wavefront stage took them, over all blocks, one a cycle, never before
  // golden + K after their issue (a read port taken by a row read or another
  // block's wavefront delays them by the cycles it was taken).
  void inorder_check(const std::vector<uint32_t>& ids) {
    std::vector<const Rec*> rd;
    for (auto id : ids)
      if (recs_[id].read)
        rd.push_back(&recs_[id]);
    std::sort(rd.begin(), rd.end(), [](const Rec* x, const Rec* y) { return x->accept < y->accept; });
    uint64_t prev = 0;
    for (auto* r : rd) {
      this->check(r->rsp >= r->accept + r->gold + k_, "%s: answered at +%ld after issue, at least +%ld", r->name,
                  (long)(r->rsp - r->accept), (long)r->gold + k_);
      this->check(rd.front() == r || r->rsp > prev, "%s: answered at +%ld, not after the previous read (+%ld)",
                  r->name, (long)(r->rsp - r->accept), (long)prev - (long)r->accept);
      prev = r->rsp;
    }
  }

  void log(const char* phase, const Rec& r) {
    if (csv_)
      std::fprintf(csv_, "%s,%u,%u,%u,%llu,%llu,%llu\n", phase, r.id, r.blk, r.cost,
                   (unsigned long long)(r.accept - r.present),
                   (unsigned long long)(r.read ? r.rsp - r.accept : 0),
                   (unsigned long long)(r.lacc ? r.commit - r.accept : 0));
  }

  // Fills the window with the pattern: 64-bit stores over every lane word.
  void preload() {
    std::vector<uint32_t> ids;
    const uint32_t step = nlanes_ * 8;
    for (uint64_t base = 0; base < WINDOW; base += step) {
      ids.push_back(this->submit(0, strided("preload", K_ST, 8, nlanes_, nlanes_, 8, 1, base)));
      if (ids.size() == NUM_TAGS / 2) {
        this->drain(ids);
        ids.clear();
      }
    }
    this->drain(ids);
  }

  // One request at a time: latency and the commit of its counted lanes.
  void isolated() {
    auto cases = directed(nlanes_, lacc_ok_);
    for (auto& c : cases) {
      uint32_t id = this->submit(0, c);
      this->drain({id});
      for (uint32_t i = 0; i < recs_[id].cost; ++i)
        this->tick();
      Rec& r = recs_[id];
      this->check(r.accept == r.present + accept_delay(r.cas), "%s: accepted %ld cycles after it was presented to an idle port (%ld)",
                  r.name, (long)(r.accept - r.present), 0);
      this->latency_check(r);
      this->log("iso", r);
    }
  }

#if VX_CFG_EXT_A_ENABLED && defined(VX_CFG_EXT_ZACAS_ENABLE)
  // A CAS whose lane pairs alternate between a comparand equal to the word and
  // one that differs: every lane returns the old word, only the matching lanes
  // take the first beat's value, and a beat taken for the other would show.
  void cas_swap() {
    Req c = strided("cas swap", K_CAS, 4, nlanes_, nlanes_, 4);
    Req l = strided("cas swap check", K_LD, 4, nlanes_, nlanes_, 4);
    c.own = l.own = true;
    for (uint32_t i = 0; i < nlanes_; ++i) {
      const uint64_t p = word_value(c.addr[i] / WORD_SIZE);
      const bool match = ((i >> 1) & 1) == 0;
      c.val[i] = ~p;
      c.cmp[i] = match ? p : (p ^ 0x0101010101010101ull);
      c.exp[i] = p;
      l.exp[i] = match ? ~p : p;
    }
    Req s = strided("cas swap restore", K_ST, 4, nlanes_, nlanes_, 4);
    for (const Req* r : {&c, &l, &s}) {
      const uint32_t id = this->submit(0, *r);
      this->drain({id});
      this->latency_check(recs_[id]);
      this->log("cas", recs_[id]);
    }
  }
#endif

  // Back-to-back copies: the wavefront stage is busy `cost` cycles per request.
  void occupancy() {
    auto cases = directed(nlanes_, lacc_ok_);
    for (auto& c : cases) {
      std::vector<uint32_t> ids;
      for (int i = 0; i < 4; ++i)
        ids.push_back(this->submit(0, c));
      this->drain(ids);
      for (size_t i = 1; i < ids.size(); ++i) {
        const Rec& a = recs_[ids[i - 1]];
        const Rec& b = recs_[ids[i]];
        this->check(b.accept - a.accept == a.cost, "%s: next request %ld cycles later, expected the cost %ld",
                    c.name, (long)(b.accept - a.accept), a.cost);
        this->latency_check(b);
      }
      for (auto id : ids)
        this->log("occ", recs_[id]);
    }
  }

  // Back-to-back row reads on an idle array: one grant a cycle, the pattern.
  void row_reads() {
    row_gnts_.clear();
    for (uint32_t i = 0; i < 8; ++i)
      this->push_row(i * 3);
    this->drain({});
    this->check(row_gnts_.size() == 8, "%s: %ld row reads answered, expected %ld", "rows", (long)row_gnts_.size(), 8);
    for (size_t i = 1; i < row_gnts_.size(); ++i)
      this->check(row_gnts_[i] - row_gnts_[i - 1] == 1, "%s: row grants %ld cycles apart, expected %ld", "rows",
                  (long)(row_gnts_[i] - row_gnts_[i - 1]), 1);
  }

  // TCU row reads on the read port ahead of the LSU's wavefronts: accepts
  // follow the port model (SramPortRef) cycle for cycle.
  void ext_scenario(const std::vector<Req>& reqs, const std::vector<bool>& ext) {
    SramPortRef ref(knob_port_cfg());
    const uint64_t t0 = d_.now();
    std::vector<uint32_t> ids;
    for (uint32_t k = 0; k < reqs.size(); ++k) {
      ids.push_back(this->submit(0, reqs[k]));
      ref.push_wf(WF_SHARED, WF_READ, cost(reqs[k], cfg_, nlanes_), k);
    }
    std::vector<uint64_t> pick, fin(reqs.size(), 0);
    uint32_t nfin = 0;
    for (uint64_t c = 0; nfin < reqs.size() || c < ext.size(); ++c) {
      if (c < ext.size() && ext[c]) {
        ref.push_read(RD_TCU, 1);
        this->push_row((uint32_t)c * 5 + 1);
      }
      auto g = ref.tick();
      if (g.wf_pick == WF_SHARED)
        pick.push_back(t0 + c);
      if (g.wf_done) {
        fin[g.wf_id] = t0 + c;
        ++nfin;
      }
      if (c < ext.size())
        this->tick();
    }
    this->drain(ids);
    for (uint32_t k = 0; k < ids.size(); ++k) {
      const Rec& r = recs_[ids[k]];
      this->check(r.accept == pick[k], "%s: accepted at +%ld with row reads ahead, expected +%ld", r.name,
                  (long)(r.accept - t0), (long)(pick[k] - t0));
      this->check((long)(r.rsp - fin[k]) == RSP_PATH, "%s: answered %ld cycles after its last read grant, expected %ld",
                  r.name, (long)(r.rsp - fin[k]), RSP_PATH);
      this->log("ext", r);
    }
    this->inorder_check(ids);
  }

  void ext_client() {
    Req one = strided("ext ld32 stride 1", K_LD, 4, nlanes_, nlanes_, 4);
    for (uint32_t len : {1u, 3u}) {
      for (uint32_t lead : {0u, 1u, 2u}) {
        std::vector<bool> ext(lead, false);
        for (uint32_t i = 0; i < len; ++i)
          ext.push_back(true);
        ext.push_back(false);
        this->ext_scenario({one, one, one}, ext);
      }
    }
    Req four = strided("ext ld32 stride 8", K_LD, 4, nlanes_, nlanes_, 32);
    this->ext_scenario({four, one}, {false, true, true, false});
  }

  // Several LSU blocks share one wavefront stage, round robin.
  void blocks() {
    for (uint32_t s : {1u, 2u}) {
      Req r = strided("ld32 stride " + std::to_string(s), K_LD, 4, nlanes_, nlanes_, s * 4);
      std::vector<uint32_t> ids;
      for (int i = 0; i < 8; ++i)
        for (uint32_t b = 0; b < nblocks_; ++b)
          ids.push_back(this->submit(b, r));
      this->drain(ids);
      std::vector<uint64_t> acc;
      for (auto id : ids)
        acc.push_back(recs_[id].accept);
      std::sort(acc.begin(), acc.end());
      uint32_t c = cost(r, cfg_, nlanes_);
      for (size_t i = 1; i < acc.size(); ++i)
        this->check(acc[i] - acc[i - 1] == c, "%s: accepts %ld apart over the blocks, expected %ld", r.name,
                    (long)(acc[i] - acc[i - 1]), c);
      for (int i = 0; i + 1 < 8; ++i)
        for (uint32_t b = 0; b < nblocks_; ++b)
          for (uint32_t b2 = 0; b2 < nblocks_; ++b2)
            this->check(recs_[ids[i * nblocks_ + b]].accept < recs_[ids[(i + 1) * nblocks_ + b2]].accept,
                        "%s: block %ld served twice before block %ld", "round robin", b, b2);
      for (auto id : ids)
        this->log("blk", recs_[id]);
    }
  }

  // Random requests on every block at random times, with random row reads:
  // latency, data, port rate.
  void random(uint32_t seed, uint32_t n) {
    std::mt19937 rng(seed);
    std::vector<uint32_t> ids;
    uint32_t issued = 0;
    while (issued < n) {
      for (uint32_t b = 0; b < nblocks_ && issued < n; ++b) {
        if (live_[b].size() < NUM_TAGS / 2 && rng() % 4 == 0) {
          ids.push_back(this->submit(b, random_req(rng, nlanes_, lacc_ok_)));
          ++issued;
        }
      }
      if (rng() % 16 == 0 && rows_.size() < ROW_TAGS / 2)
        this->push_row(rng() % 512);
      this->tick();
    }
    this->drain(ids);
    std::vector<std::pair<uint64_t, uint32_t>> acc;
    for (auto id : ids) {
      acc.push_back({recs_[id].accept - accept_delay(recs_[id].cas), recs_[id].cost});
      this->log("rnd", recs_[id]);
    }
    this->inorder_check(ids);
    std::sort(acc.begin(), acc.end());
    for (size_t i = 1; i < acc.size(); ++i)
      this->check(acc[i].first - acc[i - 1].first >= acc[i - 1].second,
                  "%s: two requests start %ld cycles apart, the first costs %ld", "random port", (long)(acc[i].first - acc[i - 1].first),
                  acc[i - 1].second);
  }

  // The cycle-by-cycle port model beside the bench: the same pushes in the
  // same cycles, its grants collected per client. The one-lane atomic
  // clients and the LSU atomics own disjoint bank pairs (DXA the first
  // quarter, DSMEM the second, the LSU the upper half), so each RD_AMO
  // grant names the clients it served.
  // SramPortRef run beside the drivers, with the two client registers in
  // front of the write port: the cp.async stage (one warp write, taken the
  // cycle after the previous one's last grant, requesting from the cycle after
  // it is taken, acknowledged the cycle after its last grant) and the DSMEM
  // merge register (slots of one row join while it is open; it closes on the
  // packet end, a full merge, or the first cycle no slot fits, and is written
  // as one WR_DSMEM grant). An mbarrier op's lane-update joins the AMO path
  // the cycle after the wavefront stage passes the op to the op queue.
  struct PortShadow {
    struct Slot {
      uint32_t row;
      bool last;
    };
    SramPortRef ref;
    uint64_t t0;
    SramPortCfg ref_cfg;
    std::vector<uint64_t> pick, row[ROW_RD_CLIENTS], wr, amo[AMO1_CLIENTS];
    std::deque<uint32_t> amo_pair[AMO1_CLIENTS];
    uint32_t max_pairs = 0;
    std::deque<uint32_t> cpa_q;
    bool cpa_stage = false, cpa_push = false;
    uint32_t cpa_left = 0, cpa_cycles = 0;
    uint64_t cpa_free = 0;
    std::vector<uint64_t> cpa_acc, cpa_ack;
    std::deque<Slot> ds_q;
    bool mrg_valid = false, mrg_open = false, mrg_req = false;
    uint32_t mrg_row = 0, mrg_slots = 0, ds_writes = 0;
    std::vector<uint64_t> ds_acc, ds_busy;
    std::map<uint32_t, std::vector<SramAmoLane>> mbar_ops;
    std::map<uint32_t, uint64_t> fin;  // each item's last wavefront grant
    std::vector<SramAmoLane> mbar_next;
    bool mbar_pend = false;

    static constexpr uint32_t MRG_SLOTS = VX_CFG_SM_SRAM_DSMEM_WR_BYTES / VX_CFG_N5_SLOT_BYTES;

    PortShadow(const SramPortCfg& c, uint64_t now) : ref(c), t0(now), ref_cfg(c) {}

    void tick() {
      const uint64_t c = t0 + ref.cycle();
      if (mbar_pend) {
        ref.push_amo_op(AMO_MBAR, mbar_next);
        mbar_pend = false;
      }
      if (cpa_push) {
        ref.push_write(WR_CPASYNC, cpa_cycles);
        cpa_push = false;
      }
      if (!cpa_stage && !cpa_q.empty() && c >= cpa_free) {
        cpa_acc.push_back(c);
        cpa_stage = cpa_push = true;
        cpa_left = cpa_cycles = cpa_q.front();
        cpa_q.pop_front();
      }
      if (mrg_valid && !mrg_open && !mrg_req) {
        ref.push_write(WR_DSMEM, 1);
        mrg_req = true;
      }
      auto g = ref.tick();
      if (g.wf_pick == WF_SHARED)
        pick.push_back(c);
      if (g.rd < ROW_RD_CLIENTS)
        row[g.rd].push_back(c);
      if (g.wr == WR_DXA)
        wr.push_back(c);
      if (g.rd == RD_AMO) {
        max_pairs = std::max<uint32_t>(max_pairs, __builtin_popcount(g.amo_pairs));
        for (uint32_t a = 0; a < AMO1_CLIENTS; ++a) {
          if (!amo_pair[a].empty() && ((g.amo_pairs >> amo_pair[a].front()) & 1)) {
            amo[a].push_back(c);
            amo_pair[a].pop_front();
          }
        }
      }
      if (g.wf_done) {
        fin[g.wf_id] = c;
        auto it = mbar_ops.find(g.wf_id);
        if (it != mbar_ops.end()) {
          mbar_next = it->second;
          mbar_pend = true;
          mbar_ops.erase(it);
        }
      }
      if (g.wr == WR_CPASYNC && --cpa_left == 0) {
        cpa_stage = false;
        cpa_free = c + 1;
        cpa_ack.push_back(c + 1);
      }
      const bool gnt = (g.wr == WR_DSMEM);
      const bool was = mrg_valid;
      ds_writes += gnt ? 1 : 0;
      const bool fit = !ds_q.empty() && mrg_valid && mrg_open && ds_q.front().row == mrg_row && mrg_slots < MRG_SLOTS;
      if (!ds_q.empty() && (fit || !mrg_valid || gnt)) {
        const Slot sl = ds_q.front();
        ds_q.pop_front();
        ds_acc.push_back(c);
        if (fit) {
          ++mrg_slots;
        } else {
          mrg_row = sl.row;
          mrg_slots = 1;
          mrg_req = false;
        }
        mrg_open = !sl.last && mrg_slots < MRG_SLOTS;
        mrg_valid = true;
      } else if (gnt) {
        mrg_valid = false;
        mrg_req = false;
      } else if (mrg_valid && mrg_open) {
        mrg_open = false;
      }
      if (mrg_valid != was)
        ds_busy.push_back(c + 1);
    }

    bool idle() const {
      return ref.idle() && !cpa_stage && !cpa_push && cpa_q.empty() && ds_q.empty() && !mrg_valid && !mbar_pend &&
             mbar_ops.empty();
    }
  };

  static constexpr uint32_t PAIRS = VX_CFG_LMEM_NUM_BANKS / 2;
  static constexpr uint32_t ROW_B = VX_CFG_SM_DATAPATH_BYTES;
  static constexpr uint32_t WIN_ROWS = WINDOW / ROW_B;

  static uint64_t pair_addr(uint32_t row, uint32_t pair) {
    return (uint64_t)(row % WIN_ROWS) * ROW_B + (uint64_t)pair * 2 * VX_CFG_LMEM_BANK_BYTES;
  }

  // An LSU atomic of 8-B lanes on distinct rows, every lane in the upper half of the bank pairs.
  Req upper_atomic(std::mt19937& rng, uint32_t lanes) {
    Req r;
    r.name = "port atom";
    r.kind = K_ADD;
    r.bytes = 8;
    const uint32_t base = rng() % WIN_ROWS;
    for (uint32_t i = 0; i < lanes && i < nlanes_; ++i) {
      r.mask |= 1u << i;
      r.addr[i] = pair_addr(base + 7 * i, PAIRS / 2 + rng() % (PAIRS / 2));
    }
    return r;
  }

  void shadow_lsu(PortShadow& s, const Req& r, uint32_t id) {
    if (r.kind == K_ADD) {
      std::vector<SramAmoLane> lanes;
      for (uint32_t i = 0; i < nlanes_; ++i) {
        if ((r.mask >> i) & 1)
          lanes.push_back(SramAmoLane{(uint32_t)((r.addr[i] % ROW_B) / VX_CFG_LMEM_BANK_BYTES), 2});
      }
      s.ref.push_wf(WF_SHARED, WF_AMO, 0, id, lanes);
    } else {
      s.ref.push_wf(WF_SHARED, r.kind == K_ST ? WF_WRITE : WF_READ, cost(r, cfg_, nlanes_), id);
    }
  }

  void shadow_amo1(PortShadow& s, uint32_t client, uint32_t row, uint32_t pair) {
    s.ref.push_amo(client == 0 ? AMO_DXA : AMO_DSMEM, pair);
    s.amo_pair[client].push_back(pair);
    this->push_amo1(client, pair_addr(row, pair));
  }

  void grant_check(const char* what, uint32_t client, const std::vector<uint64_t>& got,
                   const std::vector<uint64_t>& exp, uint64_t t0) {
    this->check(got.size() == exp.size(), "%s: %ld grants, expected %ld", what, (long)got.size(), (long)exp.size());
    for (size_t i = 0; i < std::min(got.size(), exp.size()); ++i) {
      if (got[i] != exp[i]) {
        this->check(false, "%s: grant at +%ld, the port model's at +%ld", what, (long)(got[i] - t0),
                    (long)(exp[i] - t0));
        break;
      }
    }
    if (csv_) {
      for (auto g : got)
        std::fprintf(csv_, "gnt,%s,%u,%llu\n", what, client, (unsigned long long)(g - t0));
    }
  }

  void shadow_check(const PortShadow& s, const std::vector<uint32_t>& ids, const char* phase) {
    this->check(ids.size() == s.pick.size(), "%s: %ld LSU accepts in the port model, %ld sent", phase,
                (long)s.pick.size(), (long)ids.size());
    for (size_t k = 0; k < std::min(ids.size(), s.pick.size()); ++k) {
      const Rec& r = recs_[ids[k]];
      this->check(r.accept == s.pick[k], "%s: accepted at +%ld, the port model's at +%ld", r.name,
                  (long)(r.accept - s.t0), (long)(s.pick[k] - s.t0));
      auto f = s.fin.find((uint32_t)k);
      if (r.read && !r.amo && !r.mbar && f != s.fin.end())
        this->check((long)(r.rsp - f->second) == RSP_PATH, "%s: answered %ld cycles after its last read grant, expected %ld",
                    r.name, (long)(r.rsp - f->second), RSP_PATH);
      this->log(phase, r);
    }
    static const char* rd_name[ROW_RD_CLIENTS] = {"rd tcu", "rd dxa", "rd dsmem"};
    for (uint32_t k = 0; k < ROW_RD_CLIENTS; ++k)
      this->grant_check(rd_name[k], k, row_gnt_c_[k], s.row[k], s.t0);
    this->grant_check("wr dxa", 0, wr_gnts_, s.wr, s.t0);
    static const char* amo_name[AMO1_CLIENTS] = {"amo dxa", "amo dsmem"};
    for (uint32_t a = 0; a < AMO1_CLIENTS; ++a)
      this->grant_check(amo_name[a], a, amo_gnt_c_[a], s.amo[a], s.t0);
    this->grant_check("cpa taken", 0, cpa_acc_c_, s.cpa_acc, s.t0);
    this->grant_check("cpa acked", 0, cpa_ack_c_, s.cpa_ack, s.t0);
    this->grant_check("dsmem slot taken", 0, ds_acc_c_, s.ds_acc, s.t0);
    this->grant_check("dsmem busy edge", 0, ds_busy_c_, s.ds_busy, s.t0);
  }

  void shadow_clear() {
    for (auto& v : row_gnt_c_) v.clear();
    for (auto& v : amo_gnt_c_) v.clear();
    wr_gnts_.clear();
    cpa_acc_c_.clear();
    cpa_ack_c_.clear();
    ds_acc_c_.clear();
    ds_busy_c_.clear();
  }

  // Every row client, the DXA row writes, DSMEM packets, cp.async warp
  // writes, the one-lane atomics and LSU loads, stores and atomics on block 0
  // at random: per cycle the read, write and AMO grants equal SramPortRef's,
  // and the DSMEM writes number dsmem_write_cycles per packet.
  void ports(uint32_t seed, uint32_t ncycles) {
    SramPortCfg pc = knob_port_cfg();
    pc.amo_wb_delay = AMO_WB_DELAY;
    std::mt19937 rng(seed * 7919 + 13);
    this->shadow_clear();
    PortShadow s(pc, d_.now());
    std::vector<uint32_t> ids;
    uint32_t ds_gold = 0;
    for (uint64_t c = 0; c < ncycles || !s.idle(); ++c) {
      if (c < ncycles) {
        if (live_[0].size() < NUM_TAGS / 2 && rng() % 3 == 0) {
          Req r = (rng() % 4 == 0) ? this->upper_atomic(rng, 1 + rng() % 4) : random_req(rng, nlanes_, false);
          if (r.kind == K_CAS || (r.kind == K_ADD && r.name != "port atom"))
            r.kind = K_LD;
          this->shadow_lsu(s, r, (uint32_t)ids.size());
          ids.push_back(this->submit(0, r));
        }
        for (uint32_t k = 0; k < ROW_RD_CLIENTS; ++k) {
          if (rows_.size() < ROW_TAGS / 2 && rng() % 8 == 0) {
            s.ref.push_read(k, 1);
            this->push_row(rng() % WIN_ROWS, k);
          }
        }
        if (wr_pend_ < 32 && rng() % 8 == 0) {
          s.ref.push_write(WR_DXA, 1);
          this->push_wr(rng() % WIN_ROWS);
        }
        for (uint32_t a = 0; a < AMO1_CLIENTS; ++a) {
          if (amo_live_.size() < ROW_TAGS / 4 && rng() % 8 == 0)
            this->shadow_amo1(s, a, rng() % WIN_ROWS, a * (PAIRS / 4) + rng() % (PAIRS / 4));
        }
        if (cpa_live_.size() < NUM_TAGS / 2 && rng() % 8 == 0) {
          Req r = random_req(rng, nlanes_, false);
          r.kind = K_ST;
          r.name = "cpa";
          s.cpa_q.push_back(cost(r, cfg_, nlanes_));
          this->push_cpa(r);
        }
        if (ds_pend_ < 32 && rng() % 12 == 0)
          ds_gold += this->packet(s, rng);
      }
      s.tick();
      this->tick();
    }
    this->drain(ids);
    this->shadow_check(s, ids, "prt");
    this->check(s.ds_writes == ds_gold, "%s: %ld DSMEM writes, dsmem_write_cycles gives %ld", "dsmem", s.ds_writes,
                ds_gold);
  }

  // One DSMEM packet: one or two rows, each a run of consecutive slots.
  // Returns its write count by dsmem_write_cycles.
  uint32_t packet(PortShadow& s, std::mt19937& rng) {
    constexpr uint32_t SLOT = VX_CFG_N5_SLOT_BYTES;
    constexpr uint32_t SPR = ROW_B / SLOT;
    std::vector<uint32_t> rows, slots;
    const uint32_t nruns = (rng() % 4 == 0) ? 2 : 1;
    for (uint32_t k = 0; k < nruns; ++k) {
      const uint32_t row = rng() % WIN_ROWS;
      const uint32_t first = rng() % SPR;
      const uint32_t n = 1 + rng() % (SPR - first);
      for (uint32_t j = 0; j < n; ++j) {
        rows.push_back(row);
        slots.push_back(first + j);
      }
    }
    for (size_t i = 0; i < rows.size(); ++i) {
      const bool last = (i + 1 == rows.size());
      s.ds_q.push_back({rows[i], last});
      this->push_ds(rows[i], slots[i], last);
    }
    return dsmem_write_cycles(rows.data(), (uint32_t)rows.size(), SLOT, s.ref_cfg);
  }

  // An LSU atomic, a DXA completion and a DSMEM AMO on three bank pairs, the
  // completion offset 0..5 cycles: one RD_AMO grant serves every pair whose
  // head is ready, so the atomic is never held up.
  void amo_together() {
    SramPortCfg pc = knob_port_cfg();
    pc.amo_wb_delay = AMO_WB_DELAY;
    uint32_t most = 0;
    for (uint32_t d = 0; d < 6; ++d) {
      this->shadow_clear();
      PortShadow s(pc, d_.now());
      Req r;
      r.name = "atom with clients +" + std::to_string(d);
      r.kind = K_ADD;
      r.bytes = 8;
      r.mask = 1;
      r.addr[0] = pair_addr(3, PAIRS - 1);
      std::vector<uint32_t> ids;
      for (uint64_t c = 0; c <= d || !s.idle(); ++c) {
        if (c == 0) {
          this->shadow_lsu(s, r, 0);
          ids.push_back(this->submit(0, r));
        }
        if (c == d) {
          this->shadow_amo1(s, 0, 5, 0);
          this->shadow_amo1(s, 1, 9, PAIRS / 4);
        }
        s.tick();
        this->tick();
      }
      this->drain(ids);
      this->shadow_check(s, ids, "amo3");
      this->latency_check(recs_[ids[0]]);
      most = std::max(most, s.max_pairs);
    }
    this->check(most == 3, "%s: at most %ld bank pairs in one RD_AMO grant, expected %ld", "amo together", most, 3);
#ifdef VX_CFG_EXT_MBAR_ENABLE
    // An mbarrier arrive ahead of the atomic, its word on a fourth bank pair:
    // its lane-update joins the same RD_AMO grant.
    mword_.assign(MBAR_WORDS, 0);
    this->mbar_init(std::vector<uint32_t>(MBAR_WORDS, 64));
    const uint32_t w = PAIRS / 2;
    const uint32_t wbank = (uint32_t)((mbar_addr(w) % ROW_B) / VX_CFG_LMEM_BANK_BYTES);
    most = 0;
    for (uint32_t d = 0; d < 6; ++d) {
      this->shadow_clear();
      PortShadow s(pc, d_.now());
      Req r;
      r.name = "atom with clients and arrive +" + std::to_string(d);
      r.kind = K_ADD;
      r.bytes = 8;
      r.mask = 1;
      r.addr[0] = pair_addr(3, PAIRS - 1);
      std::vector<uint32_t> ids;
      for (uint64_t c = 0; c <= d || !s.idle(); ++c) {
        if (c == 0) {
          s.ref.push_wf(WF_SHARED, WF_AMO, 0, 0);
          s.mbar_ops[0] = std::vector<SramAmoLane>(1, SramAmoLane{wbank, 2});
          ids.push_back(this->submit_mbar(0, "arrive with clients +" + std::to_string(d), w, mbr::OP_ARRIVE, 1));
          this->shadow_lsu(s, r, 1);
          ids.push_back(this->submit(0, r));
        }
        if (c == d) {
          this->shadow_amo1(s, 0, 5, 0);
          this->shadow_amo1(s, 1, 9, PAIRS / 4);
        }
        s.tick();
        this->tick();
      }
      this->drain(ids);
      this->shadow_check(s, ids, "amo4");
      for (auto id : ids)
        this->latency_check(recs_[id]);
      most = std::max(most, s.max_pairs);
    }
    this->check(most == 4, "%s: at most %ld bank pairs in one RD_AMO grant, expected %ld", "amo with arrive", most, 4);
#endif
  }

  // A new split between launches holds from the next cycle: the first access
  // after it is taken at once and answered as on an idle array (no switch
  // cost), and the rows below the smaller shared size keep their data. Each
  // model's own split at that size equals l1_slot_ref's.
  void cfg_switch() {
    const uint32_t kib[] = {VX_CFG_SMEM_CONFIG_KB_0, VX_CFG_SMEM_CONFIG_KB_4, VX_CFG_SMEM_PER_SM_MAX / 1024};
    for (uint32_t k : kib) {
      const ArrayCfg ac = array_cfg(k * 1024);
      this->check(d_.set_cfg(ac), "%s: the model's split at %ld KiB differs from l1_slot_ref's (%ld)", "cfg switch", k,
                  0);
      const uint32_t top = std::min<uint32_t>(ac.s_rows, WIN_ROWS) - 1;
      const Req r = strided("ld at the top shared row after the switch to " + std::to_string(k) + " KiB", K_LD, 4,
                            nlanes_, nlanes_, 4, 1, (uint64_t)top * ROW_B);
      const uint32_t id = this->submit(0, r);
      this->drain({id});
      const Rec& q = recs_[id];
      this->check(q.accept == q.present, "%s: taken %ld cycles after it was presented (%ld)", q.name,
                  (long)(q.accept - q.present), 0);
      this->latency_check(q);
      this->log("cfg", q);
      this->push_row(top, RD_TCU);
      this->drain({});
    }
  }

  // ---- the global leg ----
  struct GlPend {
    gl::Req r;
    uint32_t id = 0, got = 0;
  };
  struct GlRd {
    uint64_t line;
    uint32_t secs, slot;
    bool operator==(const GlRd& o) const { return line == o.line && secs == o.secs && slot == o.slot; }
  };

  uint32_t gl_slot_of(uint32_t row) const { return dir_.cfg().sram_bytes / dir_.cfg().row_bytes - 1 - row; }

  void gl_event(const Event& e) {
    if (e.type == EV_GL_MEMRD) {
      gl_reads_.push_back({e.tag, e.cnt, 0});
      return;
    }
    if (e.type == EV_GL_FILL) {
      const uint32_t slot = this->gl_slot_of(e.cnt);
      this->check(slot < dir_.lines(), "%s: L1 fill into row %ld, outside the L1's %ld rows", "global", e.cnt,
                  dir_.lines());
      gl_fills_.push_back(slot);
      if (!gl_seq_)
        gl_csv_l_.push_back(slot);
      ++gl_nfills_;
      return;
    }
    if (e.type == EV_GL_FLUSHED) {
      gl_flushed_ = true;
      return;
    }
    auto it = gl_pend_[e.blk % 8].find(e.tag);
    if (e.blk >= nblocks_ || it == gl_pend_[e.blk].end()) {
      this->check(false, "%s: event on block %ld for no request (tag %ld)", "global", e.blk, e.tag);
      return;
    }
    GlPend& p = it->second;
    gl_progress_ = true;
    if (e.type == EV_GL_ACC) {
      if (p.r.kind == gl::K_ST)
        this->gl_retire(e.blk, it);
      return;
    }
    this->check(e.data_ok && (e.cnt & ~p.r.mask & ~p.got) == 0 && (e.cnt & p.got) == 0,
                "%s: data, or lanes 0x%lx, wrong (id %ld)", p.r.name, (long)e.cnt, (long)p.id);
    if (!e.data_ok)
      this->gl_report(p, e);
    p.got |= e.cnt;
    if ((p.got & p.r.mask) == p.r.mask)
      this->gl_retire(e.blk, it);
  }

  // A wrong global lane: the value expected, the value seen, and the lines of
  // the global pools whose current value at that offset is the one seen.
  void gl_report(const GlPend& p, const Event& e) {
    for (const auto& x : d_.gl_bad()) {
      if (x.blk != e.blk || x.tag != e.tag)
        continue;
      if (gl_reported_ >= 24)
        return;
      ++gl_reported_;
      const uint64_t a = p.r.addr[x.lane];
      const uint32_t off = uint32_t(a % gl::LINE);
      const uint64_t got = (x.word >> (8 * (a % gl::WORD))) & gl::value_mask(p.r.bytes);
      std::string who;
      const uint64_t pools[][2] = {{gl::GL_LINE0 + 65536, dir_.lines() * 2 + 8}, {gl::GL_LINE0 + 131072, dir_.lines()}};
      for (const auto& pl : pools) {
        for (uint64_t l = pl[0]; l < pl[0] + pl[1]; ++l) {
          if (l != a / gl::LINE && prog_.load(l * gl::LINE + off, p.r.bytes) == got)
            who += " " + std::to_string(l - gl::GL_LINE0);
        }
      }
      std::printf("  global %s #%u block %u tag %u lane %u line %llu +%u: got 0x%llx expected 0x%llx, cycle %llu; "
                  "value of lines:%s\n", p.r.name.c_str(), p.id, x.blk, x.tag, x.lane,
                  (unsigned long long)(a / gl::LINE - gl::GL_LINE0), off, (unsigned long long)got,
                  (unsigned long long)p.r.exp[x.lane], (unsigned long long)d_.now(), who.empty() ? " none" : who.c_str());
    }
  }

  void gl_retire(uint32_t b, typename std::map<uint32_t, GlPend>::iterator it) {
    if (!gl_seq_)
      gl_csv_g_[it->second.id] = std::to_string(b) + "," + it->second.r.name;
    gl_free_[b].push_back(it->first);
    gl_pend_[b].erase(it);
  }

  gl::Req gl_one(uint32_t kind, uint64_t line, uint32_t off, uint32_t bytes, const char* name) {
    gl::Req r;
    r.kind = kind;
    r.bytes = bytes;
    r.mask = 1;
    r.addr[0] = line * gl::LINE + off;
    r.val[0] = gl_rng_();
    r.name = name;
    return r;
  }

  // Lanes on distinct words of the given lines, round robin over them.
  gl::Req gl_lanes(uint32_t kind, const std::vector<uint64_t>& ls, uint32_t n, uint32_t bytes, const char* name) {
    gl::Req r;
    r.kind = kind;
    r.bytes = bytes;
    r.name = name;
    const uint32_t nw = gl::LINE / gl::WORD;
    std::vector<std::vector<uint32_t>> pick(ls.size());
    for (auto& p : pick) {
      for (uint32_t w = 0; w < nw; ++w)
        p.push_back(w);
      std::shuffle(p.begin(), p.end(), gl_rng_);
    }
    n = std::min<uint32_t>(std::min(n, nlanes_), uint32_t(ls.size() * nw));
    for (uint32_t i = 0; i < n; ++i) {
      const uint32_t k = i % ls.size();
      const uint32_t sub = (bytes < gl::WORD) ? (gl_rng_() % (gl::WORD / bytes)) * bytes : 0;
      r.mask |= 1u << i;
      r.addr[i] = ls[k] * gl::LINE + pick[k][i / ls.size()] * gl::WORD + sub;
      r.val[i] = gl_rng_();
    }
    return r;
  }

  // The returned values in program order; a block stores only to lines no
  // other block touches while it runs, so the order across blocks is free.
  void gl_submit(uint32_t b, gl::Req r) {
    for (uint32_t i = 0; i < nlanes_; ++i) {
      if (!(r.mask >> i & 1))
        continue;
      r.exp[i] = prog_.load(r.addr[i], r.bytes);
      if (r.kind == gl::K_ST)
        prog_.store(r.addr[i], r.bytes, r.val[i]);
    }
    while (gl_free_[b].empty())
      this->gl_tick();
    const uint32_t tag = gl_free_[b].front();
    gl_free_[b].pop_front();
    GlPend p;
    p.r = r;
    p.id = gl_n_++;
    gl_pend_[b][tag] = p;
    d_.push_gl(b, r, tag);
  }

  bool gl_live() const {
    for (uint32_t b = 0; b < nblocks_; ++b) {
      if (!gl_pend_[b].empty())
        return true;
    }
    return d_.gl_busy();
  }

  // One cycle, with the row, write and shared clients kept busy beside the L1
  // (their own generator: the global requests do not follow the models' timing).
  void gl_tick() {
    if (gl_bg_) {
      if (rows_.size() < 2 && bg_rng_() % 3 == 0) {
        const uint32_t row = bg_rng_();
        this->push_row(row, bg_rng_() % ROW_RD_CLIENTS);
      }
      if (wr_pend_ == 0 && bg_rng_() % 5 == 0)
        this->push_wr(bg_rng_());
      const uint32_t b = bg_rng_() % nblocks_;
      if (live_[b].empty() && bg_rng_() % 4 == 0) {
        Req r = random_req(bg_rng_, nlanes_, false);
        if (!is_amo(r) && !is_mbar(r))
          gl_bg_ids_.push_back(this->submit(b, r));
      }
    }
    gl_progress_ = false;
    this->tick();
    gl_idle_ = gl_progress_ ? 0 : gl_idle_ + 1;
    if (gl_idle_ == 200000) {
      std::printf("  HANG: no global progress for %u cycles\nFAILED!\n", gl_idle_);
      std::exit(1);
    }
  }

  void gl_drain() {
    while (this->gl_live())
      this->gl_tick();
    const bool bg = gl_bg_;
    gl_bg_ = false;
    this->drain(gl_bg_ids_);
    gl_bg_ids_.clear();
    gl_bg_ = bg;
  }

  void gl_flush() {
    this->gl_drain();
    gl_flushed_ = false;
    d_.gl_flush();
    while (!gl_flushed_)
      this->gl_tick();
    this->gl_drain();
    dir_.invalidate_all();
  }

  uint32_t gl_alloc(uint64_t l, bool cacheable, uint32_t secs) {
    for (int k = 0; k < 1000; ++k) {
      auto a = dir_.allocate(l, cacheable, 0, secs);
      if (a.ok) {
        if (a.kind == vortex::l1_slot_ref::L1SlotDir::K_AGE)
          gl_draws_.push_back(dir_.last_draw_index());
        return a.slot;
      }
      dir_.tick();
    }
    std::printf("  golden: no allocation for line 0x%llx\nFAILED!\n", (unsigned long long)l);
    std::exit(1);
  }

  // The line reads and fill slots of one access run alone (as l1h100_cases' golden).
  std::vector<GlRd> gl_golden(const gl::Req& r) {
    std::vector<uint64_t> ls;
    std::map<uint64_t, std::pair<uint64_t, uint64_t>> wm;
    std::map<uint64_t, uint32_t> secs;
    const auto& c = dir_.cfg();
    for (uint32_t i = 0; i < nlanes_; ++i) {
      if (!(r.mask >> i & 1))
        continue;
      const uint64_t l = r.addr[i] / gl::LINE;
      const uint32_t off = uint32_t(r.addr[i] % gl::LINE);
      if (!wm.count(l))
        ls.push_back(l);
      wm[l].first |= vortex::l1_slot_ref::l1_touched_words(off, r.bytes, c);
      wm[l].second |= vortex::l1_slot_ref::l1_full_words(off, r.bytes, c);
      for (uint32_t b = off; b < off + r.bytes; ++b)
        secs[l] |= 1u << (b / gl::SECTOR);
    }
    std::vector<GlRd> rds;
    for (auto l : this->gl_tag_order(ls)) {
      if (r.kind == gl::K_ST) {
        if (!dir_.lookup(l, 0).tag)
          this->gl_alloc(l, true, 0);
        dir_.store(l, wm[l].second);
        continue;
      }
      if (dir_.lookup(l, wm[l].first).hit)
        continue;
      const uint32_t slot = this->gl_alloc(l, true, secs[l]);
      rds.push_back({l, secs[l], slot});
      dir_.fill(slot, secs[l]);
      dir_.fill_done(slot);
      dir_.replay_done(slot);
    }
    return rds;
  }

  // The line reads seen since gl_reads_/gl_fills_ were cleared, each with the
  // slot its fill wrote; a missing or extra fill shows as a last odd entry.
  std::vector<GlRd> gl_observed() const {
    std::vector<GlRd> got;
    for (size_t i = 0; i < gl_reads_.size(); ++i)
      got.push_back({gl_reads_[i].line, gl_reads_[i].secs, i < gl_fills_.size() ? gl_fills_[i] : ~0u});
    if (gl_fills_.size() != gl_reads_.size())
      got.push_back({gl::GL_LINE0, 0, uint32_t(gl_fills_.size())});
    return got;
  }

  // The order the tags take a request's lines: one line per tag bank and
  // cycle (each bank's first left in lane order), the lowest bank first.
  std::vector<uint64_t> gl_tag_order(std::vector<uint64_t> ls) const {
    std::vector<uint64_t> out;
    while (!ls.empty()) {
      std::map<uint32_t, uint64_t> pass;
      std::vector<uint64_t> rest;
      for (auto l : ls) {
        if (!pass.emplace(dir_.bank_of(l), l).second)
          rest.push_back(l);
      }
      for (const auto& x : pass)
        out.push_back(x.second);
      ls.swap(rest);
    }
    return out;
  }

  static std::string gl_fmt(const std::vector<GlRd>& v) {
    std::string s;
    char b[64];
    for (auto& x : v) {
      std::snprintf(b, sizeof(b), "%s%llx:%x@%u", s.empty() ? "" : "|", (unsigned long long)(x.line - gl::GL_LINE0),
                    x.secs, x.slot);
      s += b;
    }
    return s.empty() ? "hit" : s;
  }

  // Sequential global accesses on random blocks over twice the L1's lines,
  // the array's other clients busy beside them: hit or miss, line reads with
  // their sectors and the slot each fill writes against L1SlotDir.
  void global_seq(uint32_t seed, uint32_t n) {
    this->check(d_.set_cfg(reset_cfg()), "%s: the model's split differs from l1_slot_ref's (%ld, %ld)", "global", 0, 0);
    gl_rng_.seed(seed + 101);
    bg_rng_.seed(seed + 102);
    gl_seq_ = true;
    gl_bg_ = true;
    this->gl_flush();
    const uint64_t L = gl::GL_LINE0 + 4096;
    const uint32_t pool = dir_.lines() * 2;
    for (uint32_t k = 0; k < n; ++k) {
      const uint64_t l = L + gl_rng_() % pool;
      const uint32_t x = gl_rng_() % 10;
      const uint32_t w = (gl_rng_() % (gl::LINE / gl::WORD)) * gl::WORD;
      const uint32_t n_w = 1 + gl_rng_() % (gl::LINE / gl::WORD);
      const uint32_t bytes = (gl_rng_() & 1) ? gl::WORD : 4;
      const gl::Req r = (x == 0) ? this->gl_one(gl::K_ST, l, w, gl::WORD, "gseq-st")
                      : (x < 3)  ? this->gl_lanes(gl::K_LD, {l}, n_w, bytes, "gseq-line")
                                 : this->gl_one(gl::K_LD, l, w, gl::WORD, "gseq-ld");
      const auto exp = this->gl_golden(r);
      gl_reads_.clear();
      gl_fills_.clear();
      const uint32_t b = gl_rng_() % nblocks_;
      this->gl_submit(b, r);
      this->gl_drain();
      const auto got = this->gl_observed();
      this->check(got == exp, "%s (access %ld, block %ld)",
                  r.name + ": line reads and fill slots " + gl_fmt(got) + ", golden " + gl_fmt(exp), (long)k, (long)b);
      for (int t = 0; t < 8; ++t)
        dir_.tick();
      this->check(dir_.check() == 0, "%s: golden pool invariant %ld (access %ld)", "global", dir_.check(), (long)k);
      if (csv_)
        std::fprintf(csv_, "S,%u,%s,%s\n", k, r.name.c_str(), gl_fmt(got).c_str());
    }
    gl_seq_ = false;
    gl_bg_ = false;
  }

  // Overlapped global requests from every block, the other clients busy: each
  // block loads and stores lines of its own and loads read-only lines shared
  // by all, about twice the L1's lines in all; data in program order, every
  // fill in the L1's rows, the fill slot sequence in the CSV.
  void global_mix(uint32_t seed, uint32_t n) {
    gl_rng_.seed(seed + 202);
    bg_rng_.seed(seed + 203);
    gl_bg_ = true;
    this->gl_flush();
    const uint32_t own = std::max<uint32_t>(4, dir_.lines() / nblocks_);
    const uint64_t OWN0 = gl::GL_LINE0 + 65536, RO0 = gl::GL_LINE0 + 131072;
    for (uint32_t k = 0; k < n; ++k) {
      const uint32_t b = gl_rng_() % nblocks_;
      const uint32_t x = gl_rng_() % 8;
      const uint32_t nl = 1 + gl_rng_() % 2;
      std::vector<uint64_t> ls;
      for (uint32_t i = 0; i < nl; ++i)
        ls.push_back((x < 5) ? OWN0 + b + uint64_t(nblocks_) * (gl_rng_() % own) : RO0 + gl_rng_() % dir_.lines());
      const uint32_t lanes = 1 + gl_rng_() % nlanes_;
      const uint32_t bytes = (gl_rng_() & 1) ? gl::WORD : 4;
      this->gl_submit(b, (x < 2) ? this->gl_lanes(gl::K_ST, ls, lanes, bytes, "gmix-st")
                         : (x < 5) ? this->gl_lanes(gl::K_LD, ls, lanes, bytes, "gmix-own")
                                   : this->gl_lanes(gl::K_LD, ls, lanes, bytes, "gmix-ro"));
      if (gl_rng_() % 16 == 0) {
        for (uint32_t t = gl_rng_() % 32; t > 0; --t)
          this->gl_tick();
      }
    }
    this->gl_drain();
    gl_bg_ = false;
    // Outcomes by request id and the refill slot sequence apart, so the two models' CSVs differ only where the
    // outcomes or the slot order do, not where their timing interleaves them.
    if (csv_) {
      for (const auto& g : gl_csv_g_)
        std::fprintf(csv_, "G,%u,%s\n", g.first, g.second.c_str());
      for (size_t i = 0; i < gl_csv_l_.size(); ++i)
        std::fprintf(csv_, "L,%zu,%u\n", i, gl_csv_l_[i]);
    }
    gl_csv_g_.clear();
    gl_csv_l_.clear();
  }

  // An access run alone, its line reads and fill slots against the golden.
  void gl_seq_one(const gl::Req& r, uint32_t b, uint32_t k) {
    const auto exp = this->gl_golden(r);
    gl_reads_.clear();
    gl_fills_.clear();
    this->gl_submit(b, r);
    this->gl_drain();
    const auto got = this->gl_observed();
    this->check(got == exp, "%s (access %ld, block %ld)",
                r.name + ": line reads and fill slots " + gl_fmt(got) + ", golden " + gl_fmt(exp), (long)k, (long)b);
    this->gl_settle(k);
  }

  void gl_settle(uint32_t k) {
    for (int t = 0; t < 8; ++t)
      dir_.tick();
    this->check(dir_.check() == 0, "%s: golden pool invariant %ld (access %ld)", "global", dir_.check(), (long)k);
  }

  // Overlapped accesses whose order through the tags is the models' choice:
  // the golden takes the misses in the order their line reads were seen.
  void gl_observed_check(const char* name, uint32_t k) {
    const auto got = this->gl_observed();
    std::vector<GlRd> exp;
    for (const auto& x : gl_reads_) {
      const uint32_t slot = this->gl_alloc(x.line, true, x.secs);
      exp.push_back({x.line, x.secs, slot});
      dir_.fill(slot, x.secs);
      dir_.fill_done(slot);
      dir_.replay_done(slot);
    }
    this->check(got == exp, "%s (case %ld, golden %ld)",
                std::string(name) + ": line reads and fill slots " + gl_fmt(got) + ", golden " + gl_fmt(exp), (long)k,
                (long)exp.size());
    this->gl_settle(k);
  }

  // The line an allocation of l takes the slot of, on a copy of the golden.
  uint64_t gl_victim(uint64_t l, uint32_t secs) const {
    using Dir = vortex::l1_slot_ref::L1SlotDir;
    Dir p = dir_;
    for (int k = 0; k < 1000; ++k) {
      const auto a = p.allocate(l, true, 0, secs);
      if (a.ok)
        return (a.kind == Dir::K_AGE || a.kind == Dir::K_INSET) ? a.victim_line : 0;
      p.tick();
    }
    return 0;
  }

  // The first i such that single-line misses to first+i and first+i+1, after
  // misses to the lines before them, draw the same old-table index.
  uint32_t gl_same_draw(uint64_t first, uint32_t cap) const {
    using Dir = vortex::l1_slot_ref::L1SlotDir;
    Dir p = dir_;
    uint32_t prev = Dir::NONE;
    for (uint32_t i = 0; i < cap; ++i) {
      uint32_t idx = Dir::NONE;
      for (int k = 0;; ++k) {
        const auto a = p.allocate(first + i, true, 0, 1);
        if (a.ok) {
          if (a.kind == Dir::K_AGE)
            idx = p.last_draw_index();
          p.fill(a.slot, 1);
          p.fill_done(a.slot);
          p.replay_done(a.slot);
          break;
        }
        if (k == 1000)
          return Dir::NONE;
        p.tick();
      }
      if (idx != Dir::NONE && idx == prev)
        return i - 1;
      prev = idx;
      for (int t = 0; t < 8; ++t)
        p.tick();
    }
    return Dir::NONE;
  }

  // Directed global cases on a full L1 (every miss an age-list draw), golden
  // in step: (a) two draws of the same old-table index back to back, as one
  // two-line request and as two requests a cycle apart: the second takes the
  // slot the first moved into that index; (b) the victim of a miss looked up
  // 0..3 cycles after the miss, alone and behind row-read background; (c) with
  // several LSU blocks, a hit on a line and a store-allocate evicting it from
  // another block, 0..3 cycles apart. The victim's data must come back.
  void global_directed(uint32_t seed) {
    using Dir = vortex::l1_slot_ref::L1SlotDir;
    this->check(d_.set_cfg(reset_cfg()), "%s: the model's split differs from l1_slot_ref's (%ld, %ld)", "directed", 0,
                0);
    gl_rng_.seed(seed + 301);
    bg_rng_.seed(seed + 302);
    gl_seq_ = true;
    this->gl_flush();
    uint64_t next = gl::GL_LINE0 + 196608;
    uint32_t k = 0;
    for (uint32_t i = 0; i < dir_.lines() + 8; ++i, ++k)
      this->gl_seq_one(this->gl_one(gl::K_LD, next++, 0, gl::WORD, "gdir-fill"), k % nblocks_, k);
    if (csv_)
      std::fprintf(csv_, "D,fill,%u\n", dir_.lines() + 8);

    for (uint32_t v = 0; v < 2; ++v) {
      const uint32_t f = this->gl_same_draw(next, 8 * dir_.lines() + 64);
      if (f == Dir::NONE) {
        this->check(false, "%s: no two misses within %ld draw the same index (variant %ld)", "gdir-same", 8 * dir_.lines(),
                    v);
        break;
      }
      for (uint32_t j = 0; j < f; ++j, ++k)
        this->gl_seq_one(this->gl_one(gl::K_LD, next++, 0, gl::WORD, "gdir-walk"), k % nblocks_, k);
      const uint64_t l0 = next++, l1 = next++;
      const uint32_t b = nblocks_ - 1;
      gl_draws_.clear();
      std::vector<GlRd> exp;
      gl_reads_.clear();
      gl_fills_.clear();
      if (v == 0) {
        const gl::Req r = this->gl_lanes(gl::K_LD, {l0, l1}, 2, gl::WORD, "gdir-same-req");
        exp = this->gl_golden(r);
        this->gl_submit(b, r);
      } else {
        const gl::Req r0 = this->gl_one(gl::K_LD, l0, 0, gl::WORD, "gdir-same-a");
        const gl::Req r1 = this->gl_one(gl::K_LD, l1, 0, gl::WORD, "gdir-same-b");
        exp = this->gl_golden(r0);
        const auto e1 = this->gl_golden(r1);
        exp.insert(exp.end(), e1.begin(), e1.end());
        this->gl_submit(b, r0);
        this->gl_submit(b, r1);
      }
      this->gl_drain();
      this->check(gl_draws_.size() == 2 && gl_draws_[0] == gl_draws_[1],
                  "%s: the golden's draws are not one index twice (%ld draws, variant %ld)", "gdir-same",
                  (long)gl_draws_.size(), v);
      const auto got = this->gl_observed();
      this->check(got == exp, "%s (variant %ld, after %ld misses)",
                  std::string("gdir-same: line reads and fill slots ") + gl_fmt(got) + ", golden " + gl_fmt(exp), v, f);
      if (csv_)
        std::fprintf(csv_, "D,same,%u,%u,%s\n", v, f, gl_fmt(got).c_str());
      this->gl_settle(k++);
    }

    // From here the hit-or-miss of an overlapped access is the models' timing:
    // the CSV names the cases only.
    for (uint32_t bg = 0; bg < 2; ++bg) {
      for (uint32_t d = 0; d < 4; ++d, ++k) {
        const uint64_t y = next++;
        const uint64_t x = this->gl_victim(y, 1);
        this->check(x != 0, "%s: a miss with no victim (case %ld, %ld)", "gdir-victim", k, 0);
        if (x == 0)
          break;
        const gl::Req ry = this->gl_one(gl::K_LD, y, 0, gl::WORD, "gdir-victim-y");
        const gl::Req rx = this->gl_lanes(gl::K_LD, {x}, nlanes_, gl::WORD, "gdir-victim-x");
        gl_reads_.clear();
        gl_fills_.clear();
        gl_bg_ = (bg != 0);
        this->gl_submit(0, ry);
        for (uint32_t t = 0; t < d; ++t)
          this->gl_tick();
        this->gl_submit(nblocks_ - 1, rx);
        this->gl_drain();
        gl_bg_ = false;
        this->check(!gl_reads_.empty() && gl_reads_[0].line == y, "%s: the miss's line read is not the first (case %ld, %ld)",
                    "gdir-victim", k, (long)gl_reads_.size());
        this->gl_observed_check("gdir-victim", k);
        if (csv_)
          std::fprintf(csv_, "D,victim,%u,%u\n", bg, d);
      }
    }

    if (nblocks_ > 1) {
      for (uint32_t d = 0; d < 4; ++d, ++k) {
        const uint64_t y = next++;
        const uint64_t x = this->gl_victim(y, 0);
        this->check(x != 0, "%s: a store-allocate with no victim (case %ld, %ld)", "gdir-stalloc", k, 0);
        if (x == 0)
          break;
        const gl::Req rx = this->gl_lanes(gl::K_LD, {x}, nlanes_, gl::WORD, "gdir-stalloc-x");
        const gl::Req ry = this->gl_lanes(gl::K_ST, {y}, 1 + gl_rng_() % nlanes_, gl::WORD, "gdir-stalloc-y");
        // One in-order stream: X's lookup first; a refill of its missing
        // sectors keeps its slot in flight while Y's store-allocate draws.
        std::vector<GlRd> exp;
        uint64_t need = 0;
        for (uint32_t i = 0; i < nlanes_; ++i) {
          if (rx.mask >> i & 1)
            need |= vortex::l1_slot_ref::l1_touched_words(uint32_t(rx.addr[i] % gl::LINE), rx.bytes, dir_.cfg());
        }
        const uint32_t all = (1u << (gl::LINE / gl::SECTOR)) - 1;
        uint32_t sx = Dir::NONE;
        if (!dir_.lookup(x, need).hit) {
          sx = this->gl_alloc(x, true, all);
          exp.push_back({x, all, sx});
        }
        this->gl_golden(ry);
        if (sx != Dir::NONE) {
          dir_.fill(sx, all);
          dir_.fill_done(sx);
          dir_.replay_done(sx);
        }
        gl_reads_.clear();
        gl_fills_.clear();
        this->gl_submit(0, rx);
        for (uint32_t t = 0; t < d; ++t)
          this->gl_tick();
        this->gl_submit(1, ry);
        this->gl_drain();
        const auto got = this->gl_observed();
        this->check(got == exp, "%s (case %ld, offset %ld)",
                    std::string("gdir-stalloc: line reads and fill slots ") + gl_fmt(got) + ", golden " + gl_fmt(exp),
                    k, d);
        this->gl_settle(k);
        if (csv_)
          std::fprintf(csv_, "D,stalloc,%u,%s\n", d, gl_fmt(got).c_str());
      }
    }
    gl_seq_ = false;
  }

  // A phase flip or an open try_wait, reported on the word's first bank.
  struct MbarEvt {
    uint32_t id, bank, row, flip;
  };

  void mbar_event(const Event& e, uint64_t c) {
    if (mevt_.empty()) {
      this->check(false, "%s: event on bank %ld with no op expecting one (row %ld)", "mbarrier", e.blk, e.tag);
      return;
    }
    const MbarEvt x = mevt_.front();
    mevt_.pop_front();
    const Rec& r = recs_[x.id];
    this->check(e.blk == x.bank, "%s: event on bank %ld, expected bank %ld", r.name, e.blk, x.bank);
    this->check(e.tag == x.row, "%s: event for row %ld, expected row %ld", r.name, e.tag, x.row);
    this->check(e.cnt == x.flip, "%s: event flip %ld, expected %ld", r.name, e.cnt, x.flip);
    if (csv_)
      std::fprintf(csv_, "evt,%u,%u,%u,%llu\n", x.id, e.blk, e.cnt, (unsigned long long)(c - r.accept));
  }

#ifdef VX_CFG_EXT_MBAR_ENABLE
  static uint64_t mbar_addr(uint32_t w) { return MBAR_BASE + 8ull * w; }

  // One lane-0 op on word w: response word, response time and bank event
  // from mbar_ref on the bench's copy of the word, which ops update in order.
  uint32_t submit_mbar(uint32_t blk, const std::string& name, uint32_t w, uint32_t op, uint64_t operand,
                       bool lacc = false) {
    Req r;
    r.name = name;
    r.kind = K_MBAR;
    r.mop = op;
    r.bytes = 8;
    r.mask = 1;
    r.lacc = lacc && lacc_ok_;
    r.own = true;
    r.addr[0] = mbar_addr(w);
    r.val[0] = operand;
    const uint64_t old = mword_[w];
    const auto res = mbr::apply(op, old, operand);
    this->check(res.err == mbr::ERR_NONE, "%s: the bench's op is an mbarrier error (%ld, word %ld)", name, res.err, w);
    r.exp[0] = res.ret_word;
    if (res.writes)
      mword_[w] = res.new_word;
    const uint32_t id = this->submit(blk, r);
    const uint32_t bank = (uint32_t)(r.addr[0] / VX_CFG_LMEM_BANK_BYTES) % VX_CFG_LMEM_NUM_BANKS;
    const uint32_t row = (uint32_t)(r.addr[0] / VX_CFG_SM_DATAPATH_BYTES);
    if (res.writes && res.flip)
      mevt_.push_back({id, bank, row, 1});
    if (op == mbr::OP_TRY_WAIT && !mbr::phase_done(old, operand))
      mevt_.push_back({id, bank, row, 0});
    return id;
  }

  // The copy unit's lane-0 op on word w through the cp.async port, as
  // submit_mbar models it; its acknowledgement is its response.
  uint32_t submit_cpa_mbar(const std::string& name, uint32_t w, uint32_t op, uint64_t operand) {
    Req r;
    r.name = name;
    r.kind = K_MBAR;
    r.mop = op;
    r.bytes = 8;
    r.mask = 1;
    r.own = true;
    r.addr[0] = mbar_addr(w);
    r.val[0] = operand;
    const uint64_t old = mword_[w];
    const auto res = mbr::apply(op, old, operand);
    this->check(res.err == mbr::ERR_NONE, "%s: the bench's op is an mbarrier error (%ld, word %ld)", name, res.err, w);
    if (res.writes)
      mword_[w] = res.new_word;
    Rec rec;
    rec.id = (uint32_t)recs_.size();
    rec.cost = cost(r, cfg_, nlanes_);
    rec.gold = latency(r, cfg_, nlanes_);
    rec.read = true;
    rec.mbar = true;
    rec.cpa = true;
    rec.present = d_.now();
    rec.name = name;
    recs_.push_back(rec);
    const uint32_t tag = next_cpa_tag_;
    cpa_rec_[tag] = rec.id;
    this->push_cpa(r);
    const uint32_t bank = (uint32_t)(r.addr[0] / VX_CFG_LMEM_BANK_BYTES) % VX_CFG_LMEM_NUM_BANKS;
    const uint32_t row = (uint32_t)(r.addr[0] / VX_CFG_SM_DATAPATH_BYTES);
    if (res.writes && res.flip)
      mevt_.push_back({rec.id, bank, row, 1});
    return rec.id;
  }

  // Plain 64-bit stores of each word's initial value (vx_mbar.h's format).
  void mbar_init(const std::vector<uint32_t>& counts) {
    for (uint32_t w0 = 0; w0 < counts.size(); w0 += nlanes_) {
      Req r = strided("mbar init", K_ST, 8, 0, nlanes_, 8);
      r.own = true;
      for (uint32_t i = 0; i < nlanes_ && w0 + i < counts.size(); ++i) {
        r.mask |= 1u << i;
        r.addr[i] = mbar_addr(w0 + i);
        r.val[i] = mbr::init(counts[w0 + i]);
        mword_[w0 + i] = r.val[i];
      }
      this->drain({this->submit(0, r)});
    }
  }

  // 64-bit loads of every word: the values the ops left.
  void mbar_words(const char* phase) {
    for (uint32_t w0 = 0; w0 < MBAR_WORDS; w0 += nlanes_) {
      Req r = strided("mbar words", K_LD, 8, 0, nlanes_, 8);
      r.own = true;
      for (uint32_t i = 0; i < nlanes_ && w0 + i < MBAR_WORDS; ++i) {
        r.mask |= 1u << i;
        r.addr[i] = mbar_addr(w0 + i);
        r.exp[i] = mword_[w0 + i];
      }
      const uint32_t id = this->submit(0, r);
      this->drain({id});
      this->latency_check(recs_[id]);
      this->log(phase, recs_[id]);
    }
  }

  // Each op alone on an idle array: taken at once, answered a one-pass
  // atomic's time plus the op queue's stage after issue (+ K, as for loads),
  // whatever its kind or phase.
  void mbar_isolated() {
    mword_.assign(MBAR_WORDS, 0);
    std::vector<uint32_t> counts(MBAR_WORDS, 1);
    counts[0] = 2;
    counts[3] = 64;
    this->mbar_init(counts);
    const uint64_t tok = mword_[0];
    struct Op {
      const char* name;
      uint32_t op;
      uint64_t operand;
      bool lacc;
    };
    const Op ops[] = {
      {"test_wait open phase", mbr::OP_TEST_WAIT, tok, false},
      {"try_wait open phase", mbr::OP_TRY_WAIT, tok, false},
      {"arrive", mbr::OP_ARRIVE, 1, false},
      {"expect_tx", mbr::OP_EXPECT_TX, 64, false},
      {"arrive_tx", mbr::OP_ARRIVE_TX, 32, false},
      {"complete_tx completes the phase", mbr::OP_COMPLETE_TX, 96, false},
      {"test_wait completed phase", mbr::OP_TEST_WAIT, tok, false},
      {"try_wait completed phase", mbr::OP_TRY_WAIT, tok, false},
      {"arrive_drop", mbr::OP_ARRIVE, 1 | mbr::FLAG_DROP, false},
      {"arrive completes the phase, counted", mbr::OP_ARRIVE, 1, true},
    };
    for (const auto& o : ops) {
      const uint32_t id = this->submit_mbar(0, o.name, 0, o.op, o.operand, o.lacc);
      this->drain({id});
      Rec& r = recs_[id];
      this->check(r.accept == r.present, "%s: accepted %ld cycles after it was presented to an idle port (%ld)",
                  r.name, (long)(r.accept - r.present), 0);
      this->latency_check(r);
      this->log("mbi", r);
    }
    this->mbar_words("mbw");
  }

  // Back-to-back ops from one block: the op queue takes one a cycle and
  // its head takes one AMO pass a cycle, so every op is accepted the cycle
  // after the previous one and answered in order at issue + golden + K.
  void mbar_stream(const std::string& name, const std::vector<std::pair<uint32_t, std::pair<uint32_t, uint64_t>>>& ops) {
    const uint64_t t0 = d_.now();
    std::vector<uint32_t> ids;
    for (const auto& o : ops)
      ids.push_back(this->submit_mbar(0, name, o.first, o.second.first, o.second.second));
    this->drain(ids);
    std::vector<int64_t> a(ids.size()), rsp(ids.size());
    for (size_t k = 0; k < ids.size(); ++k) {
      a[k] = k ? a[k - 1] + 1 : 0;
      rsp[k] = std::max<int64_t>(a[k] + (int64_t)recs_[ids[k]].gold + k_, k ? rsp[k - 1] + 1 : 0);
    }
    for (size_t k = 0; k < ids.size(); ++k) {
      const Rec& r = recs_[ids[k]];
      this->check((int64_t)(r.accept - t0) == a[k], "%s: accepted at +%ld, expected +%ld", r.name,
                  (long)(r.accept - t0), (long)a[k]);
      this->check((int64_t)(r.rsp - t0) == rsp[k], "%s: answered at +%ld, expected +%ld", r.name,
                  (long)(r.rsp - t0), (long)rsp[k]);
      this->log("mbs", r);
    }
  }

  void mbar_streams() {
    using O = std::pair<uint32_t, std::pair<uint32_t, uint64_t>>;
    std::vector<O> v;
    for (int i = 0; i < 8; ++i)
      v.push_back({1, {mbr::OP_COMPLETE_TX, 1}});
    this->mbar_stream("complete_tx stream", v);
    v.clear();
    for (int i = 0; i < 2 * VX_CFG_MBAR_OP_QUEUE; ++i)
      v.push_back({1, {mbr::OP_EXPECT_TX, 1}});
    this->mbar_stream("expect_tx stream", v);
    v.clear();
    for (int i = 0; i < 8; ++i) {
      v.push_back({2, {mbr::OP_EXPECT_TX, 1}});
      v.push_back({2, {mbr::OP_COMPLETE_TX, 1}});
    }
    this->mbar_stream("expect_tx / complete_tx interleaved", v);
    v.clear();
    for (int i = 0; i < 2 * VX_CFG_MBAR_OP_QUEUE; ++i)
      v.push_back({3, {mbr::OP_ARRIVE, 1}});
    this->mbar_stream("arrive stream", v);
    v.clear();
    for (uint32_t w = 4; w < MBAR_WORDS; ++w)
      v.push_back({w, {mbr::OP_ARRIVE, 1}});
    this->mbar_stream("arrive over bank pairs", v);
    this->mbar_words("mbw");
  }

  // One warp's phase checks over every lane: lane i checks word w[i] with token
  // tok[i]. Checks of one word, op and token phase share one AMO pass, so a
  // request on one word and phase answers at the single-lane golden; every lane
  // gets its own word's result.
  void mbar_check_lanes(const std::string& name, uint32_t op, const std::vector<uint32_t>& w,
                        const std::vector<uint64_t>& tok) {
    Req r;
    r.name = name;
    r.kind = K_MBAR;
    r.mop = op;
    r.bytes = 8;
    r.own = true;
    bool one_pass = true;
    for (uint32_t i = 0; i < nlanes_; ++i) {
      r.mask |= 1u << i;
      r.addr[i] = mbar_addr(w[i]);
      r.val[i] = tok[i];
      const auto res = mbr::apply(op, mword_[w[i]], tok[i]);
      this->check(res.err == mbr::ERR_NONE && !res.writes, "%s: lane %ld is not a pure phase check", name, i, 0);
      this->check(op == mbr::OP_TEST_WAIT || mbr::phase_done(mword_[w[i]], tok[i]),
                  "%s: lane %ld would register a waiter", name, i, 0);
      r.exp[i] = res.ret_word;
      one_pass = one_pass && w[i] == w[0] && mbr::phase(tok[i]) == mbr::phase(tok[0]);
    }
    const uint32_t id = this->submit(0, r);
    this->drain({id});
    const Rec& rc = recs_[id];
    if (one_pass) {
      this->latency_check(rc);
    } else {
      const long extra = (long)(rc.rsp - rc.accept) - (long)rc.gold - k_;
      this->check(extra >= 0 && extra < (long)nlanes_, "%s: %ld cycles past golden + K, expected 0..%ld", name, extra,
                  (long)nlanes_ - 1);
    }
    this->log("mbr", rc);
  }

  void mbar_riders() {
    mword_.assign(MBAR_WORDS, 0);
    std::vector<uint32_t> counts(MBAR_WORDS, 2);
    counts[1] = 1;
    this->mbar_init(counts);
    const uint64_t open0 = mword_[0], done1 = mword_[1];
    this->drain({this->submit_mbar(0, "riders: complete word 1's phase", 1, mbr::OP_ARRIVE, 1)});
    const uint64_t open1 = mword_[1];
    std::vector<uint32_t> w0(nlanes_, 0), w1(nlanes_, 1), w01(nlanes_);
    std::vector<uint64_t> t0(nlanes_, open0), t1d(nlanes_, done1), t1o(nlanes_, open1), tph(nlanes_), t01(nlanes_);
    for (uint32_t i = 0; i < nlanes_; ++i) {
      tph[i] = (i & 1) ? open1 : done1;
      w01[i] = (i < nlanes_ / 2) ? 0 : 1;
      t01[i] = w01[i] ? done1 : open0;
    }
    this->mbar_check_lanes("riders: test_wait open phase, one word", mbr::OP_TEST_WAIT, w0, t0);
    this->mbar_check_lanes("riders: test_wait completed phase, one word", mbr::OP_TEST_WAIT, w1, t1d);
    this->mbar_check_lanes("riders: try_wait completed phase, one word", mbr::OP_TRY_WAIT, w1, t1d);
    this->mbar_check_lanes("riders: test_wait open phase after a completion", mbr::OP_TEST_WAIT, w1, t1o);
    if (nlanes_ > 1) {
      this->mbar_check_lanes("riders: test_wait two token phases, one word", mbr::OP_TEST_WAIT, w1, tph);
      this->mbar_check_lanes("riders: test_wait two words", mbr::OP_TEST_WAIT, w01, t01);
    }
    this->mbar_words("mbw");
  }

  // cp.async.mbarrier.arrive beside an LSU mbarrier op on one word: the copy
  // unit's op enters the same op queue, ahead of the LSU's in a shared cycle,
  // otherwise in arrival order, one a cycle; each answers in the time it
  // takes alone, which for both is one trip through the queue and the AMO path.
  void mbar_cpa_arrive() {
    mword_.assign(MBAR_WORDS, 0);
    std::vector<uint32_t> counts(MBAR_WORDS, 4);
    counts[0] = 2;
    counts[1] = 2;
    counts[3] = 1;
    this->mbar_init(counts);
    const uint32_t la = this->submit_mbar(0, "cpa arrive: LSU arrive alone", 2, mbr::OP_ARRIVE, 1);
    this->drain({la});
    const uint32_t ca = this->submit_cpa_mbar("cpa arrive: copy unit's arrive alone", 2, mbr::OP_ARRIVE, 1);
    this->drain({ca});
    const long lat_l = (long)(recs_[la].rsp - recs_[la].accept);
    const long lat_c = (long)(recs_[ca].rsp - recs_[ca].accept);
    this->check(recs_[ca].accept == recs_[ca].present, "%s: accepted %ld cycles after it was presented to an idle port (%ld)",
                recs_[ca].name, (long)(recs_[ca].accept - recs_[ca].present), 0);
    this->check(lat_c == lat_l, "%s: answered %ld cycles after issue, the LSU's arrive %ld", recs_[ca].name, lat_c, lat_l);
    this->latency_check(recs_[la]);
    this->log("mca", recs_[la]);
    this->log("mca", recs_[ca]);

    struct Two {
      uint32_t first, second;
    };
    auto pair_check = [&](const Two& t) {
      const Rec& f = recs_[t.first];
      const Rec& s = recs_[t.second];
      const long lat_f = (long)(f.rsp - f.accept), lat_s = (long)(s.rsp - s.accept);
      this->check(s.accept == f.accept + 1, "%s: accepted %ld cycles after the op ahead of it, expected %ld", s.name,
                  (long)(s.accept - f.accept), 1);
      this->check(lat_f == (f.cpa ? lat_c : lat_l), "%s: answered %ld cycles after issue, alone %ld", f.name, lat_f,
                  f.cpa ? lat_c : lat_l);
      this->check(lat_s == (s.cpa ? lat_c : lat_l), "%s: answered %ld cycles after issue, alone %ld", s.name, lat_s,
                  s.cpa ? lat_c : lat_l);
      this->log("mca", f);
      this->log("mca", s);
    };
    // One cycle: the copy unit's arrive, then the LSU's, which completes the phase.
    {
      const uint32_t c = this->submit_cpa_mbar("cpa arrive: copy unit first, same cycle", 0, mbr::OP_ARRIVE, 1);
      const uint32_t l = this->submit_mbar(0, "cpa arrive: LSU second, same cycle", 0, mbr::OP_ARRIVE, 1);
      this->drain({c, l});
      pair_check({c, l});
    }
    // The LSU's arrive a cycle ahead; the copy unit's completes the phase.
    {
      const uint32_t l = this->submit_mbar(0, "cpa arrive: LSU first", 1, mbr::OP_ARRIVE, 1);
      this->tick();
      const uint32_t c = this->submit_cpa_mbar("cpa arrive: copy unit second, completes", 1, mbr::OP_ARRIVE, 1);
      this->drain({l, c});
      pair_check({l, c});
    }
    // The counted form: the thread's expect_tx(1), then the copy unit's complete_tx(1).
    {
      this->drain({this->submit_mbar(0, "cpa arrive: expect_tx before the copy", 3, mbr::OP_EXPECT_TX, 1)});
      const uint32_t c = this->submit_cpa_mbar("cpa arrive: copy unit's complete_tx", 3, mbr::OP_COMPLETE_TX, 1);
      this->drain({c});
      this->log("mca", recs_[c]);
    }
    this->mbar_words("mcw");
  }
#endif

  struct RowRec {
    uint64_t gnt;
    bool granted;
  };
  using AmoRec = RowRec;

  Driver& d_;
  uint32_t nlanes_, nblocks_;
  FILE* csv_;
  bool lacc_ok_;
  SmemCfg cfg_;
  std::vector<Rec> recs_;
  std::map<uint32_t, uint32_t> live_[8];
  std::deque<uint32_t> pend_commit_[3];
  std::map<uint32_t, RowRec> rows_;
  std::vector<uint64_t> row_gnts_;
  std::vector<uint64_t> row_gnt_c_[ROW_RD_CLIENTS];
  std::vector<uint64_t> wr_gnts_;
  std::map<uint32_t, AmoRec> amo_live_;
  std::vector<uint64_t> amo_gnt_c_[AMO1_CLIENTS];
  uint32_t wr_pend_ = 0;
  uint32_t next_wr_tag_ = 0, next_amo_tag_ = 0;
  std::set<uint32_t> cpa_live_;
  std::map<uint32_t, uint32_t> cpa_rec_;
  std::vector<uint64_t> cpa_acc_c_, cpa_ack_c_;
  uint32_t next_cpa_tag_ = 0;
  uint32_t ds_pend_ = 0, next_ds_tag_ = 0;
  bool ds_busy_ = false;
  std::vector<uint64_t> ds_acc_c_, ds_busy_c_;
  int amo_lat_ = -1;
  std::deque<MbarEvt> mevt_;
#ifdef VX_CFG_EXT_MBAR_ENABLE
  std::vector<uint64_t> mword_;
#endif
  uint32_t next_tag_[8] = {};
  uint32_t next_row_tag_ = 0;
  uint32_t checks_ = 0, errors_ = 0;
  // The bench drives the array's own ports: the golden stages outside the array are not on its path.
  int k_ = -(int)cfg_.pipe;
  int row_lat_ = -1;
  vortex::l1_slot_ref::L1SlotDir dir_;
  gl::MemImage prog_;
  std::map<uint32_t, GlPend> gl_pend_[8];
  std::deque<uint32_t> gl_free_[8];
  std::vector<GlRd> gl_reads_;
  std::vector<uint32_t> gl_fills_;
  std::vector<uint32_t> gl_draws_;
  std::vector<uint32_t> gl_bg_ids_;
  std::mt19937 gl_rng_, bg_rng_;
  uint32_t gl_n_ = 0, gl_nfills_ = 0, gl_idle_ = 0, gl_reported_ = 0;
  bool gl_seq_ = false, gl_bg_ = false, gl_flushed_ = false, gl_progress_ = false;
  std::map<uint32_t, std::string> gl_csv_g_;
  std::vector<uint32_t> gl_csv_l_;
};

} // namespace sm_sram_cases
