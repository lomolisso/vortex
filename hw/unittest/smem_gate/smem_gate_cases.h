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

// The cases and checks of the shared-memory port bench, run unchanged by the
// RTL testbench (VX_smem_gate) and its SimX twin (SmemGate) through a driver
// with the same cycle contract. Every expectation comes from smem_bank_ref.h at
// the build's knobs. The unit adds a constant to every read latency (its share
// of the pipeline overhead), so read latencies are checked as golden + K, K
// measured on the first case; occupancy, order and data are checked exactly.
// Each run also writes a CSV of every request's present/accept/response cycle,
// which must be identical between the two models.

#include <smem_bank_ref.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace smem_gate_cases {

using namespace vortex::smem_bank_ref;

constexpr uint32_t MAX_LANES  = 32;
constexpr uint32_t NUM_TAGS   = 64;     // the bench's tag field: 6 bits per block
constexpr uint32_t SINK_DELAY = 2;      // bank round trip of the ideal sink
constexpr uint64_t WINDOW     = 1 << 16; // byte window the cases address

enum Kind : uint32_t { K_LD = 0, K_ST = 1, K_ADD = 2, K_CAS = 3 };

struct Req {
  std::string name;
  uint32_t kind  = K_LD;
  uint32_t mask  = 0;
  uint32_t bytes = 4;
  bool     lacc  = false;
  uint64_t addr[MAX_LANES] = {};
};

enum EvType : uint32_t { EV_ACCEPT = 0, EV_RSP = 1, EV_FOLD = 2 };

struct Event {
  uint32_t type;
  uint32_t blk;
  uint32_t tag;
  uint32_t out_mask;  // ACCEPT: lanes sent to the banks
  bool     data_ok;   // RSP: every lane answered with its word
  uint32_t fold = 0;  // FOLD: lanes the block's latest accept folded away, for the DLMEM commit counter
};

inline SmemCfg knob_cfg() {
  return SmemCfg{VX_CFG_LMEM_NUM_BANKS, VX_CFG_LMEM_BANK_BYTES, VX_CFG_LMEM_LATENCY, VX_CFG_LMEM_LATENCY_64,
                 VX_CFG_LMEM_WAVEFRONT_PENALTY, VX_CFG_LMEM_WIDE_OFFSET_64, VX_CFG_LMEM_ATOM_EXTRA,
                 VX_CFG_LMEM_CAS_EXTRA, VX_CFG_LMEM_CAS_PASSES};
}

// The value the ideal sink returns for a lane word (lane-word address).
inline uint64_t word_value(uint64_t word) {
  return (word + 1) * 0x9E3779B97F4A7C15ull;
}

inline bool is_read(const Req& r) { return r.kind != K_ST; }

inline uint32_t cost(const Req& r, const SmemCfg& c, uint32_t nlanes) {
  uint32_t n;
  if (r.kind == K_ADD || r.kind == K_CAS) {
    n = smem_atom_passes(r.addr, r.mask, nlanes, r.bytes, r.kind == K_CAS, c);
  } else {
    n = smem_wavefronts(r.addr, r.mask, nlanes, r.bytes, c);
  }
  return n ? n : 1;
}

inline uint32_t latency(const Req& r, const SmemCfg& c, uint32_t nlanes) {
  if (r.kind == K_ADD || r.kind == K_CAS) {
    bool cas = (r.kind == K_CAS);
    return smem_atom_latency(smem_atom_passes(r.addr, r.mask, nlanes, r.bytes, cas, c), cas, c);
  }
  if (r.kind == K_LD) {
    uint32_t d = smem_wavefronts(r.addr, r.mask, nlanes, r.bytes, c);
    return smem_load_latency(d, r.bytes, smem_distinct_words(r.addr, r.mask, nlanes, r.bytes), c);
  }
  return 0;
}

// Lanes the banks see: a load's lanes on one lane word reach them once;
// a 32-bit add on the high half of an 8-B bank word rides with the next
// unpaired lane on its low half.
template <uint32_t WORD_SIZE>
uint32_t out_mask(const Req& r, uint32_t nlanes) {
  if (r.kind == K_ADD && r.bytes == 4) {
    uint32_t m = r.mask, taken = 0;
    for (uint32_t j = 0; j < nlanes; ++j) {
      if (!(r.mask >> j & 1) || (r.addr[j] & 7) != 4)
        continue;
      for (uint32_t i = 0; i < nlanes; ++i) {
        if ((r.mask >> i & 1) && !(taken >> i & 1) && (r.addr[i] & 7) == 0 && (r.addr[i] >> 3) == (r.addr[j] >> 3)) {
          taken |= 1u << i;
          m &= ~(1u << j);
          break;
        }
      }
    }
    return m;
  }
  if (r.kind != K_LD)
    return r.mask;
  uint32_t m = 0;
  for (uint32_t j = 0; j < nlanes; ++j) {
    if (!(r.mask >> j & 1))
      continue;
    bool dup = false;
    for (uint32_t i = 0; i < j && !dup; ++i)
      dup = (r.mask >> i & 1) && (r.addr[i] / WORD_SIZE == r.addr[j] / WORD_SIZE);
    if (!dup)
      m |= 1u << j;
  }
  return m;
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
  r.lacc = lacc_ok && r.kind == K_LD && (rng() % 8 == 0);
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
  uint32_t id = 0, blk = 0, tag = 0, cost = 0, gold = 0, out_mask = 0, fold = 0, fold_seen = 0;
  uint64_t present = 0, accept = 0, rsp = 0;
  bool accepted = false, responded = false, data_ok = true, read = false;
  std::string name;
};

// Driver contract (both models): push() queues a request at a block's input,
// held valid from the next cycle() until accepted; set_ext() is the other
// clients' request line for the next cycle(); cycle() applies the inputs,
// samples the outputs into events, and advances one clock. The ideal sink
// answers each read SINK_DELAY cycles after the gate sends it.
template <typename Driver, uint32_t WORD_SIZE>
class Runner {
public:
  Runner(Driver& d, uint32_t nlanes, uint32_t nblocks, FILE* csv, bool lacc_ok)
    : d_(d), nlanes_(nlanes), nblocks_(nblocks), csv_(csv), lacc_ok_(lacc_ok), cfg_(knob_cfg()) {}

  int run(uint32_t seed, uint32_t random_reqs) {
    this->isolated();
    this->occupancy();
    this->ext_client();
    if (nblocks_ > 1)
      this->blocks();
    this->random(seed, random_reqs);
    for (const auto& r : recs_)
      this->check(r.fold_seen == r.fold, "%s: folded lanes counted %ld, expected %ld", r.name, r.fold_seen, r.fold);
    std::printf("smem_gate: %u checks, %u errors (K=%d)\n", checks_, errors_, k_);
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
      std::printf("smem_gate: stopping after %u errors\nFAILED!\n", errors_);
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
    rec.out_mask = out_mask<WORD_SIZE>(r, nlanes_);
    if (r.lacc)
      rec.fold = __builtin_popcount(r.mask) - __builtin_popcount(rec.out_mask);
    rec.present = d_.now();
    rec.name = r.name;
    d_.push(blk, r, rec.tag);
    live_[blk][rec.tag] = rec.id;
    recs_.push_back(rec);
    return rec.id;
  }

  void tick() {
    for (auto& e : d_.cycle()) {
      // A model may report the folded lanes after the accept cycle; they
      // belong to the block's latest accept.
      if (e.type == EV_FOLD) {
        auto la = last_acc_.find(e.blk);
        if (la == last_acc_.end())
          this->check(false, "%s: %ld folded lanes on block %ld before any accept", "fold", e.fold, e.blk);
        else
          recs_[la->second].fold_seen += e.fold;
        continue;
      }
      auto it = live_[e.blk].find(e.tag);
      if (it == live_[e.blk].end()) {
        this->check(false, "%s: event for no request on block %ld tag %ld", "event", e.blk, e.tag);
        continue;
      }
      Rec& r = recs_[it->second];
      const uint64_t c = d_.now() - 1;
      if (e.type == EV_ACCEPT) {
        r.accepted = true;
        r.accept = c;
        last_acc_[e.blk] = it->second;
        this->check(e.out_mask == r.out_mask, "%s: banks see lanes %lx, expected %lx", r.name, e.out_mask, r.out_mask);
        if (!r.read)
          live_[e.blk].erase(it);
      } else {
        r.responded = true;
        r.rsp = c;
        r.data_ok = e.data_ok;
        this->check(e.data_ok, "%s: response data or mask wrong (id %ld, %ld)", r.name, r.id, 0);
        live_[e.blk].erase(it);
      }
    }
  }

  bool done(const std::vector<uint32_t>& ids) const {
    for (auto id : ids) {
      const Rec& r = recs_[id];
      if (!r.accepted || (r.read && !r.responded))
        return false;
    }
    return true;
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

  // Overlapping reads: a block's reads leave in the order they were taken,
  // one a cycle, each at golden + K after its issue at the earliest.
  void inorder_check(const std::vector<uint32_t>& ids) {
    for (uint32_t b = 0; b < nblocks_; ++b) {
      std::vector<const Rec*> rd;
      for (auto id : ids)
        if (recs_[id].blk == b && recs_[id].read)
          rd.push_back(&recs_[id]);
      std::sort(rd.begin(), rd.end(), [](const Rec* x, const Rec* y) { return x->accept < y->accept; });
      uint64_t prev = 0;
      for (auto* r : rd) {
        uint64_t exp = std::max<uint64_t>(r->accept + r->gold + k_, rd.front() == r ? 0 : prev + 1);
        this->check(r->rsp == exp, "%s: answered at +%ld after issue, expected +%ld (in order)", r->name,
                    (long)(r->rsp - r->accept), (long)(exp - r->accept));
        prev = r->rsp;
      }
    }
  }

  void log(const char* phase, const Rec& r) {
    if (csv_)
      std::fprintf(csv_, "%s,%u,%u,%u,%llu,%llu,%llu\n", phase, r.id, r.blk, r.cost,
                   (unsigned long long)(r.accept - r.present),
                   (unsigned long long)(r.read ? r.rsp - r.accept : 0), (unsigned long long)r.out_mask);
  }

  // One request at a time: latency and the lanes the banks see.
  void isolated() {
    auto cases = directed(nlanes_, lacc_ok_);
    bool first = true;
    for (auto& c : cases) {
      uint32_t id = this->submit(0, c);
      this->drain({id});
      // The port stays busy for the request's cost after it was taken.
      for (uint32_t i = 0; i < recs_[id].cost; ++i)
        this->tick();
      Rec& r = recs_[id];
      if (first) {
        k_ = (int)((long)(r.rsp - r.accept) - (long)r.gold);
        first = false;
      }
      this->check(r.accept == r.present, "%s: accepted %ld cycles after it was presented to an idle port (%ld)",
                  r.name, (long)(r.accept - r.present), 0);
      this->latency_check(r);
      this->log("iso", r);
    }
  }

  // Back-to-back copies: the port is busy `cost` cycles per request.
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

  // Another client's request in cycle c holds the LSU off in cycle c + 1,
  // including the cycles of a request's own occupancy.
  void ext_client() {
    Req one = strided("ld32 stride 1", K_LD, 4, nlanes_, nlanes_, 4);
    for (uint32_t len : {1u, 3u}) {
      for (uint32_t lead : {0u, 1u, 2u}) {
        std::vector<bool> ext;
        for (uint32_t i = 0; i < lead; ++i)
          ext.push_back(false);
        for (uint32_t i = 0; i < len; ++i)
          ext.push_back(true);
        ext.push_back(false);
        uint64_t t0 = d_.now();
        uint32_t id = this->submit(0, one);
        for (bool e : ext) {
          d_.set_ext(e);
          this->tick();
        }
        d_.set_ext(false);
        this->drain({id});
        // The first cycle from t0 whose previous cycle had no other request.
        uint64_t exp = t0;
        for (uint64_t c = t0;; ++c) {
          bool prev = (c > t0) && (c - 1 - t0 < ext.size()) && ext[c - 1 - t0];
          if (!prev) { exp = c; break; }
        }
        const Rec& r = recs_[id];
        this->check(r.accept == exp, "%s: accepted at +%ld with other clients ahead, expected +%ld", "ext",
                    (long)(r.accept - t0), (long)(exp - t0));
        this->latency_check(r);
        this->log("ext", r);
      }
    }
    // A cost-4 request: other clients' cycles extend its occupancy.
    Req four = strided("ld32 stride 8", K_LD, 4, nlanes_, nlanes_, 32);
    if (cost(four, cfg_, nlanes_) > 2) {
      uint64_t t0 = d_.now();
      uint32_t a = this->submit(0, four);
      uint32_t b = this->submit(0, one);
      std::vector<bool> ext = {false, true, true, false};
      for (bool e : ext) {
        d_.set_ext(e);
        this->tick();
      }
      d_.set_ext(false);
      this->drain({a, b});
      const Rec& ra = recs_[a];
      const Rec& rb = recs_[b];
      this->check(ra.accept == t0, "%s: accepted at +%ld, expected +%ld", "ext occupancy", (long)(ra.accept - t0), 0);
      this->check(rb.accept - ra.accept == ra.cost + 2, "%s: next request %ld cycles later, expected cost + 2 = %ld",
                  "ext occupancy", (long)(rb.accept - ra.accept), (long)ra.cost + 2);
      this->log("ext", ra);
      this->log("ext", rb);
    }
  }

  // Several LSU blocks share one wavefront a cycle, round robin.
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
      // round robin: each block's i-th request goes before any block's (i+1)-th
      for (int i = 0; i + 1 < 8; ++i)
        for (uint32_t b = 0; b < nblocks_; ++b)
          for (uint32_t b2 = 0; b2 < nblocks_; ++b2)
            this->check(recs_[ids[i * nblocks_ + b]].accept < recs_[ids[(i + 1) * nblocks_ + b2]].accept,
                        "%s: block %ld served twice before block %ld", "round robin", b, b2);
      for (auto id : ids)
        this->log("blk", recs_[id]);
    }
  }

  // Random requests on every block at random times: latency, data, port rate.
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
      d_.set_ext(rng() % 16 == 0);
      this->tick();
    }
    d_.set_ext(false);
    this->drain(ids);
    std::vector<std::pair<uint64_t, uint32_t>> acc;
    for (auto id : ids) {
      acc.push_back({recs_[id].accept, recs_[id].cost});
      this->log("rnd", recs_[id]);
    }
    this->inorder_check(ids);
    std::sort(acc.begin(), acc.end());
    for (size_t i = 1; i < acc.size(); ++i)
      this->check(acc[i].first - acc[i - 1].first >= acc[i - 1].second,
                  "%s: two requests %ld cycles apart, the first costs %ld", "random port", (long)(acc[i].first - acc[i - 1].first),
                  acc[i - 1].second);
  }

  Driver& d_;
  uint32_t nlanes_, nblocks_;
  FILE* csv_;
  bool lacc_ok_;
  SmemCfg cfg_;
  std::vector<Rec> recs_;
  std::map<uint32_t, uint32_t> live_[8];
  std::map<uint32_t, uint32_t> last_acc_;
  uint32_t next_tag_[8] = {};
  uint32_t checks_ = 0, errors_ = 0;
  int k_ = 0;
};

} // namespace smem_gate_cases
