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

// VX_cpa_unit testbench. The bench plays the LSU slice (cp.async issue events
// with their packets, global read responses split into random lane subsets
// after random delays) and the local memory behind the unit's port (random
// request back-pressure, every lane's store ack after its own random delay,
// so acks return out of order and in random groups). A request the port has
// not taken must be presented again unchanged: the real adapter takes a
// packet's lanes as their outputs free up.
//
// Directed runs pin the timing: a store ack at T unlocks a satisfied
// wait_group at T+3 and presents a waiting arrive at T+2; the 9th copy of a
// warp, the 9th pending group, a second arrive and a copy with every slot in
// use stall at issue until the state that blocks them clears; an arrive
// whose warp keeps the port busy with later copies (more than 16 of them,
// past the sequence numbers' range) still leaves once its copy is acked. Random runs mix
// copies of 4 and 8 bytes at every src-size (zero fill), commits, waits and
// arrives over all warps, and check every written byte, each arrive against
// the acks of the warp's earlier copies, each unlock against the groups still
// pending, and that everything drains.

#include "vl_simulator.h"
#include "VVX_cpa_unit_top.h"
#include "VX_config.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#ifndef VX_CFG_EXT_MBAR_ENABLE
#error "the cp.async unit needs an mbarrier build"
#endif

static constexpr uint32_t clog2(uint32_t n) {
  uint32_t r = 0;
  while ((1u << r) < n) {
    ++r;
  }
  return r;
}

static constexpr uint32_t NL       = VX_CFG_NUM_LSU_LANES;
static constexpr uint32_t NT       = VX_CFG_NUM_THREADS;
static constexpr uint32_t NW       = VX_CFG_NUM_WARPS;
static constexpr uint32_t NPIDS    = NT / NL;
static constexpr uint32_t LMEM_LOG = VX_CFG_LMEM_LOG_SIZE;
static constexpr uint32_t LMEM_SZ  = 1u << LMEM_LOG;
static constexpr uint32_t PENDING  = VX_CFG_LSU_PENDING_SIZE;
static constexpr uint32_t TAG_BITS = clog2(PENDING);
static constexpr uint32_t SLOTS    = std::min(PENDING, (1u << TAG_BITS) - 1);
static constexpr uint32_t ARR_TAG  = (1u << TAG_BITS) - 1;
static constexpr uint32_t DEPTH    = 8;
static constexpr uint32_t GROUPS   = 8;

enum : uint32_t { OP_COPY = 0, OP_COMMIT = 1, OP_WAIT = 2, OP_ARRIVE = 3 };
static constexpr uint32_t AMO_MBAR_ARRIVE = 10, AMO_MBAR_COMPLETE_TX = 13;

static std::string strf(const char* fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  return buf;
}

template <typename T>
static void set_bits(T& sig, uint32_t lsb, uint32_t width, uint64_t value) {
  if constexpr (std::is_integral_v<T>) {
    uint64_t mask = (width >= 64) ? ~0ull : ((1ull << width) - 1);
    uint64_t v = static_cast<uint64_t>(sig);
    v = (v & ~(mask << lsb)) | ((value & mask) << lsb);
    sig = static_cast<T>(v);
  } else {
    uint32_t i = 0;
    while (i < width) {
      uint32_t bit = lsb + i;
      uint32_t w = bit / 32;
      uint32_t o = bit % 32;
      uint32_t n = std::min(32 - o, width - i);
      uint32_t m = (n == 32) ? 0xffffffffu : ((1u << n) - 1);
      uint32_t chunk = static_cast<uint32_t>(value >> i) & m;
      sig[w] = (sig[w] & ~(m << o)) | (chunk << o);
      i += n;
    }
  }
}

template <typename T>
static uint64_t get_bits(const T& sig, uint32_t lsb, uint32_t width) {
  if constexpr (std::is_integral_v<T>) {
    uint64_t mask = (width >= 64) ? ~0ull : ((1ull << width) - 1);
    return (static_cast<uint64_t>(sig) >> lsb) & mask;
  } else {
    uint64_t value = 0;
    uint32_t i = 0;
    while (i < width) {
      uint32_t bit = lsb + i;
      uint32_t w = bit / 32;
      uint32_t o = bit % 32;
      uint32_t n = std::min(32 - o, width - i);
      uint32_t m = (n == 32) ? 0xffffffffu : ((1u << n) - 1);
      value |= static_cast<uint64_t>((sig[w] >> o) & m) << i;
      i += n;
    }
    return value;
  }
}

///////////////////////////////////////////////////////////////////////////////

struct Lane {
  bool     active = false;
  uint32_t dst = 0;       // local-memory byte offset
  uint32_t ss = 0;        // src-size of this uop
  uint32_t align = 0;     // source byte offset inside its word
  uint64_t word = 0;      // the global word the read returns
  bool     answered = false;
  uint64_t landed = 0;    // cycle the response landed
  bool     written = false;
  bool     acked = false;
  uint64_t ack_cycle = 0;
};

struct Copy {
  uint32_t wid = 0;
  bool     size8 = false;
  uint32_t slot = 0;
  bool     closed = false;
  Lane     lanes[NPIDS][NL];
  uint32_t lanes_left = 0;   // lanes not yet acknowledged
  uint64_t done_cycle = 0;   // cycle of the last ack
};

// One instruction of a warp's program.
struct Op {
  uint32_t kind = OP_COPY;
  bool     size8 = false;
  bool     noinc = false;
  uint32_t n = 0;
  uint32_t tmask = 0;                // NT bits
  uint32_t addr[NT] = {};            // copy destination / arrive mbarrier (byte offsets)
  uint32_t ss[NT] = {};
  uint32_t align[NT] = {};
  uint64_t word[NT] = {};
};

struct Pending {
  uint64_t due;
  uint32_t tag;
  uint32_t lane;
};

struct Options {
  uint32_t seed = 1;
  uint32_t ops = 1500;
};

class Bench {
public:
  explicit Bench(const Options& o) : opts_(o), rng_(o.seed) {
    auto* t = sim_.operator->();
    this->clear_inputs();
    t->reset = 1;
    for (int i = 0; i < 4; ++i) {
      tick_ = sim_.step(tick_, 2);
    }
    t->reset = 0;
  }

  int run() {
    std::printf("cpa_unit: lanes=%u threads=%u warps=%u pids=%u slots=%u tag_bits=%u lmem=%u seed=%u\n",
                NL, NT, NW, NPIDS, SLOTS, TAG_BITS, LMEM_SZ, opts_.seed);
    this->directed_timing();
    this->directed_fifo_full();
    this->directed_groups_full();
    this->directed_arrive_busy();
    this->directed_slots_full();
    this->directed_arrive_starve();
    this->directed_zero_fill();
    this->random_run(opts_.seed);
    if (failures_ != 0) {
      std::printf("FAILED: %u check(s) failed\n", failures_);
      return 1;
    }
    std::printf("PASSED\n");
    return 0;
  }

private:
  ///////////////////////////////////////////////////////////////////////////
  // Harness

  void fail(const std::string& m) {
    if (failures_ < 20) {
      std::printf("FAIL [cycle %" PRIu64 "] %s\n", cycle_, m.c_str());
    }
    ++failures_;
  }

  uint32_t rnd(uint32_t n) { return static_cast<uint32_t>(rng_() % n); }
  bool chance(uint32_t pct) { return this->rnd(100) < pct; }

  void clear_inputs() {
    auto* t = sim_.operator->();
    t->iss_valid = 0;
    t->iss_op = 0;
    t->iss_noinc = 0;
    t->iss_wid = 0;
    t->iss_pid = 0;
    t->iss_sop = 0;
    t->iss_eop = 0;
    t->iss_tmask = 0;
    t->iss_size8 = 0;
    t->iss_n = 0;
    t->rsp_valid = 0;
    t->rsp_mask = 0;
    t->req_ready = 0;
    t->ack_valid = 0;
    t->ack_mask = 0;
    t->ack_tag = 0;
  }

  // The issue the bench offers this cycle: one packet of one instruction.
  struct Packet {
    uint32_t wid, pid;
    bool sop, eop;
    uint32_t mask;
    const Op* op;
  };

  // Drives one cycle: the offered packet (if any), the response to land (if
  // any) and the local-memory side; returns whether the packet was taken.
  bool cycle(const std::optional<Packet>& pkt, bool req_ready, bool ack_ok = true) {
    auto* t = sim_.operator->();
    this->clear_inputs();
    if (pkt) {
      const Op& op = *pkt->op;
      t->iss_valid = 1;
      t->iss_op = op.kind;
      t->iss_noinc = op.noinc;
      t->iss_wid = pkt->wid;
      t->iss_pid = pkt->pid;
      t->iss_sop = pkt->sop;
      t->iss_eop = pkt->eop;
      t->iss_tmask = pkt->mask;
      t->iss_size8 = op.size8;
      t->iss_n = op.n;
      for (uint32_t j = 0; j < NL; ++j) {
        uint32_t k = pkt->pid * NL + j;
        set_bits(t->iss_addr, j * LMEM_LOG, LMEM_LOG, op.addr[k]);
        set_bits(t->iss_ss, j * 4, 4, op.ss[k]);
      }
    }
    // global responses: at most one per cycle, a random subset of a due packet's lanes
    std::optional<std::pair<Copy*, uint32_t>> land;
    uint32_t land_mask = 0;
    for (auto& [key, due] : rsp_due_) {
      auto sc = slot_copy_.find(key.first);
      if (hold_rsp_ || due > cycle_ || sc == slot_copy_.end()) {
        continue;
      }
      Copy* c = sc->second;
      uint32_t p = key.second;
      uint32_t avail = 0;
      for (uint32_t j = 0; j < NL; ++j) {
        if (c->lanes[p][j].active && !c->lanes[p][j].answered) {
          avail |= 1u << j;
        }
      }
      if (avail == 0) {
        continue;
      }
      uint32_t m = avail & (chance(60) ? avail : this->rnd(1u << NL));
      if (m == 0) {
        m = avail & (~avail + 1);
      }
      land = std::make_pair(c, p);
      land_mask = m;
      break;
    }
    if (land) {
      Copy* c = land->first;
      uint32_t p = land->second;
      t->rsp_valid = 1;
      t->rsp_slot = c->slot;
      t->rsp_pid = p;
      t->rsp_mask = land_mask;
      for (uint32_t j = 0; j < NL; ++j) {
        set_bits(t->rsp_data, j * 64, 64, c->lanes[p][j].word);
        set_bits(t->rsp_align, j * 3, 3, c->lanes[p][j].align);
      }
    }
    // store acks: one packet of due lanes sharing a tag
    std::vector<size_t> ack_idx;
    if (ack_ok) {
      for (size_t i = 0; i < acks_.size(); ++i) {
        if (acks_[i].due <= cycle_ && (ack_idx.empty() || acks_[i].tag == acks_[ack_idx[0]].tag)) {
          bool dup = false;
          for (size_t q : ack_idx) {
            dup |= (acks_[q].lane == acks_[i].lane);
          }
          if (!dup && (ack_idx.empty() || chance(70))) {
            ack_idx.push_back(i);
          }
        }
      }
    }
    if (!ack_idx.empty()) {
      t->ack_valid = 1;
      t->ack_tag = acks_[ack_idx[0]].tag;
      uint32_t m = 0;
      for (size_t q : ack_idx) {
        m |= 1u << acks_[q].lane;
      }
      t->ack_mask = m;
    }
    t->req_ready = req_ready;
    t->eval();

    // observe: a request not taken must come back unchanged
    {
      Snap now;
      now.valid = t->req_valid;
      now.rw = t->req_rw;
      now.mask = static_cast<uint32_t>(t->req_mask);
      now.tag = static_cast<uint32_t>(t->req_tag);
      for (uint32_t j = 0; j < NL; ++j) {
        if (now.mask & (1u << j)) {
          now.lanes.push_back({get_bits(t->req_addr, j * 32, 32), get_bits(t->req_data, j * 64, 64),
                               get_bits(t->req_byteen, j * 8, 8), get_bits(t->req_amo_op, j * 8, 8)});
        }
      }
      if (stalled_ && !(now == last_req_)) {
        this->fail("a local-memory request changed before it was taken");
      }
      stalled_ = now.valid && !req_ready;
      last_req_ = now;
    }
    bool took = pkt && t->iss_ready;
    if (took) {
      this->on_issue(*pkt, static_cast<uint32_t>(t->iss_slot));
    }
    if (land) {
      Copy* c = land->first;
      for (uint32_t j = 0; j < NL; ++j) {
        if (land_mask & (1u << j)) {
          c->lanes[land->second][j].answered = true;
          c->lanes[land->second][j].landed = cycle_;
        }
      }
    }
    if (t->req_valid && req_ready) {
      this->on_request();
    }
    if (!ack_idx.empty()) {
      uint32_t tag = acks_[ack_idx[0]].tag;
      std::sort(ack_idx.begin(), ack_idx.end());
      for (auto it = ack_idx.rbegin(); it != ack_idx.rend(); ++it) {
        uint32_t lane = acks_[*it].lane;
        acks_.erase(acks_.begin() + static_cast<long>(*it));
        if (tag != ARR_TAG) {
          this->on_ack(tag, lane);
        }
      }
    }
    if (t->unlock_valid) {
      this->on_unlock(static_cast<uint32_t>(t->unlock_wid));
    }

    tick_ = sim_.step(tick_, 2);
    ++cycle_;
    return took;
  }

  ///////////////////////////////////////////////////////////////////////////
  // Model

  void on_issue(const Packet& p, uint32_t slot) {
    const Op& op = *p.op;
    WarpState& w = warps_[p.wid];
    switch (op.kind) {
    case OP_COPY: {
      Copy* c;
      if (p.sop) {
        if (slot >= SLOTS || slot_copy_.count(slot)) {
          this->fail(strf("copy of warp %u took slot %u, which is out of range or in use", p.wid, slot));
        }
        copies_.push_back(std::make_unique<Copy>());
        c = copies_.back().get();
        c->wid = p.wid;
        c->size8 = op.size8;
        c->slot = slot;
        slot_copy_[slot] = c;
        w.copies.push_back(c);
        w.open = c;
        if (w.outstanding() > DEPTH) {
          this->fail(strf("warp %u has %u copies outstanding", p.wid, w.outstanding()));
        }
        if (slot_copy_.size() > SLOTS) {
          this->fail(strf("%zu copies hold slots", slot_copy_.size()));
        }
      } else {
        c = w.open;
        if (c == nullptr || c->slot != slot) {
          this->fail(strf("copy packet of warp %u on slot %u, not its open copy's", p.wid, slot));
          return;
        }
      }
      for (uint32_t j = 0; j < NL; ++j) {
        if (p.mask & (1u << j)) {
          uint32_t k = p.pid * NL + j;
          Lane& l = c->lanes[p.pid][j];
          l.active = true;
          l.dst = op.addr[k];
          l.ss = op.ss[k];
          l.align = op.align[k];
          l.word = op.word[k];
          ++c->lanes_left;
        }
      }
      if (p.eop) {
        c->closed = true;
        w.open = nullptr;
      }
      uint32_t delay = fixed_delays_ ? 3 : (chance(20) ? 1 + this->rnd(40) : 1 + this->rnd(6));
      rsp_due_[{slot, p.pid}] = cycle_ + delay;
      ++n_copy_pkts_;
      break;
    }
    case OP_COMMIT:
      if (p.eop) {
        w.groups.push_back(w.copies.size());
        ++n_commits_;
      }
      break;
    case OP_WAIT:
      if (p.eop) {
        w.waiting = true;
        w.wait_n = op.n;
        w.wait_cycle = cycle_;
        ++n_waits_;
      }
      break;
    case OP_ARRIVE: {
      if (p.sop) {
        w.arrives.push_back({});
        Arrive& a = w.arrives.back();
        a.copies_before = w.copies.size();
        a.noinc = op.noinc;
        a.issue_cycle = cycle_;
      }
      Arrive& a = w.arrives.back();
      for (uint32_t j = 0; j < NL; ++j) {
        if (p.mask & (1u << j)) {
          uint32_t k = p.pid * NL + j;
          a.lanes.push_back(op.addr[k] >> 3);
        }
      }
      if (p.eop) {
        a.closed = true;
        ++n_arrives_;
      }
      break;
    }
    }
  }

  void on_request() {
    auto* t = sim_.operator->();
    uint32_t mask = static_cast<uint32_t>(t->req_mask);
    uint32_t tag = static_cast<uint32_t>(t->req_tag);
    if (mask == 0) {
      this->fail("request with no lanes");
    }
    if (!t->req_rw) {
      // an arrive
      if (tag != ARR_TAG) {
        this->fail(strf("arrive request with tag %u", tag));
      }
      for (uint32_t j = 0; j < NL; ++j) {
        if (!(mask & (1u << j))) {
          continue;
        }
        uint32_t addr = static_cast<uint32_t>(get_bits(t->req_addr, j * 32, 32));
        uint64_t data = get_bits(t->req_data, j * 64, 64);
        uint32_t be = static_cast<uint32_t>(get_bits(t->req_byteen, j * 8, 8));
        uint32_t av = static_cast<uint32_t>(get_bits(t->req_amo_valid, j, 1));
        uint32_t aop = static_cast<uint32_t>(get_bits(t->req_amo_op, j * 8, 8));
        this->match_arrive(addr, data, be, av, aop);
        acks_.push_back({cycle_ + (fixed_delays_ ? 4 : 1 + this->rnd(4)), ARR_TAG, j});
      }
      ++n_arrive_pkts_;
      return;
    }
    // a data write: every lane must be a landed, unwritten lane of the slot
    auto it = slot_copy_.find(tag);
    if (it == slot_copy_.end()) {
      this->fail(strf("write with tag %u, no copy holds that slot", tag));
      return;
    }
    Copy* c = it->second;
    bool found_pid = false;
    for (uint32_t p = 0; p < NPIDS && !found_pid; ++p) {
      // the packet whose lanes these are: destinations do not overlap
      bool all = true;
      for (uint32_t j = 0; j < NL; ++j) {
        if (mask & (1u << j)) {
          const Lane& l = c->lanes[p][j];
          uint32_t addr = static_cast<uint32_t>(get_bits(t->req_addr, j * 32, 32));
          uint32_t be = static_cast<uint32_t>(get_bits(t->req_byteen, j * 8, 8));
          uint32_t want_be = ((1u << (c->size8 ? 8 : 4)) - 1) << (l.dst & 7);
          all &= l.active && l.answered && !l.written && (addr == (l.dst >> 3)) && (be == want_be);
        }
      }
      if (!all) {
        continue;
      }
      found_pid = true;
      for (uint32_t j = 0; j < NL; ++j) {
        if (!(mask & (1u << j))) {
          continue;
        }
        Lane& l = c->lanes[p][j];
        if (l.landed >= cycle_) {
          this->fail(strf("slot %u pid %u lane %u written in its landing cycle", tag, p, j));
        }
        uint32_t sz = c->size8 ? 8 : 4;
        uint64_t v = l.word >> (8 * l.align);
        uint64_t keep = (l.ss >= 8) ? ~0ull : ((1ull << (8 * l.ss)) - 1);
        v &= keep;
        if (sz == 4) {
          v &= 0xffffffffull;
        }
        uint32_t off = l.dst & 7;
        uint64_t want_data = v << (8 * off);
        uint32_t want_be = ((1u << sz) - 1) << off;
        uint32_t addr = static_cast<uint32_t>(get_bits(t->req_addr, j * 32, 32));
        uint64_t data = get_bits(t->req_data, j * 64, 64);
        uint32_t be = static_cast<uint32_t>(get_bits(t->req_byteen, j * 8, 8));
        uint32_t av = static_cast<uint32_t>(get_bits(t->req_amo_valid, j, 1));
        uint64_t data_m = 0, want_m = 0;
        for (uint32_t b = 0; b < 8; ++b) {
          if (be & (1u << b)) {
            data_m |= data & (0xffull << (8 * b));
          }
          if (want_be & (1u << b)) {
            want_m |= want_data & (0xffull << (8 * b));
          }
        }
        if (addr != (l.dst >> 3) || be != want_be || data_m != want_m || av) {
          this->fail(strf("write slot %u pid %u lane %u: addr 0x%x be 0x%x data 0x%016" PRIx64
                          " amo %u, expected addr 0x%x be 0x%x data 0x%016" PRIx64 " (ss %u align %u)",
                          tag, p, j, addr, be, data_m, av, l.dst >> 3, want_be, want_m, l.ss, l.align));
        }
        n_bytes_ += sz;
        n_zero_bytes_ += sz - std::min(sz, l.ss);
        l.written = true;
        acks_.push_back({cycle_ + (fixed_delays_ ? 4 : 1 + (chance(25) ? this->rnd(30) : this->rnd(5))), tag, j});
      }
    }
    if (!found_pid) {
      this->fail(strf("write slot %u mask 0x%x matches no landed, unwritten packet", tag, mask));
    }
    ++n_write_pkts_;
  }

  void on_ack(uint32_t slot, uint32_t lane) {
    auto it = slot_copy_.find(slot);
    if (it == slot_copy_.end()) {
      this->fail(strf("ack for free slot %u", slot));
      return;
    }
    Copy* c = it->second;
    // the lowest pid whose lane is written and not acked
    for (uint32_t p = 0; p < NPIDS; ++p) {
      Lane& l = c->lanes[p][lane];
      if (l.written && !l.acked) {
        l.acked = true;
        l.ack_cycle = cycle_;
        if (--c->lanes_left == 0 && c->closed) {
          c->done_cycle = cycle_;
          slot_copy_.erase(it);
        }
        return;
      }
    }
    this->fail(strf("ack for slot %u lane %u with no write outstanding", slot, lane));
  }

  void match_arrive(uint32_t addr, uint64_t data, uint32_t be, uint32_t av, uint32_t aop) {
    // the lowest warp with a matching unsent lane in its pending arrive
    for (uint32_t w = 0; w < NW; ++w) {
      WarpState& ws = warps_[w];
      for (Arrive& a : ws.arrives) {
        if (a.sent_all) {
          continue;
        }
        auto it = std::find(a.lanes.begin(), a.lanes.end(), addr);
        if (it == a.lanes.end()) {
          break;
        }
        uint32_t want_op = a.noinc ? AMO_MBAR_ARRIVE : AMO_MBAR_COMPLETE_TX;
        if (data != 1 || be != 0xff || !av || aop != want_op) {
          this->fail(strf("arrive of warp %u: data %" PRIu64 " be 0x%x amo %u op %u, expected op %u", w, data, be,
                          av, aop, want_op));
        }
        for (size_t i = 0; i < a.copies_before; ++i) {
          const Copy* c = ws.copies[i];
          if (!c->closed || c->lanes_left != 0 || c->done_cycle >= cycle_) {
            this->fail(strf("arrive of warp %u left before copy %zu of the warp was acknowledged", w, i));
            break;
          }
        }
        if (a.first_cycle == 0) {
          a.first_cycle = cycle_;
        }
        a.lanes.erase(it);
        if (a.lanes.empty()) {
          a.sent_all = true;
        }
        ++n_arrive_lanes_;
        return;
      }
    }
    this->fail(strf("arrive at word 0x%x matches no pending arrive", addr));
  }

  void on_unlock(uint32_t wid) {
    if (wid >= NW || !warps_[wid].waiting) {
      this->fail(strf("unlock of warp %u, which is not waiting", wid));
      return;
    }
    WarpState& w = warps_[wid];
    // groups beyond the N newest must have completed before this cycle
    uint32_t pending = 0;
    for (size_t g : w.groups) {
      bool done = true;
      for (size_t i = 0; i < g; ++i) {
        const Copy* c = w.copies[i];
        done &= c->closed && (c->lanes_left == 0) && (c->done_cycle < cycle_);
      }
      pending += !done;
    }
    if (pending > w.wait_n) {
      this->fail(strf("unlock of warp %u with %u groups pending, waiting for %u", wid, pending, w.wait_n));
    }
    w.waiting = false;
    w.unlock_cycle = cycle_;
    ++n_unlocks_;
  }

  struct Arrive {
    std::vector<uint32_t> lanes;
    size_t   copies_before = 0;
    bool     noinc = false;
    bool     closed = false;
    bool     sent_all = false;
    uint64_t issue_cycle = 0;
    uint64_t first_cycle = 0;
  };

  struct WarpState {
    std::vector<Copy*> copies;
    std::vector<size_t> groups;   // copies count at each commit
    std::deque<Arrive> arrives;
    Copy* open = nullptr;
    bool waiting = false;
    uint32_t wait_n = 0;
    uint64_t wait_cycle = 0;
    uint64_t unlock_cycle = 0;
    uint32_t outstanding() const {
      uint32_t n = 0;
      for (const Copy* c : copies) {
        n += (c->lanes_left != 0) || !c->closed;
      }
      return n;
    }
  };

  void reset_model() {
    for (auto& w : warps_) {
      w = WarpState{};
    }
  }

  // Runs idle cycles until the unit has nothing left to do.
  void drain(const char* ctx, uint32_t limit = 2000) {
    for (uint32_t i = 0; i < limit; ++i) {
      bool busy = !acks_.empty() || !slot_copy_.empty();
      for (auto& w : warps_) {
        busy |= w.waiting;
        for (auto& a : w.arrives) {
          busy |= !a.sent_all;
        }
      }
      if (!busy) {
        // the heads pop one entry a cycle
        const uint32_t all = (NW >= 32) ? ~0u : ((1u << NW) - 1);
        for (uint32_t k = 0; k < DEPTH + 2 && (static_cast<uint32_t>(sim_->idle) & all) != all; ++k) {
          this->cycle(std::nullopt, true);
        }
        if ((static_cast<uint32_t>(sim_->idle) & all) != all) {
          this->fail(strf("%s: drained, but warps 0x%x are not idle", ctx, ~static_cast<uint32_t>(sim_->idle) & all));
        }
        this->cycle(std::nullopt, true);
        this->cycle(std::nullopt, true);
        return;
      }
      this->cycle(std::nullopt, true);
    }
    this->fail(strf("%s: the unit did not drain", ctx));
  }

  ///////////////////////////////////////////////////////////////////////////
  // Op builders

  uint32_t next_dst_ = 0;
  uint32_t alloc_dst(uint32_t sz) {
    next_dst_ = (next_dst_ + sz - 1) & ~(sz - 1);
    if (next_dst_ + sz > LMEM_SZ / 2) {
      next_dst_ = 0;
    }
    uint32_t a = next_dst_;
    next_dst_ += sz;
    return a;
  }

  Op copy_op(bool size8, uint32_t tmask, std::optional<uint32_t> ss = std::nullopt) {
    Op op;
    op.kind = OP_COPY;
    op.size8 = size8;
    op.tmask = tmask;
    uint32_t sz = size8 ? 8 : 4;
    for (uint32_t k = 0; k < NT; ++k) {
      if (!(tmask & (1u << k))) {
        continue;
      }
      op.addr[k] = this->alloc_dst(sz);
      op.ss[k] = ss ? *ss : this->rnd(sz + 1);
      op.align[k] = size8 ? 0 : 4 * this->rnd(2);
      op.word[k] = rng_();
    }
    return op;
  }

  Op simple_op(uint32_t kind, uint32_t n = 0) {
    Op op;
    op.kind = kind;
    op.n = n;
    op.tmask = 1;
    return op;
  }

  // The mbarriers of each warp are its own, so every arrive lane names its warp.
  Op arrive_op(uint32_t wid, uint32_t tmask, bool noinc) {
    Op op;
    op.kind = OP_ARRIVE;
    op.noinc = noinc;
    op.tmask = tmask;
    for (uint32_t k = 0; k < NT; ++k) {
      op.addr[k] = LMEM_SZ / 2 + 8 * (wid * 32 + this->rnd(32));
    }
    return op;
  }

  // The packets of an instruction: its non-empty pids in order.
  static std::vector<Packet> packets(uint32_t wid, const Op& op) {
    std::vector<uint32_t> pids;
    for (uint32_t p = 0; p < NPIDS; ++p) {
      if ((op.tmask >> (p * NL)) & ((1u << NL) - 1)) {
        pids.push_back(p);
      }
    }
    std::vector<Packet> v;
    for (size_t i = 0; i < pids.size(); ++i) {
      v.push_back({wid, pids[i], i == 0, i + 1 == pids.size(), (op.tmask >> (pids[i] * NL)) & ((1u << NL) - 1), &op});
    }
    return v;
  }

  // Issues the whole instruction now, cycle by cycle; fails if it stalls.
  void issue_now(uint32_t wid, const Op& op, const char* what) {
    for (const Packet& p : packets(wid, op)) {
      if (!this->cycle(p, true)) {
        this->fail(strf("%s: warp %u stalled at issue", what, wid));
      }
    }
  }

  // Issues the whole instruction, waiting out stalls.
  void issue_wait(uint32_t wid, const Op& op, const char* what) {
    for (const Packet& p : packets(wid, op)) {
      bool took = false;
      for (int i = 0; i < 1000 && !took; ++i) {
        took = this->cycle(p, true);
      }
      if (!took) {
        this->fail(strf("%s: warp %u never issued", what, wid));
      }
    }
  }

  ///////////////////////////////////////////////////////////////////////////
  // Directed

  uint32_t full_mask() const { return (NT >= 32) ? ~0u : ((1u << (NT & 31)) - 1); }

  // One copy, its group, a wait for it and an arrive behind it; the copy's
  // single store ack is placed at a known cycle.
  void directed_timing() {
    const uint32_t before = failures_;
    this->reset_model();
    ops_.clear();
    ops_.push_back(this->copy_op(true, 1, 8));
    ops_.push_back(this->simple_op(OP_COMMIT));
    ops_.push_back(this->simple_op(OP_WAIT, 0));
    ops_.push_back(this->arrive_op(0, 1, false));
    this->issue_now(0, ops_[0], "timing copy");
    this->issue_now(0, ops_[1], "timing commit");
    this->issue_now(0, ops_[2], "timing wait");
    this->issue_now(0, ops_[3], "timing arrive");
    // land and write the lane, ack it at a known cycle
    for (int i = 0; i < 100 && acks_.empty(); ++i) {
      this->cycle(std::nullopt, true, false);
    }
    if (acks_.empty()) {
      this->fail("timing: the copy was never written");
      return;
    }
    acks_[0].due = cycle_;
    const uint64_t t_ack = cycle_;
    this->cycle(std::nullopt, true);
    for (int i = 0; i < 10; ++i) {
      this->cycle(std::nullopt, true);
    }
    const WarpState& w = warps_[0];
    if (w.unlock_cycle != t_ack + 3) {
      this->fail(strf("timing: ack at %" PRIu64 ", unlock at %" PRIu64 ", expected ack + 3", t_ack, w.unlock_cycle));
    }
    if (w.arrives.empty() || w.arrives[0].first_cycle != t_ack + 2) {
      this->fail(strf("timing: ack at %" PRIu64 ", arrive presented at %" PRIu64 ", expected ack + 2", t_ack,
                      w.arrives.empty() ? 0 : w.arrives[0].first_cycle));
    }
    this->drain("timing");
    std::printf("directed timing (ack -> arrive +2, unlock +3): %s\n", (failures_ == before) ? "ok" : "FAILED");
  }

  void directed_fifo_full() {
    const uint32_t before = failures_;
    this->reset_model();
    ops_.clear();
    const uint32_t n = std::min(DEPTH, SLOTS);
    for (uint32_t i = 0; i <= n; ++i) {
      ops_.push_back(this->copy_op(false, 1, 4));
    }
    hold_rsp_ = true;
    for (uint32_t i = 0; i < n; ++i) {
      this->issue_now(0, ops_[i], "fifo fill");
    }
    // the next copy stalls while nothing completes
    auto first = packets(0, ops_[n])[0];
    for (int i = 0; i < 20; ++i) {
      if (this->cycle(first, true, false)) {
        this->fail("fifo: a copy issued past a full FIFO / slot table");
        break;
      }
    }
    hold_rsp_ = false;
    bool took = false;
    for (int i = 0; i < 400 && !took; ++i) {
      took = this->cycle(first, true);
    }
    if (!took) {
      this->fail("fifo: the stalled copy never issued");
    }
    this->drain("fifo");
    std::printf("directed FIFO full (%u copies): %s\n", n, (failures_ == before) ? "ok" : "FAILED");
  }

  void directed_groups_full() {
    const uint32_t before = failures_;
    this->reset_model();
    ops_.clear();
    ops_.reserve(2 * GROUPS + 2);
    hold_rsp_ = true;
    const uint32_t n = std::min(GROUPS, std::min(DEPTH, SLOTS));
    for (uint32_t i = 0; i < n; ++i) {
      ops_.push_back(this->copy_op(true, 1, 8));
      ops_.push_back(this->simple_op(OP_COMMIT));
    }
    for (uint32_t i = 0; i < 2 * n; ++i) {
      this->issue_now(1 % NW, ops_[i], "groups fill");
    }
    // empty groups up to GROUPS pending need copies; with fewer, commits keep issuing
    ops_.push_back(this->simple_op(OP_COMMIT));
    auto pkt = packets(1 % NW, ops_.back())[0];
    bool stalled = !this->cycle(pkt, true, false);
    if ((n == GROUPS) != stalled) {
      this->fail(strf("groups: commit with %u groups pending %s", n, stalled ? "stalled" : "issued"));
    }
    hold_rsp_ = false;
    if (stalled) {
      bool took = false;
      for (int i = 0; i < 400 && !took; ++i) {
        took = this->cycle(pkt, true);
      }
      if (!took) {
        this->fail("groups: the stalled commit never issued");
      }
    }
    this->drain("groups");
    std::printf("directed group boundaries full: %s\n", (failures_ == before) ? "ok" : "FAILED");
  }

  void directed_arrive_busy() {
    const uint32_t before = failures_;
    this->reset_model();
    ops_.clear();
    ops_.push_back(this->copy_op(false, full_mask(), 3));
    ops_.push_back(this->arrive_op(NW - 1, full_mask(), true));
    ops_.push_back(this->arrive_op(NW - 1, full_mask(), false));
    hold_rsp_ = true;
    this->issue_now(NW - 1, ops_[0], "arrive copy");
    this->issue_now(NW - 1, ops_[1], "arrive 1");
    auto pkt = packets(NW - 1, ops_[2])[0];
    for (int i = 0; i < 10; ++i) {
      if (this->cycle(pkt, true, false)) {
        this->fail("arrive: a second arrive issued while one is pending");
        break;
      }
    }
    hold_rsp_ = false;
    bool took = false;
    for (int i = 0; i < 400 && !took; ++i) {
      took = this->cycle(pkt, true);
    }
    if (!took) {
      this->fail("arrive: the second arrive never issued");
    } else {
      for (size_t i = 1; i < packets(NW - 1, ops_[2]).size(); ++i) {
        this->cycle(packets(NW - 1, ops_[2])[i], true);
      }
    }
    this->drain("arrive");
    std::printf("directed second arrive: %s\n", (failures_ == before) ? "ok" : "FAILED");
  }

  void directed_slots_full() {
    const uint32_t before = failures_;
    this->reset_model();
    ops_.clear();
    ops_.reserve(SLOTS + 1);
    hold_rsp_ = true;
    for (uint32_t i = 0; i < SLOTS; ++i) {
      ops_.push_back(this->copy_op(true, 1, 8));
      this->issue_now(i % NW, ops_.back(), "slots fill");
    }
    ops_.push_back(this->copy_op(true, 1, 8));
    // a warp whose FIFO has room, if there is one
    const uint32_t w = (SLOTS % NW);
    auto pkt = packets(w, ops_.back())[0];
    for (int i = 0; i < 10; ++i) {
      if (this->cycle(pkt, true, false)) {
        this->fail("slots: a copy issued with every slot in use");
        break;
      }
    }
    hold_rsp_ = false;
    bool took = false;
    for (int i = 0; i < 400 && !took; ++i) {
      took = this->cycle(pkt, true);
    }
    if (!took) {
      this->fail("slots: the stalled copy never issued");
    }
    this->drain("slots");
    std::printf("directed slots full (%u slots): %s\n", SLOTS, (failures_ == before) ? "ok" : "FAILED");
  }

  // A copy, a noinc arrive behind it, then 24 more copies of the same warp
  // while the port takes one request in 4 cycles, responses and acks at fixed
  // delays: their writes always wait, yet the arrive leaves within two port
  // slots of its copy's ack.
  void directed_arrive_starve() {
    const uint32_t before = failures_;
    this->reset_model();
    fixed_delays_ = true;
    ops_.clear();
    const uint32_t K = 24;
    ops_.reserve(K + 2);
    ops_.push_back(this->copy_op(true, 1, 8));
    ops_.push_back(this->arrive_op(0, 1, true));
    for (uint32_t i = 0; i < K; ++i) {
      ops_.push_back(this->copy_op(true, 1, 8));
    }
    for (const Op& op : ops_) {
      for (const Packet& p : packets(0, op)) {
        bool took = false;
        for (int i = 0; i < 2000 && !took; ++i) {
          took = this->cycle(p, (cycle_ % 4) == 0);
        }
        if (!took) {
          this->fail("arrive starve: warp 0 never issued");
        }
      }
    }
    for (int i = 0; i < 400; ++i) {
      this->cycle(std::nullopt, (cycle_ % 4) == 0);
    }
    const WarpState& w = warps_[0];
    const Copy* c0 = w.copies.empty() ? nullptr : w.copies[0];
    const uint64_t first = w.arrives.empty() ? 0 : w.arrives[0].first_cycle;
    if (c0 == nullptr || c0->done_cycle == 0 || first == 0 || first > c0->done_cycle + 2 + 8) {
      this->fail(strf("arrive starve: copy 0 acked at %" PRIu64 ", arrive sent at %" PRIu64 ", expected within 10 cycles",
                      c0 ? c0->done_cycle : 0, first));
    }
    this->drain("arrive starve");
    fixed_delays_ = false;
    std::printf("directed arrive under write pressure (%u copies behind it): %s\n", K,
                (failures_ == before) ? "ok" : "FAILED");
  }

  // Every cp-size x src-size x source offset x destination offset.
  void directed_zero_fill() {
    const uint32_t before = failures_;
    this->reset_model();
    ops_.clear();
    ops_.reserve(64);
    uint32_t n = 0;
    for (uint32_t sz : {4u, 8u}) {
      for (uint32_t ss = 0; ss <= sz; ++ss) {
        for (uint32_t al = 0; al < 8; al += sz) {
          for (uint32_t doff = 0; doff < 8; doff += sz) {
            Op op = this->copy_op(sz == 8, 1, ss);
            op.align[0] = al;
            op.addr[0] = (this->alloc_dst(8) & ~7u) | doff;
            ops_.push_back(op);
            ++n;
          }
        }
      }
    }
    for (const Op& op : ops_) {
      this->issue_wait(n % NW, op, "zero fill");
      for (int i = 0; i < 3; ++i) {
        this->cycle(std::nullopt, true);
      }
    }
    this->drain("zero fill");
    std::printf("directed zero fill (%u copies): %s\n", n, (failures_ == before) ? "ok" : "FAILED");
  }

  ///////////////////////////////////////////////////////////////////////////
  // Random

  void random_run(uint32_t seed) {
    const uint32_t before = failures_;
    rng_.seed(seed * 2654435761u + 7);
    this->reset_model();
    ops_.clear();
    ops_.reserve(opts_.ops + 8);
    // per warp: the instructions still to issue, and the packets of the current one
    std::vector<std::deque<size_t>> prog(NW);
    for (uint32_t i = 0; i < opts_.ops; ++i) {
      uint32_t w = this->rnd(NW);
      uint32_t r = this->rnd(100);
      uint32_t tmask = this->rnd(full_mask()) + 1;
      if (r < 60) {
        ops_.push_back(this->copy_op(this->chance(50), tmask));
      } else if (r < 75) {
        ops_.push_back(this->simple_op(OP_COMMIT));
      } else if (r < 85) {
        ops_.push_back(this->simple_op(OP_WAIT, this->chance(20) ? this->rnd(32) : this->rnd(4)));
      } else {
        ops_.push_back(this->arrive_op(w, tmask, this->chance(50)));
      }
      prog[w].push_back(ops_.size() - 1);
    }
    std::vector<std::deque<Packet>> cur(NW);
    uint64_t start = cycle_;
    uint32_t idle = 0;
    while (true) {
      bool left = false;
      std::vector<uint32_t> ready_w;
      for (uint32_t w = 0; w < NW; ++w) {
        if (cur[w].empty() && !prog[w].empty() && !warps_[w].waiting) {
          const Op& op = ops_[prog[w].front()];
          prog[w].pop_front();
          for (const Packet& p : packets(w, op)) {
            cur[w].push_back(p);
          }
        }
        left |= !cur[w].empty() || !prog[w].empty();
        if (!cur[w].empty()) {
          ready_w.push_back(w);
        }
      }
      if (!left) {
        break;
      }
      std::optional<Packet> pkt;
      uint32_t pw = 0;
      if (!ready_w.empty() && this->chance(70)) {
        pw = ready_w[this->rnd(static_cast<uint32_t>(ready_w.size()))];
        pkt = cur[pw].front();
      }
      if (this->cycle(pkt, this->chance(75))) {
        cur[pw].pop_front();
        idle = 0;
      } else if (++idle > 5000) {
        this->fail("random: no issue for 5000 cycles");
        break;
      }
      if (cycle_ - start > 400000) {
        this->fail("random: run too long");
        break;
      }
    }
    this->drain("random", 5000);
    std::printf("random seed=%u: %u ops, %" PRIu64 " copy packets, %" PRIu64 " writes, %" PRIu64 " bytes (%" PRIu64
                " zero), %" PRIu64 " commits, %" PRIu64 " waits, %" PRIu64 " unlocks, %" PRIu64 " arrives (%" PRIu64
                " lanes), %" PRIu64 " cycles: %s\n",
                seed, opts_.ops, n_copy_pkts_, n_write_pkts_, n_bytes_, n_zero_bytes_, n_commits_, n_waits_,
                n_unlocks_, n_arrives_, n_arrive_lanes_, cycle_ - start, (failures_ == before) ? "ok" : "FAILED");
  }

  bool hold_rsp_ = false;
  bool fixed_delays_ = false;  // responses at +3, acks at +4

  struct Snap {
    bool valid = false, rw = false;
    uint32_t mask = 0, tag = 0;
    std::vector<std::array<uint64_t, 4>> lanes;
    bool operator==(const Snap& o) const {
      return valid == o.valid && rw == o.rw && mask == o.mask && tag == o.tag && lanes == o.lanes;
    }
  };
  bool stalled_ = false;
  Snap last_req_;

  Options opts_;
  std::mt19937_64 rng_;
  vl_simulator<VVX_cpa_unit_top> sim_;
  uint64_t tick_ = 0;
  uint64_t cycle_ = 0;
  uint32_t failures_ = 0;

  std::vector<Op> ops_;
  std::vector<std::unique_ptr<Copy>> copies_;
  std::map<uint32_t, Copy*> slot_copy_;
  std::map<std::pair<uint32_t, uint32_t>, uint64_t> rsp_due_;
  std::vector<Pending> acks_;
  WarpState warps_[NW];

  uint64_t n_copy_pkts_ = 0, n_write_pkts_ = 0, n_bytes_ = 0, n_zero_bytes_ = 0;
  uint64_t n_commits_ = 0, n_waits_ = 0, n_unlocks_ = 0, n_arrives_ = 0, n_arrive_lanes_ = 0;
  uint64_t n_arrive_pkts_ = 0;
};

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Options opts;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--seed=", 0) == 0) {
      opts.seed = static_cast<uint32_t>(std::strtoul(a.c_str() + 7, nullptr, 0));
    } else if (a.rfind("--ops=", 0) == 0) {
      opts.ops = static_cast<uint32_t>(std::strtoul(a.c_str() + 6, nullptr, 0));
    } else if (a.rfind("+verilator", 0) == 0) {
      continue;
    } else {
      std::printf("usage: %s [--seed=N] [--ops=N]\n", argv[0]);
      return 2;
    }
  }
  Bench b(opts);
  return b.run();
}
