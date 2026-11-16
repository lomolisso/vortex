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

// Golden model of an H100-shaped SM's ports: the egress item costs, the
// request FIFO and response queue drained round robin per half-beat, the
// SM-to-SM injection credit loop, the single-sector issue limits, the
// weighted ingress gate and the SM-to-SM packet slot rule. SimX, the RTL
// testbench and the host test run this one model; shapes come from SmPortCfg,
// never VX_config.h.
//
// One cycle of either model: pushes test their admission against the state
// at the start of the cycle and become visible to the next cycle's tick; a
// tick deals the cycle's half-beats (egress) or sectors (ingress) to the
// heads present at its start, then lets each paid head whose sink is ready
// leave, at most one item per queue per cycle.

#include <algorithm>
#include <cstdint>
#include <deque>
#include <vector>

namespace vortex {
namespace sm_port_ref {

enum class SmItem {
  WriteLine,     // global write: the written bytes of one line
  ReadLine,      // global read request of one line (any sectors)
  AtomicLanes,   // global atomic, lanes not packed
  PrefetchLine,  // L2 prefetch of one line
  TmaStoreLine,  // TMA bulk-store data of one line
  DsmemStore,    // SM-to-SM store packet (one destination line)
  DsmemLoadReq,  // SM-to-SM load request packet
  DsmemAtomic,   // SM-to-SM atomic request packet
  DsmemResp,     // SM-to-SM reply (load data, atomic returns)
  MbarRemote     // remote mbarrier operation
};

struct SmPortCfg {
  uint32_t egress_bytes;      // egress port bytes per cycle
  uint32_t beat_bytes;        // one beat = one sector = one SM-to-SM slot
  uint32_t hb_wr_sector;      // half-beats per full written sector
  uint32_t hb_wr_masked;      // half-beats per byte-masked written sector
  uint32_t hb_rd_line;        // half-beats per read request line
  uint32_t hb_pf_line;        // half-beats per prefetch line
  uint32_t hb_atom_lane;      // half-beats per global atomic lane
  uint32_t hb_dsmem_slot;     // half-beats per SM-to-SM slot
  uint32_t hb_dsmem_ld_line;  // half-beats per SM-to-SM load request line
  uint32_t hb_tma_st_sector;  // half-beats per TMA bulk-store sector
  uint32_t hb_mbar;           // half-beats per remote mbarrier operation
  uint32_t min_slots;         // slots of a store packet with few sectors
  uint32_t atom_extra_slots;  // request slots of an atomic packet beyond its sectors
  uint32_t inj_credits;       // injection credits
  uint32_t inj_return;        // cycles until a used credit returns
  uint32_t eject_bytes;       // SM-to-SM arrivals per cycle
  uint32_t ingress_bytes;     // ingress port bytes per cycle
  uint32_t w_fill;            // ingress weight of fills (L2 and TMA replies)
  uint32_t w_dsmem;           // ingress weight of SM-to-SM arrivals
  uint32_t issue_w_pm;        // single-sector write requests per 1000 cycles
  uint32_t issue_r_pm;        // single-sector read requests per 1000 cycles
  uint32_t req_fifo_bytes;    // request FIFO capacity, in beats of the queued items
  uint32_t rsp_slots;         // response queue capacity, in slots
};

inline uint32_t popcount64(uint64_t v) {
  uint32_t n = 0;
  while (v) {
    v &= v - 1;
    ++n;
  }
  return n;
}

// Sectors of a line with any / every byte enabled.
inline uint32_t touched_sectors(const uint8_t* byteen, uint32_t line_bytes, uint32_t sector_bytes) {
  uint32_t n = 0;
  for (uint32_t s = 0; s < line_bytes / sector_bytes; ++s) {
    bool any = false;
    for (uint32_t b = 0; b < sector_bytes; ++b)
      any |= (byteen[s * sector_bytes + b] != 0);
    n += any;
  }
  return n;
}

inline uint32_t full_sectors(const uint8_t* byteen, uint32_t line_bytes, uint32_t sector_bytes) {
  uint32_t n = 0;
  for (uint32_t s = 0; s < line_bytes / sector_bytes; ++s) {
    bool all = true;
    for (uint32_t b = 0; b < sector_bytes; ++b)
      all &= (byteen[s * sector_bytes + b] != 0);
    n += all;
  }
  return n;
}

inline uint32_t dsmem_packet_slots(uint32_t touched, SmItem kind, const SmPortCfg& cfg) {
  switch (kind) {
  case SmItem::DsmemStore:   return (touched > cfg.min_slots) ? touched : cfg.min_slots;
  case SmItem::DsmemLoadReq: return 1;
  case SmItem::DsmemAtomic:  return touched + cfg.atom_extra_slots;
  case SmItem::DsmemResp:    return touched;
  case SmItem::MbarRemote:   return 1;
  default:                   return 0;
  }
}

// Bytes beyond the slots (st.async / red.async extras), in half-beats, rounded up.
inline uint32_t extra_halfbeats(uint32_t extra_bytes, const SmPortCfg& cfg) {
  uint32_t hb_bytes = cfg.beat_bytes / 2;
  return (extra_bytes + hb_bytes - 1) / hb_bytes;
}

// An item's cost on the egress port, in half-beats. byteen has line_bytes
// entries (0 / non-zero); sectors are beat_bytes wide.
inline uint32_t sm_item_halfbeats(SmItem kind, const uint8_t* byteen, uint32_t line_bytes,
                                  uint32_t active_lanes, uint32_t extra_bytes, const SmPortCfg& cfg) {
  uint32_t sb = cfg.beat_bytes;
  switch (kind) {
  case SmItem::WriteLine: {
    uint32_t t = touched_sectors(byteen, line_bytes, sb);
    uint32_t f = full_sectors(byteen, line_bytes, sb);
    return f * cfg.hb_wr_sector + (t - f) * cfg.hb_wr_masked;
  }
  case SmItem::ReadLine:     return cfg.hb_rd_line;
  case SmItem::AtomicLanes:  return active_lanes * cfg.hb_atom_lane;
  case SmItem::PrefetchLine: return cfg.hb_pf_line;
  case SmItem::TmaStoreLine: return touched_sectors(byteen, line_bytes, sb) * cfg.hb_tma_st_sector;
  case SmItem::DsmemLoadReq: return cfg.hb_dsmem_ld_line + extra_halfbeats(extra_bytes, cfg);
  case SmItem::MbarRemote:   return cfg.hb_mbar;
  case SmItem::DsmemStore:
  case SmItem::DsmemAtomic:
  case SmItem::DsmemResp: {
    uint32_t t = touched_sectors(byteen, line_bytes, sb);
    return dsmem_packet_slots(t, kind, cfg) * cfg.hb_dsmem_slot + extra_halfbeats(extra_bytes, cfg);
  }
  }
  return 0;
}

// Injection credits of an SM-to-SM item: one per slot it occupies.
inline uint32_t credit_slots(uint32_t hb) {
  return (hb + 1) / 2;
}

struct SmEgressItem {
  uint32_t hb = 0;      // cost in half-beats
  uint32_t slots = 0;   // injection credits it takes (0: an L2-side item)
  bool     l2 = false;  // leaves toward the L2 (else toward the SM-to-SM network)
  uint8_t  single = 0;  // 1: single-sector write, 2: single-sector read
  uint64_t id = 0;
};

class SmEgressRef {
public:
  enum { REQ = 0, RSP = 1 };

  struct Pops {
    bool req = false;
    bool rsp = false;
    SmEgressItem req_item;
    SmEgressItem rsp_item;
  };

  explicit SmEgressRef(const SmPortCfg& cfg)
    : cfg_(cfg)
    , units_(2 * cfg.egress_bytes / cfg.beat_bytes)
    , req_cap_hb_(2 * cfg.req_fifo_bytes / cfg.beat_bytes)
    , req_cap_items_(2 * cfg.req_fifo_bytes / cfg.beat_bytes)
    , rsp_cap_hb_(2 * cfg.rsp_slots)
    , rsp_cap_items_(cfg.rsp_slots)
    , ret_(cfg.inj_return, 0) {
    this->reset();
  }

  void reset() {
    for (int k = 0; k < 2; ++k) {
      q_[k].clear();
      occ_hb_[k] = 0;
      paid_[k] = 0;
    }
    staged_[REQ].clear();
    staged_[RSP].clear();
    turn_ = REQ;
    credits_ = cfg_.inj_credits;
    std::fill(ret_.begin(), ret_.end(), 0);
    ret_pos_ = 0;
    tok_w_ = 1000 + cfg_.issue_w_pm;
    tok_r_ = 1000 + cfg_.issue_r_pm;
  }

  uint32_t units() const { return units_; }
  uint32_t req_cap_items() const { return req_cap_items_; }
  uint32_t req_cap_hb() const { return req_cap_hb_; }
  uint32_t rsp_cap_items() const { return rsp_cap_items_; }
  uint32_t rsp_cap_hb() const { return rsp_cap_hb_; }

  bool req_can_push(const SmEgressItem& it) const {
    if (!staged_[REQ].empty())
      return false;
    if (occ_hb_[REQ] + it.hb > req_cap_hb_ || q_[REQ].size() >= req_cap_items_)
      return false;
    if (it.single == 1 && tok_w_ < 1000)
      return false;
    if (it.single == 2 && tok_r_ < 1000)
      return false;
    return true;
  }

  bool rsp_can_push(const SmEgressItem& it) const {
    if (!staged_[RSP].empty())
      return false;
    return (occ_hb_[RSP] + it.hb <= rsp_cap_hb_) && (q_[RSP].size() < rsp_cap_items_);
  }

  void req_push(const SmEgressItem& it) {
    staged_[REQ].push_back(it);
    occ_hb_[REQ] += it.hb;
    if (it.single == 1)
      tok_w_ -= 1000;
    if (it.single == 2)
      tok_r_ -= 1000;
  }

  void rsp_push(const SmEgressItem& it) {
    staged_[RSP].push_back(it);
    occ_hb_[RSP] += it.hb;
  }

  bool empty() const {
    return q_[REQ].empty() && q_[RSP].empty() && staged_[REQ].empty() && staged_[RSP].empty();
  }

  uint32_t credits() const { return credits_; }

  Pops tick(bool req_l2_ready, bool req_dsm_ready, bool rsp_ready) {
    credits_ += ret_[ret_pos_];
    ret_[ret_pos_] = 0;
    uint32_t used = 0;

    for (uint32_t u = 0; u < units_; ++u) {
      bool cand[2];
      for (int k = 0; k < 2; ++k)
        cand[k] = this->candidate(k);
      int g;
      if (cand[REQ] && cand[RSP]) {
        g = turn_;
      } else if (cand[REQ]) {
        g = REQ;
      } else if (cand[RSP]) {
        g = RSP;
      } else {
        break;
      }
      const auto& h = q_[g].front();
      if (this->needs_credit(h, paid_[g])) {
        --credits_;
        ++used;
      }
      ++paid_[g];
      turn_ = 1 - g;
    }
    ret_[ret_pos_] = used;
    ret_pos_ = (ret_pos_ + 1) % ret_.size();

    Pops pops;
    if (!q_[REQ].empty() && paid_[REQ] >= q_[REQ].front().hb) {
      const auto& h = q_[REQ].front();
      if (h.l2 ? req_l2_ready : req_dsm_ready) {
        pops.req = true;
        pops.req_item = h;
        occ_hb_[REQ] -= h.hb;
        paid_[REQ] = 0;
        q_[REQ].pop_front();
      }
    }
    if (!q_[RSP].empty() && paid_[RSP] >= q_[RSP].front().hb && rsp_ready) {
      const auto& h = q_[RSP].front();
      pops.rsp = true;
      pops.rsp_item = h;
      occ_hb_[RSP] -= h.hb;
      paid_[RSP] = 0;
      q_[RSP].pop_front();
    }

    tok_w_ = std::min<int64_t>(tok_w_ + cfg_.issue_w_pm, 1000 + cfg_.issue_w_pm);
    tok_r_ = std::min<int64_t>(tok_r_ + cfg_.issue_r_pm, 1000 + cfg_.issue_r_pm);

    for (int k = 0; k < 2; ++k) {
      for (auto& it : staged_[k])
        q_[k].push_back(it);
      staged_[k].clear();
    }
    return pops;
  }

private:
  // A unit that starts a slot of an SM-to-SM item takes an injection credit.
  static bool needs_credit(const SmEgressItem& it, uint32_t paid) {
    return (paid % 2) == 0 && (paid / 2) < it.slots;
  }

  bool candidate(int k) const {
    if (q_[k].empty())
      return false;
    const auto& h = q_[k].front();
    if (paid_[k] >= h.hb)
      return false;
    if (needs_credit(h, paid_[k]) && credits_ == 0)
      return false;
    return true;
  }

  SmPortCfg cfg_;
  uint32_t units_;
  uint32_t req_cap_hb_;
  uint32_t req_cap_items_;
  uint32_t rsp_cap_hb_;
  uint32_t rsp_cap_items_;
  std::deque<SmEgressItem> q_[2];
  std::vector<SmEgressItem> staged_[2];
  uint32_t occ_hb_[2];
  uint32_t paid_[2];
  int turn_;
  uint32_t credits_;
  std::vector<uint32_t> ret_;
  uint32_t ret_pos_;
  int64_t tok_w_;
  int64_t tok_r_;
};

// The ingress gate. Sources: fills (L2 and TMA replies), SM-to-SM request
// arrivals and SM-to-SM reply arrivals; the last two share the ejection cap
// and the SM-to-SM weight. A fill head may be prepaid up to one cycle of
// sectors beyond its need, carried to the next head; an idle source keeps
// nothing.
class SmIngressRef {
public:
  enum { FILL = 0, DSM_REQ = 1, DSM_RSP = 2, NSRC = 3 };

  explicit SmIngressRef(const SmPortCfg& cfg)
    : cfg_(cfg)
    , units_(cfg.ingress_bytes / cfg.beat_bytes)
    , eject_(cfg.eject_bytes / cfg.beat_bytes)
    , wheel_(cfg.w_fill + cfg.w_dsmem) {
    this->reset();
  }

  void reset() {
    for (int s = 0; s < NSRC; ++s)
      bank_[s] = 0;
    wpos_ = 0;
    dturn_ = DSM_REQ;
  }

  uint32_t units() const { return units_; }
  uint32_t eject() const { return eject_; }

  // valid/sectors: each source's head at the start of the cycle; ready: its sink.
  // pass[s]: the head of source s leaves this cycle.
  void tick(const bool valid[NSRC], const uint32_t sectors[NSRC], const bool ready[NSRC], bool pass[NSRC]) {
    for (int s = 0; s < NSRC; ++s) {
      if (!valid[s])
        bank_[s] = 0;
    }
    uint32_t dsm_used = 0;
    for (uint32_t u = 0; u < units_; ++u) {
      bool fc = valid[FILL] && sectors[FILL] != 0 && bank_[FILL] < sectors[FILL] + units_ - 1;
      bool rc = valid[DSM_REQ] && bank_[DSM_REQ] < sectors[DSM_REQ] && dsm_used < eject_;
      bool pc = valid[DSM_RSP] && bank_[DSM_RSP] < sectors[DSM_RSP] && dsm_used < eject_;
      bool dc = rc || pc;
      if (!fc && !dc)
        break;
      bool pref_fill = (wpos_ < cfg_.w_fill);
      bool to_fill = fc && (pref_fill || !dc);
      if (to_fill) {
        ++bank_[FILL];
      } else {
        int d = (rc && pc) ? dturn_ : (rc ? DSM_REQ : DSM_RSP);
        ++bank_[d];
        ++dsm_used;
        dturn_ = (d == DSM_REQ) ? DSM_RSP : DSM_REQ;
      }
      wpos_ = (wpos_ + 1) % wheel_;
    }
    for (int s = 0; s < NSRC; ++s) {
      pass[s] = valid[s] && bank_[s] >= sectors[s] && ready[s];
      if (pass[s])
        bank_[s] -= sectors[s];
    }
  }

private:
  SmPortCfg cfg_;
  uint32_t units_;
  uint32_t eject_;
  uint32_t wheel_;
  uint32_t bank_[NSRC];
  uint32_t wpos_;
  int dturn_;
};

} // namespace sm_port_ref
} // namespace vortex
