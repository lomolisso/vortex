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

// Golden model of an H100-shaped SM's ports: the egress link sends one flit
// (one sector) per cycle per EGRESS_BYTES / FLIT, each request a packet of
// flits by the SM port's format; the request FIFO and the response queue share
// the link round robin per flit; the ingress gate takes fills and the SM-to-SM
// ejection link. SimX, the RTL testbench and the host test run this one
// model; shapes come from SmPortCfg, never VX_config.h.
//
// One cycle of either model: pushes test their admission against the state
// at the start of the cycle and become visible to the next cycle's tick; a
// tick sends the cycle's flits of the heads present at its start, then lets
// each head whose last flit is sent and whose sink is ready leave, at most one
// packet per queue per cycle.

#include <algorithm>
#include <cstdint>
#include <deque>
#include <vector>

namespace vortex {
namespace sm_port_ref {

// The port's output register toward the L2 and its input register from it
// (VX_sm_port out_buf / in_buf, two entries each): one stage each way on every
// L2 access.
constexpr uint32_t SM_PORT_OUT_STAGES = 1;
constexpr uint32_t SM_PORT_IN_STAGES  = 1;
constexpr uint32_t SM_PORT_REG_DEPTH  = 2;

// The request FIFO is the serialiser's skid: the packet on the link and the
// next one, so the link never idles on its own ready loop. A backlog waits
// where it is made (the L1's egress queue and MSHRs, the LSU queue).
constexpr uint32_t SM_REQ_FIFO_DEPTH = 2;

enum class SmItem {
  WriteLine,  // global write of one line (L1 write-through, TMA bulk store)
  ReadLine,   // global read or L2 prefetch of one line (any sectors)
  Atomic,     // global atomic, one lane with its operands
  Fence,      // fence toward the L2
  DsmemMsg    // SM-to-SM message: one request, reply or ack, header + one word
};

struct SmPortCfg {
  uint32_t egress_bytes;   // egress link bytes per cycle
  uint32_t flit_bytes;     // one flit = one sector
  uint32_t eject_bytes;    // SM-to-SM ejection link bytes per cycle
  uint32_t ingress_bytes;  // ingress port bytes per cycle
  uint32_t w_fill;         // ingress weight of fills (L2 and TMA replies)
  uint32_t w_dsmem;        // ingress weight of SM-to-SM arrivals
  uint32_t rsp_slots;      // response queue entries
};

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

// A packet's flits; the header rides in the link's side field. A write sends
// its touched sectors, then the line's byte mask as one more flit when a
// touched sector is partial; any other packet is one flit (a read's or a
// fence's header, an atomic's operand sector, an SM-to-SM message).
inline uint32_t sm_packet_flits(SmItem kind, const uint8_t* byteen, uint32_t line_bytes, const SmPortCfg& cfg) {
  if (kind != SmItem::WriteLine)
    return 1;
  uint32_t t = touched_sectors(byteen, line_bytes, cfg.flit_bytes);
  uint32_t f = full_sectors(byteen, line_bytes, cfg.flit_bytes);
  return std::max<uint32_t>(t + (t != f ? 1 : 0), 1);
}

struct SmEgressItem {
  uint32_t flits = 0;   // packet length on the link
  bool     l2 = false;  // leaves toward the L2 (else toward the SM-to-SM network)
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
    , units_(cfg.egress_bytes / cfg.flit_bytes)
    , cap_{SM_REQ_FIFO_DEPTH, cfg.rsp_slots} {
    this->reset();
  }

  void reset() {
    for (int k = 0; k < 2; ++k) {
      q_[k].clear();
      staged_[k].clear();
      sent_[k] = 0;
    }
    turn_ = REQ;
  }

  uint32_t units() const { return units_; }
  uint32_t req_cap() const { return cap_[REQ]; }
  uint32_t rsp_cap() const { return cap_[RSP]; }

  bool req_can_push() const { return this->can_push(REQ); }
  bool rsp_can_push() const { return this->can_push(RSP); }
  void req_push(const SmEgressItem& it) { staged_[REQ].push_back(it); }
  void rsp_push(const SmEgressItem& it) { staged_[RSP].push_back(it); }

  bool empty() const {
    return q_[REQ].empty() && q_[RSP].empty() && staged_[REQ].empty() && staged_[RSP].empty();
  }

  Pops tick(bool req_l2_ready, bool req_dsm_ready, bool rsp_ready) {
    for (uint32_t u = 0; u < units_; ++u) {
      bool cand[2];
      for (int k = 0; k < 2; ++k)
        cand[k] = !q_[k].empty() && sent_[k] < q_[k].front().flits;
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
      ++sent_[g];
      turn_ = 1 - g;
    }

    Pops pops;
    if (!q_[REQ].empty() && sent_[REQ] >= q_[REQ].front().flits) {
      const auto& h = q_[REQ].front();
      if (h.l2 ? req_l2_ready : req_dsm_ready) {
        pops.req = true;
        pops.req_item = h;
        sent_[REQ] = 0;
        q_[REQ].pop_front();
      }
    }
    if (!q_[RSP].empty() && sent_[RSP] >= q_[RSP].front().flits && rsp_ready) {
      pops.rsp = true;
      pops.rsp_item = q_[RSP].front();
      sent_[RSP] = 0;
      q_[RSP].pop_front();
    }

    for (int k = 0; k < 2; ++k) {
      for (auto& it : staged_[k])
        q_[k].push_back(it);
      staged_[k].clear();
    }
    return pops;
  }

private:
  bool can_push(int k) const {
    return staged_[k].empty() && q_[k].size() < cap_[k];
  }

  SmPortCfg cfg_;
  uint32_t units_;
  uint32_t cap_[2];
  std::deque<SmEgressItem> q_[2];
  std::vector<SmEgressItem> staged_[2];
  uint32_t sent_[2];
  int turn_;
};

// The ingress gate. Sources: fills (L2 and TMA replies), SM-to-SM request
// arrivals and SM-to-SM reply arrivals; the last two come over the one
// ejection link (EJECT_BYTES / FLIT flits per cycle, one flit per message) and
// share the SM-to-SM weight. A fill head may be prepaid up to one cycle of
// sectors beyond its need, carried to the next head; an idle source keeps
// nothing.
class SmIngressRef {
public:
  enum { FILL = 0, DSM_REQ = 1, DSM_RSP = 2, NSRC = 3 };

  explicit SmIngressRef(const SmPortCfg& cfg)
    : cfg_(cfg)
    , units_(cfg.ingress_bytes / cfg.flit_bytes)
    , eject_(cfg.eject_bytes / cfg.flit_bytes)
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
