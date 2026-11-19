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

// Scenarios of the SM port bench, shared by the RTL testbench and its SimX
// twin. Each scenario's expectation is a structural formula (flits per packet
// by the port's format, one link of EGRESS_BYTES / FLIT flits per cycle, the
// memory side's blocks per cycle, the ejection link's flits per cycle), checked
// to equal sm_port_ref.h's steady state on the same stream; H100 rates of the
// same streams are printed as H100REF lines (SMPORT_flit_h100.md), never judged.

#include <sm_port_ref.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <functional>
#include <string>
#include <vector>

namespace sm_port_cases {

using namespace vortex::sm_port_ref;

constexpr uint32_t LINE    = VX_CFG_DCACHE_LINE_SIZE;
constexpr uint32_t FLIT    = VX_CFG_SM_EGRESS_BEAT;
constexpr uint32_t SECTORS = LINE / FLIT;
constexpr uint64_t WARM    = 2000;
constexpr uint64_t WINDOW  = 12000;

inline SmPortCfg knob_cfg() {
  SmPortCfg c{};
  c.egress_bytes  = VX_CFG_SM_EGRESS_BYTES;
  c.flit_bytes    = VX_CFG_SM_EGRESS_BEAT;
  c.eject_bytes   = VX_CFG_SM2SM_EJECT_BYTES;
  c.ingress_bytes = VX_CFG_SM_INGRESS_BYTES;
  c.w_fill        = VX_CFG_SM_INGRESS_FILL_WEIGHT;
  c.w_dsmem       = VX_CFG_SM_INGRESS_DSMEM_WEIGHT;
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  c.rsp_slots     = VX_CFG_DLMEM_INGRESS_SLOTS;
#else
  c.rsp_slots     = 1;
#endif
  return c;
}

// An L1TEX source's line request.
struct LineReq {
  bool rw = false;
  bool amo = false;
  std::array<uint8_t, LINE> be{};
};

// nsect sectors from sector 0, the first `bytes` bytes of each enabled.
inline LineReq sectors_req(bool rw, uint32_t nsect, uint32_t bytes) {
  LineReq r;
  r.rw = rw;
  for (uint32_t s = 0; s < nsect; ++s)
    for (uint32_t b = 0; b < bytes; ++b)
      r.be[s * FLIT + b] = 1;
  return r;
}

inline LineReq atomic_req() {
  LineReq r = sectors_req(false, 1, 4);
  r.amo = true;
  return r;
}

// `bytes` bytes at byte `off` of each sector in `sects`.
inline LineReq scattered_req(bool rw, bool amo, std::initializer_list<uint32_t> sects, uint32_t off, uint32_t bytes) {
  LineReq r;
  r.rw = rw;
  r.amo = amo;
  for (uint32_t s : sects)
    for (uint32_t b = 0; b < bytes; ++b)
      r.be[s * FLIT + off + b] = 1;
  return r;
}

inline uint32_t popcnt(const std::array<uint8_t, LINE>& be) {
  uint32_t n = 0;
  for (auto b : be) n += b;
  return n;
}

inline SmItem kind_of(const LineReq& r) {
  return r.amo ? SmItem::Atomic : (r.rw ? SmItem::WriteLine : SmItem::ReadLine);
}

inline uint32_t flits_of(const LineReq& r, const SmPortCfg& cfg) {
  return sm_packet_flits(kind_of(r), r.be.data(), LINE, cfg);
}

inline SmEgressItem egress_item(const LineReq& r, const SmPortCfg& cfg) {
  SmEgressItem it;
  it.flits = flits_of(r, cfg);
  it.l2 = true;
  return it;
}

inline SmEgressItem dsm_item() {
  SmEgressItem it;
  it.flits = 1;
  it.l2 = false;
  return it;
}

// Memory blocks of mem_blk bytes a line request touches (at least one).
inline uint32_t blocks_touched(const LineReq& lr, uint32_t mem_blk) {
  uint32_t n = 0;
  for (uint32_t b = 0; b < LINE; b += mem_blk) {
    bool any = false;
    for (uint32_t i = b; i < b + mem_blk; ++i)
      any |= (lr.be[i] != 0);
    n += any ? 1 : 0;
  }
  return std::max<uint32_t>(n, 1);
}

struct Scenario {
  std::string name;
  std::vector<LineReq> l1;      // cycled; empty = idle
  std::vector<LineReq> tma;
  bool dsm_req = false;         // this SM's SM-to-SM messages, one per cycle offered
  bool dsm_rsp = false;         // replies this SM serves
  bool arr_req = false;         // SM-to-SM requests arriving for this SM
  uint32_t mem_delay = 8;       // reads answered after this many cycles
  std::function<bool(uint64_t)> dsm_out_ready = [](uint64_t) { return true; };
};

// Class payloads per cycle in the measured window.
struct Rates {
  double l2_bytes = 0, l2_lines = 0, l2_reqs = 0, dsm_req = 0, dsm_rsp = 0;
  double fill_sectors = 0, arr_msgs = 0;
};

// The golden's steady state on the scenario's offered load (sinks ready).
// mem_blk: memory blocks behind the port (a line-to-block adapter sending one
// block per cycle), 0 when the bench drives the port's line interface.
inline Rates golden(const Scenario& s, const SmPortCfg& cfg_, uint32_t mem_blk) {
  SmEgressRef eg(cfg_);
  SmIngressRef ig(cfg_);
  Rates r;
  size_t i1 = 0, i2 = 0;
  uint32_t turn = 0;
  uint32_t backlog = 0;
  std::deque<uint64_t> fills;
  for (uint64_t c = 0; c < WARM + WINDOW; ++c) {
    if (backlog)
      --backlog;
    std::vector<std::pair<int, SmEgressItem>> cand;
    if (!s.l1.empty()) cand.push_back({0, egress_item(s.l1[i1 % s.l1.size()], cfg_)});
    if (!s.tma.empty()) cand.push_back({1, egress_item(s.tma[i2 % s.tma.size()], cfg_)});
    if (s.dsm_req) cand.push_back({2, dsm_item()});
    for (size_t k = 0; k < cand.size(); ++k) {
      auto& x = cand[(turn + k) % cand.size()];
      if (eg.req_can_push()) {
        SmEgressItem it = x.second;
        // The source in the low bits, its stream position above: a popped
        // packet may be up to the FIFO depth behind the latest push.
        it.id = (uint64_t)x.first | ((uint64_t)(x.first == 1 ? i2 : i1) << 2);
        eg.req_push(it);
        if (x.first == 0) ++i1;
        if (x.first == 1) ++i2;
        turn = (uint32_t)((turn + k + 1) % cand.size());
        break;
      }
    }
    if (s.dsm_rsp && eg.rsp_can_push())
      eg.rsp_push(dsm_item());
    auto p = eg.tick(backlog == 0, true, true);
    const bool win = (c >= WARM);
    const LineReq* lr = nullptr;
    if (p.req && p.req_item.l2) {
      size_t k = (size_t)(p.req_item.id >> 2);
      lr = ((p.req_item.id & 3) == 0) ? &s.l1[k % s.l1.size()] : &s.tma[k % s.tma.size()];
    }
    if (lr) {
      uint32_t nblk = mem_blk ? blocks_touched(*lr, mem_blk) : 1;
      if (mem_blk)
        backlog += nblk;
      if (!lr->rw)
        fills.push_back(c + nblk + s.mem_delay);
    }
    {
      bool valid[SmIngressRef::NSRC] = {!fills.empty() && fills.front() <= c, s.arr_req, false};
      uint32_t sectors[SmIngressRef::NSRC] = {SECTORS, 1, 0};
      bool ready[SmIngressRef::NSRC] = {true, true, true};
      bool pass[SmIngressRef::NSRC];
      ig.tick(valid, sectors, ready, pass);
      if (pass[SmIngressRef::FILL]) {
        fills.pop_front();
        if (win) r.fill_sectors += SECTORS;
      }
      if (pass[SmIngressRef::DSM_REQ] && win)
        r.arr_msgs += 1;
    }
    if (!win)
      continue;
    if (p.req) {
      if (lr) {
        r.l2_reqs += 1;
        r.l2_lines += lr->rw ? 0 : 1;
        r.l2_bytes += lr->rw ? popcnt(lr->be) : 0;
      } else {
        r.dsm_req += 1;
      }
    }
    if (p.rsp)
      r.dsm_rsp += 1;
  }
  for (double* v : {&r.l2_bytes, &r.l2_lines, &r.l2_reqs, &r.dsm_req, &r.dsm_rsp, &r.fill_sectors, &r.arr_msgs})
    *v /= WINDOW;
  return r;
}

// The scenarios: the L1TEX sources alone, both sources, the ingress, and in
// DLMEM builds the SM-to-SM classes.
inline std::vector<Scenario> scenarios(bool dlmem) {
  const uint32_t S = SECTORS;
  std::vector<Scenario> v;
  Scenario s;
  s = Scenario{}; s.name = "full-sector writes"; s.l1 = {sectors_req(true, S, FLIT)}; v.push_back(s);
  s = Scenario{}; s.name = "byte-masked sectors"; s.l1 = {sectors_req(true, S, 4)}; v.push_back(s);
  s = Scenario{}; s.name = "read lines"; s.l1 = {sectors_req(false, S, FLIT)}; v.push_back(s);
  s = Scenario{}; s.name = "single-sector writes"; s.l1 = {sectors_req(true, 1, FLIT)}; v.push_back(s);
  s = Scenario{}; s.name = "single masked-sector writes"; s.l1 = {sectors_req(true, 1, 4)}; v.push_back(s);
  s = Scenario{}; s.name = "single-sector reads"; s.l1 = {sectors_req(false, 1, FLIT)}; v.push_back(s);
  s = Scenario{}; s.name = "single-lane atomics"; s.l1 = {atomic_req()}; v.push_back(s);
  s = Scenario{}; s.name = "L1 and TMA writes"; s.l1 = {sectors_req(true, S, FLIT)}; s.tma = {sectors_req(true, S, FLIT)}; v.push_back(s);
  if (S >= 4) {
    // Gaps between the touched sectors and operands away from byte 0.
    s = Scenario{}; s.name = "scattered sectors";
    s.l1 = {scattered_req(true, false, {1, 3}, 8, 4), scattered_req(true, false, {0, 2}, 0, FLIT),
            scattered_req(false, true, {3}, 20, 4), scattered_req(false, false, {2}, 0, FLIT)};
    v.push_back(s);
  }
  s = Scenario{}; s.name = "fills alone"; s.l1 = {sectors_req(false, S, FLIT)}; s.mem_delay = 4; v.push_back(s);
  if (dlmem) {
    s = Scenario{}; s.name = "SM-to-SM messages alone"; s.dsm_req = true; v.push_back(s);
    s = Scenario{}; s.name = "both classes"; s.l1 = {sectors_req(true, S, FLIT)}; s.dsm_rsp = true; v.push_back(s);
    s = Scenario{}; s.name = "SM-to-SM arrivals alone"; s.arr_req = true; v.push_back(s);
    s = Scenario{}; s.name = "fills and arrivals"; s.l1 = {sectors_req(false, S, FLIT)}; s.arr_req = true;
    s.mem_delay = 4; v.push_back(s);
  }
  return v;
}

struct Expect {
  std::string what;
  double got, exp, tol_rel, tol_abs;
};

// Packets per cycle when the link binds: its flits per cycle over the packet's
// flits, capped by one admission per cycle and, with an adapter behind the
// port, one memory block per cycle.
inline double packet_rate(const LineReq& r, const SmPortCfg& cfg, uint32_t mem_blk) {
  double units = cfg.egress_bytes / cfg.flit_bytes;
  // The serialiser sends one packet at a time: a packet takes whole link cycles.
  double rate = 1.0 / std::ceil(flits_of(r, cfg) / units);
  if (mem_blk)
    rate = std::min(rate, 1.0 / blocks_touched(r, mem_blk));
  return rate;
}

inline void h100ref(const char* metric, double h100, double vortex) {
  std::printf("H100REF metric=%s h100=%.3f vortex=%.3f\n", metric, h100, vortex);
}

// What each scenario must measure: the structural formula, which the golden
// reproduces on the same stream (a mismatch of the two is a golden error).
inline std::vector<Expect> expectations(const Scenario& s, const Rates& r, const SmPortCfg& cfg, uint32_t mem_blk) {
  Rates g = golden(s, cfg, mem_blk);
  const std::string& n = s.name;
  const double units = cfg.egress_bytes / cfg.flit_bytes;
  const double eject = cfg.eject_bytes / cfg.flit_bytes;
  std::vector<Expect> v;
  auto add = [&](const std::string& what, double got, double formula, double gold, double tol) {
    v.push_back({n + ": " + what, got, formula, tol, 0});
    v.push_back({n + ": golden " + what, gold, formula, tol, 0});
  };
  if (n == "scattered sectors") {
    // One packet at a time on the link: the stream's packets over their link
    // cycles. Behind an adapter a packet leaves the link once the blocks of the
    // one before it have gone, one per cycle, while its own flits crossed.
    double cyc = 0;
    for (size_t k = 0; k < s.l1.size(); ++k) {
      double link = std::ceil(flits_of(s.l1[(k + 1) % s.l1.size()], cfg) / units);
      cyc += mem_blk ? std::max<double>(link, blocks_touched(s.l1[k], mem_blk)) : link;
    }
    add("requests/cycle", r.l2_reqs, s.l1.size() / cyc, g.l2_reqs, 0.01);
    return v;
  }
  if (!s.l1.empty() && s.tma.empty() && !s.dsm_rsp && !s.arr_req) {
    const LineReq& q = s.l1[0];
    double pr = packet_rate(q, cfg, mem_blk);
    std::printf("  %s: %u flit(s) per packet\n", n.c_str(), flits_of(q, cfg));
    if (n == "fills alone") {
      double fill = std::min<double>(cfg.ingress_bytes / cfg.flit_bytes, pr * SECTORS);
      add("sectors/cycle", r.fill_sectors, fill, g.fill_sectors, 0.02);
    } else if (q.rw) {
      add("B/cycle", r.l2_bytes, pr * popcnt(q.be), g.l2_bytes, 0.01);
    } else {
      add("requests/cycle", r.l2_reqs, pr, g.l2_reqs, 0.01);
    }
    if (n == "full-sector writes")
      h100ref("sm_port_full_line_store_Bpc", 31.92, r.l2_bytes);
    if (n == "byte-masked sectors")
      h100ref("sm_port_masked_line_flits", 5, flits_of(q, cfg));
    if (n == "single-sector writes") {
      h100ref("sm_port_1sector_store_alloc_rpc", 0.617, r.l2_reqs);
      h100ref("sm_port_1sector_store_noalloc_rpc", 0.997, r.l2_reqs);
    }
    if (n == "single masked-sector writes")
      h100ref("sm_port_1sector_masked_flits", 2, flits_of(q, cfg));
    if (n == "single-sector reads") {
      h100ref("sm_port_1sector_read_rpc", 0.996, r.l2_reqs);
      h100ref("sm_port_read_flits_per_request", 0.5, flits_of(q, cfg));
    }
    if (n == "single-lane atomics")
      h100ref("sm_port_1lane_red_rpc", 1.000, r.l2_reqs);
    return v;
  }
  if (n == "L1 and TMA writes") {
    add("B/cycle", r.l2_bytes, units * cfg.flit_bytes, g.l2_bytes, 0.01);
    return v;
  }
  if (n == "SM-to-SM messages alone") {
    add("messages/cycle", r.dsm_req, std::min(1.0, units), g.dsm_req, 0.01);
    h100ref("sm_port_dsmem_store_Bpc_sd_lanes", 21.3, r.dsm_req * 8);
    return v;
  }
  if (n == "both classes") {
    // Round robin per flit: each class takes every other flit.
    add("request flits/cycle", r.l2_bytes / cfg.flit_bytes, units / 2, g.l2_bytes / cfg.flit_bytes, 0.01);
    add("replies/cycle", r.dsm_rsp, units / 2, g.dsm_rsp, 0.01);
    return v;
  }
  if (n == "SM-to-SM arrivals alone") {
    add("messages/cycle", r.arr_msgs, eject, g.arr_msgs, 0.02);
    return v;
  }
  if (n == "fills and arrivals") {
    double ratio = (r.arr_msgs > 0) ? r.fill_sectors / r.arr_msgs : 0;
    v.push_back({n + ": fill sectors : arrivals", ratio, g.fill_sectors / g.arr_msgs, 0.02, 0});
    v.push_back({n + ": arrivals/cycle within the ejection link", std::min(r.arr_msgs, eject), r.arr_msgs, 0, 1e-9});
    return v;
  }
  return v;
}

} // namespace sm_port_cases
