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

// Scenarios of the SM port bench and their golden rates (sm_port_ref.h at the
// build's knobs), shared by the RTL testbench and its SimX twin.

#include <sm_port_ref.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <functional>
#include <string>
#include <vector>

namespace sm_port_cases {

using namespace vortex::sm_port_ref;

constexpr uint32_t LINE    = VX_CFG_DCACHE_LINE_SIZE;
constexpr uint32_t BEAT    = VX_CFG_SM_EGRESS_BEAT;
constexpr uint32_t SECTORS = LINE / BEAT;
constexpr uint64_t WARM    = 2000;
constexpr uint64_t WINDOW  = 12000;

inline SmPortCfg knob_cfg() {
  SmPortCfg c{};
  c.egress_bytes     = VX_CFG_SM_EGRESS_BYTES;
  c.beat_bytes       = VX_CFG_SM_EGRESS_BEAT;
  c.hb_wr_sector     = VX_CFG_SM_HB_WRITE_SECTOR;
  c.hb_wr_masked     = VX_CFG_SM_HB_WRITE_MASKED;
  c.hb_rd_line       = VX_CFG_SM_HB_READ_LINE;
  c.hb_pf_line       = VX_CFG_SM_HB_PREFETCH_LINE;
  c.hb_atom_lane     = VX_CFG_SM_HB_ATOM_LANE;
  c.hb_dsmem_slot    = VX_CFG_SM_HB_DSMEM_SLOT;
  c.hb_dsmem_ld_line = VX_CFG_SM_HB_DSMEM_LD_LINE;
  c.hb_tma_st_sector = VX_CFG_SM_HB_TMA_ST_SECTOR;
  c.hb_mbar          = VX_CFG_SM_HB_MBAR;
  c.min_slots        = VX_CFG_N5_MIN_SLOTS;
  c.atom_extra_slots = VX_CFG_DSMEM_ATOM_EXTRA_SLOTS;
  c.inj_credits      = VX_CFG_SM2SM_INJ_CREDITS;
  c.inj_return       = VX_CFG_SM2SM_INJ_RETURN;
  c.eject_bytes      = VX_CFG_SM2SM_EJECT_BYTES;
  c.ingress_bytes    = VX_CFG_SM_INGRESS_BYTES;
  c.w_fill           = VX_CFG_SM_INGRESS_FILL_WEIGHT;
  c.w_dsmem          = VX_CFG_SM_INGRESS_DSMEM_WEIGHT;
  c.issue_w_pm       = VX_CFG_SM_SECTOR_ISSUE_W_PERMILLE;
  c.issue_r_pm       = VX_CFG_SM_SECTOR_ISSUE_R_PERMILLE;
  c.req_fifo_bytes   = VX_CFG_SM_REQ_FIFO_BYTES;
  c.rsp_slots        = VX_CFG_SM_RSP_QUEUE_SLOTS;
  return c;
}

// An L1TEX source's line request.
struct LineReq {
  bool rw = false;
  bool amo = false;
  std::array<uint8_t, LINE> be{};
};

inline LineReq sectors_req(bool rw, uint32_t nsect, uint32_t bytes_per_sector) {
  LineReq r;
  r.rw = rw;
  for (uint32_t s = 0; s < nsect; ++s)
    for (uint32_t b = 0; b < bytes_per_sector; ++b)
      r.be[s * BEAT + b] = 1;
  return r;
}

inline uint32_t popcnt(const std::array<uint8_t, LINE>& be) {
  uint32_t n = 0;
  for (auto b : be) n += b;
  return n;
}

// The golden cost of a source request, as the port charges it.
inline SmEgressItem egress_item(uint32_t src, const LineReq& r, const SmPortCfg& cfg) {
  SmEgressItem it;
  uint32_t touched = touched_sectors(r.be.data(), LINE, BEAT);
  if (r.amo) {
    it.hb = cfg.hb_atom_lane;
  } else if (r.rw) {
    it.hb = sm_item_halfbeats(src == 0 ? SmItem::WriteLine : SmItem::TmaStoreLine, r.be.data(), LINE, 0, 0, cfg);
    it.single = (src == 0 && touched == 1) ? 1 : 0;
  } else {
    it.hb = cfg.hb_rd_line;
    it.single = (src == 0 && touched == 1) ? 2 : 0;
  }
  it.l2 = true;
  return it;
}

inline SmEgressItem dsm_item(uint32_t hb) {
  SmEgressItem it;
  it.hb = hb;
  it.slots = (hb + 1) / 2;
  it.l2 = false;
  return it;
}

struct Scenario {
  std::string name;
  bool needs_dlmem = false;
  std::vector<LineReq> l1;      // cycled; empty = idle
  std::vector<LineReq> tma;
  uint32_t dsm_req_hb = 0;      // 0 = idle
  uint32_t dsm_rsp_hb = 0;
  uint32_t arr_req_hb = 0;
  uint32_t mem_delay = 8;       // reads answered after this many cycles
  std::function<bool(uint64_t)> dsm_out_ready = [](uint64_t) { return true; };
};

// Class payloads per cycle in the measured window.
struct Rates {
  double l2_bytes = 0, l2_lines = 0, l2_reqs = 0, dsm_req = 0, dsm_rsp = 0;
  double fill_sectors = 0, arr_sectors = 0;
};

// Memory blocks of mem_blk bytes a line request touches.
inline uint32_t blocks_touched(const LineReq& lr, uint32_t mem_blk) {
  uint32_t n = 0;
  for (uint32_t b = 0; b < LINE; b += mem_blk) {
    bool any = false;
    for (uint32_t i = b; i < b + mem_blk; ++i)
      any |= (lr.be[i] != 0);
    n += any ? 1 : 0;
  }
  return n;
}

// The golden's steady state on the scenario's offered load (sinks ready).
// mem_blk: bytes per cycle the memory side moves (a line-to-block adapter
// behind the port), 0 when the bench drives the port's line interface.
inline Rates golden(const Scenario& s, const SmPortCfg& cfg_, uint32_t mem_blk) {
  SmEgressRef eg(cfg_);
  SmIngressRef ig(cfg_);
  Rates r;
  size_t i1 = 0, i2 = 0;
  uint32_t turn = 0;
  uint32_t backlog = 0;
  std::deque<uint64_t> fills;
  const uint32_t arr_sectors = (s.arr_req_hb + 1) / 2;
  for (uint64_t c = 0; c < WARM + WINDOW; ++c) {
    if (backlog)
      --backlog;
    // One push a cycle over the request-class sources, round robin as the port admits.
    std::vector<std::pair<int, SmEgressItem>> cand;
    if (!s.l1.empty()) cand.push_back({0, egress_item(0, s.l1[i1 % s.l1.size()], cfg_)});
    if (!s.tma.empty()) cand.push_back({1, egress_item(1, s.tma[i2 % s.tma.size()], cfg_)});
    if (s.dsm_req_hb) cand.push_back({2, dsm_item(s.dsm_req_hb)});
    for (size_t k = 0; k < cand.size(); ++k) {
      auto& x = cand[(turn + k) % cand.size()];
      if (eg.req_can_push(x.second)) {
        SmEgressItem it = x.second;
        it.id = (uint64_t)x.first;
        eg.req_push(it);
        if (x.first == 0) ++i1;
        if (x.first == 1) ++i2;
        turn = (uint32_t)((turn + k + 1) % cand.size());
        break;
      }
    }
    if (s.dsm_rsp_hb && eg.rsp_can_push(dsm_item(s.dsm_rsp_hb)))
      eg.rsp_push(dsm_item(s.dsm_rsp_hb));
    auto p = eg.tick(backlog == 0, true, true);
    const bool win = (c >= WARM);
    if (p.req && p.req_item.l2) {
      const LineReq& lr = (p.req_item.id == 0) ? s.l1[(i1 + s.l1.size() - 1) % s.l1.size()] : s.tma[0];
      uint32_t nblk = mem_blk ? blocks_touched(lr, mem_blk) : 1;
      if (mem_blk)
        backlog += nblk;
      if (!lr.rw)
        fills.push_back(c + nblk + s.mem_delay);
    }
    {
      bool valid[SmIngressRef::NSRC] = {!fills.empty() && fills.front() <= c, s.arr_req_hb != 0, false};
      uint32_t sectors[SmIngressRef::NSRC] = {SECTORS, arr_sectors, 0};
      bool ready[SmIngressRef::NSRC] = {true, true, true};
      bool pass[SmIngressRef::NSRC];
      ig.tick(valid, sectors, ready, pass);
      if (pass[SmIngressRef::FILL]) {
        fills.pop_front();
        if (win) r.fill_sectors += SECTORS;
      }
      if (pass[SmIngressRef::DSM_REQ] && win)
        r.arr_sectors += arr_sectors;
    }
    if (!win)
      continue;
    if (p.req) {
      if (p.req_item.l2) {
        const LineReq& lr = (p.req_item.id == 0) ? s.l1[(i1 + s.l1.size() - 1) % s.l1.size()] : s.tma[0];
        r.l2_reqs += 1;
        r.l2_lines += lr.rw ? 0 : 1;
        r.l2_bytes += lr.rw ? popcnt(lr.be) : 0;
      } else {
        r.dsm_req += 1;
      }
    }
    if (p.rsp)
      r.dsm_rsp += 1;
  }
  for (double* v : {&r.l2_bytes, &r.l2_lines, &r.l2_reqs, &r.dsm_req, &r.dsm_rsp, &r.fill_sectors, &r.arr_sectors})
    *v /= WINDOW;
  return r;
}


// The scenarios: the L1TEX sources alone, both sources, the ingress, and in
// DLMEM builds the SM-to-SM classes.
inline std::vector<Scenario> scenarios(const SmPortCfg& cfg, bool dlmem) {
  const uint32_t S = SECTORS;
  std::vector<Scenario> v;
  Scenario s;
  s = Scenario{}; s.name = "full-sector writes"; s.l1 = {sectors_req(true, S, BEAT)}; v.push_back(s);
  s = Scenario{}; s.name = "byte-masked sectors"; s.l1 = {sectors_req(true, S, 4)}; v.push_back(s);
  s = Scenario{}; s.name = "read lines"; s.l1 = {sectors_req(false, S, BEAT)}; v.push_back(s);
  s = Scenario{}; s.name = "single-sector writes"; s.l1 = {sectors_req(true, 1, BEAT)}; v.push_back(s);
  s = Scenario{}; s.name = "single-sector reads"; s.l1 = {sectors_req(false, 1, BEAT)}; v.push_back(s);
  s = Scenario{}; s.name = "L1 and TMA writes"; s.l1 = {sectors_req(true, S, BEAT)}; s.tma = {sectors_req(true, S, BEAT)}; v.push_back(s);
  s = Scenario{}; s.name = "fills alone"; s.l1 = {sectors_req(false, S, BEAT)}; s.mem_delay = 4; v.push_back(s);
  if (dlmem) {
    const uint32_t full_hb = S * cfg.hb_dsmem_slot;
    s = Scenario{}; s.name = "DSMEM stores alone"; s.needs_dlmem = true; s.dsm_req_hb = full_hb; v.push_back(s);
    s = Scenario{}; s.name = "both classes"; s.needs_dlmem = true; s.l1 = {sectors_req(true, S, BEAT)}; s.dsm_rsp_hb = full_hb; v.push_back(s);
    s = Scenario{}; s.name = "DSMEM arrivals alone"; s.needs_dlmem = true; s.arr_req_hb = full_hb; v.push_back(s);
    s = Scenario{}; s.name = "fills and arrivals"; s.needs_dlmem = true; s.l1 = {sectors_req(false, S, BEAT)};
    s.arr_req_hb = full_hb; s.mem_delay = 4; v.push_back(s);
  }
  return v;
}

struct Expect {
  std::string what;
  double got, exp, tol_rel, tol_abs;
};

// What each scenario must measure, from the golden on the same stream.
inline std::vector<Expect> expectations(const Scenario& s, const Rates& r, const SmPortCfg& cfg, uint32_t mem_blk) {
  Rates g = golden(s, cfg, mem_blk);
  const std::string& n = s.name;
  if (n == "full-sector writes" || n == "L1 and TMA writes")
    return {{n + ": B/cycle", r.l2_bytes, g.l2_bytes, 0.01, 0}};
  if (n == "byte-masked sectors")
    return {{n + ": sectors/cycle", r.l2_bytes / 4, g.l2_bytes / 4, 0.01, 0}};
  if (n == "read lines")
    return {{n + ": lines/cycle", r.l2_lines, g.l2_lines, 0.01, 0}};
  if (n == "single-sector writes")
    return {{n + ": requests/cycle", r.l2_reqs, cfg.issue_w_pm / 1000.0, 0, 0.01}};
  if (n == "single-sector reads")
    return {{n + ": requests/cycle", r.l2_reqs, cfg.issue_r_pm / 1000.0, 0, 0.01}};
  if (n == "fills alone")
    return {{n + ": sectors/cycle", r.fill_sectors, std::min<double>(cfg.ingress_bytes / BEAT, g.l2_lines * SECTORS), 0.02, 0}};
  if (n == "DSMEM stores alone")
    return {{n + ": B/cycle (credit loop)", r.dsm_req * LINE, g.dsm_req * LINE, 0.01, 0}};
  if (n == "both classes")
    return {{n + ": requests B/cycle", r.l2_bytes, g.l2_bytes, 0.01, 0},
            {n + ": replies B/cycle", r.dsm_rsp * LINE, g.dsm_rsp * LINE, 0.01, 0}};
  if (n == "DSMEM arrivals alone")
    return {{n + ": B/cycle", r.arr_sectors * BEAT, (double)cfg.eject_bytes, 0.02, 0}};
  if (n == "fills and arrivals") {
    double ratio = (r.arr_sectors > 0) ? r.fill_sectors / r.arr_sectors : 0;
    double exp = mem_blk ? g.fill_sectors / g.arr_sectors : (double)cfg.w_fill / cfg.w_dsmem;
    return {{n + ": fill : DSMEM sectors", ratio, exp, 0.02, 0}};
  }
  return {};
}

} // namespace sm_port_cases
