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

// SimX twin of hw/unittest/sm_port: SmPort through the same scenarios and
// golden expectations (sm_port_cases.h). A source sends a line request as its
// touched memory blocks, every one but the last marked by sm_port_more(); the
// memory answers all blocks of a request in one cycle. SmPort also splits
// each item into memory blocks on its memory side (the RTL's adapter), one
// block per cycle, so the twin measures payloads in bytes and lines and its
// golden carries that adapter.

#include "mem/sm_port.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>

#include "sm_port_cases.h"

using namespace vortex;
using namespace sm_port_cases;

#ifndef VX_CFG_H100_SM_ENABLE
#error "the sm_port twin tests the H100 SM path: build with -DVX_CFG_H100_SM_ENABLE"
#endif

constexpr uint32_t BLOCK = VX_CFG_MEM_BLOCK_SIZE;
constexpr uint32_t NSRC  = 2;
#ifdef VX_CFG_EXT_DLMEM_ENABLE
constexpr bool DLMEM = true;
#else
constexpr bool DLMEM = false;
#endif

static_assert(LINE % BLOCK == 0, "an SM port line is whole memory blocks");

class Twin {
public:
  Twin() : cfg_(knob_cfg()) {}

  int run() {
    std::printf("sm_port (SimX): egress %u B/cycle, beat %u B, line %u B%s\n", cfg_.egress_bytes, cfg_.beat_bytes, LINE,
                DLMEM ? ", DLMEM" : "");
    for (const auto& s : scenarios(cfg_, DLMEM)) {
      auto r = this->scenario(s);
      for (const auto& e : expectations(s, r, cfg_, BLOCK))
        this->expect(e.what, e.got, e.exp, e.tol_rel, e.tol_abs);
    }
    std::printf("sm_port: %u checks, %u errors\n", checks_, errors_);
    return errors_ ? 1 : 0;
  }

private:
  void expect(const std::string& what, double got, double exp, double tol_rel, double tol_abs) {
    double tol = std::max(std::fabs(exp) * tol_rel, tol_abs);
    bool ok = std::fabs(got - exp) <= tol + 1e-9;
    ++checks_;
    if (!ok) ++errors_;
    std::printf("  %-52s measured %9.4f expected %9.4f (+-%.4f) %s\n", what.c_str(), got, exp, tol, ok ? "ok" : "MISMATCH");
  }

  // A source request's blocks: those its byte enables touch (block 0 for a
  // read of none), lowest first, all but the last marked.
  static std::vector<MemReq> blocks_of(const LineReq& r, uint32_t tag) {
    std::vector<MemReq> v;
    for (uint32_t b = 0; b < LINE / BLOCK; ++b) {
      uint64_t be = 0;
      for (uint32_t i = 0; i < BLOCK; ++i)
        be |= uint64_t(r.be[b * BLOCK + i]) << i;
      if (be == 0)
        continue;
      MemReq q(r.rw ? MemOp::ST : (r.amo ? MemOp::AMO_ADD : MemOp::LD), uint64_t(b) * BLOCK, nullptr, be, tag);
      v.push_back(q);
    }
    if (v.empty())
      v.push_back(MemReq(r.rw ? MemOp::ST : MemOp::LD, 0, nullptr, 0, tag));
    for (size_t i = 0; i + 1 < v.size(); ++i)
      sm_port_more(v[i]);
    return v;
  }

  Rates scenario(const Scenario& s) {
    auto& platform = SimPlatform::instance();
    std::printf("%s\n", s.name.c_str());
    auto port = SmPort::Create(("sm_port_" + std::to_string(n_++)).c_str(), SmPort::Config{NSRC, LINE, 0, 0});
    platform.reset();
    Rates r;
    std::deque<std::pair<uint64_t, MemRsp>> mem_rsp;
    std::deque<uint32_t> served;
    #ifdef VX_CFG_EXT_DLMEM_ENABLE
    uint32_t rsp_tag = 0;
    #endif
    uint32_t tag = 0;
    for (uint64_t c = 0; c < WARM + WINDOW; ++c) {
      const bool win = (c >= WARM);
      for (uint32_t i = 0; i < NSRC; ++i) {
        const auto& src = i ? s.tma : s.l1;
        if (src.empty() || !port->SrcReqIn.at(i).empty())
          continue;
        for (auto& b : blocks_of(src[0], tag++))
          port->SrcReqIn.at(i).send(b, 0);
      }
      while (!mem_rsp.empty() && mem_rsp.front().first <= c && !port->MemRspIn.full()) {
        port->MemRspIn.send(mem_rsp.front().second, 0);
        mem_rsp.pop_front();
      }
    #ifdef VX_CFG_EXT_DLMEM_ENABLE
      if (s.dsm_req_hb && port->DsmReqIn.empty()) {
        DlmemReqPkt p;
        p.pkt_hb = s.dsm_req_hb;
        p.tag = tag & 0xff;
        port->DsmReqIn.send(p, 0);
      }
      // Each reply answers an arriving one-slot request this SM served.
      if (s.dsm_rsp_hb && port->ArrReqIn.empty()) {
        DlmemReqPkt p;
        p.pkt_hb = 1;
        p.rsp_hb = s.dsm_rsp_hb;
        p.src = 0;
        p.tag = ++rsp_tag;
        port->ArrReqIn.send(p, 0);
      }
      if (s.dsm_rsp_hb && port->DsmRspIn.empty() && !served.empty()) {
        DlmemRspPkt p;
        p.dst = 0;
        p.pkt_hb = s.dsm_rsp_hb;
        p.tag = served.front();
        port->DsmRspIn.send(p, 0);
        served.pop_front();
      }
      if (s.arr_req_hb && port->ArrReqIn.empty()) {
        DlmemReqPkt p;
        p.pkt_hb = s.arr_req_hb;
        port->ArrReqIn.send(p, 0);
      }
    #endif
      platform.tick();
      while (!port->MemReqOut.empty()) {
        MemReq q = port->MemReqOut.peek();
        port->MemReqOut.pop();
        uint32_t bytes = (uint32_t)__builtin_popcountll(q.byteen);
        if (win) {
          if (q.is_write())
            r.l2_bytes += bytes;
          else
            r.l2_lines += double(BLOCK) / LINE;
        }
        if (!q.is_write())
          mem_rsp.push_back({c + s.mem_delay, MemRsp(q.tag, q.hart_id, 0, std::make_shared<mem_block_t>())});
      }
      for (uint32_t i = 0; i < NSRC; ++i) {
        while (!port->SrcRspOut.at(i).empty()) {
          if (win)
            r.fill_sectors += double(BLOCK) / BEAT;
          port->SrcRspOut.at(i).pop();
        }
      }
    #ifdef VX_CFG_EXT_DLMEM_ENABLE
      while (!port->DsmReqOut.empty()) {
        if (win) r.dsm_req += 1;
        port->DsmReqOut.pop();
      }
      while (!port->DsmRspOut.empty()) {
        if (win) r.dsm_rsp += 1;
        port->DsmRspOut.pop();
      }
      while (!port->ArrReqOut.empty()) {
        if (s.dsm_rsp_hb)
          served.push_back(port->ArrReqOut.peek().tag);
        else if (win)
          r.arr_sectors += (s.arr_req_hb + 1) / 2;
        port->ArrReqOut.pop();
      }
    #endif
    }
    // Requests: the single-sector scenarios' are one sector (one block) each.
    r.l2_reqs = r.l2_lines * LINE / BLOCK + r.l2_bytes / BEAT;
    for (double* v : {&r.l2_bytes, &r.l2_lines, &r.l2_reqs, &r.dsm_req, &r.dsm_rsp, &r.fill_sectors, &r.arr_sectors})
      *v /= WINDOW;
    return r;
  }

  SmPortCfg cfg_;
  uint32_t n_ = 0;
  uint32_t checks_ = 0, errors_ = 0;
};

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--csv=", 6)) continue;
    std::printf("usage: %s [--csv=file]\n", argv[0]);
    return 2;
  }
  Twin t;
  int rc = t.run();
  std::printf(rc ? "FAILED!\n" : "PASSED!\n");
  return rc;
}
