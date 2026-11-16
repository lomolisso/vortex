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

// VX_sm_port testbench: synthetic sources, ideal sinks. Every scenario is
// checked against sw/common/sm_port_ref.h at the build's knobs:
// - the egress: each cycle the port's admitted items are fed to SmEgressRef,
//   and every item must leave the port a constant number of cycles after the
//   golden pops it (cycle for cycle); the steady-state payload per cycle must
//   equal the golden's on the same stream;
// - the ingress: the gate's sectors per cycle per class against SmIngressRef's
//   steady state on the same offered load;
// - a DSMEM head that cannot leave holds the L2 request behind it.
// --csv=<file> writes each item's leave cycle per scenario.

#include "vl_simulator.h"
#include "VVX_sm_port_top.h"
#include "VX_config.h"

#include "sm_port_cases.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

using namespace sm_port_cases;

#ifndef VX_CFG_H100_SM_ENABLE
#error "sm_port tests the H100 SM path: build with -DVX_CFG_H100_SM_ENABLE"
#endif

constexpr uint32_t NSRC    = 2;
#ifdef VX_CFG_EXT_DLMEM_ENABLE
constexpr bool DLMEM = true;
#else
constexpr bool DLMEM = false;
#endif

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

class Bench {
public:
  Bench(FILE* csv) : cfg_(knob_cfg()), csv_(csv) {}

  int run() {
    std::printf("sm_port: egress %u B/cycle, beat %u B, line %u B, credits %u per %u, ingress %u, ejection %u, weights %u:%u%s\n",
                cfg_.egress_bytes, cfg_.beat_bytes, LINE, cfg_.inj_credits, cfg_.inj_return, cfg_.ingress_bytes,
                cfg_.eject_bytes, cfg_.w_fill, cfg_.w_dsmem, DLMEM ? ", DLMEM" : "");
    for (const auto& s : scenarios(cfg_, DLMEM)) {
      auto r = this->scenario(s);
      for (const auto& e : expectations(s, r, cfg_, 0))
        this->expect(e.what, e.got, e.exp, e.tol_rel, e.tol_abs);
    }
    if (DLMEM) {
      this->head_of_line();
    }
    std::printf("sm_port: %u checks, %u errors\n", checks_, errors_);
    return errors_ ? 1 : 0;
  }

private:
  void expect(const std::string& what, double got, double exp, double tol_rel, double tol_abs = 0) {
    double tol = std::max(std::fabs(exp) * tol_rel, tol_abs);
    bool ok = std::fabs(got - exp) <= tol + 1e-9;
    ++checks_;
    if (!ok) ++errors_;
    std::printf("  %-52s measured %9.4f expected %9.4f (+-%.4f) %s\n", what.c_str(), got, exp, tol, ok ? "ok" : "MISMATCH");
  }

  void check(bool ok, const std::string& what) {
    ++checks_;
    if (!ok) {
      ++errors_;
      std::printf("  MISMATCH %s\n", what.c_str());
    }
  }

  void reset(VVX_sm_port_top* t) {
    t->reset = 1;
    for (int i = 0; i < 8; ++i)
      tick_ = sim_.step(tick_, 2);
    t->reset = 0;
  }

  // Drives the scenario; returns its rates in the window. Every egress leave
  // is checked against the golden fed with the port's own admissions.
  Rates scenario(const Scenario& s) {
    auto* t = sim_.operator->();
    this->reset(t);
    std::printf("%s\n", s.name.c_str());
    SmEgressRef eg(cfg_);
    Rates r;
    size_t i1 = 0, i2 = 0;
    uint32_t tag = 0;
    std::deque<std::pair<uint64_t, uint64_t>> mem_rsp;  // (due, tag)
    std::deque<uint64_t> gold_req, gold_rsp;             // golden pop cycles
    std::vector<uint64_t> rtl_req, rtl_rsp;               // RTL leave cycles
    std::map<uint64_t, uint32_t> rtl_hb;
    for (uint64_t c = 0; c < WARM + WINDOW; ++c) {
      const LineReq l1 = s.l1.empty() ? LineReq{} : s.l1[i1 % s.l1.size()];
      const LineReq tm = s.tma.empty() ? LineReq{} : s.tma[i2 % s.tma.size()];
      set_bits(t->src_valid, 0, 1, !s.l1.empty());
      set_bits(t->src_valid, 1, 1, !s.tma.empty());
      set_bits(t->src_rw, 0, 1, l1.rw);
      set_bits(t->src_rw, 1, 1, tm.rw);
      set_bits(t->src_amo, 0, 1, l1.amo);
      set_bits(t->src_amo, 1, 1, tm.amo);
      for (uint32_t b = 0; b < LINE; ++b) {
        set_bits(t->src_byteen, b, 1, l1.be[b]);
        set_bits(t->src_byteen, LINE + b, 1, tm.be[b]);
      }
      set_bits(t->src_tag, 0, 8, tag);
      set_bits(t->src_tag, 8, 8, tag);
      set_bits(t->src_rsp_ready, 0, NSRC, 3);
      t->mem_req_ready = 1;
      const bool mv = !mem_rsp.empty() && mem_rsp.front().first <= c;
      t->mem_rsp_valid = mv;
      set_bits(t->mem_rsp_tag, 0, 64, mv ? mem_rsp.front().second : 0);
      t->dsm_req_in_valid = s.dsm_req_hb != 0;
      t->dsm_req_in_hb = s.dsm_req_hb;
      t->dsm_req_in_tag = tag & 0xff;
      t->dsm_req_out_ready = s.dsm_out_ready(c);
      t->dsm_rsp_in_valid = s.dsm_rsp_hb != 0;
      t->dsm_rsp_in_hb = s.dsm_rsp_hb;
      t->dsm_rsp_in_tag = tag & 0xff;
      t->dsm_rsp_out_ready = 1;
      t->arr_req_in_valid = s.arr_req_hb != 0;
      t->arr_req_in_hb = s.arr_req_hb;
      t->arr_req_out_ready = 1;
      t->arr_rsp_in_valid = 0;
      t->arr_rsp_in_hb = 0;
      t->arr_rsp_out_ready = 1;
      t->eval();

      // Admissions this cycle, fed to the golden.
      int admitted = -1;
      for (uint32_t i = 0; i < NSRC; ++i) {
        if (get_bits(t->src_valid, i, 1) && get_bits(t->src_ready, i, 1)) {
          this->check(admitted < 0, s.name + ": two admissions in one cycle");
          SmEgressItem it = egress_item(i, i == 0 ? l1 : tm, cfg_);
          this->check(eg.req_can_push(it), s.name + ": the port admitted an item the golden refuses at cycle " + std::to_string(c));
          eg.req_push(it);
          admitted = (int)i;
          if (i == 0) ++i1; else ++i2;
        }
      }
      if (t->dsm_req_in_valid && t->dsm_req_in_ready) {
        this->check(admitted < 0, s.name + ": two admissions in one cycle");
        SmEgressItem it = dsm_item(s.dsm_req_hb);
        this->check(eg.req_can_push(it), s.name + ": the port admitted a packet the golden refuses at cycle " + std::to_string(c));
        eg.req_push(it);
      }
      if (t->dsm_rsp_in_valid && t->dsm_rsp_in_ready)
        eg.rsp_push(dsm_item(s.dsm_rsp_hb));
      auto p = eg.tick(true, s.dsm_out_ready(c), true);
      if (p.req) gold_req.push_back(c);
      if (p.rsp) gold_rsp.push_back(c);

      // Leaves and sinks.
      const bool win = (c >= WARM);
      if (t->mem_req_valid) {
        rtl_req.push_back(c);
        uint32_t bytes = 0;
        for (uint32_t b = 0; b < LINE; ++b) bytes += (uint32_t)get_bits(t->mem_req_byteen, b, 1);
        if (win) {
          r.l2_reqs += 1;
          r.l2_lines += t->mem_req_rw ? 0 : 1;
          r.l2_bytes += t->mem_req_rw ? bytes : 0;
        }
        if (!t->mem_req_rw)
          mem_rsp.push_back({c + s.mem_delay, get_bits(t->mem_req_tag, 0, 64)});
      }
      if (mv && t->mem_rsp_ready)
        mem_rsp.pop_front();
      if (t->dsm_req_out_valid && t->dsm_req_out_ready) {
        rtl_req.push_back(c);
        if (win) r.dsm_req += 1;
      }
      if (t->dsm_rsp_out_valid) {
        rtl_rsp.push_back(c);
        if (win) r.dsm_rsp += 1;
      }
      for (uint32_t i = 0; i < NSRC; ++i)
        if (get_bits(t->src_rsp_valid, i, 1) && win) r.fill_sectors += SECTORS;
      if (t->arr_req_out_valid && win)
        r.arr_sectors += (s.arr_req_hb + 1) / 2;
      tick_ = sim_.step(tick_, 2);
      ++tag;
    }
    for (double* v : {&r.l2_bytes, &r.l2_lines, &r.l2_reqs, &r.dsm_req, &r.dsm_rsp, &r.fill_sectors, &r.arr_sectors})
      *v /= WINDOW;

    // Cycle for cycle: every leave a constant offset after its golden pop.
    for (int k = 0; k < 2; ++k) {
      const auto& g = k ? gold_rsp : gold_req;
      const auto& o = k ? rtl_rsp : rtl_req;
      size_t n = std::min(g.size(), o.size());
      if (n == 0)
        continue;
      this->check(std::max(g.size(), o.size()) - n <= 4, s.name + ": the port and the golden left different numbers of items");
      int64_t off = (int64_t)o[0] - (int64_t)g[0];
      uint32_t bad = 0;
      for (size_t i = 0; i < n; ++i)
        bad += ((int64_t)o[i] - (int64_t)g[i]) != off;
      this->check(bad == 0, s.name + (k ? ": replies" : ": requests") + " leave off the golden's cycles (" + std::to_string(bad) + " of " + std::to_string(n) + ")");
      if (csv_)
        for (size_t i = 0; i < n; ++i)
          std::fprintf(csv_, "%s,%d,%zu,%lld\n", s.name.c_str(), k, i, (long long)(g[i] - g[0]));
    }
    return r;
  }

  // A DSMEM head that cannot leave holds the L2 read queued behind it.
  void head_of_line() {
    auto* t = sim_.operator->();
    this->reset(t);
    std::printf("head of line\n");
    const uint64_t BLOCK = 200;
    // Enough packets ahead of the read that one stays at the FIFO head
    // after the port's two output buffers fill.
    const uint32_t NDSM = 4;
    uint64_t read_at = 0, dsm_at = 0;
    uint32_t dsm_sent = 0;
    bool rd_sent = false;
    for (uint64_t c = 0; c < BLOCK + 100; ++c) {
      t->src_valid = 0;
      t->dsm_req_in_valid = 0;
      if (dsm_sent < NDSM) {
        t->dsm_req_in_valid = 1;
        t->dsm_req_in_hb = 2 * cfg_.hb_dsmem_slot;
        t->dsm_req_in_tag = 1;
      } else if (!rd_sent) {
        set_bits(t->src_valid, 0, 1, 1);
        set_bits(t->src_rw, 0, 1, 0);
        set_bits(t->src_amo, 0, 1, 0);
        for (uint32_t b = 0; b < LINE; ++b) set_bits(t->src_byteen, b, 1, 1);
        set_bits(t->src_tag, 0, 8, 2);
      }
      t->mem_req_ready = 1;
      t->mem_rsp_valid = 0;
      t->dsm_req_out_ready = (c >= BLOCK);
      t->dsm_rsp_in_valid = 0;
      t->arr_req_in_valid = 0;
      t->arr_rsp_in_valid = 0;
      t->dsm_rsp_out_ready = 1;
      t->arr_req_out_ready = 1;
      t->arr_rsp_out_ready = 1;
      set_bits(t->src_rsp_ready, 0, NSRC, 3);
      t->eval();
      if (t->dsm_req_in_valid && t->dsm_req_in_ready) ++dsm_sent;
      if (get_bits(t->src_valid, 0, 1) && get_bits(t->src_ready, 0, 1)) rd_sent = true;
      if (t->mem_req_valid && !read_at) read_at = c;
      if (t->dsm_req_out_valid && t->dsm_req_out_ready && !dsm_at) dsm_at = c;
      tick_ = sim_.step(tick_, 2);
    }
    this->check(dsm_at >= BLOCK, "head of line: the DSMEM packet left while its sink was blocked");
    this->check(read_at != 0 && read_at >= dsm_at, "head of line: the L2 read left before the DSMEM head ahead of it (read "
                + std::to_string(read_at) + ", DSMEM " + std::to_string(dsm_at) + ")");
  }

  SmPortCfg cfg_;
  FILE* csv_;
  vl_simulator<VVX_sm_port_top> sim_;
  uint64_t tick_ = 0;
  uint32_t checks_ = 0, errors_ = 0;
};

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  const char* csv_path = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--csv=", 6)) csv_path = argv[i] + 6;
    else if (!strncmp(argv[i], "+", 1)) continue;
    else { std::printf("usage: %s [--csv=file]\n", argv[0]); return 2; }
  }
  Verilated::commandArgs(argc, argv);
  FILE* csv = csv_path ? std::fopen(csv_path, "w") : nullptr;
  Bench b(csv);
  int rc = b.run();
  if (csv)
    std::fclose(csv);
  std::printf(rc ? "FAILED!\n" : "PASSED!\n");
  return rc;
}
