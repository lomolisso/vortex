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

// st.async / red.async bench of the DLMEM endpoints: async lanes offered at
// each core's first LSU port, every one through the network, its own core's
// included, and checked where the destination's ingress performs them. Each
// request, like an instruction, is all stores or all adds. It has no
// simulator dependency, so the RTL testbench and the SimX test run the same
// schedule and write the same latency rows.
//
// Checks, per lane: the destination's local-memory port takes the access (a
// store, or an add for red.async) once, then, in a later cycle than the
// access's response, one complete_tx of the lane's bytes on its mbarrier word
// (hart 0, all byte lanes); its source then ejects one ack from the response
// network. No ack reaches the store tracker, every lane's credit returns, and
// every endpoint drains.
//
// CSV rows (src,dst,op,seq,issue_cycle,done_cycle), seq numbered per source
// in schedule order of its lanes, three rows per lane: ASYNC_ACC and ASYNC_CX
// finish when the access and the complete_tx fire on the destination's
// local-memory port; ASYNC_ACK, numbered by the source's acks in arrival
// order, finishes when the source ejects its n-th ack (dst = src, issue = the
// issue of the source's n-th lane). issue_cycle is the first cycle the lane's
// request is offered.

#pragma once

#include <algorithm>
#include <cinttypes>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <fstream>
#include <random>
#include <string>
#include <tuple>
#include <vector>

// The including file provides the VX_CFG_* configuration first.

namespace async {

static_assert(VX_CFG_NUM_CORES % VX_CFG_SOCKET_SIZE == 0, "NUM_CORES must be a multiple of SOCKET_SIZE");

constexpr uint32_t SOCKET_SIZE = VX_CFG_SOCKET_SIZE;
constexpr uint32_t NUM_SOCKETS = VX_CFG_NUM_CORES / VX_CFG_SOCKET_SIZE;
constexpr uint32_t NCC         = NUM_SOCKETS * SOCKET_SIZE;
constexpr uint32_t NB          = VX_CFG_NUM_LSU_BLOCKS;
constexpr uint32_t NL          = VX_CFG_NUM_LSU_LANES;
constexpr uint32_t NP          = NCC * NB;
constexpr uint32_t NW          = VX_CFG_NUM_WARPS;
constexpr uint32_t LMEM_AW     = VX_CFG_LMEM_LOG_SIZE - 3;
constexpr uint32_t CREDITS     = VX_CFG_DLMEM_STORE_CREDITS;
constexpr uint32_t LRSP_DELAY  = 2;

constexpr uint32_t log2_up(uint32_t n) {
  uint32_t b = 0;
  while ((1u << b) < n) {
    ++b;
  }
  return b;
}
constexpr uint32_t NW_BITS = log2_up(NW);
constexpr uint32_t NT_BITS = log2_up(VX_CFG_NUM_THREADS);

struct Lane {
  uint32_t src = 0;
  uint32_t req = 0;       // the request it belongs to
  uint32_t lane = 0;      // its LSU lane
  uint32_t seq = 0;       // per source, in issue order
  uint32_t dst = 0;
  bool     red = false;   // red.async.add, else st.async
  bool     wide = false;  // .b64/.u64, else .b32/.u32
  uint32_t waddr = 0;     // destination word
  uint32_t mbar = 0;      // destination mbarrier word
  uint8_t  byteen = 0;
  uint64_t data = 0;
  uint32_t hart = 0;
  uint64_t issue = ~0ull;
  uint64_t acc = ~0ull;
  uint64_t acc_rsp = ~0ull;
  uint64_t cx = ~0ull;
};

struct Req {
  uint32_t src = 0;
  uint32_t mask = 0;
  uint32_t wid = 0;
  uint64_t release = 0;
  std::vector<Lane*> lanes = std::vector<Lane*>(NL, nullptr);
};

// What the bench drives in one cycle, before the clock edge: at most one
// request per core, on its first LSU port.
struct In {
  std::vector<const Req*> req = std::vector<const Req*>(NCC, nullptr);
  std::vector<uint8_t>  lmem_ready = std::vector<uint8_t>(NCC, 1);
  std::vector<uint8_t>  lrsp_valid = std::vector<uint8_t>(NCC, 0);
  std::vector<uint32_t> lrsp_tag = std::vector<uint32_t>(NCC, 0);
};

struct LmemV {
  bool     fire = false;
  bool     rw = false;
  bool     amo = false;
  bool     uns = false;
  uint32_t amo_op = 0;
  uint8_t  byteen = 0;
  uint32_t waddr = 0;
  uint32_t hart = 0;
  uint32_t tag = 0;
  uint64_t data = 0;
};

// What the bench observes: handshakes of the cycle, then state after the edge.
struct Out {
  std::vector<uint8_t> req_fire = std::vector<uint8_t>(NCC, 0);
  std::vector<LmemV>   lmem = std::vector<LmemV>(NCC);
  std::vector<uint8_t> ack_ej = std::vector<uint8_t>(NCC, 0);    // an ack ejected at this source
  std::vector<uint8_t> data_ej = std::vector<uint8_t>(NCC, 0);   // a data response ejected
  std::vector<uint8_t> wt_ack = std::vector<uint8_t>(NCC, 0);    // an ack decremented the store tracker
  std::vector<uint8_t> eg_empty = std::vector<uint8_t>(NCC, 0);
  std::vector<uint8_t> in_empty = std::vector<uint8_t>(NCC, 0);
};

class Dut {
public:
  virtual ~Dut() = default;
  virtual void cycle(const In& in, Out& out) = 0;
  // The local memory's atomic op codes of an add and of an mbarrier complete_tx.
  virtual uint32_t amo_add() const = 0;
  virtual uint32_t amo_complete_tx() const = 0;
};

struct Options {
  std::string mode = "async";   // async (latency schedule), async-random
  uint32_t seed = 1;
  uint32_t reqs = 0;
  uint32_t watchdog = 2000;
};

inline std::vector<std::string> all_modes() {
  return {"async", "async-random"};
}

class Bench {
public:
  Bench(Dut& dut, const Options& opts) : dut_(dut), opts_(opts), rng_(opts.seed) {
    queue_.resize(NCC);
    pending_.resize(NCC);
    lrsp_.resize(NCC);
    acks_.resize(NCC);
    seq_.assign(NCC, 0);
  }

  uint32_t run() {
    generate();
    const uint64_t limit = 1000000;
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
    for (int i = 0; i < 16 && failures_ == 0; ++i) {
      clock();
    }
    for (uint32_t c = 0; c < NCC && failures_ == 0; ++c) {
      if (!out_.eg_empty[c] || !out_.in_empty[c]) {
        fail(strf("core %u: endpoints not empty after the run (egress %d ingress %d)", c, out_.eg_empty[c],
                  out_.in_empty[c]));
      }
      if (acks_[c].size() != n_lanes_src_[c]) {
        fail(strf("core %u: %zu acks for %u lanes", c, acks_[c].size(), n_lanes_src_[c]));
      }
    }
    return failures_;
  }

  uint64_t cycles() const { return done_cycle_; }
  size_t lanes() const { return lanes_.size(); }
  uint64_t max_inflight() const { return max_inflight_; }
  const std::string& first_failure() const { return first_failure_; }

  void write_csv(const std::string& path) const {
    struct Row { uint32_t src, dst; const char* op; uint32_t seq; uint64_t issue, done; };
    std::vector<Row> rows;
    std::vector<std::vector<const Lane*>> by_src(NCC);
    for (auto& l : lanes_) {
      by_src[l.src].push_back(&l);
    }
    for (uint32_t s = 0; s < NCC; ++s) {
      auto& v = by_src[s];
      std::sort(v.begin(), v.end(), [](const Lane* a, const Lane* b) { return a->seq < b->seq; });
      for (auto* l : v) {
        rows.push_back({s, l->dst, l->red ? "RED_ASYNC_ACC" : "ST_ASYNC_ACC", 3 * l->seq, l->issue, l->acc});
        rows.push_back({s, l->dst, l->red ? "RED_ASYNC_CX" : "ST_ASYNC_CX", 3 * l->seq + 1, l->issue, l->cx});
      }
      for (size_t n = 0; n < acks_[s].size() && n < v.size(); ++n) {
        rows.push_back({s, s, "ASYNC_ACK", uint32_t(3 * n + 2), v[n]->issue, acks_[s][n]});
      }
    }
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
      return std::tie(a.src, a.seq) < std::tie(b.src, b.seq);
    });
    std::ofstream ofs(path);
    ofs << "src,dst,op,seq,issue_cycle,done_cycle\n";
    for (auto& r : rows) {
      ofs << r.src << "," << r.dst << "," << r.op << "," << r.seq << "," << r.issue << "," << r.done << "\n";
    }
  }

private:
  static std::string strf(const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    return buf;
  }

  void fail(const std::string& msg) {
    if (failures_ == 0) {
      first_failure_ = strf("[cycle %" PRIu64 "] ", cycle_) + msg;
      std::printf("FAIL %s\n", first_failure_.c_str());
    }
    ++failures_;
  }

  uint32_t below(uint32_t n) {
    return std::uniform_int_distribution<uint32_t>(0, n - 1)(rng_);
  }

  bool chance(uint32_t pct) {
    return below(100) < pct;
  }

  // A request from core s of warp wid: its lanes' (dst, word, mbarrier, kind).
  void add_req(uint32_t s, uint32_t wid, uint64_t release,
               const std::vector<std::tuple<uint32_t, uint32_t, uint32_t, bool, bool>>& lanes) {
    reqs_.emplace_back();
    Req& r = reqs_.back();
    r.src = s;
    r.wid = wid;
    r.release = release;
    for (uint32_t l = 0; l < lanes.size() && l < NL; ++l) {
      lanes_.emplace_back();
      Lane& x = lanes_.back();
      std::tie(x.dst, x.waddr, x.mbar, x.red, x.wide) = lanes[l];
      x.src = s;
      x.req = uint32_t(reqs_.size() - 1);
      x.lane = l;
      x.waddr &= (1u << LMEM_AW) - 1;
      x.mbar &= (1u << LMEM_AW) - 1;
      uint32_t half = (x.waddr + l) & 1;
      x.byteen = x.wide ? 0xff : (half ? 0xf0 : 0x0f);
      uint64_t v = (uint64_t(s) << 48) ^ (uint64_t(lanes_.size()) << 24) ^ 0x0123456789abull;
      x.data = x.wide ? v : (half ? (v << 32) : (v & 0xffffffffull));
      x.hart = (s << (NW_BITS + NT_BITS)) | (wid << NT_BITS) | l;
      r.mask |= 1u << l;
      r.lanes[l] = &x;
      ++n_lanes_src_[s];
    }
  }

  void generate() {
    n_lanes_src_.assign(NCC, 0);
    const std::string& m = opts_.mode;
    if (m == "async") {
      // Phase A: one lane at a time, every source into every core, its own
      // included, both kinds and widths, far enough apart that none meet.
      uint64_t t = 8;
      uint32_t k = 0;
      for (uint32_t s = 0; s < NCC; ++s) {
        for (uint32_t d = 0; d < NCC; ++d, ++k) {
          add_req(s, k % NW, t, {{d, 16 * k, 1000 + 2 * k, (k & 1) != 0, (k & 2) != 0}});
          t += 40;
        }
      }
      // Phase B: every source at once, all its lanes, into core 0 and the
      // last core, each with its own mbarrier: credits, switches and ingress
      // slots contend. An instruction has one kind, so each request does:
      // one of each per source, in alternating order.
      for (uint32_t s = 0; s < NCC; ++s) {
        for (uint32_t i = 0; i < 2; ++i) {
          bool red = ((s & 1) != 0) != (i != 0);
          std::vector<std::tuple<uint32_t, uint32_t, uint32_t, bool, bool>> v;
          for (uint32_t l = 0; l < NL; ++l) {
            uint32_t d = (l & 1) ? (NCC - 1) : 0;
            v.push_back({d, 512 + 16 * s + l, 2000 + s, red, (l & 2) != 0});
          }
          add_req(s, (s + i) % NW, t, v);
        }
      }
    } else if (m == "async-random") {
      uint32_t n = opts_.reqs ? opts_.reqs : 150;
      for (uint32_t i = 0; i < n; ++i) {
        uint32_t s = below(NCC);
        uint32_t lanes = 1 + below(NL);
        bool red = chance(40);
        std::vector<std::tuple<uint32_t, uint32_t, uint32_t, bool, bool>> v;
        for (uint32_t l = 0; l < lanes; ++l) {
          v.push_back({below(NCC), below(1u << LMEM_AW), below(1u << LMEM_AW), red, chance(50)});
        }
        add_req(s, below(NW), 8 + below(4 * n), v);
      }
      lmem_pct_ = 70;
      rsp_delay_max_ = 6;
    } else {
      fail("unknown mode " + m);
    }
    // Each core issues in release order; lanes take sequence numbers in issue order.
    std::vector<Req*> order;
    for (auto& r : reqs_) {
      order.push_back(&r);
    }
    std::stable_sort(order.begin(), order.end(), [](const Req* a, const Req* b) {
      return std::tie(a->src, a->release) < std::tie(b->src, b->release);
    });
    for (auto* r : order) {
      queue_[r->src].push_back(r);
      for (uint32_t l = 0; l < NL; ++l) {
        if (r->lanes[l]) {
          r->lanes[l]->seq = seq_[r->src]++;
        }
      }
    }
  }

  bool outstanding() const {
    if (n_done_ < lanes_.size()) {
      return true;
    }
    for (uint32_t c = 0; c < NCC; ++c) {
      if (acks_[c].size() < n_lanes_src_[c]) {
        return true;
      }
    }
    return false;
  }

  void clock() {
    In in;
    for (uint32_t c = 0; c < NCC; ++c) {
      auto& q = queue_[c];
      if (!q.empty() && q.front()->release <= cycle_) {
        in.req[c] = q.front();
        for (auto* l : q.front()->lanes) {
          if (l && l->issue == ~0ull) {
            l->issue = cycle_;
          }
        }
      }
      in.lmem_ready[c] = (lmem_pct_ >= 100) ? 1 : chance(lmem_pct_);
      auto& r = lrsp_[c];
      in.lrsp_valid[c] = !r.empty() && std::get<0>(r.front()) <= cycle_;
      in.lrsp_tag[c] = in.lrsp_valid[c] ? std::get<1>(r.front()) : 0;
    }

    dut_.cycle(in, out_);

    for (uint32_t c = 0; c < NCC; ++c) {
      if (out_.req_fire[c]) {
        if (!in.req[c]) {
          fail(strf("core %u: LSU port fired with nothing offered", c));
          continue;
        }
        queue_[c].pop_front();
        last_progress_ = cycle_;
      }
      if (in.lrsp_valid[c]) {
        Lane* l = std::get<2>(lrsp_[c].front());
        if (l) {
          l->acc_rsp = cycle_;
        }
        lrsp_[c].pop_front();
      }
      if (out_.wt_ack[c]) {
        fail(strf("core %u: an async ack reached the store tracker", c));
      }
      if (out_.data_ej[c]) {
        fail(strf("core %u: a data response for an async access", c));
      }
      if (out_.ack_ej[c]) {
        if (acks_[c].size() >= n_lanes_src_[c]) {
          fail(strf("core %u: an ack without an async lane", c));
        }
        acks_[c].push_back(cycle_);
        last_progress_ = cycle_;
      }
      if (out_.lmem[c].fire) {
        check_lmem(c, out_.lmem[c]);
      }
    }
    ++cycle_;
  }

  // An access at core c must be some lane's, into c, not yet performed; a
  // complete_tx must follow the response of a performed lane's access.
  void check_lmem(uint32_t c, const LmemV& q) {
    bool is_cx = q.amo && (q.amo_op == dut_.amo_complete_tx());
    uint32_t delay = (rsp_delay_max_ > LRSP_DELAY) ? (LRSP_DELAY + below(rsp_delay_max_ - LRSP_DELAY + 1)) : LRSP_DELAY;
    if (is_cx) {
      Lane* hit = nullptr;
      for (auto* l : pending_[c]) {
        if (l->cx == ~0ull && l->acc_rsp != ~0ull && l->mbar == q.waddr) {
          hit = l;
          break;
        }
      }
      if (!hit) {
        fail(strf("core %u: complete_tx on word 0x%x with no performed lane awaiting it", c, q.waddr));
        return;
      }
      uint64_t bytes = hit->wide ? 8 : 4;
      if (q.rw || q.uns || q.byteen != 0xff || q.hart != 0 || q.data != bytes) {
        fail(strf("core %u: complete_tx on word 0x%x: rw %d uns %d byteen 0x%x hart %u data %" PRIu64
                  ", expected %" PRIu64 " bytes of hart 0", c, q.waddr, q.rw, q.uns, q.byteen, q.hart, q.data, bytes));
      }
      if (!(hit->acc_rsp < cycle_)) {
        fail(strf("core %u: complete_tx on word 0x%x before its access's response", c, q.waddr));
      }
      hit->cx = cycle_;
      pending_[c].erase(std::find(pending_[c].begin(), pending_[c].end(), hit));
      ++n_done_;
      last_progress_ = cycle_;
      lrsp_[c].push_back({cycle_ + delay, q.tag, nullptr});
      return;
    }
    Lane* hit = nullptr;
    for (auto& l : lanes_) {
      if (l.dst == c && l.acc == ~0ull && l.issue != ~0ull && l.waddr == q.waddr && l.data == q.data
       && l.byteen == q.byteen) {
        hit = &l;
        break;
      }
    }
    if (!hit) {
      fail(strf("core %u: local-memory access at word 0x%x is no issued lane's", c, q.waddr));
      return;
    }
    bool ok = hit->red ? (!q.rw && q.amo && q.amo_op == dut_.amo_add() && !q.uns) : (q.rw && !q.amo);
    if (!ok || q.hart != hit->hart) {
      fail(strf("core %u: access at word 0x%x: rw %d amo %d op %u uns %d hart 0x%x, expected %s of hart 0x%x", c,
                q.waddr, q.rw, q.amo, q.amo_op, q.uns, q.hart, hit->red ? "an add" : "a store", hit->hart));
    }
    hit->acc = cycle_;
    pending_[c].push_back(hit);
    last_progress_ = cycle_;
    lrsp_[c].push_back({cycle_ + delay, q.tag, hit});
    max_inflight_ = std::max<uint64_t>(max_inflight_, pending_[c].size());
  }

  Dut& dut_;
  Options opts_;
  std::mt19937 rng_;
  Out out_;
  std::deque<Req> reqs_;
  std::deque<Lane> lanes_;
  std::vector<std::deque<Req*>> queue_;
  std::vector<std::vector<Lane*>> pending_;
  std::vector<std::deque<std::tuple<uint64_t, uint32_t, Lane*>>> lrsp_;
  std::vector<std::vector<uint64_t>> acks_;
  std::vector<uint32_t> seq_;
  std::vector<uint32_t> n_lanes_src_;
  uint32_t lmem_pct_ = 100;
  uint32_t rsp_delay_max_ = LRSP_DELAY;
  uint64_t n_done_ = 0;
  uint64_t max_inflight_ = 0;
  uint64_t cycle_ = 0;
  uint64_t last_progress_ = 0;
  uint64_t done_cycle_ = 0;
  uint32_t failures_ = 0;
  std::string first_failure_;
};

} // namespace async
