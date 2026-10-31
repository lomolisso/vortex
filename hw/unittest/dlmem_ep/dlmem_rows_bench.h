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

// Row-flit bench of the DLMEM endpoints: the posted flits of remote DXA
// copies, injected at each socket's row port and checked where each core's
// ingress delivers them, a data word on the local memory's DMA port, a copy's
// completion as a complete_tx on its local-memory port. It has no simulator
// dependency, so the RTL testbench and the SimX test run the same schedule and
// write the same latency rows.
//
// Checks: every flit arrives once, at its core, with its address, data, byte
// mask and slot; the flits from one socket to one core arrive in send order,
// so a completion reaches the local memory after every data word sent before
// it, in a later cycle; nothing answers a row flit on the response network;
// every ingress drains.

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

namespace rows {

static_assert(VX_CFG_NUM_CORES % VX_CFG_SOCKET_SIZE == 0, "NUM_CORES must be a multiple of SOCKET_SIZE");

constexpr uint32_t SOCKET_SIZE = VX_CFG_SOCKET_SIZE;
constexpr uint32_t NUM_SOCKETS = VX_CFG_NUM_CORES / VX_CFG_SOCKET_SIZE;
constexpr uint32_t NCC         = NUM_SOCKETS * SOCKET_SIZE;
constexpr uint32_t NW          = VX_CFG_NUM_WARPS;
constexpr uint32_t NBANKS      = VX_CFG_LMEM_NUM_BANKS;
constexpr uint32_t WS          = 8;
constexpr uint32_t LMEM_AW     = VX_CFG_LMEM_LOG_SIZE - 3;
constexpr uint32_t LRSP_DELAY  = 2;

struct Flit {
  uint32_t sock = 0;      // sending socket
  uint32_t seq = 0;       // per socket, in send order
  uint32_t copy = 0;
  bool     compl_ = false;
  uint32_t dst = 0;
  uint32_t src = 0;       // issuing core, carried in the packet
  uint32_t slot = 0;
  uint32_t addr = 0;      // word address; a completion's mbarrier word
  uint8_t  byteen = 0;
  uint64_t data = 0;      // a completion's box bytes
  uint64_t release = 0;
  uint64_t issue = ~0ull;
  uint64_t done = ~0ull;
};

// What the bench drives in one cycle, before the clock edge.
struct In {
  std::vector<const Flit*> row = std::vector<const Flit*>(NUM_SOCKETS, nullptr);
  std::vector<uint8_t>  dma_ready = std::vector<uint8_t>(NCC, 1);
  std::vector<uint8_t>  lmem_ready = std::vector<uint8_t>(NCC, 1);
  std::vector<uint8_t>  lrsp_valid = std::vector<uint8_t>(NCC, 0);
  std::vector<uint32_t> lrsp_tag = std::vector<uint32_t>(NCC, 0);
  std::vector<uint32_t> slot_valid = std::vector<uint32_t>(NCC, (NW >= 32) ? ~0u : ((1u << NW) - 1));
};

struct DmaV {
  bool     fire = false;
  uint32_t addr = 0;                     // DMA word: NBANKS words
  std::vector<uint64_t> word = std::vector<uint64_t>(NBANKS, 0);
  std::vector<uint8_t>  byteen = std::vector<uint8_t>(NBANKS, 0);
  uint32_t slot = 0;
  bool     remote = false;
  bool     last = false;
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
  std::vector<uint8_t> row_fire = std::vector<uint8_t>(NUM_SOCKETS, 0);
  std::vector<DmaV>    dma = std::vector<DmaV>(NCC);
  std::vector<LmemV>   lmem = std::vector<LmemV>(NCC);
  std::vector<uint8_t> rsp_inj = std::vector<uint8_t>(NCC, 0);
  std::vector<uint8_t> in_empty = std::vector<uint8_t>(NCC, 0);
};

class Dut {
public:
  virtual ~Dut() = default;
  virtual void cycle(const In& in, Out& out) = 0;
  // The local memory's atomic op code of an mbarrier complete_tx.
  virtual uint32_t amo_complete_tx() const = 0;
};

struct Options {
  std::string mode = "rows";   // rows (latency schedule), rows-random, rows-retired
  uint32_t seed = 1;
  uint32_t copies = 0;
  uint32_t watchdog = 2000;
};

inline std::vector<std::string> all_modes() {
  return {"rows", "rows-random"};
}

class Bench {
public:
  Bench(Dut& dut, const Options& opts) : dut_(dut), opts_(opts), rng_(opts.seed) {
    queue_.resize(NUM_SOCKETS);
    inflight_.assign(NUM_SOCKETS, std::vector<std::deque<Flit*>>(NCC));
    lrsp_.resize(NCC);
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
      if (!out_.in_empty[c]) {
        fail(strf("core %u: ingress not empty after the run", c));
      }
    }
    return failures_;
  }

  uint64_t cycles() const { return done_cycle_; }
  size_t flits() const { return flits_.size(); }
  uint64_t compls() const { return n_compl_; }
  uint64_t stalls() const { return n_stall_; }
  const std::string& first_failure() const { return first_failure_; }

  // issue_cycle: the first cycle the flit is offered at its socket's row
  // port; done_cycle: a data word's DMA write, a completion's complete_tx on
  // the local-memory port.
  void write_csv(const std::string& path) const {
    std::vector<const Flit*> v;
    for (auto& f : flits_) {
      v.push_back(&f);
    }
    std::sort(v.begin(), v.end(), [](const Flit* a, const Flit* b) {
      return std::tie(a->sock, a->seq) < std::tie(b->sock, b->seq);
    });
    std::ofstream ofs(path);
    ofs << "src,dst,op,seq,issue_cycle,done_cycle\n";
    for (auto* f : v) {
      ofs << f->sock << "," << f->dst << "," << (f->compl_ ? "DXA_COMPL" : "DXA_DATA") << "," << f->seq << ","
          << f->issue << "," << f->done << "\n";
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

  // A copy from socket s into core d: n data words from word w0 (each with a
  // nonzero byte mask), then its completion on mbarrier word mb.
  void add_copy(uint32_t s, uint32_t d, uint32_t slot, uint32_t w0, uint32_t n, uint64_t release, bool random_masks) {
    uint32_t copy = n_copies_++;
    uint32_t src = s * SOCKET_SIZE + (copy % SOCKET_SIZE);
    for (uint32_t i = 0; i <= n; ++i) {
      Flit f;
      f.sock = s;
      f.copy = copy;
      f.compl_ = (i == n);
      f.dst = d;
      f.src = src;
      f.slot = slot;
      f.release = release;
      if (f.compl_) {
        f.addr = ((1u << LMEM_AW) - 1 - (copy % 64)) & ((1u << LMEM_AW) - 1);
        f.byteen = 0xff;
        f.data = (uint64_t(copy) << 16) | (uint64_t(n) * WS);   // the box bytes, tagged with the copy
      } else {
        f.addr = (w0 + i) & ((1u << LMEM_AW) - 1);
        f.byteen = random_masks ? uint8_t(1 + below(255)) : uint8_t((i % 3 == 1) ? 0x0f : (i % 3 == 2) ? 0xf0 : 0xff);
        f.data = (uint64_t(copy) << 40) ^ (uint64_t(i) << 20) ^ 0x5a5a00001234ull ^ (uint64_t(d) << 56);
      }
      flits_.push_back(f);
    }
  }

  void generate() {
    const std::string& m = opts_.mode;
    if (m == "rows") {
      // Phase A: one copy at a time, every socket into every core, with
      // releases far enough apart that no two copies meet.
      uint64_t t = 8;
      for (uint32_t s = 0; s < NUM_SOCKETS; ++s) {
        for (uint32_t d = 0; d < NCC; ++d) {
          add_copy(s, d, (s + d) % NW, 16 * (s * NCC + d), 3, t, false);
          t += 40;
        }
      }
      // Phase B: every socket at once into core 0 and into the last core,
      // contending at the crossbar, the switches and the ingresses.
      for (uint32_t s = 0; s < NUM_SOCKETS; ++s) {
        add_copy(s, 0, s % NW, 512 + 32 * s, 6, t, false);
        add_copy(s, NCC - 1, (s + 1) % NW, 1024 + 32 * s, 6, t, false);
      }
    } else if (m == "rows-random") {
      uint32_t n = opts_.copies ? opts_.copies : 200;
      for (uint32_t i = 0; i < n; ++i) {
        uint32_t s = below(NUM_SOCKETS);
        uint32_t d = below(NCC);
        uint32_t slot = below(NW);
        uint32_t w0 = below(1u << LMEM_AW);
        uint32_t words = 1 + below(12);
        uint64_t release = 8 + below(4 * n);
        add_copy(s, d, slot, w0, words, release, true);
      }
      dma_pct_ = 70;
      lmem_pct_ = 80;
    } else if (m == "rows-retired") {
      // The last core's slot 0 is free: the copy into it must stop the run.
      retired_core_ = NCC - 1;
      add_copy(0, NCC - 1, 0, 64, 2, 8, false);
    } else {
      fail("unknown mode " + m);
    }
    // Each socket sends in release order.
    std::vector<Flit*> order;
    for (auto& f : flits_) {
      order.push_back(&f);
    }
    std::stable_sort(order.begin(), order.end(), [](const Flit* a, const Flit* b) {
      return std::tie(a->sock, a->release, a->copy) < std::tie(b->sock, b->release, b->copy);
    });
    std::vector<uint32_t> seq(NUM_SOCKETS, 0);
    for (auto* f : order) {
      f->seq = seq[f->sock]++;
      queue_[f->sock].push_back(f);
    }
  }

  bool outstanding() const {
    return n_done_ < flits_.size();
  }

  bool chance(uint32_t pct) {
    return below(100) < pct;
  }

  void clock() {
    In in;
    for (uint32_t s = 0; s < NUM_SOCKETS; ++s) {
      auto& q = queue_[s];
      if (!q.empty() && q.front()->release <= cycle_) {
        in.row[s] = q.front();
        if (q.front()->issue == ~0ull) {
          q.front()->issue = cycle_;
        }
      }
    }
    for (uint32_t c = 0; c < NCC; ++c) {
      in.dma_ready[c] = (dma_pct_ >= 100) ? 1 : chance(dma_pct_);
      in.lmem_ready[c] = (lmem_pct_ >= 100) ? 1 : chance(lmem_pct_);
      auto& r = lrsp_[c];
      in.lrsp_valid[c] = !r.empty() && r.front().first <= cycle_;
      in.lrsp_tag[c] = in.lrsp_valid[c] ? r.front().second : 0;
      if (c == retired_core_) {
        in.slot_valid[c] &= ~1u;
      }
    }

    dut_.cycle(in, out_);

    for (uint32_t s = 0; s < NUM_SOCKETS; ++s) {
      if (out_.row_fire[s]) {
        if (!in.row[s]) {
          fail(strf("socket %u: row port fired with nothing offered", s));
          continue;
        }
        Flit* f = queue_[s].front();
        queue_[s].pop_front();
        inflight_[s][f->dst].push_back(f);
        last_progress_ = cycle_;
      } else if (in.row[s]) {
        ++n_stall_;
      }
    }
    for (uint32_t c = 0; c < NCC; ++c) {
      if (in.lrsp_valid[c]) {
        lrsp_[c].pop_front();
      }
      if (out_.rsp_inj[c]) {
        fail(strf("core %u: a response left the ingress for a posted flit", c));
      }
      const auto& d = out_.dma[c];
      if (d.fire) {
        check_dma(c, d);
      }
      const auto& l = out_.lmem[c];
      if (l.fire) {
        check_lmem(c, l);
      }
    }
    ++cycle_;
  }

  // The next flit of some socket's stream into core c, if it matches. A data
  // word is the stream's first one: completions ahead of it may still wait
  // between the ingress and the local-memory port, which the ingress left in
  // order. A completion must be the stream's head.
  Flit* match_head(uint32_t c, bool compl_, uint32_t addr, uint64_t data) {
    for (uint32_t s = 0; s < NUM_SOCKETS; ++s) {
      auto& q = inflight_[s][c];
      for (auto it = q.begin(); it != q.end(); ++it) {
        if ((*it)->compl_ != compl_) {
          if (compl_) {
            break;
          }
          continue;
        }
        if ((*it)->addr != addr || (*it)->data != data) {
          break;
        }
        Flit* f = *it;
        q.erase(it);
        return f;
      }
    }
    return nullptr;
  }

  void check_dma(uint32_t c, const DmaV& d) {
    uint32_t lanes = 0, lane = 0;
    for (uint32_t b = 0; b < NBANKS; ++b) {
      if (d.byteen[b]) {
        ++lanes;
        lane = b;
      }
    }
    if (lanes != 1) {
      fail(strf("core %u: DMA write with %u bank lanes", c, lanes));
      return;
    }
    uint32_t waddr = d.addr * NBANKS + lane;
    Flit* f = match_head(c, false, waddr, d.word[lane]);
    if (!f) {
      fail(strf("core %u: DMA write of word 0x%x is no socket's next flit into this core", c, waddr));
      return;
    }
    if (d.byteen[lane] != f->byteen || d.slot != f->slot || !d.remote || d.last) {
      fail(strf("core %u: DMA write of word 0x%x: byteen 0x%x data 0x%" PRIx64 " slot %u remote %d last %d, "
                "expected 0x%x 0x%" PRIx64 " slot %u", c, waddr, d.byteen[lane], d.word[lane], d.slot,
                d.remote, d.last, f->byteen, f->data, f->slot));
    }
    finish(f);
  }

  void check_lmem(uint32_t c, const LmemV& l) {
    Flit* f = match_head(c, true, l.waddr, l.data);
    if (!f) {
      fail(strf("core %u: local-memory request at word 0x%x is no socket's next completion into this core", c, l.waddr));
      return;
    }
    if (l.rw || !l.amo || l.amo_op != dut_.amo_complete_tx() || l.uns || l.byteen != 0xff || l.hart != 0
     || l.data != f->data) {
      fail(strf("core %u: completion at word 0x%x: rw %d amo %d op %u uns %d byteen 0x%x hart %u data %" PRIu64
                ", expected a complete_tx of %" PRIu64 " bytes", c, l.waddr, l.rw, l.amo, l.amo_op, l.uns,
                l.byteen, l.hart, l.data, f->data));
    }
    for (auto& g : flits_) {
      if (g.copy == f->copy && !g.compl_ && !(g.done < cycle_)) {
        fail(strf("core %u: copy %u's completion reached the local memory before its word 0x%x", c, f->copy, g.addr));
        break;
      }
    }
    lrsp_[c].push_back({cycle_ + LRSP_DELAY, l.tag});
    ++n_compl_;
    finish(f);
  }

  void finish(Flit* f) {
    f->done = cycle_;
    ++n_done_;
    last_progress_ = cycle_;
  }

  Dut& dut_;
  Options opts_;
  std::mt19937 rng_;
  Out out_;
  std::deque<Flit> flits_;
  std::vector<std::deque<Flit*>> queue_;
  std::vector<std::vector<std::deque<Flit*>>> inflight_;
  std::vector<std::deque<std::pair<uint64_t, uint32_t>>> lrsp_;
  uint32_t dma_pct_ = 100;
  uint32_t lmem_pct_ = 100;
  uint32_t retired_core_ = ~0u;
  uint32_t n_copies_ = 0;
  uint64_t n_done_ = 0;
  uint64_t n_compl_ = 0;
  uint64_t n_stall_ = 0;
  uint64_t cycle_ = 0;
  uint64_t last_progress_ = 0;
  uint64_t done_cycle_ = 0;
  uint32_t failures_ = 0;
  std::string first_failure_;
};

} // namespace rows
