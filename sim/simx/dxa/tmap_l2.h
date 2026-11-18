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

#include "types.h"

#if defined(DXA_MBAR_ENABLE) && defined(VX_CFG_H100_FE_ENABLE)
#define DXA_TMAP_L2_ENABLE

#include <array>
#include <deque>
#include <ostream>
#include <stdexcept>
#include <vector>
#include <dxa_tmap.h>

namespace vortex {

// Level 2 of the tensor-map cache, one per cluster (the H100's GPC-level
// constant cache), shared by the cluster's sockets. It holds 128-B lines in
// L1-line sectors; the sockets' DXA cores read level 3 on its misses and fill
// it, and the one on port 0 loads the launch's parameter window into it. As
// VX_tmap_l2: one operation at a time through tag read, compare, data read and
// answer, one L1-line beat per pass; other operations go ahead of a parked
// lookup, and no lookup runs while the window loads.
struct TmapL2Req {
  enum Op : uint8_t { LOOKUP, FILL, INVAL, FLUSH };
  Op       op    = LOOKUP;
  uint64_t addr  = 0;      // LOOKUP/INVAL: the map; FILL: an L1 line; FLUSH: the parameter window
  bool     param = false;  // FILL from the parameter window
  std::array<uint8_t, VX_CFG_L1_LINE_SIZE> data{};
};

struct TmapL2Rsp {
  enum Kind : uint8_t { HIT, MISS, PRELOAD };
  Kind     kind = MISS;
  uint64_t addr = 0;       // PRELOAD: the parameter window
  std::array<uint32_t, TMAP_WORDS> w{};
};

inline std::ostream& operator<<(std::ostream& os, const TmapL2Req& r) {
  return os << "op=" << int(r.op) << ", addr=0x" << std::hex << r.addr << std::dec << ", param=" << r.param;
}

inline std::ostream& operator<<(std::ostream& os, const TmapL2Rsp& r) {
  return os << "kind=" << int(r.kind) << ", addr=0x" << std::hex << r.addr << std::dec;
}

class TmapL2 : public SimObject<TmapL2> {
public:
  using Ptr = std::shared_ptr<TmapL2>;

  static constexpr uint32_t LINE   = 128;
  static constexpr uint32_t SECTOR = VX_CFG_L1_LINE_SIZE;
  static constexpr uint32_t SECTORS = LINE / SECTOR;
  static constexpr uint32_t MAP_BYTES = TMAP_WORDS * 4;
  static constexpr uint32_t WAYS = VX_CFG_TMAP_L2_WAYS;
  static constexpr uint32_t SETS = VX_CFG_TMAP_L2_ENTRIES / VX_CFG_TMAP_L2_WAYS;
  // The FSM's cycles: a lookup beat passes tag, compare, data and answer; a
  // miss answers from compare; a fill writes at compare; an invalidate takes
  // tag and compare per line; a flush answers the cycle after it starts.
  static constexpr uint32_t BEAT_CYCLES = 4;
  static constexpr uint32_t MISS_CYCLES = 3;
  static constexpr uint32_t FILL_CYCLES = 3;

  std::vector<SimChannel<TmapL2Req>> ReqIn;
  std::vector<SimChannel<TmapL2Rsp>> RspOut;

  TmapL2(const SimContext& ctx, const char* name, uint32_t num_ports)
    : SimObject<TmapL2>(ctx, name)
    , ReqIn(num_ports, this)
    , RspOut(num_ports, this)
    , lookup_(num_ports)
    , sets_(SETS)
  {
    if (SECTOR > LINE || (LINE % SECTOR) != 0 || (MAP_BYTES % SECTOR) != 0)
      throw std::runtime_error("the tensor-map level 2 needs L1 lines that divide its 128-B lines and a map");
  }

protected:
  struct Line {
    bool     valid  = false;
    uint64_t tag    = 0;
    uint32_t sectors = 0;
    std::array<uint8_t, LINE> data{};
  };

  struct Set {
    std::array<Line, WAYS> ways;
    std::array<uint8_t, WAYS> plru{};  // tree nodes 1..WAYS-1: the half that holds the victim
    uint32_t fifo = 0;
  };

  struct Lookup {
    bool     valid = false;
    uint64_t addr  = 0;
    uint64_t ready = 0;  // parked the cycle it arrives, eligible the next
  };

  void on_reset() {
    for (auto& s : sets_) {
      s = Set();
    }
    for (auto& l : lookup_) {
      l = Lookup();
    }
    pending_ = 0;
    rr_ = 0;
    op_rr_ = 0;
    idle_at_ = 0;
  }

  void on_tick() {
    const uint64_t now = SimPlatform::instance().cycles();
    for (uint32_t p = 0; p < ReqIn.size(); ++p) {
      auto& ch = ReqIn.at(p);
      if (!ch.empty() && ch.peek().op == TmapL2Req::LOOKUP && !lookup_.at(p).valid) {
        lookup_.at(p) = {true, ch.peek().addr, now + 1};
        ch.pop();
      }
    }
    if (now < idle_at_)
      return;
    // Other operations first, round robin; then one lookup, round robin.
    for (uint32_t i = 0; i < ReqIn.size(); ++i) {
      const uint32_t p = (op_rr_ + i) % ReqIn.size();
      auto& ch = ReqIn.at(p);
      if (ch.empty() || ch.peek().op == TmapL2Req::LOOKUP)
        continue;
      const auto r = ch.peek();
      ch.pop();
      op_rr_ = (p + 1) % ReqIn.size();
      switch (r.op) {
      case TmapL2Req::FILL:
        this->fill(r);
        idle_at_ = now + FILL_CYCLES;
        break;
      case TmapL2Req::INVAL:
        idle_at_ = now + 1 + 2 * this->inval(r.addr);
        break;
      default:
        // Every socket sees the launch; port 0 loads the window.
        idle_at_ = now + 1;
        if (p == 0) {
          this->flush(r.addr);
          idle_at_ = now + 2;
        }
        break;
      }
      return;
    }
    if (pending_ != 0)
      return;
    for (uint32_t i = 0; i < lookup_.size(); ++i) {
      const uint32_t p = (rr_ + i) % lookup_.size();
      if (!lookup_[p].valid || lookup_[p].ready > now)
        continue;
      lookup_[p].valid = false;
      rr_ = (p + 1) % lookup_.size();
      this->lookup(p, lookup_[p].addr, now);
      break;
    }
  }

private:
  static uint64_t line_of(uint64_t a) { return a / LINE; }

  Line* find(uint64_t line, uint32_t* way = nullptr) {
    auto& s = sets_[line % SETS];
    for (uint32_t w = 0; w < WAYS; ++w) {
      if (s.ways[w].valid && s.ways[w].tag == line) {
        if (way) *way = w;
        return &s.ways[w];
      }
    }
    return nullptr;
  }

  void touch(uint64_t line, uint32_t way) {
    auto& s = sets_[line % SETS];
    uint32_t node = 1;
    for (uint32_t span = WAYS / 2; span != 0; span /= 2) {
      const uint32_t bit = (way & span) ? 1 : 0;
      s.plru[node] = uint8_t(bit ^ 1);
      node = 2 * node + bit;
    }
  }

  uint32_t victim(uint64_t line) {
    auto& s = sets_[line % SETS];
    for (uint32_t w = 0; w < WAYS; ++w) {
      if (!s.ways[w].valid)
        return w;
    }
    if (VX_CFG_TMAP_L2_REPL == 0) {
      const uint32_t w = s.fifo;
      s.fifo = (s.fifo + 1) % WAYS;
      return w;
    }
    uint32_t node = 1, way = 0;
    for (uint32_t span = WAYS / 2; span != 0; span /= 2) {
      const uint32_t bit = s.plru[node];
      way |= bit ? span : 0;
      node = 2 * node + bit;
    }
    return way;
  }

  // One beat per L1 line from the one holding the map's first byte; a beat
  // whose sector is absent answers MISS.
  void lookup(uint32_t port, uint64_t addr, uint64_t now) {
    TmapL2Rsp rsp;
    rsp.kind = TmapL2Rsp::HIT;
    const uint64_t first = addr & ~uint64_t(SECTOR - 1);
    const uint32_t beats = uint32_t((addr - first + MAP_BYTES - 1) / SECTOR + 1);
    uint32_t done = 0;
    for (; done < beats; ++done) {
      const uint64_t a = first + uint64_t(done) * SECTOR;
      const Line* l = this->find(line_of(a));
      const uint32_t sec = uint32_t(a % LINE) / SECTOR;
      if (!l || !(l->sectors & (1u << sec))) {
        rsp.kind = TmapL2Rsp::MISS;
        break;
      }
    }
    uint64_t at = now + uint64_t(done) * BEAT_CYCLES;
    if (rsp.kind == TmapL2Rsp::HIT) {
      for (uint32_t b = 0; b < MAP_BYTES; ++b) {
        uint32_t way = 0;
        const uint64_t a = addr + b;
        const Line* l = this->find(line_of(a), &way);
        rsp.w[b / 4] |= uint32_t(l->data[a % LINE]) << (8 * (b % 4));
        if ((a % LINE) == 0 || b == 0)
          this->touch(line_of(a), way);
      }
      idle_at_ = at + 1;
    } else {
      at += MISS_CYCLES;
      idle_at_ = at + 1;
    }
    DT(3, this->name() << " lookup: port=" << port << ", map=0x" << std::hex << addr << std::dec
       << ", " << (rsp.kind == TmapL2Rsp::MISS ? "miss" : "hit") << ", answer=" << at);
    // The response crossing's register is the answer register of the last beat.
    RspOut.at(port).send(rsp, at - now - 1);
  }

  void fill(const TmapL2Req& r) {
    const uint64_t line = line_of(r.addr);
    uint32_t way = 0;
    Line* l = this->find(line, &way);
    if (!l) {
      way = this->victim(line);
      l = &sets_[line % SETS].ways[way];
      *l = Line();
      l->valid = true;
      l->tag   = line;
    }
    const uint32_t off = uint32_t(r.addr % LINE);
    std::copy(r.data.begin(), r.data.end(), l->data.begin() + off);
    l->sectors |= 1u << (off / SECTOR);
    this->touch(line, way);
    if (r.param) {
      __assert(pending_ != 0, this->name() << " parameter-window fill without a load");
      --pending_;
    }
  }

  // Returns the lines visited.
  uint32_t inval(uint64_t addr) {
    for (uint64_t ln = line_of(addr); ln <= line_of(addr + MAP_BYTES - 1); ++ln) {
      if (Line* l = this->find(ln)) {
        l->valid = false;
      }
    }
    DT(3, this->name() << " inval: map=0x" << std::hex << addr << std::dec);
    return uint32_t(line_of(addr + MAP_BYTES - 1) - line_of(addr) + 1);
  }

  void flush(uint64_t window) {
    for (auto& s : sets_) {
      s = Set();
    }
    const uint64_t first = window & ~uint64_t(SECTOR - 1);
    pending_ = uint32_t((window + VX_CFG_TMAP_PARAM_BYTES - first + SECTOR - 1) / SECTOR);
    TmapL2Rsp rsp;
    rsp.kind = TmapL2Rsp::PRELOAD;
    rsp.addr = window;
    RspOut.at(0).send(rsp, 0);
    DT(3, this->name() << " flush: window=0x" << std::hex << window << std::dec << ", lines=" << pending_);
  }

  std::vector<Lookup> lookup_;
  std::vector<Set>    sets_;
  uint32_t            pending_ = 0;  // parameter-window sectors still to come
  uint32_t            rr_ = 0;
  uint32_t            op_rr_ = 0;
  uint64_t            idle_at_ = 0;  // the first cycle the FSM can start an operation

  friend class SimObject<TmapL2>;
};

} // namespace vortex

#endif
