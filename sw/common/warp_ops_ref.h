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

// Golden model of the warp-op unit (ELECT, MATCH, REDUX): lane results and the
// cycle timing of its structure. Shared by the RTL bench, SimX and the test
// apps' host code; independent of VX_config.h.
//
// An op is a 5-bit code {kind[1:0], sub[2:0]}: kind = funct7[1:0] of the
// 0x18-0x1B encodings with REDUX's URd bit 5 (funct7[0]) dropped, sub = funct3.

#include <algorithm>
#include <cstdint>
#ifndef __VORTEX__
#include <vector>
#endif

namespace vortex {
namespace warp_ops_ref {

enum Code : uint32_t {
  REDUX_ADD     = 0x00,
  REDUX_MIN_S   = 0x01,
  REDUX_MAX_S   = 0x02,
  REDUX_MIN_U   = 0x03,
  REDUX_MAX_U   = 0x04,
  REDUX_AND     = 0x05,
  REDUX_OR      = 0x06,
  REDUX_XOR     = 0x07,
  ELECT_P       = 0x10,
  ELECT_L       = 0x11,
  MATCH_ANY_B32 = 0x18,
  MATCH_ANY_B64 = 0x19,
  MATCH_ALL_B32 = 0x1A,
  MATCH_ALL_B64 = 0x1B,
};

inline uint32_t code_of(uint32_t funct7, uint32_t funct3) {
  const uint32_t kind = (funct7 & 0x2) ? (funct7 & 0x3) : 0;
  return (kind << 3) | (funct3 & 0x7);
}

inline bool is_redux(uint32_t c)     { return c <= REDUX_XOR; }
inline bool is_elect(uint32_t c)     { return c == ELECT_P || c == ELECT_L; }
inline bool is_match_any(uint32_t c) { return c == MATCH_ANY_B32 || c == MATCH_ANY_B64; }
inline bool is_match_all(uint32_t c) { return c == MATCH_ALL_B32 || c == MATCH_ALL_B64; }
inline bool is_b64(uint32_t c)       { return c == MATCH_ANY_B64 || c == MATCH_ALL_B64; }
inline bool is_valid(uint32_t c) {
  return is_redux(c) || is_elect(c) || is_match_any(c) || is_match_all(c);
}
// ELECT.L and REDUX write a uniform register (the first active lane's value).
inline bool writes_ur(uint32_t c)    { return is_redux(c) || c == ELECT_L; }

// ELECT.L's value when no member lane is active.
constexpr uint64_t NO_LEADER = ~uint64_t(0);

// Reduction-tree levels over n lanes: one pipeline stage per level.
inline uint32_t tree_depth(uint32_t n) {
  uint32_t d = 0;
  while ((1u << d) < n)
    ++d;
  return d;
}

// Cycles from accept to result on an idle unit: MATCH.ANY makes one comparator-row pass per
// distinct value k in the operand stage and leaves from there; the other ops make one pass in
// the operand stage and then cross the tree.
inline uint32_t latency(uint32_t n, uint32_t c, uint32_t k) {
  return is_match_any(c) ? k : 1 + tree_depth(n);
}

// Cycles the operand stage stays busy with an op when its result is taken at once.
inline uint32_t occupancy(uint32_t c, uint32_t k) {
  return is_match_any(c) ? k : 1;
}

///////////////////////////////////////////////////////////////////////////////
// Lane results. Lanes are 0..n-1, tmask bit i = lane i active; rs1/rs2 hold the
// XLEN-wide register values. A b64 value is rs1 at XLEN 64 and {rs2, rs1} at
// XLEN 32. Inactive lanes of rd are left as they are.

inline uint64_t b64_value(uint32_t xlen, uint64_t rs1, uint64_t rs2) {
  return (xlen == 64) ? rs1 : ((rs2 & 0xffffffffull) << 32) | (rs1 & 0xffffffffull);
}

inline uint64_t key_of(uint32_t c, uint32_t xlen, uint64_t rs1, uint64_t rs2) {
  return is_b64(c) ? b64_value(xlen, rs1, rs2) : (rs1 & 0xffffffffull);
}

inline uint64_t to_xlen(uint32_t xlen, uint64_t v) {
  return (xlen == 64) ? v : (v & 0xffffffffull);
}

inline uint64_t sext32(uint32_t xlen, uint32_t v) {
  return to_xlen(xlen, (uint64_t)(int64_t)(int32_t)v);
}

inline uint32_t distinct_values(uint32_t c, uint32_t xlen, uint32_t n, uint64_t tmask,
                                const uint64_t* rs1, const uint64_t* rs2) {
  uint32_t k = 0;
  for (uint32_t i = 0; i < n; ++i) {
    if (!((tmask >> i) & 1))
      continue;
    bool first = true;
    for (uint32_t j = 0; j < i && first; ++j)
      first = !(((tmask >> j) & 1) && key_of(c, xlen, rs1[j], rs2[j]) == key_of(c, xlen, rs1[i], rs2[i]));
    k += first;
  }
  return std::max<uint32_t>(1, k);
}

inline uint32_t redux_step(uint32_t c, uint32_t a, uint32_t b) {
  switch (c) {
  case REDUX_ADD:   return a + b;
  case REDUX_MIN_S: return ((int32_t)a < (int32_t)b) ? a : b;
  case REDUX_MAX_S: return ((int32_t)a > (int32_t)b) ? a : b;
  case REDUX_MIN_U: return (a < b) ? a : b;
  case REDUX_MAX_U: return (a > b) ? a : b;
  case REDUX_AND:   return a & b;
  case REDUX_OR:    return a | b;
  default:          return a ^ b;
  }
}

// REDUX results are 32-bit, sign-extended to XLEN; masks and lane indices are
// zero-extended; ELECT.P writes 1 in the leader lane, 0 in the others.
inline void execute(uint32_t c, uint32_t xlen, uint32_t n, uint64_t tmask,
                    const uint64_t* rs1, const uint64_t* rs2, uint64_t* rd) {
  if (is_redux(c)) {
    bool first = true;
    uint32_t acc = 0;
    for (uint32_t i = 0; i < n; ++i) {
      if (!((tmask >> i) & 1))
        continue;
      acc = first ? (uint32_t)rs1[i] : redux_step(c, acc, (uint32_t)rs1[i]);
      first = false;
    }
    for (uint32_t i = 0; i < n; ++i)
      if ((tmask >> i) & 1)
        rd[i] = sext32(xlen, acc);
    return;
  }
  if (is_elect(c)) {
    int leader = -1;
    for (uint32_t i = 0; i < n && leader < 0; ++i) {
      if (((tmask >> i) & 1) && ((rs1[i] >> i) & 1))
        leader = (int)i;
    }
    for (uint32_t i = 0; i < n; ++i) {
      if (!((tmask >> i) & 1))
        continue;
      if (c == ELECT_P)
        rd[i] = ((int)i == leader) ? 1 : 0;
      else
        rd[i] = (leader < 0) ? to_xlen(xlen, NO_LEADER) : (uint64_t)leader;
    }
    return;
  }
  for (uint32_t i = 0; i < n; ++i) {
    if (!((tmask >> i) & 1))
      continue;
    const uint64_t key = key_of(c, xlen, rs1[i], rs2[i]);
    uint64_t same = 0;
    bool all = true;
    for (uint32_t j = 0; j < n; ++j) {
      if (!((tmask >> j) & 1))
        continue;
      if (key_of(c, xlen, rs1[j], rs2[j]) == key)
        same |= uint64_t(1) << j;
      else
        all = false;
    }
    rd[i] = is_match_any(c) ? same : (all ? tmask : 0);
  }
}

///////////////////////////////////////////////////////////////////////////////
// Timing of VX_alu_warp's structure, cycle by cycle. The operand stage holds one op; a tree op
// moves on to the tree's first level when the tree advances, a MATCH.ANY stays there for its k
// passes and offers its result on the last one. The tree advances unless its last level holds a
// result the port does not take; the tree's result goes first when both are ready.

#ifndef __VORTEX__
template <typename Tag>
class Pipe {
public:
  struct Fire {
    bool valid = false;
    Tag tag{};
  };

  explicit Pipe(uint32_t n) : tree_(tree_depth(n)) {}

  void reset() {
    s0_ = Slot{};
    for (auto& t : tree_)
      t = Slot{};
  }

  bool empty() const {
    if (s0_.valid)
      return false;
    for (auto& t : tree_) {
      if (t.valid)
        return false;
    }
    return true;
  }

  // The op whose result the unit offers this cycle (the tree's first).
  Fire offered() const {
    const Slot& last = tree_.back();
    if (last.valid)
      return Fire{true, last.tag};
    if (s0_.valid && s0_.any && s0_.left == 1)
      return Fire{true, s0_.tag};
    return Fire{};
  }

  // Whether an offered op would be accepted this cycle, given whether the port takes the
  // offered result.
  bool ready(bool out_ready) const {
    return !s0_.valid || this->s0_leaves(out_ready);
  }

  // One clock edge: `out_ready` = the port takes offered(); `in` (may be null) = the op offered
  // at the input, accepted when ready(out_ready). Returns whether it was accepted.
  bool step(bool out_ready, const Tag* in, uint32_t code, uint32_t k) {
    const bool en = this->tree_advances(out_ready);
    const bool leave = s0_.valid && this->s0_leaves(out_ready);
    const bool accept = in && (!s0_.valid || leave);
    if (en) {
      for (size_t i = tree_.size() - 1; i > 0; --i)
        tree_[i] = tree_[i - 1];
      tree_[0] = (s0_.valid && !s0_.any) ? s0_ : Slot{};
    }
    if (accept) {
      s0_ = Slot{true, is_match_any(code), is_match_any(code) ? std::max<uint32_t>(k, 1) : 1, *in};
    } else if (leave) {
      s0_ = Slot{};
    } else if (s0_.valid && s0_.any && s0_.left > 1) {
      --s0_.left;
    }
    return accept;
  }

private:
  struct Slot {
    bool valid = false;
    bool any = false;
    uint32_t left = 0;
    Tag tag{};
  };

  bool tree_advances(bool out_ready) const {
    return !tree_.back().valid || out_ready;
  }

  bool s0_leaves(bool out_ready) const {
    if (s0_.any)
      return s0_.left == 1 && !tree_.back().valid && out_ready;
    return this->tree_advances(out_ready);
  }

  Slot s0_;
  std::vector<Slot> tree_;
};

struct Req {
  uint64_t arrive;
  uint32_t code;
  uint32_t k;
};

struct Event {
  uint64_t accept;
  uint64_t done;
};

// The accept and done (result taken) cycle of each op of `ports` in-order streams into one unit:
// the input picks round robin among the ports whose head has arrived, starting after the port it
// last accepted from (VX_stream_arb "R", reset seed ports - 1); `out_ready(port, cycle)` is that
// port's result ready, the result going to the port its op came from.
template <typename ReadyFn>
std::vector<std::vector<Event>> schedule_ports(uint32_t n, const std::vector<std::vector<Req>>& reqs,
                                               ReadyFn out_ready) {
  const uint32_t ports = (uint32_t)reqs.size();
  Pipe<uint64_t> pipe(n);
  std::vector<std::vector<Event>> ev(ports);
  std::vector<size_t> head(ports, 0);
  size_t total = 0, done = 0;
  for (uint32_t p = 0; p < ports; ++p) {
    ev[p].assign(reqs[p].size(), Event{0, 0});
    total += reqs[p].size();
  }
  uint32_t rr = ports - 1;
  for (uint64_t cyc = 0; done < total; ++cyc) {
    const auto f = pipe.offered();
    const bool rdy = f.valid && out_ready(uint32_t(f.tag >> 32), cyc);
    if (rdy) {
      ev[f.tag >> 32][uint32_t(f.tag)].done = cyc;
      ++done;
    }
    uint32_t sel = 0;
    bool offer = false;
    for (uint32_t i = 1; i <= ports && !offer; ++i) {
      sel = (rr + i) % ports;
      offer = head[sel] < reqs[sel].size() && reqs[sel][head[sel]].arrive <= cyc;
    }
    const uint64_t tag = offer ? (uint64_t(sel) << 32) | head[sel] : 0;
    const Req* r = offer ? &reqs[sel][head[sel]] : nullptr;
    if (pipe.step(rdy, offer ? &tag : nullptr, r ? r->code : 0, r ? r->k : 1)) {
      ev[sel][head[sel]++].accept = cyc;
      rr = sel;
    }
  }
  return ev;
}

// One in-order stream; `out_ready(cycle)` is the result port's ready.
template <typename ReadyFn>
std::vector<Event> schedule(uint32_t n, const std::vector<Req>& reqs, ReadyFn out_ready) {
  return schedule_ports(n, std::vector<std::vector<Req>>{reqs},
                        [&](uint32_t, uint64_t c) { return out_ready(c); })[0];
}
#endif

} // namespace warp_ops_ref
} // namespace vortex
