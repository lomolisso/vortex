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

// Golden model of an mbarrier, the 64-bit local-memory word that the local
// memory banks update in their atomic unit. Shared by SimX, the testbenches and
// the test apps' host code; independent of VX_config.h.
//
// Layout: pending[19:0] expected[39:20] tx[60:40] (signed bytes) [62:61] zero phase[63].

#include <cstdint>

namespace vortex {
namespace mbar_ref {

// amo_op codes; the custom-2 (0x5B) funct3 is the code minus OP_BASE.
enum Op : uint32_t {
  OP_ARRIVE      = 10,  // operand: arrival count
  OP_ARRIVE_TX   = 11,  // operand: bytes; tx += bytes, then one arrival
  OP_EXPECT_TX   = 12,  // operand: bytes
  OP_COMPLETE_TX = 13,  // operand: bytes
  OP_TEST_WAIT   = 14,  // operand: token
  OP_TRY_WAIT    = 15,  // operand: token; same bank behaviour as TEST_WAIT
};
constexpr uint32_t OP_BASE = OP_ARRIVE;

enum Err : uint32_t {
  ERR_NONE      = 0,
  ERR_UNDERFLOW = 1,  // more arrivals than pending
  ERR_TX_RANGE  = 2,  // tx left [-TX_MAX, TX_MAX]
};

// Assert texts, identical in the RTL and SimX.
constexpr const char* MSG_UNDERFLOW = "mbarrier pending underflow";
constexpr const char* MSG_TX_RANGE  = "mbarrier tx out of range";
constexpr const char* MSG_ALIGN     = "mbarrier op is not a naturally aligned doubleword";
constexpr const char* MSG_GLOBAL    = "mbarrier op on a global address";
constexpr const char* MSG_REMOTE    = "remote mbarrier ops are not supported yet";

constexpr uint32_t COUNT_BITS = 20;
constexpr uint32_t TX_BITS    = 21;
constexpr uint64_t COUNT_MAX  = (uint64_t(1) << COUNT_BITS) - 1;
constexpr int64_t  TX_MAX     = (int64_t(1) << (TX_BITS - 1)) - 1;

inline uint32_t pending(uint64_t w)  { return uint32_t(w & COUNT_MAX); }
inline uint32_t expected(uint64_t w) { return uint32_t((w >> COUNT_BITS) & COUNT_MAX); }
inline int64_t  tx(uint64_t w) {
  uint64_t f = (w >> 40) & ((uint64_t(1) << TX_BITS) - 1);
  return (f >> (TX_BITS - 1)) ? int64_t(f) - (int64_t(1) << TX_BITS) : int64_t(f);
}
inline uint32_t phase(uint64_t w)    { return uint32_t(w >> 63); }

// Fields are truncated to their widths, so an erroneous op still yields a defined word.
inline uint64_t pack(uint64_t pend, uint64_t exp, int64_t t, uint32_t ph) {
  return (pend & COUNT_MAX)
       | ((exp & COUNT_MAX) << COUNT_BITS)
       | ((uint64_t(t) & ((uint64_t(1) << TX_BITS) - 1)) << 40)
       | (uint64_t(ph & 1) << 63);
}

// What a plain store writes to initialise the object; count in [1, COUNT_MAX].
inline uint64_t init(uint32_t count) {
  return pack(count, count, 0, 0);
}

inline bool is_wait(uint32_t op) {
  return op == OP_TEST_WAIT || op == OP_TRY_WAIT;
}

// True once the phase the token was taken in has completed.
inline bool phase_done(uint64_t word, uint64_t token) {
  return phase(word) != phase(token);
}

struct Result {
  uint64_t new_word;  // written back when writes
  uint64_t ret_word;  // returned to rd: the old word, or 0/1 for a wait
  bool     writes;
  bool     flip;      // the op completed the phase
  uint32_t err;       // Err bits
};

inline Result apply(uint32_t op, uint64_t old, uint64_t operand) {
  Result r{old, old, false, false, ERR_NONE};
  if (is_wait(op)) {
    r.ret_word = phase_done(old, operand) ? 1 : 0;
    return r;
  }

  int64_t pend = pending(old);
  int64_t t    = tx(old);
  // A byte count of 2^TX_BITS or more cannot leave tx in range.
  const bool big = (operand >> TX_BITS) != 0;
  const int64_t bytes = int64_t(operand & ((uint64_t(1) << TX_BITS) - 1));

  switch (op) {
  case OP_ARRIVE:
    if (operand > uint64_t(pend)) r.err |= ERR_UNDERFLOW;
    pend -= int64_t(operand & COUNT_MAX);
    break;
  case OP_ARRIVE_TX:
    if (big) r.err |= ERR_TX_RANGE;
    t += bytes;
    if (pend == 0) r.err |= ERR_UNDERFLOW;
    pend -= 1;
    break;
  case OP_EXPECT_TX:
    if (big) r.err |= ERR_TX_RANGE;
    t += bytes;
    break;
  case OP_COMPLETE_TX:
    if (big) r.err |= ERR_TX_RANGE;
    t -= bytes;
    break;
  default:
    r.err |= ERR_UNDERFLOW | ERR_TX_RANGE;
    return r;
  }
  if (t > TX_MAX || t < -TX_MAX) r.err |= ERR_TX_RANGE;

  // Fields wrap at their widths, as in the bank; only an erroneous op can wrap.
  pend = int64_t(uint64_t(pend) & COUNT_MAX);
  t    = tx(pack(0, 0, t, 0));
  uint32_t ph = phase(old);
  // The completion rule follows every writing op, expect_tx included: a
  // complete_tx may arrive first and leave tx negative.
  if (pend == 0 && t == 0) {
    ph ^= 1;
    pend = expected(old);
    r.flip = true;
  }
  r.new_word = pack(uint64_t(pend), expected(old), t, ph);
  r.writes = true;
  return r;
}

} // namespace mbar_ref
} // namespace vortex
