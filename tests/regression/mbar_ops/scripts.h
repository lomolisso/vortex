#ifndef _MBAR_OPS_SCRIPTS_H_
#define _MBAR_OPS_SCRIPTS_H_

// The mbarrier scripts of the H100 semantics probe (H100 NVL, CUDA 12.9.1,
// sm_90a), one thread per script, with the value the H100
// returned at each recorded step: the pending count of the word an arrive
// returned (mbarrier.pending_count), or test_wait.parity's 0/1. -1: not
// recorded. The Vortex-only scripts at the end (h100 false) carry the values
// PTX defines instead. Shared by the mbar_ops host code and the mbar_ref unit
// test.

#include "common.h"
#include <mbar_ref.h>

#include <cstdint>
#include <vector>

namespace mbar_ops {

struct Step {
  uint32_t kind, side, n;
  int64_t  h100;
};

struct Script {
  const char*       name;
  std::vector<Step> steps;
  bool              h100;
};

constexpr int64_t NA = -1;

inline Step init(uint32_t n)                  { return {K_INIT, SIDE_OWN, n, NA}; }
inline Step arr(uint32_t n, int64_t v, uint32_t side = SIDE_PEER) { return {K_ARR, side, n, v}; }
inline Step drop(uint32_t n, int64_t v, uint32_t side = SIDE_PEER) { return {K_DROP, side, n, v}; }
inline Step nc(uint32_t n, int64_t v)         { return {K_NC, SIDE_PEER, n, v}; }
inline Step dnc(uint32_t n, int64_t v)        { return {K_DNC, SIDE_PEER, n, v}; }
inline Step droptx(uint32_t b)                { return {K_DROPTX, SIDE_PEER, b, NA}; }
inline Step ctx(uint32_t b)                   { return {K_CTX, SIDE_PEER, b, NA}; }
inline Step clsdrop(uint32_t n)               { return {K_CLSDROP, SIDE_PEER, n, NA}; }
inline Step tw(uint32_t par, int64_t v)       { return {K_TW, SIDE_OWN, par, v}; }
inline Step waitp(uint32_t par, int64_t v)    { return {K_WAITP, SIDE_OWN, par, v}; }

// The arrives of the H100 scripts run on the peer in the remote modes; the
// waits always on the owner, as PTX allows waits only on the own object.
inline std::vector<Script> scripts() {
  std::vector<Script> s;
  // arrive_drop, then arrives: phase 1 and 2 expect 3.
  s.push_back({"drop.A", {init(4), drop(1, 4), tw(0, 0), arr(1, 3), tw(0, 0), arr(2, 2), tw(0, 1),
                          arr(2, 3), tw(1, 0), arr(1, 1), tw(1, 1), arr(3, 3), tw(0, 1)}, true});
  s.push_back({"drop.B", {init(8), drop(3, 8), tw(0, 0), arr(5, 5), tw(0, 1), arr(4, 5), tw(1, 0),
                          arr(1, 1), tw(1, 1)}, true});
  // The drop completes the phase: pending reloads from the reduced count.
  s.push_back({"drop.C", {init(2), arr(1, 2), drop(1, 1), tw(0, 1), arr(1, 1), tw(1, 1)}, true});
  s.push_back({"drop.D", {init(5), drop(2, 5), drop(1, 3), tw(0, 0), arr(2, 2), tw(0, 1), arr(1, 2),
                          tw(1, 0), arr(1, 1), tw(1, 1)}, true});
  // Rank 1 drops on rank 0's object after rank 0's arrive (H100: over DSMEM).
  s.push_back({"drop.remote", {init(2), arr(1, 2, SIDE_OWN), drop(1, NA), waitp(0, 1), arr(1, 1, SIDE_OWN),
                               tw(1, 1), arr(1, 1, SIDE_OWN), tw(0, 1)}, true});
  s.push_back({"nocomplete.arrive", {init(4), nc(2, 4), tw(0, 0), nc(1, 2), tw(0, 0), arr(1, 1), tw(0, 1),
                                     arr(4, 4), tw(1, 1)}, true});
  s.push_back({"nocomplete.drop", {init(4), dnc(1, 4), tw(0, 0), arr(3, 3), tw(0, 1), arr(2, 3), tw(1, 0),
                                   arr(1, 1), tw(1, 1)}, true});
  {
    Script q{"pending_count.seq", {init(20)}, true};
    for (int i = 0; i < 21; ++i)
      q.steps.push_back(arr(1, i < 20 ? 20 - i : 20));
    s.push_back(q);
  }
  // Vortex only: arrive_drop.expect_tx, and cuda::barrier::arrive_and_drop().
  s.push_back({"drop.tx", {init(2), droptx(16), tw(0, 0), ctx(16), tw(0, 0), arr(1, 1), tw(0, 1),
                           arr(1, 1), tw(1, 1)}, false});
  s.push_back({"drop.cls", {init(3), arr(1, 3), clsdrop(1), tw(0, 0), arr(1, 1), tw(0, 1), arr(2, 2),
                            tw(1, 1)}, false});
  return s;
}

inline uint64_t operand(const Step& st) {
  namespace mr = vortex::mbar_ref;
  switch (st.kind) {
  case K_DROP:    case K_CLSDROP: case K_DROPTX: return st.n | mr::FLAG_DROP;
  case K_NC:      return st.n | mr::FLAG_NOCOMPLETE;
  case K_DNC:     return st.n | mr::FLAG_DROP | mr::FLAG_NOCOMPLETE;
  case K_TW:      case K_WAITP: return uint64_t(st.n & 1) << 63;
  default:        return st.n;
  }
}

inline uint32_t op_code(const Step& st) {
  namespace mr = vortex::mbar_ref;
  switch (st.kind) {
  case K_DROPTX: return mr::OP_ARRIVE_TX;
  case K_CTX:    return mr::OP_COMPLETE_TX;
  case K_TW:     case K_WAITP: return mr::OP_TEST_WAIT;
  default:       return mr::OP_ARRIVE;
  }
}

// What the device records at each step, by mbar_ref: the returned word, the
// wait's 0/1, or 0. Also the final word, and false on any mbar_ref error.
inline bool replay(const Script& sc, std::vector<uint64_t>& rec, uint64_t& word) {
  namespace mr = vortex::mbar_ref;
  bool ok = true;
  word = 0;
  rec.clear();
  for (const Step& st : sc.steps) {
    if (st.kind == K_INIT) {
      word = mr::init(st.n);
      rec.push_back(0);
      continue;
    }
    const mr::Result r = mr::apply(op_code(st), word, operand(st));
    ok &= (r.err == mr::ERR_NONE);
    if (st.kind == K_WAITP)
      ok &= (r.ret_word == 1);
    rec.push_back(st.kind == K_CLSDROP ? 0 : r.ret_word);
    if (r.writes)
      word = r.new_word;
  }
  return ok;
}

// The H100's value of a recorded step, from what the device recorded.
inline int64_t h100_view(const Step& st, uint64_t rec) {
  if (st.kind == K_TW || st.kind == K_WAITP)
    return int64_t(rec);
  return int64_t(vortex::mbar_ref::pending(rec));
}

} // namespace mbar_ops

#endif
