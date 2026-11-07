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

// BgUnit, the bulk async-groups of one core. A driver ticking before it plays
// the DXA unit (group events in their send cycle) and the socket's engine
// (completions); a sink ticking after it records the unlocks and the state
// the DXA unit sees.
//
// Timing: an event or a completion in cycle T shows in the state the DXA unit
// sees at T+1 (the open group's sequence, stall_op while its group has 63
// operations not yet performed or with 8 committed groups, stall_commit with
// 8); a wait satisfied in cycle T, on
// the counts registered before it, reaches the sink at T+2, one warp a cycle,
// lowest first, so a completion in cycle T releases a wait at T+3 and a wait
// that is satisfied on arrival is released at T+2; the head retires one
// complete group a cycle. The pending flags the exit hold and the core's busy
// read (sampled by the driver, before the unit ticks): an operation in cycle
// T raises them from T+1, the completion that clears the last count in T
// drops them from T+1; rd_pending per warp, wr_pending for every warp.
//
// Semantics: wait_group N waits for the writes, wait_group.read N for the
// source reads, of every committed group but the N newest; N >= 8 and
// operations not yet committed never hold a wait; the sequence wraps at 8.

#include "bg_unit.h"
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace vortex;

#ifndef DXA_MBAR_ENABLE
#error "the bulk-group unit needs a DXA build with mbarriers"
#endif

namespace {

int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

class BgProbe {
public:
  struct Sample {
    std::vector<bool> rd;
    bool wr = false;
  };
  std::map<uint64_t, Sample> samples;
};

class Driver : public SimObject<Driver> {
public:
  SimChannel<DxaBgEvt>   EvtOut;
  SimChannel<DxaBgCompl> ComplOut;
  std::multimap<uint64_t, std::function<void()>> script;
  bool active = false;
  const BgUnit* bg = nullptr;
  BgProbe probe;

  Driver(const SimContext& ctx, const char* name)
    : SimObject<Driver>(ctx, name), EvtOut(this, 4), ComplOut(this, 4) {}

protected:
  void on_reset() {}
  void on_tick() {
    if (!active)
      return;
    if (bg) {
      BgProbe::Sample x;
      for (uint32_t w = 0; w < VX_CFG_NUM_WARPS; ++w)
        x.rd.push_back(bg->rd_pending(w));
      x.wr = bg->wr_pending();
      probe.samples[SimPlatform::instance().cycles()] = x;
    }
    auto r = script.equal_range(SimPlatform::instance().cycles());
    for (auto it = r.first; it != r.second; ++it) {
      it->second();
    }
  }
  friend class SimObject<Driver>;
};

class Sink : public SimObject<Sink> {
public:
  SimChannel<uint32_t>   UnlockIn;
  SimChannel<DxaBgState> StateIn;
  std::vector<std::pair<uint64_t, uint32_t>>   unlocks;
  std::vector<std::pair<uint64_t, DxaBgState>> states;

  Sink(const SimContext& ctx, const char* name)
    : SimObject<Sink>(ctx, name), UnlockIn(this, 4), StateIn(this, 2 * VX_CFG_NUM_WARPS) {}

  // The state of warp w the DXA unit sees in cycle c.
  DxaBgState state_at(uint32_t w, uint64_t c) const {
    DxaBgState s;
    s.wid = w;
    for (auto& e : states) {
      if (e.first <= c && e.second.wid == w) s = e.second;
    }
    return s;
  }

  // The first cycle the warp's state matched `f`, or ~0.
  uint64_t first(uint32_t w, std::function<bool(const DxaBgState&)> f) const {
    for (auto& e : states) {
      if (e.second.wid == w && f(e.second)) return e.first;
    }
    return ~uint64_t(0);
  }

protected:
  void on_reset() {}
  void on_tick() {
    uint64_t c = SimPlatform::instance().cycles();
    while (!UnlockIn.empty()) {
      unlocks.push_back({c, UnlockIn.peek()});
      UnlockIn.pop();
    }
    while (!StateIn.empty()) {
      states.push_back({c, StateIn.peek()});
      StateIn.pop();
    }
  }
  friend class SimObject<Sink>;
};

std::vector<std::shared_ptr<Driver>> g_drivers;

struct Rig {
  std::shared_ptr<Driver> drv;
  std::shared_ptr<BgUnit> bg;
  std::shared_ptr<Sink>   sink;

  explicit Rig(const std::string& name) {
    drv  = Driver::Create((name + "-drv").c_str());
    bg   = BgUnit::Create((name + "-bg").c_str());
    sink = Sink::Create((name + "-sink").c_str());
    drv->EvtOut.bind(&bg->EvtIn);
    drv->ComplOut.bind(&bg->ComplIn);
    bg->UnlockOut.bind(&sink->UnlockIn);
    bg->StateOut.bind(&sink->StateIn);
    drv->bg = bg.get();
    g_drivers.push_back(drv);
  }

  void at(uint64_t c, std::function<void()> f) { drv->script.emplace(c, std::move(f)); }
  void evt(uint64_t c, uint32_t wid, DxaBgEvt::Kind kind, uint32_t n = 0) {
    this->at(c, [this, wid, kind, n] {
      DxaBgEvt e;
      e.wid = wid;
      e.kind = kind;
      e.n = n;
      drv->EvtOut.send(e, 0);
    });
  }
  void compl_(uint64_t c, uint32_t wid, uint32_t seq, bool rd, bool wr) {
    this->at(c, [this, wid, seq, rd, wr] {
      DxaBgCompl e;
      e.wid = wid;
      e.seq = seq;
      e.read_done = rd;
      e.write_done = wr;
      drv->ComplOut.send(e, 0);
    });
  }

  void run(uint64_t cycles) {
    auto& p = SimPlatform::instance();
    for (auto& d : g_drivers) d->active = (d == drv);
    p.reset();
    for (uint64_t i = 0; i < cycles; ++i) {
      p.tick();
    }
  }

  bool rd_at(uint32_t w, uint64_t c) const { return drv->probe.samples.at(c).rd.at(w); }
  bool wr_at(uint64_t c) const { return drv->probe.samples.at(c).wr; }

  uint64_t unlock_at(size_t k) const {
    return (k < sink->unlocks.size()) ? sink->unlocks[k].first : ~uint64_t(0);
  }
};

std::map<std::string, std::unique_ptr<Rig>> g_rigs;
Rig& rig(const std::string& name) { return *g_rigs.at(name); }

constexpr auto OP = DxaBgEvt::OP;
constexpr auto COMMIT = DxaBgEvt::COMMIT;
constexpr auto WAIT = DxaBgEvt::WAIT;
constexpr auto WAIT_READ = DxaBgEvt::WAIT_READ;

///////////////////////////////////////////////////////////////////////////////

// A wait with nothing committed, or with N >= 8, is released at T+2; so is
// one with only uncommitted operations.
void test_wait_free() {
  Rig& r = rig("free");
  r.evt(2, 1, WAIT, 0);
  r.evt(4, 2, OP);
  r.evt(5, 2, WAIT, 0);
  r.evt(6, 3, OP);
  r.evt(7, 3, COMMIT);
  r.evt(8, 3, WAIT, 8);
  r.evt(9, 3, OP);
  r.evt(10, 3, WAIT_READ, 31);
  r.run(20);
  auto& u = r.sink->unlocks;
  CHECK(u.size() == 4);
  if (u.size() == 4) {
    CHECK(u[0].first == 4 && u[0].second == 1);
    CHECK(u[1].first == 7 && u[1].second == 2);
    CHECK(u[2].first == 10 && u[2].second == 3);
    CHECK(u[3].first == 12 && u[3].second == 3);
  }
  std::printf("free waits: released at +2 with nothing committed, with N >= 8 and past uncommitted operations\n");
}

// wait_group 0 on one group of two operations: the read wait goes after the
// last read_done, the write wait after the last write_done, each at +3.
void test_wait_done() {
  Rig& r = rig("done");
  r.evt(1, 0, OP);
  r.evt(2, 0, OP);
  r.evt(3, 0, COMMIT);
  r.evt(4, 0, WAIT_READ, 0);
  r.compl_(6, 0, 0, true, false);
  r.compl_(9, 0, 0, true, false);
  r.evt(14, 0, WAIT, 0);
  r.compl_(16, 0, 0, false, true);
  r.compl_(20, 0, 0, false, true);
  r.run(30);
  auto& u = r.sink->unlocks;
  CHECK(u.size() == 2);
  if (u.size() == 2) {
    CHECK(u[0].first == 12 && u[0].second == 0);
    CHECK(u[1].first == 23 && u[1].second == 0);
  }
  // seq 1 from the commit's +1; the group retires the cycle after its last completion.
  CHECK(r.sink->first(0, [](const DxaBgState& s) { return s.seq == 1; }) == 4);
  std::printf("wait done: wait_group.read after the reads, wait_group after the writes, completion+3\n");
}

// wait_group N leaves the N newest groups pending; completions in the same
// event count once each; an older group's completion releases out of order.
void test_wait_n() {
  Rig& r = rig("n");
  for (uint32_t g = 0; g < 3; ++g) {
    r.evt(1 + 2 * g, 5 % VX_CFG_NUM_WARPS, OP);
    r.evt(2 + 2 * g, 5 % VX_CFG_NUM_WARPS, COMMIT);
  }
  const uint32_t w = 5 % VX_CFG_NUM_WARPS;
  r.evt(8, w, WAIT, 1);
  r.compl_(10, w, 2, true, true);   // the newest: not needed
  r.compl_(12, w, 1, true, true);   // the middle
  r.compl_(15, w, 0, true, true);   // the oldest: the wait goes
  r.evt(20, w, OP);                  // an open group's operation never holds a wait
  r.evt(21, w, WAIT, 0);
  r.run(30);
  auto& u = r.sink->unlocks;
  CHECK(u.size() == 2);
  if (u.size() == 2) {
    CHECK(u[0].first == 18 && u[0].second == w);
    CHECK(u[1].first == 23 && u[1].second == w);
  }
  std::printf("wait N: the N newest groups stay pending, uncommitted operations are not waited for\n");
}

// Two waits satisfied in one cycle, one by a completion and one on arrival:
// one unlock a cycle, lowest warp first.
void test_unlock_order() {
  Rig& r = rig("ul");
  const uint32_t a = 3 % VX_CFG_NUM_WARPS, b = 1;
  r.evt(2, b, OP);
  r.evt(7, b, COMMIT);
  r.evt(11, b, WAIT, 0);
  r.compl_(13, b, 0, true, true);
  r.evt(14, a, WAIT, 0);
  r.run(24);
  auto& u = r.sink->unlocks;
  CHECK(u.size() == 2);
  if (u.size() == 2) {
    CHECK(u[0].first == 16 && u[0].second == b);
    CHECK(u[1].first == 17 && u[1].second == a);
  }
  std::printf("unlock order: one warp a cycle, lowest first\n");
}

// stall_op once the open group holds 63 operations not yet performed; it
// drops as soon as one of them completes, so a group takes any number of
// operations (65 here), and the commit clears it. With 8 committed groups
// pending both stalls, until the head retires.
void test_capacity() {
  Rig& r = rig("cap");
  const uint32_t w = 2;
  for (uint32_t i = 0; i < 63; ++i) {
    r.evt(1 + i, w, OP);
  }
  r.compl_(70, w, 0, true, true);
  r.evt(72, w, OP);                 // the 64th
  r.compl_(74, w, 0, true, true);
  r.evt(76, w, OP);                 // the 65th
  r.evt(80, w, COMMIT);
  for (uint32_t g = 1; g < 8; ++g) {
    r.evt(80 + g, w, COMMIT);       // 7 empty groups behind group 0
  }
  for (uint32_t i = 0; i < 63; ++i) {
    r.compl_(90 + i, w, 0, true, true);
  }
  r.run(170);
  auto* s = r.sink.get();
  CHECK(s->first(w, [](const DxaBgState& x) { return x.stall_op; }) == 64);
  CHECK(!s->state_at(w, 63).stall_op && s->state_at(w, 64).stall_op);
  CHECK(s->state_at(w, 70).stall_op && !s->state_at(w, 71).stall_op);
  CHECK(!s->state_at(w, 72).stall_op && s->state_at(w, 73).stall_op);
  CHECK(!s->state_at(w, 75).stall_op && s->state_at(w, 77).stall_op);
  CHECK(!s->state_at(w, 81).stall_op && s->state_at(w, 81).seq == 1);
  CHECK(!s->state_at(w, 87).stall_commit);
  CHECK(s->state_at(w, 88).seq == 0 && s->state_at(w, 88).stall_commit && s->state_at(w, 88).stall_op);
  // the last completion in cycle 152, group 0 retires at 153, seen at 154
  CHECK(s->state_at(w, 153).stall_commit && !s->state_at(w, 154).stall_commit && !s->state_at(w, 154).stall_op);
  CHECK(!s->state_at(w, 169).stall_op && !s->state_at(w, 169).stall_commit);
  std::printf("capacity: stall_op at 63 pending operations, cleared by a completion (65 in one group) "
              "or the commit; both stalls with 8 groups, at +1\n");
}

// The head retires one complete group a cycle; the sequence wraps at 8 and
// a reused slot counts again, also completions that come before the commit.
void test_retire_wrap() {
  Rig& r = rig("wrap");
  const uint32_t w = 0;
  for (uint32_t g = 0; g < 8; ++g) {
    r.evt(1 + 2 * g, w, OP);
    r.evt(2 + 2 * g, w, COMMIT);
  }
  for (uint32_t g = 0; g < 8; ++g) {
    r.compl_(20 + g, w, g, true, true);
  }
  r.evt(30, w, OP);
  r.compl_(31, w, 0, true, false);
  r.compl_(32, w, 0, false, true);
  r.evt(33, w, COMMIT);
  r.evt(34, w, WAIT, 0);
  r.run(40);
  auto* s = r.sink.get();
  CHECK(s->state_at(w, 16).seq == 7 && !s->state_at(w, 16).stall_commit);
  CHECK(s->state_at(w, 17).seq == 0 && s->state_at(w, 17).stall_commit && s->state_at(w, 17).stall_op);
  CHECK(s->first(w, [](const DxaBgState& x) { return x.seq == 0 && !x.stall_commit; }) == 22);
  CHECK(s->state_at(w, 34).seq == 1);
  CHECK(r.sink->unlocks.size() == 1 && r.unlock_at(0) == 36);
  std::printf("retire and wrap: one group a cycle, the sequence wraps, early completions of the open group count\n");
}

// The pending flags: an operation at T raises rd_pending of its warp and
// wr_pending from T+1; the completion that clears the last count at T drops
// them from T+1, also beside a new operation; another warp's write keeps
// wr_pending.
void test_pending() {
  Rig& r = rig("pend");
  const uint32_t v = VX_CFG_NUM_WARPS - 1;
  r.evt(2, 0, OP);
  r.compl_(5, 0, 0, true, false);
  r.compl_(8, 0, 0, false, true);
  r.evt(10, 0, OP);
  r.evt(11, 0, OP);
  r.compl_(11, 0, 0, true, true);
  r.compl_(13, 0, 0, true, true);
  r.evt(16, 0, OP);
  r.evt(17, v, OP);
  r.compl_(18, 0, 0, true, true);
  r.compl_(20, v, 0, true, true);
  r.run(24);
  CHECK(!r.rd_at(0, 2) && !r.wr_at(2) && r.rd_at(0, 3) && r.wr_at(3));
  CHECK(r.rd_at(0, 5) && !r.rd_at(0, 6) && r.wr_at(6));
  CHECK(r.wr_at(8) && !r.wr_at(9));
  CHECK(r.rd_at(0, 12) && r.wr_at(12) && r.rd_at(0, 13) && !r.rd_at(0, 14) && !r.wr_at(14));
  CHECK(r.rd_at(0, 18) && !r.rd_at(0, 19) && r.rd_at(v, 19) && r.wr_at(19));
  CHECK(r.rd_at(v, 20) && !r.rd_at(v, 21) && !r.wr_at(21));
  for (uint32_t w = 1; w < v; ++w) {
    for (uint64_t c = 0; c < 24; ++c)
      CHECK(!r.rd_at(w, c));
  }
  std::printf("pending flags: raised at +1 by an operation, dropped at +1 by the last completion\n");
}

} // namespace

int main() {
  for (const char* n : {"free", "done", "n", "ul", "cap", "wrap", "pend"}) {
    g_rigs[n] = std::make_unique<Rig>(n);
  }
  test_wait_free();
  test_wait_done();
  test_wait_n();
  test_unlock_order();
  test_capacity();
  test_retire_wrap();
  test_pending();
  if (g_fail) {
    std::printf("FAILED (%d)\n", g_fail);
    return 1;
  }
  std::printf("PASSED\n");
  return 0;
}
