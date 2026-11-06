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

// CpaUnit, the cp.async completion unit of one LSU block. A driver ticking
// before it plays the LSU slice (issue events and read responses in the same
// cycle) and a sink ticking after it records its write requests, unlocks and
// the registered status the slice sees.
//
// Timing: a response's lanes land in its cycle T and leave as one write
// packet at T+1, the lowest (slot, pid) first; an ack in cycle T makes its
// FIFO entry poppable at T+1, the head pops one entry per cycle, and the slot
// frees in the status the slice sees at T+1; the FIFO full (8 entries), the
// group boundaries full (8) and the pending arrive show the same way; a
// wait_group satisfied in cycle T unlocks its warp at T+1 in the scheduler,
// one warp a cycle, lowest first; an arrive leaves only after the head passed
// its tail, one packet a cycle, ahead of landed lanes, as COMPLETE_TX(1) or
// ARRIVE(1) with the all-ones tag, also while its warp keeps the port busy
// with later copies (more than 16 of them).
//
// Data: behind a real LsuMemAdapter and LocalMem, every cp-size x src-size x
// offset writes bytes [0, ss) of the source and zeros up to the cp-size,
// nothing else; an arrive updates the mbarrier word as mbar_ref says.

#include "cpa_unit.h"
#include "local_mem.h"
#include "lsu_mem_adapter.h"
#include "mem_block_pool.h"
#include "mbar_ref.h"
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace vortex;

#ifndef VX_CFG_EXT_MBAR_ENABLE
#error "the cp.async unit needs an mbarrier build"
#endif

namespace {

int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

constexpr uint32_t T     = VX_CFG_NUM_THREADS;
constexpr uint32_t L     = VX_CFG_NUM_LSU_LANES;
constexpr uint32_t NPIDS = CpaUnit::NUM_PIDS;
constexpr uint64_t LMEM  = VX_MEM_LMEM_BASE_ADDR;
constexpr uint32_t BLK   = VX_CFG_MEM_BLOCK_SIZE;

// Plays the slice: actions run in their cycle, ahead of the unit.
class Driver : public SimObject<Driver> {
public:
  SimChannel<CpaIssue> IssueOut;
  SimChannel<LsuRsp>   RspOut;
  SimChannel<LsuRsp>   AckOut;
  std::multimap<uint64_t, std::function<void()>> script;
  std::function<void()> every;  // runs each cycle, before the script
  bool active = false;

  Driver(const SimContext& ctx, const char* name)
    : SimObject<Driver>(ctx, name), IssueOut(this, 4), RspOut(this, 4), AckOut(this, 4) {}

protected:
  void on_reset() {}
  void on_tick() {
    if (!active)
      return;
    if (every)
      every();
    auto r = script.equal_range(SimPlatform::instance().cycles());
    for (auto it = r.first; it != r.second; ++it) {
      it->second();
    }
  }
  friend class SimObject<Driver>;
};

// Records what the unit sends, in the cycle a consumer ticking after it sees it.
class Sink : public SimObject<Sink> {
public:
  SimChannel<LsuReq>    ReqIn;
  SimChannel<uint32_t>  UnlockIn;
  SimChannel<CpaStatus> StatusIn;
  uint64_t hold_until = 0;  // the write port is not ready before this cycle
  uint64_t ready_div = 1;   // ... and then only one cycle in ready_div
  std::vector<std::pair<uint64_t, LsuReq>>   reqs;
  std::vector<std::pair<uint64_t, uint32_t>> unlocks;
  std::vector<std::pair<uint64_t, CpaStatus>> stats;

  Sink(const SimContext& ctx, const char* name)
    : SimObject<Sink>(ctx, name), ReqIn(this, 1), UnlockIn(this, 4), StatusIn(this, 4) {}

  // The status the slice sees in cycle c.
  CpaStatus status_at(uint64_t c) const {
    CpaStatus s = CpaStatus::idle();
    for (auto& e : stats) {
      if (e.first <= c) s = e.second;
    }
    return s;
  }

protected:
  void on_reset() {}
  void on_tick() {
    uint64_t c = SimPlatform::instance().cycles();
    if (c >= hold_until && (c % ready_div) == 0 && !ReqIn.empty()) {
      reqs.push_back({c, ReqIn.peek()});
      ReqIn.pop();
    }
    while (!UnlockIn.empty()) {
      unlocks.push_back({c, UnlockIn.peek()});
      UnlockIn.pop();
    }
    while (!StatusIn.empty()) {
      stats.push_back({c, StatusIn.peek()});
      StatusIn.pop();
    }
  }
  friend class SimObject<Sink>;
};

// Every object is created before the first tick; a run drives only its own rig.
std::vector<std::shared_ptr<Driver>> g_drivers;

struct Rig {
  std::shared_ptr<Driver>  drv;
  std::shared_ptr<CpaUnit> cpa;
  std::shared_ptr<Sink>    sink;

  explicit Rig(const std::string& name) {
    drv  = Driver::Create((name + "-drv").c_str());
    cpa  = CpaUnit::Create((name + "-cpa").c_str(), 0);
    sink = Sink::Create((name + "-sink").c_str());
    drv->IssueOut.bind(&cpa->IssueIn);
    drv->RspOut.bind(&cpa->RspIn);
    drv->AckOut.bind(&cpa->AckIn);
    cpa->ReqOut.bind(&sink->ReqIn);
    cpa->UnlockOut.bind(&sink->UnlockIn);
    cpa->StatusOut.bind(&sink->StatusIn);
    g_drivers.push_back(drv);
  }

  void at(uint64_t c, std::function<void()> f) { drv->script.emplace(c, std::move(f)); }

  void run(uint64_t cycles) {
    auto& p = SimPlatform::instance();
    for (auto& d : g_drivers) d->active = (d == drv);
    p.reset();
    for (uint64_t i = 0; i < cycles; ++i) {
      p.tick();
    }
  }

  // The cycle of the k-th write or arrive request, or ~0.
  uint64_t req_cycle(size_t k) const { return (k < sink->reqs.size()) ? sink->reqs[k].first : ~uint64_t(0); }
};

std::map<std::string, std::unique_ptr<Rig>> g_rigs;

Rig& rig(const std::string& name) {
  return *g_rigs.at(name);
}

CpaIssue copy(uint32_t wid, uint32_t slot, uint32_t size, uint32_t tmask, uint64_t dst0 = LMEM + 0x100,
              uint32_t ss = 0xff) {
  CpaIssue e;
  e.kind = CpaIssue::COPY;
  e.wid = wid;
  e.slot = slot;
  e.size = size;
  e.uuid = 100 + slot;
  e.tmask = tmask;
  e.lanes.resize(T);
  for (uint32_t t = 0; t < T; ++t) {
    e.lanes[t].dst = dst0 + t * size;
    e.lanes[t].ss = (ss == 0xff) ? size : ss;
    e.lanes[t].src_off = (t * size) % BLK;
  }
  return e;
}

CpaIssue ctrl(CpaIssue::Kind kind, uint32_t wid, uint32_t n = 0) {
  CpaIssue e;
  e.kind = kind;
  e.wid = wid;
  e.n = n;
  return e;
}

CpaIssue arrive(uint32_t wid, bool noinc, uint32_t tmask, uint64_t mbar) {
  CpaIssue e = ctrl(CpaIssue::ARRIVE, wid);
  e.noinc = noinc;
  e.tmask = tmask;
  e.addrs.assign(T, mbar);
  return e;
}

// A read response for lanes `mask` of (slot, pid), each source word = pattern.
LsuRsp rsp(uint32_t slot, uint32_t pid, uint32_t mask, uint8_t seed = 0x10) {
  LsuRsp r(L);
  r.tag = CpaUnit::rsp_tag(slot, pid);
  for (uint32_t i = 0; i < L; ++i) {
    if (!(mask >> i & 1)) continue;
    r.mask.set(i);
    auto b = make_mem_block();
    for (uint32_t k = 0; k < BLK; ++k) (*b)[k] = uint8_t(seed + k);
    r.data.at(i) = b;
  }
  return r;
}

LsuRsp ack(uint32_t tag, uint32_t mask) {
  LsuRsp r(L);
  r.tag = tag;
  for (uint32_t i = 0; i < L; ++i) {
    if (mask >> i & 1) r.mask.set(i);
  }
  return r;
}

uint32_t lane_mask(const LsuReq& r) {
  uint32_t m = 0;
  for (uint32_t i = 0; i < L; ++i) m |= uint32_t(r.mask.test(i)) << i;
  return m;
}

uint32_t all_lanes() { return (L >= 32) ? ~0u : ((1u << L) - 1); }

///////////////////////////////////////////////////////////////////////////////

// A response lands in its cycle and leaves at +1; landed lanes go lowest
// (slot, pid) first, one packet a cycle, the zero fill in the packet.
void test_write_timing() {
  Rig& r = rig("wt");
  r.at(1, [&] { r.drv->IssueOut.send(copy(1, 3, 8, 0x3, LMEM + 0x200, 5), 0); });
  r.at(2, [&] { r.drv->IssueOut.send(copy(2, 1, 4, 0x1, LMEM + 0x300), 0); });
  r.at(5, [&] { r.drv->RspOut.send(rsp(3, 0, 0x1), 0); });
  r.sink->hold_until = 8;  // the write port refuses until 8: slot 3 waits, slot 1 lands behind it
  r.at(6, [&] { r.drv->RspOut.send(rsp(1, 0, 0x1, 0x40), 0); });
  r.at(7, [&] { r.drv->RspOut.send(rsp(3, 0, 0x2), 0); });
  r.run(14);
  auto& q = r.sink->reqs;
  CHECK(q.size() == 3);
  if (q.size() != 3) return;
  // The unit offers slot 3 lane 0 from cycle 6 and holds it; at 8 the port
  // takes it; slot 1 (lower) goes next although it landed later.
  CHECK(q[0].first == 8 && q[0].second.tag == 3 && lane_mask(q[0].second) == 0x1);
  CHECK(q[1].first == 9 && q[1].second.tag == 1 && lane_mask(q[1].second) == 0x1);
  CHECK(q[2].first == 10 && q[2].second.tag == 3 && lane_mask(q[2].second) == 0x2);
  // An 8-byte lane with src-size 5: five source bytes from its block offset, three zeros.
  const auto& w = q[0].second;
  CHECK(w.op == MemOp::ST && w.flags.strsp);
  CHECK(w.addrs.at(0) == LMEM + 0x200);
  uint32_t off = (LMEM + 0x200) % BLK;
  CHECK(w.byteen.at(0) == (0xffull << off));
  for (uint32_t k = 0; k < 8; ++k) {
    uint8_t want = (k < 5) ? uint8_t(0x10 + 0 + k) : 0;
    CHECK((*w.data.at(0))[off + k] == want);
  }
  std::printf("write timing: land at T, write at T+1, lowest slot first, zero fill in the packet\n");
}

// Without back-pressure: a response landing in T is written at T+1.
void test_write_latency() {
  Rig& r = rig("wl");
  r.at(1, [&] { r.drv->IssueOut.send(copy(0, 0, 4, all_lanes() & 0x5), 0); });
  r.at(4, [&] { r.drv->RspOut.send(rsp(0, 0, all_lanes() & 0x5), 0); });
  r.run(8);
  CHECK(r.sink->reqs.size() == 1);
  CHECK(r.req_cycle(0) == 5);
  std::printf("write latency: 1 cycle\n");
}

// Acks out of order: the head waits for its own entry, then pops one entry a
// cycle; each slot frees in the status at ack+1.
void test_ooo_acks() {
  Rig& r = rig("ooo");
  r.at(1, [&] { r.drv->IssueOut.send(copy(1, 0, 8, 0x1), 0); });
  r.at(2, [&] { r.drv->IssueOut.send(copy(1, 1, 8, 0x1), 0); });
  r.at(3, [&] { r.drv->IssueOut.send(ctrl(CpaIssue::COMMIT, 1), 0); });
  r.at(4, [&] { r.drv->IssueOut.send(ctrl(CpaIssue::WAIT, 1, 0), 0); });
  r.at(10, [&] { r.drv->AckOut.send(ack(1, 0x1), 0); });
  r.at(14, [&] { r.drv->AckOut.send(ack(0, 0x1), 0); });
  r.run(24);
  auto st = [&](uint64_t c) { return r.sink->status_at(c); };
  CHECK(st(2).free_slots == ((uint64_t(1) << CpaUnit::SLOTS) - 2));
  CHECK(st(3).free_slots == ((uint64_t(1) << CpaUnit::SLOTS) - 4));
  CHECK((st(11).free_slots & 2) && !(st(10).free_slots & 2));  // slot 1 free at ack+1
  CHECK((st(15).free_slots & 1) && !(st(14).free_slots & 1));
  // Entry 0 pops at 15, entry 1 at 16: the boundary at 2 is passed at 16, the
  // wait is satisfied at 17 on the registered head, pulses at 18 and the
  // scheduler takes it at 19.
  CHECK(r.sink->unlocks.size() == 1);
  if (!r.sink->unlocks.empty()) {
    CHECK(r.sink->unlocks[0].second == 1);
    CHECK(r.sink->unlocks[0].first == 19);
  }
  std::printf("out-of-order acks: in-order pops, slot free at ack+1, the wait's unlock 3 cycles after the pop\n");
}

// The FIFO (8 entries) or the slots fill, and 8 boundaries beyond the head,
// show at +1; a wait satisfied at its issue in T reaches the scheduler at T+2;
// N >= 8 always is; a pop that passes the boundaries releases a waiting warp.
void test_full_flags() {
  {
    Rig& r = rig("fifo");
    const uint32_t n = std::min(CpaUnit::DEPTH, CpaUnit::SLOTS);
    for (uint32_t k = 0; k < n; ++k) {
      r.at(1 + k, [&r, k] { r.drv->IssueOut.send(copy(2, k, 4, 0x1), 0); });
    }
    r.run(n + 4);
    auto st = [&](uint64_t c) { return r.sink->status_at(c); };
    if (n == CpaUnit::DEPTH) {
      CHECK(!st(n).fifo_full.test(2) && st(n + 1).fifo_full.test(2));
    }
    if (n == CpaUnit::SLOTS) {
      CHECK(st(n).free_slots != 0 && st(n + 1).free_slots == 0);
    }
  }
  Rig& r = rig("grp");
  r.at(1, [&] { r.drv->IssueOut.send(copy(3, 0, 4, 0x1), 0); });
  for (uint32_t k = 0; k < CpaUnit::GROUPS; ++k) {
    r.at(2 + k, [&r] { r.drv->IssueOut.send(ctrl(CpaIssue::COMMIT, 3), 0); });
  }
  r.at(12, [&] { r.drv->IssueOut.send(ctrl(CpaIssue::WAIT, 3, 7), 0); });
  r.at(13, [&] { r.drv->IssueOut.send(ctrl(CpaIssue::WAIT, 2, 0), 0); });
  r.at(14, [&] { r.drv->IssueOut.send(ctrl(CpaIssue::WAIT, 1, 8), 0); });
  r.at(20, [&] { r.drv->AckOut.send(ack(0, 0x1), 0); });
  r.run(30);
  auto st = [&](uint64_t c) { return r.sink->status_at(c); };
  CHECK(!st(1 + CpaUnit::GROUPS).groups_full.test(3) && st(2 + CpaUnit::GROUPS).groups_full.test(3));
  // The pop at 21 drops every boundary; the status clears at 22.
  CHECK(st(21).groups_full.test(3) && !st(22).groups_full.test(3));
  CHECK(r.sink->unlocks.size() == 3);
  if (r.sink->unlocks.size() == 3) {
    CHECK(r.sink->unlocks[0].first == 15 && r.sink->unlocks[0].second == 2);
    CHECK(r.sink->unlocks[1].first == 16 && r.sink->unlocks[1].second == 1);
    // Satisfied at 22 on the registered head.
    CHECK(r.sink->unlocks[2].first == 24 && r.sink->unlocks[2].second == 3);
  }
  std::printf("full flags: FIFO, slots and boundaries full at +1; waits unlock at satisfied+2\n");
}

// Two waits satisfied in one cycle unlock one a cycle, lowest warp first.
void test_unlock_order() {
  Rig& r = rig("ul");
  r.at(1, [&] { r.drv->IssueOut.send(copy(3, 0, 4, 0x1), 0); });
  r.at(2, [&] { r.drv->IssueOut.send(ctrl(CpaIssue::COMMIT, 3), 0); });
  r.at(3, [&] { r.drv->IssueOut.send(ctrl(CpaIssue::WAIT, 3, 0), 0); });
  r.at(4, [&] { r.drv->IssueOut.send(copy(1, 1, 4, 0x1), 0); });
  r.at(5, [&] { r.drv->IssueOut.send(ctrl(CpaIssue::COMMIT, 1), 0); });
  r.at(6, [&] { r.drv->IssueOut.send(ctrl(CpaIssue::WAIT, 1, 0), 0); });
  r.at(10, [&] { r.drv->AckOut.send(ack(0, 0x1), 0); });
  r.at(10, [&] { r.drv->AckOut.send(ack(1, 0x1), 0); });
  r.run(20);
  CHECK(r.sink->unlocks.size() == 2);
  if (r.sink->unlocks.size() == 2) {
    // Both entries pop at 11, both waits are satisfied at 12.
    CHECK(r.sink->unlocks[0].first == 14 && r.sink->unlocks[0].second == 1);
    CHECK(r.sink->unlocks[1].first == 15 && r.sink->unlocks[1].second == 3);
  }
  std::printf("unlock order: one warp a cycle, lowest first\n");
}

// The arrive waits for the head to pass its tail and for the landed lanes;
// pending in the status from issue+1 until the cycle after its last packet.
void test_arrive_after_data() {
  Rig& r = rig("arr");
  const uint64_t mbar = LMEM + 0x40;
  r.at(1, [&] { r.drv->IssueOut.send(copy(1, 0, 8, 0x1), 0); });
  r.at(2, [&] { r.drv->IssueOut.send(arrive(1, false, all_lanes() & 0x3, mbar), 0); });
  r.at(3, [&] { r.drv->IssueOut.send(arrive(2, true, 0x1, mbar + 8), 0); });
  r.at(5, [&] { r.drv->RspOut.send(rsp(0, 0, 0x1), 0); });
  r.at(9, [&] { r.drv->AckOut.send(ack(0, 0x1), 0); });
  r.run(20);
  auto& q = r.sink->reqs;
  // Warp 2's arrive has no copy before it: it goes at 4. The write at 6;
  // warp 1's arrive after the pop at 10, at 11.
  CHECK(q.size() == 3);
  if (q.size() == 3) {
    CHECK(q[0].first == 4 && q[0].second.op == MemOp::MBAR_ARRIVE && q[0].second.tag == CpaUnit::ARRIVE_TAG);
    CHECK(q[1].first == 6 && q[1].second.op == MemOp::ST);
    CHECK(q[2].first == 11 && q[2].second.op == MemOp::MBAR_COMPLETE_TX);
    CHECK(lane_mask(q[2].second) == (all_lanes() & 0x3) && q[2].second.addrs.at(0) == mbar);
    uint64_t v = 0;
    std::memcpy(&v, q[2].second.data.at(0)->data() + (mbar % BLK), 8);
    CHECK(v == 1 && q[2].second.byteen.at(0) == (0xffull << (mbar % BLK)));
  }
  auto st = [&](uint64_t c) { return r.sink->status_at(c); };
  CHECK(!st(2).arrive_pending.test(1) && st(3).arrive_pending.test(1));
  CHECK(st(11).arrive_pending.test(1) && !st(12).arrive_pending.test(1));
  CHECK(st(4).arrive_pending.test(2) && !st(5).arrive_pending.test(2));
  std::printf("arrive: after the head passes its tail, behind landed data, pending until sent\n");
}

// An eligible arrive goes ahead of landed lanes; a request the port holds
// keeps the next one from being chosen until it is taken.
void test_arrive_first() {
  Rig& r = rig("ay");
  r.at(1, [&] { r.drv->IssueOut.send(copy(1, 0, 4, 0x1), 0); });
  r.at(2, [&] { r.drv->IssueOut.send(arrive(2, true, 0x1, LMEM + 0x80), 0); });
  r.at(2, [&] { r.drv->RspOut.send(rsp(0, 0, 0x1), 0); });
  r.sink->hold_until = 5;
  r.run(10);
  auto& q = r.sink->reqs;
  CHECK(q.size() == 2);
  if (q.size() == 2) {
    CHECK(q[0].first == 5 && q[0].second.op == MemOp::MBAR_ARRIVE);
    CHECK(q[1].first == 6 && q[1].second.op == MemOp::ST);
  }
  std::printf("arrive: ahead of landed lanes\n");
}

// A copy, a noinc arrive behind it, then 24 more copies of the same warp
// while the port takes one request in 4 cycles: their writes always wait,
// yet the arrive leaves within two port slots of its copy's ack, and the
// warp is idle only once everything, the arrive's answer included, is back.
void test_arrive_starve() {
  Rig& r = rig("as");
  constexpr uint32_t K = 24, INFLIGHT = 6;
  auto issued = std::make_shared<uint32_t>(0);
  auto slot_free_at = std::make_shared<std::vector<uint64_t>>(INFLIGHT, 0);
  auto rsp_due = std::make_shared<std::multimap<uint64_t, uint32_t>>();
  auto ack0 = std::make_shared<uint64_t>(0);
  r.sink->ready_div = 4;
  r.drv->every = [&r, issued, slot_free_at, rsp_due, ack0] {
    const uint64_t c = SimPlatform::instance().cycles();
    for (auto& e : r.sink->reqs) {
      if (e.first + 4 != c)
        continue;
      r.drv->AckOut.send(ack(e.second.tag, lane_mask(e.second)), 0);
      if (e.second.op == MemOp::ST) {
        if (e.second.tag == 0 && *ack0 == 0)
          *ack0 = c;
        slot_free_at->at(e.second.tag) = c + 2;
      }
    }
    auto due = rsp_due->equal_range(c);
    for (auto it = due.first; it != due.second; ++it) {
      r.drv->RspOut.send(rsp(it->second, 0, 0x1), 0);
    }
    if (c == 2) {
      r.drv->IssueOut.send(arrive(0, true, 0x1, LMEM + 0x40), 0);
      return;
    }
    if (c < 1 || *issued > K)
      return;
    const uint32_t slot = *issued % INFLIGHT;
    if (slot_free_at->at(slot) > c)
      return;
    r.drv->IssueOut.send(copy(0, slot, 8, 0x1, LMEM + 0x100 + 8 * (*issued % 16)), 0);
    slot_free_at->at(slot) = ~uint64_t(0);
    rsp_due->emplace(c + 3, slot);
    ++*issued;
  };
  r.run(600);
  r.drv->every = nullptr;
  uint64_t arr = 0, writes = 0;
  for (auto& e : r.sink->reqs) {
    if (e.second.op == MemOp::MBAR_ARRIVE && arr == 0)
      arr = e.first;
    writes += (e.second.op == MemOp::ST);
  }
  CHECK(*issued == K + 1 && writes == K + 1);
  CHECK(*ack0 != 0 && arr != 0 && arr <= *ack0 + 2 + 8);
  CHECK(r.cpa->idle(0));
  std::printf("arrive: leaves under write pressure (%u copies behind it), ack at %lu, arrive at %lu\n", K,
              (unsigned long)*ack0, (unsigned long)arr);
}

// Several packets (pids): each lands on its own, the lower pid first.
void test_pids() {
  if (NPIDS < 2) {
    std::printf("pids: skipped (one packet per warp)\n");
    return;
  }
  Rig& r = rig("pid");
  const uint32_t tm = (T >= 32) ? ~0u : ((1u << T) - 1);
  r.at(1, [&] { r.drv->IssueOut.send(copy(0, 2, 8, tm), 0); });
  r.at(3, [&] { r.drv->RspOut.send(rsp(2, 1, all_lanes()), 0); });
  r.at(3, [&] { r.drv->RspOut.send(rsp(2, 0, all_lanes()), 0); });
  r.at(8, [&] { r.drv->AckOut.send(ack(2, all_lanes()), 0); });
  r.at(9, [&] { r.drv->AckOut.send(ack(2, all_lanes()), 0); });
  r.run(14);
  auto& q = r.sink->reqs;
  CHECK(q.size() == 2);
  if (q.size() == 2) {
    CHECK(q[0].first == 4 && q[0].second.addrs.at(0) == LMEM + 0x100);
    CHECK(q[1].first == 5 && q[1].second.addrs.at(0) == LMEM + 0x100 + L * 8);
  }
  CHECK(!(r.sink->status_at(9).free_slots & 4) && (r.sink->status_at(10).free_slots & 4));

  // A uop issued as two packets: the first takes the slot and the FIFO entry,
  // the second fills them; acks of the first packet's lanes alone leave the
  // entry open, so the wait holds until the second packet's lanes are acked.
  Rig& m = rig("pkt");
  auto half = [](bool first) {
    CpaIssue e = copy(1, 3, 8, first ? all_lanes() : (all_lanes() << L));
    e.sop = first;
    e.eop = !first;
    return e;
  };
  m.at(1, [&] { m.drv->IssueOut.send(half(true), 0); });
  m.at(2, [&] { m.drv->IssueOut.send(half(false), 0); });
  m.at(3, [&] { m.drv->IssueOut.send(ctrl(CpaIssue::COMMIT, 1), 0); });
  m.at(4, [&] { m.drv->IssueOut.send(ctrl(CpaIssue::WAIT, 1, 0), 0); });
  m.at(5, [&] { m.drv->RspOut.send(rsp(3, 0, all_lanes()), 0); });
  m.at(9, [&] { m.drv->AckOut.send(ack(3, all_lanes()), 0); });
  m.at(12, [&] { m.drv->RspOut.send(rsp(3, 1, all_lanes()), 0); });
  m.at(16, [&] { m.drv->AckOut.send(ack(3, all_lanes()), 0); });
  m.run(26);
  const uint64_t all_slots = (uint64_t(1) << CpaUnit::SLOTS) - 1;
  CHECK(m.sink->status_at(3).free_slots == (all_slots & ~uint64_t(8)));
  CHECK(!(m.sink->status_at(16).free_slots & 8) && (m.sink->status_at(17).free_slots & 8));
  CHECK(m.sink->unlocks.size() == 1);
  if (!m.sink->unlocks.empty()) {
    // the entry pops at 17, the wait is satisfied at 18 and reaches the scheduler at 20
    CHECK(m.sink->unlocks[0].first == 20 && m.sink->unlocks[0].second == 1);
  }
  std::printf("pids: per packet, lowest pid first; the slot frees with its last lane; a uop's packets share one slot and entry\n");
}

///////////////////////////////////////////////////////////////////////////////
// Data, behind the adapter and a real local memory

// Ticks after the unit and before the local memory: one read port of its own.
class Reader : public SimObject<Reader> {
public:
  SimChannel<MemReq> ReqOut;
  SimChannel<MemRsp> RspIn;
  std::map<uint64_t, std::vector<uint8_t>> got;
  std::vector<uint64_t> todo;
  std::vector<uint64_t> sent;  // a read's tag is its index here

  Reader(const SimContext& ctx, const char* name) : SimObject<Reader>(ctx, name), ReqOut(this), RspIn(this) {}

protected:
  void on_reset() {}
  void on_tick() {
    if (!todo.empty() && !ReqOut.full()) {
      sent.push_back(todo.back());
      MemReq r(MemOp::LD, todo.back(), nullptr, 0, uint32_t(sent.size() - 1), 0, 1);
      ReqOut.send(r, 1);
      todo.pop_back();
    }
    while (!RspIn.empty()) {
      auto& rsp = RspIn.peek();
      got[sent.at(rsp.tag)].assign(rsp.data->data(), rsp.data->data() + BLK);
      RspIn.pop();
    }
  }
  friend class SimObject<Reader>;
};

struct DataRig {
  std::shared_ptr<Driver> drv;
  std::shared_ptr<CpaUnit> cpa;
  std::shared_ptr<LsuMemAdapter> adp;
  std::shared_ptr<Reader> rd;
  std::shared_ptr<LocalMem> lmem;
  std::shared_ptr<Sink> sink;

  DataRig() {
    drv  = Driver::Create("data-drv");
    cpa  = CpaUnit::Create("data-cpa", 0);
    adp  = LsuMemAdapter::Create("data-adp", L, 1);
    rd   = Reader::Create("data-rd");
    constexpr uint32_t NPORTS = L + 1;
    lmem = LocalMem::Create("data-lmem", LocalMem::Config{1u << VX_CFG_LMEM_LOG_SIZE, LSU_WORD_SIZE,
                                  NPORTS, NPORTS, log2ceil(VX_CFG_LMEM_NUM_BANKS), false, 0});
    sink = Sink::Create("data-sink");
    drv->IssueOut.bind(&cpa->IssueIn);
    drv->RspOut.bind(&cpa->RspIn);
    cpa->ReqOut.bind(&adp->ReqIn);
    adp->RspOut.bind(&cpa->AckIn);
    for (uint32_t c = 0; c < L; ++c) {
      adp->ReqOut.at(c).bind(&lmem->Inputs.at(c));
      lmem->Outputs.at(c).bind(&adp->RspIn.at(c));
    }
    rd->ReqOut.bind(&lmem->Inputs.at(L));
    lmem->Outputs.at(L).bind(&rd->RspIn);
    cpa->UnlockOut.bind(&sink->UnlockIn);
    cpa->StatusOut.bind(&sink->StatusIn);

    g_drivers.push_back(drv);
  }
};
std::unique_ptr<DataRig> g_data;

void test_data() {
  auto& drv  = g_data->drv;
  auto& rd   = g_data->rd;
  auto& sink = g_data->sink;
  for (auto& d : g_drivers) d->active = (d == drv);
  // One copy per (size, ss, dst word offset, src offset); each lane writes its own word.
  struct Case { uint32_t size, ss, src_off; uint64_t dst; };
  std::vector<Case> cases;
  uint64_t dst = LMEM + 0x400;
  for (uint32_t size : {4u, 8u}) {
    for (uint32_t ss = 0; ss <= size; ++ss) {
      for (uint32_t so : {0u, size, BLK - size}) {
        cases.push_back({size, ss, so, dst + ((size == 4 && (cases.size() & 1)) ? 4 : 0)});
        dst += 16;
      }
    }
  }
  // Fill every touched word with 0xAA through the reader's port first.
  uint64_t cyc = 2;
  uint32_t slot = 0;
  for (size_t k = 0; k < cases.size(); ++k, cyc += 12) {
    auto c = cases[k];
    uint32_t s = slot;
    slot = (slot + 1) % CpaUnit::SLOTS;
    drv->script.emplace(cyc, [&drv, c, s] {
      CpaIssue e = copy(0, s, c.size, 0x1);
      e.lanes[0].dst = c.dst;
      e.lanes[0].ss = c.ss;
      e.lanes[0].src_off = c.src_off;
      drv->IssueOut.send(e, 0);
    });
    drv->script.emplace(cyc + 3, [&drv, s, k] { drv->RspOut.send(rsp(s, 0, 0x1, uint8_t(k * 7)), 0); });
  }
  // An arrive of 2 lanes on an mbarrier expecting 2: the phase completes.
  const uint64_t mbar = LMEM + 0x80;
  drv->script.emplace(cyc, [&drv, mbar] {
    drv->IssueOut.send(arrive(3, true, 0x1 | (L > 1 ? 0x2 : 0), mbar), 0);
  });
  auto& p = SimPlatform::instance();
  p.reset();
  // Pre-state: 0xAA everywhere the copies write, the mbarrier initialised.
  for (auto& c : cases) {
    MemReq w(MemOp::ST, c.dst & ~uint64_t(7), make_mem_block(), 0xffull << (c.dst & 63 & ~7u), 0, 0, 0);
    std::memset(w.data->data(), 0xaa, BLK);
    rd->ReqOut.send(w, 1);
    p.tick();
    p.tick();
  }
  {
    uint32_t n = (L > 1) ? 2 : 1;
    MemReq w(MemOp::ST, mbar, make_mem_block(), 0xffull << (mbar & 63), 0, 0, 0);
    uint64_t v = mbar_ref::init(n);
    std::memcpy(w.data->data() + (mbar & 63), &v, 8);
    rd->ReqOut.send(w, 1);
    p.tick();
  }
  // The script's cycles count from this point.
  uint64_t base = p.cycles();
  std::multimap<uint64_t, std::function<void()>> shifted;
  for (auto& e : drv->script) shifted.emplace(e.first + base, e.second);
  drv->script.swap(shifted);
  for (uint64_t i = 0; i < cyc + 40; ++i) p.tick();
  for (auto& c : cases) rd->todo.push_back(c.dst & ~uint64_t(BLK - 1));
  rd->todo.push_back(mbar & ~uint64_t(BLK - 1));
  for (int i = 0; i < 400; ++i) p.tick();

  uint32_t bad = 0;
  for (size_t k = 0; k < cases.size(); ++k) {
    auto c = cases[k];
    auto& blk = rd->got[c.dst & ~uint64_t(BLK - 1)];
    if (blk.size() != BLK) { ++bad; continue; }
    uint32_t off = c.dst & (BLK - 1);
    uint32_t w0 = off & ~7u;
    for (uint32_t b = w0; b < w0 + 8; ++b) {
      uint8_t want = 0xaa;
      if (b >= off && b < off + c.size) {
        uint32_t i = b - off;
        want = (i < c.ss) ? uint8_t(uint8_t(k * 7) + c.src_off + i) : 0;
      }
      if (blk[b] != want) {
        ++bad;
        std::fprintf(stderr, "data: size=%u ss=%u src_off=%u dst=0x%llx byte %u = 0x%02x, want 0x%02x\n",
                     c.size, c.ss, c.src_off, (unsigned long long)c.dst, b, blk[b], want);
        break;
      }
    }
  }
  CHECK(bad == 0);
  {
    auto& blk = rd->got[mbar & ~uint64_t(BLK - 1)];
    uint64_t w = 0;
    if (blk.size() == BLK) std::memcpy(&w, blk.data() + (mbar & (BLK - 1)), 8);
    uint32_t n = (L > 1) ? 2 : 1;
    CHECK(w == (mbar_ref::init(n) | (uint64_t(1) << 63)));
  }
  CHECK(sink->status_at(~uint64_t(0) >> 1).free_slots == ((uint64_t(1) << CpaUnit::SLOTS) - 1));
  std::printf("data: %zu copies byte-exact (4/8 bytes x every src-size x 3 offsets), arrive completes the phase\n",
              cases.size());
}

} // namespace

int main() {
  for (const char* n : {"wt", "wl", "ooo", "fifo", "grp", "ul", "arr", "ay", "as", "pid", "pkt"}) {
    g_rigs[n] = std::make_unique<Rig>(n);
  }
  g_data = std::make_unique<DataRig>();
  test_write_timing();
  test_write_latency();
  test_ooo_acks();
  test_full_flags();
  test_unlock_order();
  test_arrive_after_data();
  test_arrive_first();
  test_arrive_starve();
  test_pids();
  test_data();
  if (g_fail) {
    std::printf("FAILED (%d)\n", g_fail);
    return 1;
  }
  std::printf("PASSED\n");
  return 0;
}
