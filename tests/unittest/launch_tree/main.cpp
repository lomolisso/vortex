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

// SimX launch bus (LaunchTree) against hand-derived cycles of the RTL's
// VX_kmu_bus_arb fan-out: round robin over the ready outputs, a message held on
// its output until eop, pinned beats routed by destination, the skids' ready
// registered (an endpoint's pop is seen a cycle later), and an internal skid
// showing its beat one cycle after it enters.

#include <cstdio>
#include <vector>
#include "launch_tree.h"

using namespace vortex;

namespace {

int errors = 0;

void expect(bool cond, const char* what) {
  if (!cond) {
    std::printf("FAIL: %s\n", what);
    ++errors;
  }
}

struct Arrival {
  uint64_t cycle;
  uint32_t leaf;
  uint32_t tag;
};

// Drives a tree one cycle at a time, the way its owner does: credits, the ready
// latch, the internal skids, then at most one new beat from the source.
struct Bench {
  LaunchTree tree;
  uint64_t now = 0;
  std::vector<Arrival> got;

  explicit Bench(const std::vector<LaunchTree::Level>& levels) : tree(levels) {}

  // Returns whether `beat` was taken this cycle.
  bool cycle(const std::vector<uint32_t>& credits, const LaunchTree::Beat* beat) {
    for (auto leaf : credits) {
      tree.credit(leaf);
    }
    auto sink = [this](uint32_t leaf, const kmu_req_t& req) {
      got.push_back({now, leaf, req.cta_id});
    };
    tree.begin_cycle();
    tree.advance(now, sink);
    bool taken = false;
    if (beat && tree.ready(*beat)) {
      tree.push(*beat, now, sink);
      taken = true;
    }
    ++now;
    return taken;
  }
};

LaunchTree::Beat beat(uint32_t tag, bool eop = true, bool pinned = false, uint32_t dest = 0) {
  LaunchTree::Beat b{};
  b.req.cta_id = tag;
  b.eop = eop;
  b.pinned = pinned;
  b.dest = dest;
  return b;
}

// One level of two skids: round robin over the ready outputs, and a full skid
// that frees only when the endpoint's credit has arrived.
void t_round_robin() {
  Bench b({LaunchTree::Level(2)});
  for (uint32_t i = 0; i < 4; ++i) {
    auto bt = beat(i);
    expect(b.cycle({}, &bt), "rr: a beat is taken while a skid has room");
  }
  expect(b.got.size() == 4, "rr: four beats reach the leaves the cycle they are taken");
  uint32_t want[4] = {0, 1, 0, 1};
  for (uint32_t i = 0; i < 4 && i < b.got.size(); ++i) {
    expect(b.got[i].leaf == want[i] && b.got[i].cycle == i, "rr: leaves alternate");
  }
  auto b4 = beat(4);
  expect(!b.cycle({}, &b4), "rr: both skids full, the beat waits");
  // the endpoint of leaf 1 pops; its credit arrives a cycle later
  expect(b.cycle({1}, &b4), "rr: the credit frees leaf 1");
  expect(b.got.back().leaf == 1 && b.got.back().tag == 4, "rr: the nearest ready output wins over the pointer");
  auto b5 = beat(5);
  expect(!b.cycle({}, &b5), "rr: no credit, no room");
  expect(b.cycle({0}, &b5), "rr: the credit frees leaf 0");
  expect(b.got.back().leaf == 0, "rr: the beat takes the freed output");
}

// A three-beat message stays on the output its first beat took, even when that
// output is full and the other one has room; the pointer moves only at eop.
void t_lock() {
  Bench b({LaunchTree::Level(2)});
  auto m0 = beat(0, false), m1 = beat(1, false), m2 = beat(2, true);
  expect(b.cycle({}, &m0) && b.cycle({}, &m1), "lock: the first two beats are taken");
  expect(b.got[0].leaf == 0 && b.got[1].leaf == 0, "lock: the message keeps output 0");
  expect(!b.cycle({}, &m2), "lock: output 0 is full and the message may not move");
  expect(b.cycle({0}, &m2), "lock: the last beat goes once output 0 has room");
  expect(b.got.back().leaf == 0, "lock: the eop beat stays on output 0");
  auto n = beat(3);
  expect(b.cycle({}, &n) && b.got.back().leaf == 1, "lock: after eop the pointer moved past output 0");
}

// Two levels of two: a beat crosses an internal skid in one cycle, and pinned
// beats ignore the pointers but advance them on eop.
void t_two_levels_pinned() {
  Bench b({LaunchTree::Level(2), LaunchTree::Level(2)});
  auto p = beat(0, true, true, 3);
  expect(b.cycle({}, &p), "pinned: taken");
  expect(b.got.empty(), "pinned: still in the level-0 skid");
  b.cycle({}, nullptr);
  expect(b.got.size() == 1 && b.got[0].leaf == 3 && b.got[0].cycle == 1, "pinned: reaches leaf 3 a cycle later");
  // level 0 sent to its output 1 and now points at 0; node 1 of level 1 points at 0
  auto q = beat(1);
  b.cycle({}, &q);
  b.cycle({}, nullptr);
  expect(b.got.size() == 2 && b.got[1].leaf == 0, "pinned: the next compute beat follows the advanced pointers");
  auto r = beat(2);
  b.cycle({}, &r);
  b.cycle({}, nullptr);
  expect(b.got.size() == 3 && b.got[2].leaf == 2, "pinned: then the other socket, at its node's pointer");
  auto s = beat(3, true, true, 0);
  b.cycle({}, &s);
  b.cycle({}, nullptr);
  expect(b.got.size() == 4 && b.got[3].leaf == 0, "pinned: routed to a busy-pointer leaf by destination");
}

// A registered single-output level is a skid of two in front of the endpoint;
// with no registered level the endpoint is one two-entry stage.
void t_registered_single() {
  Bench b({LaunchTree::Level(1, true)});
  auto a = beat(0), c = beat(1), d = beat(2);
  expect(b.cycle({}, &a) && b.cycle({}, &c), "single: two beats fill the skid");
  expect(!b.cycle({}, &d), "single: the third waits");
  expect(!b.cycle({}, &d), "single: still full without a credit");
  expect(b.cycle({0}, &d), "single: a credit frees a place");
  Bench w({LaunchTree::Level(1), LaunchTree::Level(1)});
  expect(w.tree.num_leaves() == 1, "wire: one endpoint");
  expect(w.cycle({}, &a) && w.cycle({}, &c) && !w.cycle({}, &d), "wire: the endpoint stage holds two");
}

// An internal skid's ready is registered: the cycle it drains a beat it still
// reads full to its source.
void t_registered_internal() {
  Bench b({LaunchTree::Level(2), LaunchTree::Level(2)});
  auto p = beat(0, true, true, 0);
  for (int i = 0; i < 4; ++i) {
    expect(b.cycle({}, &p), "internal: pinned beats fill leaf 0 and its skid");
  }
  expect(!b.cycle({}, &p), "internal: leaf 0 and the skid in front of it are full");
  expect(!b.cycle({0}, &p), "internal: the skid drains into the freed leaf but reads full this cycle");
  expect(b.cycle({}, &p), "internal: the next cycle it has room");
}

// Three outputs: the scan starts at the pointer and wraps.
void t_three_outputs() {
  Bench b({LaunchTree::Level(3)});
  for (uint32_t i = 0; i < 6; ++i) {
    auto bt = beat(i);
    b.cycle({}, &bt);
  }
  // leaves 0,1,2,0,1,2 are full; free 1 and 2, pointer is at 0
  auto x = beat(6);
  b.cycle({2, 1}, &x);
  expect(b.got.back().leaf == 1, "three: from pointer 0, output 1 is the nearest ready");
  auto y = beat(7);
  b.cycle({}, &y);
  expect(b.got.back().leaf == 2, "three: then output 2");
}

} // namespace

int main() {
  t_round_robin();
  t_lock();
  t_two_levels_pinned();
  t_registered_single();
  t_registered_internal();
  t_three_outputs();
  if (errors) {
    std::printf("FAILED (%d)\n", errors);
    return 1;
  }
  std::printf("PASSED\n");
  return 0;
}
