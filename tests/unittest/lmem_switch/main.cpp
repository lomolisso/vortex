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

// LocalMemSwitch in a DLMEM build: lane routing (the remote flag before the
// address test), the field copies of every leg, the all-or-nothing push under
// random per-side back-pressure, and the response merge cycle by cycle against
// a transcription of VX_rr_arbiter's 3-input table (inputs {local, global,
// remote} at indices 2, 1, 0), which no parity cell puts under contention.

#include "local_mem_switch.h"
#include "mem_block_pool.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <deque>

using namespace vortex;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static constexpr uint32_t N = 8;
static const uint64_t LMEM = VX_MEM_LMEM_BASE_ADDR;
static const uint64_t WIN  = VX_MEM_DLMEM_BASE_ADDR;
static const uint64_t GLB  = 0x10000000ull;

// VX_rr_arbiter g_lut3: state = last grant, reset to 2; requests bit i = input i.
static int rr3(uint32_t state, uint32_t req) {
  for (uint32_t k = 1; k <= 3; ++k) {
    uint32_t i = (state + k) % 3;
    if (req >> i & 1) return int(i);
  }
  return -1;
}
// The literal casez table, to cross-check rr3 itself.
static int rr3_table(uint32_t state, uint32_t r) {
  auto m = [&](const char* pat) {
    // pat: 2 state bits then 3 request bits, MSB first
    uint32_t v = (state << 3) | r;
    for (int b = 0; b < 5; ++b) {
      char c = pat[b];
      uint32_t bit = v >> (4 - b) & 1;
      if (c == '?') continue;
      if ((c - '0') != int(bit)) return false;
    }
    return true;
  };
  if (m("00001") || m("010?1") || m("10??1")) return 0;
  if (m("00?1?") || m("01010") || m("10?10")) return 1;
  if (m("0010?") || m("011??") || m("10100")) return 2;
  return -1;
}

static std::shared_ptr<mem_block_t> blk(uint8_t v) {
  auto b = make_mem_block();
  std::memset(b->data(), v, b->size());
  return b;
}

int main() {
  for (uint32_t s = 0; s < 3; ++s)
    for (uint32_t r = 0; r < 8; ++r)
      CHECK(rr3(s, r) == rr3_table(s, r));

  auto& platform = SimPlatform::instance();
  auto sw = LocalMemSwitch::Create("sw", 1);
  platform.reset();

  // ---- 1. routing and field copies ----
  {
    LsuReq r(N);
    r.op = MemOp::AMO_CAS;
    r.flags.amo_unsigned = 1;
    r.tag = 0x55; r.cid = 3; r.wid = 2; r.uuid = 77; r.cta_id = 5;
    // lane: 0 remote(win) 1 shared 2 global 3 remote(win) 4 remote with a Shared address 5 inactive remote bit
    uint64_t addrs[N] = {WIN + 0x10000, LMEM + 0x40, GLB, WIN + 0x20008, LMEM + 0x80, WIN, 0, 0};
    for (uint32_t i = 0; i < 6; ++i) {
      if (i != 5) r.mask.set(i);
      r.addrs[i] = addrs[i];
      r.data[i] = blk(uint8_t(0x10 + i));
      r.byteen[i] = 0xffull << (8 * (i & 1));
      r.amo_cmp[i] = 0x1000 + i;
      r.tids[i] = 10 + i;
    }
    r.dlmem_remote.set(0); r.dlmem_remote.set(3); r.dlmem_remote.set(4); r.dlmem_remote.set(5);
    r.dlmem_core[0] = 1; r.dlmem_core[3] = 2; r.dlmem_core[4] = 3; r.dlmem_core[5] = 1;
    sw->ReqIn.send(r, 0);
    platform.tick();
    platform.tick(); // outputs land with delay 1
    CHECK(sw->ReqIn.empty());
    CHECK(!sw->ReqOutRemote.empty());
    CHECK(!sw->ReqOutLmem.empty());
    CHECK(!sw->ReqOutDC.empty());
    if (!sw->ReqOutRemote.empty()) {
      auto q = sw->ReqOutRemote.peek(); sw->ReqOutRemote.pop();
      CHECK(q.mask.test(0) && q.mask.test(3) && q.mask.test(4) && q.mask.count() == 3);
      CHECK(q.dlmem_remote.test(0) && q.dlmem_remote.test(3) && q.dlmem_remote.test(4));
      CHECK(q.dlmem_remote.count() == 3);
      for (uint32_t i : {0u, 3u, 4u}) {
        CHECK(q.addrs[i] == addrs[i]);
        CHECK(q.data[i] == r.data[i]);
        CHECK(q.byteen[i] == r.byteen[i]);
        CHECK(q.amo_cmp[i] == r.amo_cmp[i]);
        CHECK(q.tids[i] == r.tids[i]);
        CHECK(q.dlmem_core[i] == r.dlmem_core[i]);
      }
      CHECK(q.op == MemOp::AMO_CAS && q.flags.amo_unsigned == 1);
      CHECK(q.tag == 0x55 && q.cid == 3 && q.wid == 2 && q.uuid == 77 && q.cta_id == 5);
    }
    if (!sw->ReqOutLmem.empty()) {
      auto q = sw->ReqOutLmem.peek(); sw->ReqOutLmem.pop();
      CHECK(q.mask.test(1) && q.mask.count() == 1);
      CHECK(q.addrs[1] == addrs[1] && q.data[1] == r.data[1] && q.byteen[1] == r.byteen[1]);
      CHECK(q.amo_cmp[1] == r.amo_cmp[1]);
      CHECK(q.op == MemOp::AMO_CAS && q.tag == 0x55 && q.wid == 2 && q.tids[1] == 11);
    }
    if (!sw->ReqOutDC.empty()) {
      auto q = sw->ReqOutDC.peek(); sw->ReqOutDC.pop();
      CHECK(q.mask.test(2) && q.mask.count() == 1);
      CHECK(q.addrs[2] == addrs[2] && q.data[2] == r.data[2] && q.byteen[2] == r.byteen[2]);
    }
  }

  // ---- 2. all-or-nothing under random per-side back-pressure ----
  {
    std::mt19937 rng(1);
    const uint32_t NREQ = 2000;
    uint32_t sent = 0, cyc = 0;
    std::deque<uint32_t> want_rem, want_loc, want_glb; // tags expected per side, in order
    uint32_t got_rem = 0, got_loc = 0, got_glb = 0;
    bool pending = false;
    while (cyc < 200000) {
      ++cyc;
      if (!pending && sent < NREQ) {
        LsuReq r(N);
        uint32_t sides = 1 + rng() % 7; // bit0 remote, bit1 local, bit2 global
        r.tag = sent; r.uuid = sent;
        for (uint32_t i = 0; i < N; ++i) {
          uint32_t pick = rng() % 4;
          if (pick == 3) continue;
          if (!(sides >> pick & 1)) continue;
          r.mask.set(i);
          if (pick == 0) { r.addrs[i] = WIN + 8 * i; r.dlmem_remote.set(i); r.dlmem_core[i] = 1; }
          else if (pick == 1) r.addrs[i] = LMEM + 8 * i;
          else r.addrs[i] = GLB + 8 * i;
        }
        bool hr = false, hl = false, hg = false;
        for (uint32_t i = 0; i < N; ++i) if (r.mask.test(i)) {
          if (r.dlmem_remote.test(i)) hr = true; else if (get_addr_type(r.addrs[i]) == AddrType::Shared) hl = true; else hg = true;
        }
        if (hr) want_rem.push_back(sent);
        if (hl) want_loc.push_back(sent);
        if (hg) want_glb.push_back(sent);
        if (r.mask.any()) { sw->ReqIn.send(r, 0); pending = true; }
        ++sent;
      }
      platform.tick();
      if (pending && sw->ReqIn.empty()) pending = false;
      // random sink readiness per side
      auto drain = [&](SimChannel<LsuReq>& ch, std::deque<uint32_t>& want, uint32_t& got) {
        if (!ch.empty() && (rng() % 3) != 0) {
          auto q = ch.peek(); ch.pop();
          CHECK(!want.empty());
          if (!want.empty()) { CHECK(q.tag == want.front()); want.pop_front(); }
          ++got;
        }
      };
      drain(sw->ReqOutRemote, want_rem, got_rem);
      drain(sw->ReqOutLmem, want_loc, got_loc);
      drain(sw->ReqOutDC, want_glb, got_glb);
      if (sent >= NREQ && !pending && want_rem.empty() && want_loc.empty() && want_glb.empty()) break;
    }
    CHECK(want_rem.empty() && want_loc.empty() && want_glb.empty());
    std::printf("all-or-nothing: %u requests, remote=%u local=%u global=%u pushes, %u cycles\n",
                NREQ, got_rem, got_loc, got_glb, cyc);
  }

  // ---- 2b. a full side holds the whole request: no partial push ----
  {
    for (int k = 0; k < 2; ++k) { LsuReq f(N); f.mask.set(0); f.addrs[0] = WIN; f.dlmem_remote.set(0); f.tag = 900 + k; sw->ReqIn.send(f, 0); platform.tick(); }
    platform.tick();
    CHECK(sw->ReqOutRemote.size() == 2);
    LsuReq r(N); r.tag = 999;
    r.mask.set(0); r.addrs[0] = WIN; r.dlmem_remote.set(0);
    r.mask.set(1); r.addrs[1] = LMEM;
    r.mask.set(2); r.addrs[2] = GLB;
    sw->ReqIn.send(r, 0);
    for (int c = 0; c < 5; ++c) platform.tick();
    CHECK(!sw->ReqIn.empty());
    CHECK(sw->ReqOutLmem.empty() && sw->ReqOutDC.empty());
    sw->ReqOutRemote.pop();
    platform.tick(); platform.tick();
    CHECK(sw->ReqIn.empty());
    CHECK(sw->ReqOutLmem.size() == 1 && sw->ReqOutDC.size() == 1 && sw->ReqOutRemote.size() == 2);
    while (!sw->ReqOutRemote.empty()) sw->ReqOutRemote.pop();
    sw->ReqOutLmem.pop(); sw->ReqOutDC.pop();
  }

  // ---- 3. response merge, cycle by cycle, against the RTL arbiter table ----
  {
    std::mt19937 rng(7);
    // inputs: 0 remote, 1 global(DC), 2 local
    SimChannel<LsuRsp>* in[3] = {&sw->RspInRemote, &sw->RspInDC, &sw->RspInLmem};
    std::deque<uint32_t> q[3]; // model queues (tags)
    uint32_t state = 2;        // RTL reset value; test 1 never granted a response
    uint32_t next_tag = 0, out_cnt = 0, grants[3] = {0, 0, 0}, mism = 0;
    uint32_t cap_out = 2;
    std::deque<uint32_t> model_out; // model of RspOut occupancy
    for (uint32_t cyc = 0; cyc < 20000; ++cyc) {
      // producers: push with delay 0 (seen this cycle), respecting channel capacity
      for (uint32_t i = 0; i < 3; ++i) {
        bool heavy = (cyc / 2000) % 2 == 0;
        uint32_t p = (i == 0) ? (heavy ? 30 : 70) : 80;
        if ((rng() % 100) < p && !in[i]->full()) {
          LsuRsp r(N); r.tag = (uint64_t(i) << 32) | next_tag++; r.mask.set(0);
          in[i]->send(r, 0);
          q[i].push_back(uint32_t(r.tag));
        }
      }
      // model grant for this cycle
      uint32_t req = 0;
      for (uint32_t i = 0; i < 3; ++i) if (!q[i].empty()) req |= 1u << i;
      int g = -1;
      if (model_out.size() < cap_out) {
        g = rr3(state, req);
        if (g >= 0) { model_out.push_back(q[g].front()); q[g].pop_front(); state = g; ++grants[g]; }
      }
      platform.tick();
      // consumer: random readiness, pops one per cycle
      bool rdy = (rng() % 100) < 85;
      if (rdy && !sw->RspOut.empty()) {
        auto r = sw->RspOut.peek(); sw->RspOut.pop();
        if (model_out.empty() || model_out.front() != uint32_t(r.tag)) ++mism;
        if (!model_out.empty()) model_out.pop_front();
        ++out_cnt;
      }
      // model RspOut lands with delay 1: keep the model aligned to SimX visibility
      CHECK(sw->RspOut.size() <= 2);
    }
    CHECK(mism == 0);
    std::printf("merge: %u responses out, grants remote=%u global=%u local=%u, mismatches=%u\n",
                out_cnt, grants[0], grants[1], grants[2], mism);
  }

  std::printf(g_fail ? "FAILED (%d)\n" : "PASSED\n", g_fail);
  sw.reset();
  platform.finalize();
  return g_fail ? 1 : 0;
}
