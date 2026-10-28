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
// Then LocalMem's per-bank commit pulse: one per access of an LSU port that the
// translation counted (flags.lacc), sent in the tick the bank takes it, with
// delay 1 for a load or store and delay 3 for an atomic (its write-back stage,
// registered); none for uncounted accesses or the ingress or DMA ports. With
// mbarriers: every op against mbar_ref, waits that neither write nor leave a
// read-during-write shadow, reservations, the assert texts, the waiter-table
// events (an insert for a try_wait that reads an incomplete phase, taken at
// accept+1 with its warp; a flip for an op that completes one, at accept+2;
// at most one per bank and cycle), and a random op stream on a few words. Last, the dcache's CTA fence: at an L1 that is not the
// shared level it drains an older write of another port, keeps the lines (a
// later read hits), and sends nothing below marked as a fence; a GPU fence at
// the same cache still walks.

#include "local_mem_switch.h"
#include "local_mem.h"
#include "mem_block_pool.h"
#include "cache.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <deque>
#include <functional>
#include <map>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#ifdef VX_CFG_EXT_MBAR_ENABLE
#include "mbar_ref.h"
#endif

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

// Records the tick each commit pulse is taken; created after the memory, as the core's tracker.
class CommitProbe : public SimObject<CommitProbe> {
public:
  using Ptr = std::shared_ptr<CommitProbe>;
  struct Pulse { uint64_t tick; uint32_t bank; uint32_t wid; };
  std::vector<SimChannel<uint32_t>> In;
  std::vector<Pulse> pulses;
  CommitProbe(const SimContext& ctx, const char* name, uint32_t n)
    : SimObject<CommitProbe>(ctx, name), In(n, this) {}
protected:
  void on_reset() { pulses.clear(); }
  void on_tick() {
    for (uint32_t b = 0; b < In.size(); ++b) {
      while (!In[b].empty()) {
        pulses.push_back({SimPlatform::instance().cycles(), b, In[b].peek()});
        In[b].pop();
      }
    }
  }
  friend class SimObject<CommitProbe>;
};

#ifdef VX_CFG_EXT_MBAR_ENABLE
// Records the tick each mbarrier event is taken, as the waiter table does.
class EvtProbe : public SimObject<EvtProbe> {
public:
  using Ptr = std::shared_ptr<EvtProbe>;
  struct Event { uint64_t tick; uint32_t bank; MbarBankEvt evt; };
  std::vector<SimChannel<MbarBankEvt>> In;
  std::vector<Event> events;
  EvtProbe(const SimContext& ctx, const char* name, uint32_t n)
    : SimObject<EvtProbe>(ctx, name), In(n, this) {}
protected:
  void on_reset() { events.clear(); }
  void on_tick() {
    for (uint32_t b = 0; b < In.size(); ++b) {
      while (!In[b].empty()) {
        events.push_back({SimPlatform::instance().cycles(), b, In[b].peek()});
        In[b].pop();
      }
    }
  }
  friend class SimObject<EvtProbe>;
};
#endif

// The memory below a cache: logs each request with its tick, answers reads.
class MemModel : public SimObject<MemModel> {
public:
  using Ptr = std::shared_ptr<MemModel>;
  struct Entry { uint64_t tick; MemReq req; };
  SimChannel<MemReq> ReqIn;
  SimChannel<MemRsp> RspOut;
  std::vector<Entry> log;
  MemModel(const SimContext& ctx, const char* name)
    : SimObject<MemModel>(ctx, name), ReqIn(this), RspOut(this) {}
protected:
  void on_reset() { log.clear(); }
  void on_tick() {
    while (!ReqIn.empty()) {
      auto& req = ReqIn.peek();
      log.push_back({SimPlatform::instance().cycles(), req});
      if (!req.is_write()) {
        auto b = make_mem_block();
        std::memset(b->data(), 0x5a, b->size());
        RspOut.send(MemRsp{req.tag, req.hart_id, req.uuid, b}, 4);
      }
      ReqIn.pop();
    }
  }
  friend class SimObject<MemModel>;
};

static std::shared_ptr<mem_block_t> blk(uint8_t v) {
  auto b = make_mem_block();
  std::memset(b->data(), v, b->size());
  return b;
}

static uint64_t word_at(const std::shared_ptr<mem_block_t>& b, uint64_t a) {
  uint64_t v = 0;
  if (b) {
    std::memcpy(&v, b->data() + (a & 63), 8);
  }
  return v;
}

constexpr uint32_t NLSU = 4, NB = 4, INGRESS = NLSU, DMA = NLSU + 1;

static LocalMem::Config lmem_cfg() {
  return LocalMem::Config{1u << 14, LSU_WORD_SIZE, NLSU + 2, DMA, 2, false, NLSU};
}

static uint64_t lmem_addr(uint32_t bank, uint32_t row) {
  return LMEM + (uint64_t(row) * NB + bank) * LSU_WORD_SIZE;
}

#ifdef VX_CFG_EXT_MBAR_ENABLE
// A block carrying one 8-byte word at the address's line offset.
static std::shared_ptr<mem_block_t> word_blk(uint64_t a, uint64_t v) {
  auto b = blk(0);
  std::memcpy(b->data() + (a & 63), &v, 8);
  return b;
}

// Runs `body` in a child process; returns its stderr and whether it aborted.
static std::string run_child(const std::function<void()>& body, bool* aborted) {
  int fds[2];
  if (pipe(fds) != 0) {
    std::fprintf(stderr, "pipe failed\n");
    std::exit(1);
  }
  std::fflush(stdout);
  std::fflush(stderr);
  pid_t pid = fork();
  if (pid == 0) {
    close(fds[0]);
    dup2(fds[1], STDERR_FILENO);
    body();
    _exit(0);
  }
  close(fds[1]);
  std::string err;
  char buf[256];
  ssize_t n;
  while ((n = read(fds[0], buf, sizeof(buf))) > 0) {
    err.append(buf, n);
  }
  close(fds[0]);
  int status = 0;
  waitpid(pid, &status, 0);
  *aborted = WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
  return err;
}

static MemOp mbar_memop(uint32_t op) {
  switch (op) {
  case mbar_ref::OP_ARRIVE:      return MemOp::MBAR_ARRIVE;
  case mbar_ref::OP_ARRIVE_TX:   return MemOp::MBAR_ARRIVE_TX;
  case mbar_ref::OP_EXPECT_TX:   return MemOp::MBAR_EXPECT_TX;
  case mbar_ref::OP_COMPLETE_TX: return MemOp::MBAR_COMPLETE_TX;
  case mbar_ref::OP_TEST_WAIT:   return MemOp::MBAR_TEST_WAIT;
  default:                       return MemOp::MBAR_TRY_WAIT;
  }
}

// One erroneous op on a fresh memory: the bank must abort with `msg`.
static void mbar_negative(const char* name, uint64_t init, uint32_t op, uint64_t operand,
                          uint64_t byteen, const char* msg) {
  bool aborted = false;
  std::string err = run_child([&]() {
    auto& platform = SimPlatform::instance();
    auto lmem = LocalMem::Create("neg", lmem_cfg());
    platform.reset();
    uint64_t a = lmem_addr(1, 3);
    MemReq st(MemOp::ST, a, word_blk(a, init), 0xffull << (a & 63), 0, make_hart_id(0, 1, 0), 1);
    lmem->Inputs.at(0).send(st, 1);
    for (int k = 0; k < 4; ++k) platform.tick();
    MemReq r(mbar_memop(op), a, word_blk(a, operand), byteen << (a & 63), 1, make_hart_id(0, 1, 0), 2);
    lmem->Inputs.at(0).send(r, 1);
    for (int k = 0; k < 8; ++k) {
      platform.tick();
      while (!lmem->Outputs.at(0).empty()) lmem->Outputs.at(0).pop();
      while (!lmem->CommitOut.at(1).empty()) lmem->CommitOut.at(1).pop();
      while (!lmem->MbarEvtOut.at(1).empty()) lmem->MbarEvtOut.at(1).pop();
    }
  }, &aborted);
  CHECK(aborted);
  CHECK(err.find(msg) != std::string::npos);
  if (!aborted || err.find(msg) == std::string::npos) {
    std::fprintf(stderr, "mbar negative %s: aborted=%d, stderr: %s\n", name, aborted, err.c_str());
  } else {
    std::printf("mbar negative %s: aborts with \"%s\"\n", name, msg);
  }
}
#endif

// The dcache's CTA fence, at an L1 that is not the shared level.
static __attribute__((noinline)) void test_cta_fence() {
  auto& platform = SimPlatform::instance();
  Cache::Config cfg{};
  cfg.bypass = false;
  cfg.C = 12; cfg.L = 6; cfg.S = 6; cfg.W = 3; cfg.A = 2; cfg.B = 0;
  cfg.addr_width = 64; cfg.num_inputs = 2; cfg.mem_ports = 1;
  cfg.write_back = false; cfg.write_reponse = false; cfg.mshr_size = 8;
  cfg.latency = 2; cfg.repl_policy = Cache::FIFO; cfg.is_llc = false;
  cfg.fence_shared = false;
  cfg.fence_cta = true;
  auto l1 = Cache::Create("l1", cfg);
  auto mem = MemModel::Create("mem");
  l1->mem_req_out.at(0).bind(&mem->ReqIn);
  mem->RspOut.bind(&l1->mem_rsp_in.at(0));
  platform.reset();
  for (int k = 0; k < 80; ++k) platform.tick(); // tag init

  const uint64_t X = GLB + 0x1000, Y = GLB + 0x2000;
  std::map<uint32_t, uint64_t> rsp_tick; // tag -> tick its response left the cache
  auto step = [&](int n = 1) {
    for (int k = 0; k < n; ++k) {
      platform.tick();
      for (uint32_t p = 0; p < 2; ++p) {
        while (!l1->core_rsp_out.at(p).empty()) {
          rsp_tick[uint32_t(l1->core_rsp_out.at(p).peek().tag)] = platform.cycles();
          l1->core_rsp_out.at(p).pop();
        }
      }
    }
  };
  auto read = [&](uint32_t port, uint64_t a, uint32_t tag) {
    l1->core_req_in.at(port).send(MemReq(MemOp::LD, a, nullptr, 0xff, tag), 0);
  };
  auto fence = [&](uint32_t port, bool cta, uint32_t tag) {
    MemReq f(MemOp::FLUSH, 0, nullptr, 0, tag);
    f.flags.fence = 1;
    f.flags.fence_cta = cta;
    l1->core_req_in.at(port).send(f, 0);
  };
  auto mem_reads = [&](uint64_t a) {
    uint32_t n = 0;
    for (auto& e : mem->log) n += !e.req.is_write() && (e.req.addr & ~uint64_t(63)) == a;
    return n;
  };
  auto any_fence_below = [&]() {
    bool f = false;
    for (auto& e : mem->log) f = f || e.req.flags.fence || e.req.flags.fence_cta;
    return f;
  };

  // A line the fence must keep.
  read(0, Y, 1);
  step(40);
  CHECK(rsp_tick.count(1) && mem_reads(Y) == 1);

  // An older write on port 1, then a CTA fence on port 0 in the next tick:
  // the write leaves for memory before the fence answers.
  MemReq w(MemOp::ST, X, blk(0x77), 0xff, 2);
  l1->core_req_in.at(1).send(w, 0);
  step();
  fence(0, true, 3);
  step(60);
  uint64_t w_tick = 0;
  for (auto& e : mem->log) if (e.req.is_write() && e.req.addr == X) w_tick = e.tick;
  CHECK(w_tick != 0);
  CHECK(rsp_tick.count(3) && rsp_tick[3] > w_tick);
  CHECK(!any_fence_below());

  // No walk: the line is still there.
  read(0, Y, 4);
  step(40);
  CHECK(rsp_tick.count(4) && mem_reads(Y) == 1);

  // A GPU fence at this level walks: the line is gone, and its own miss
  // carries the fence below.
  fence(1, false, 5);
  step(200);
  CHECK(rsp_tick.count(5));
  read(0, Y, 6);
  step(40);
  CHECK(rsp_tick.count(6) && mem_reads(Y) == 2);
  CHECK(any_fence_below());
  std::printf("cta fence: write sent at %llu, fence answered at %llu; %zu memory requests\n",
              (unsigned long long)w_tick, (unsigned long long)rsp_tick[3], mem->log.size());
}

int main() {
  for (uint32_t s = 0; s < 3; ++s)
    for (uint32_t r = 0; r < 8; ++r)
      CHECK(rr3(s, r) == rr3_table(s, r));

#ifdef VX_CFG_EXT_MBAR_ENABLE
  // Fork before any object exists.
  mbar_negative("underflow", mbar_ref::init(2), mbar_ref::OP_ARRIVE, 3, 0xff, mbar_ref::MSG_UNDERFLOW);
  mbar_negative("arrive_tx-underflow", mbar_ref::pack(0, 1, 8, 0), mbar_ref::OP_ARRIVE_TX, 0, 0xff,
                mbar_ref::MSG_UNDERFLOW);
  mbar_negative("tx-range", mbar_ref::init(1), mbar_ref::OP_EXPECT_TX, mbar_ref::TX_MAX + 1, 0xff,
                mbar_ref::MSG_TX_RANGE);
  mbar_negative("tx-range-big", mbar_ref::init(1), mbar_ref::OP_COMPLETE_TX, uint64_t(1) << 40, 0xff,
                mbar_ref::MSG_TX_RANGE);
  mbar_negative("align", mbar_ref::init(1), mbar_ref::OP_ARRIVE, 1, 0x0f, mbar_ref::MSG_ALIGN);
  mbar_negative("align-wait", mbar_ref::init(1), mbar_ref::OP_TEST_WAIT, 0, 0xf0, mbar_ref::MSG_ALIGN);
#endif

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

  // ---- 4. LocalMem commit pulses ----
  {
    const uint32_t word = LSU_WORD_SIZE;
    auto lmem = LocalMem::Create("lmem", lmem_cfg());
    auto probe = CommitProbe::Create("probe", NB);
    CHECK(lmem->CommitOut.size() == NB);
    for (uint32_t b = 0; b < NB; ++b) {
      lmem->CommitOut.at(b).bind(&probe->In.at(b));
    }
  #ifdef VX_CFG_EXT_MBAR_ENABLE
    auto evts = EvtProbe::Create("evts", NB);
    CHECK(lmem->MbarEvtOut.size() == NB);
    for (uint32_t b = 0; b < NB; ++b) {
      lmem->MbarEvtOut.at(b).bind(&evts->In.at(b));
    }
  #endif
    platform.reset();
    uint64_t cyc = platform.cycles();
    std::map<uint64_t, uint64_t> rsps;  // tag -> word at the request's address
    std::map<uint64_t, uint64_t> rsp_addr;
    uint64_t next_tag = 1;
    auto step = [&](uint32_t n = 1) {
      for (uint32_t k = 0; k < n; ++k) {
        platform.tick();
        ++cyc;
        for (uint32_t p = 0; p < NLSU + 2; ++p) {
          while (!lmem->Outputs.at(p).empty()) {
            auto& r = lmem->Outputs.at(p).peek();
            rsps[r.tag] = word_at(r.data, rsp_addr[r.tag]);
            lmem->Outputs.at(p).pop();
          }
        }
      }
    };
    auto addr = [&](uint32_t bank, uint32_t row) { return lmem_addr(bank, row); };
    // counted: the translation counted the access (flags.lacc); only those pulse.
    auto req = [&](MemOp op, uint32_t bank, uint32_t row, uint32_t wid, uint32_t tid, bool dma = false,
                   bool counted = true) {
      uint64_t a = addr(bank, row);
      auto data = blk(uint8_t(0x30 + wid));
      MemReq r(op, a, op == MemOp::LD ? nullptr : data,
               op == MemOp::LD ? 0 : ((uint64_t(1) << word) - 1) << (a & 63), next_tag, make_hart_id(5, wid, tid), cyc);
      rsp_addr[next_tag++] = a;
      r.flags.local = 1;
      r.flags.dma = dma;
      r.flags.lacc = counted && !dma;
      return r;
    };
    // Pulses taken in the window, as (tick - t0, bank, wid).
    auto taken = [&](size_t from) {
      std::vector<CommitProbe::Pulse> v(probe->pulses.begin() + from, probe->pulses.end());
      return v;
    };

    // A solo store: the bank writes it in tick t0+W; its pulse is still in
    // flight after that tick and is taken in tick t0+W+1.
    uint64_t t0 = cyc;
    lmem->Inputs.at(1).send(req(MemOp::ST, 2, 0, 3, 1), 1);
    uint64_t seen = 0;
    for (uint32_t k = 0; k < 8 && !seen; ++k) {
      step();
      if (lmem->CommitOut.at(2).size() != 0 && probe->pulses.empty()) {
        seen = cyc - 1;  // the tick that just ran
      }
    }
    step(4);
    CHECK(seen != 0);
    CHECK(probe->pulses.size() == 1);
    uint64_t W = seen - t0;
    if (probe->pulses.size() == 1) {
      CHECK(probe->pulses[0].bank == 2 && probe->pulses[0].wid == 3);
      CHECK(probe->pulses[0].tick == seen + 1);
    }

    // Ingress and DMA accesses: no pulse. An LSU load pulses like a store.
    size_t base = probe->pulses.size();
    t0 = cyc;
    lmem->Inputs.at(INGRESS).send(req(MemOp::ST, 0, 1, 1, 0), 1);
    lmem->Inputs.at(DMA).send(req(MemOp::ST, 1, 1, 1, 0, true), 1);
    lmem->Inputs.at(INGRESS + 0).send(req(MemOp::LD, 2, 1, 1, 0), 1);
    step(8);
    CHECK(probe->pulses.size() == base);
    t0 = cyc;
    lmem->Inputs.at(0).send(req(MemOp::LD, 3, 1, 1, 0), 1);
    step(8);
    {
      auto v = taken(base);
      CHECK(v.size() == 1);
      if (v.size() == 1) {
        CHECK(v[0].tick == t0 + W + 1 && v[0].bank == 3 && v[0].wid == 1);
      }
    }
    base = probe->pulses.size();
    t0 = cyc;
    for (uint32_t p = 0; p < NLSU; ++p) {
      lmem->Inputs.at(p).send(req(MemOp::ST, p, 2, p, p), 1);
    }
    step(8);
    {
      auto v = taken(base);
      CHECK(v.size() == NLSU);
      for (uint32_t k = 0; k < v.size(); ++k) {
        CHECK(v[k].tick == t0 + W + 1 && v[k].bank == k && v[k].wid == k);
      }
    }

    // Two LSU stores to one bank: one write per cycle, so pulses one tick apart.
    // A DMA write to the bank in the same cycle wins it and delays both.
    for (bool dma : {false, true}) {
      base = probe->pulses.size();
      t0 = cyc;
      lmem->Inputs.at(0).send(req(MemOp::ST, 1, 3, 2, 0), 1);
      lmem->Inputs.at(3).send(req(MemOp::ST, 1, 4, 1, 3), 1);
      if (dma) {
        lmem->Inputs.at(DMA).send(req(MemOp::ST, 1, 5, 0, 0, true), 1);
      }
      step(10);
      auto v = taken(base);
      CHECK(v.size() == 2);
      if (v.size() == 2) {
        CHECK(v[0].bank == 1 && v[1].bank == 1 && v[0].wid + v[1].wid == 3);
        CHECK(v[0].tick == t0 + W + 1 + dma && v[1].tick == t0 + W + 2 + dma);
      }
    }
    size_t plain_pulses = 1 + 1 + NLSU + 2 + 2;
    CHECK(probe->pulses.size() == plain_pulses);

  #if VX_CFG_EXT_A_ENABLED
    // An atomic pulses from its write-back stage: taken 3 ticks after the
    // bank takes it. Each kind, alone on bank 0.
    {
      const uint32_t hid = 2;
      std::vector<std::pair<MemOp, uint32_t>> ops = {
        {MemOp::AMO_ADD, 2}, {MemOp::AMO_LR, 1}, {MemOp::AMO_SC, 1}, {MemOp::AMO_SC, 1},
        {MemOp::AMO_CAS, 3}, {MemOp::AMO_SWAP, 0}};
      for (auto& o : ops) {
        base = probe->pulses.size();
        t0 = cyc;
        lmem->Inputs.at(o.second).send(req(o.first, 0, 6, hid, o.second), 1);
        step(10);
        auto v = taken(base);
        CHECK(v.size() == 1);
        if (v.size() == 1) {
          CHECK(v[0].tick == t0 + W + 3 && v[0].bank == 0 && v[0].wid == hid);
        }
      }
      // The SCs: the first one succeeds, the second one finds no reservation.
      CHECK(rsps[next_tag - 4] == 0 && rsps[next_tag - 3] == 1);
      // From the ingress: no pulse.
      base = probe->pulses.size();
      lmem->Inputs.at(INGRESS).send(req(MemOp::AMO_ADD, 0, 6, 1, 0), 1);
      step(10);
      CHECK(probe->pulses.size() == base);
      // An atomic, then an LSU load of another word of its bank sent a tick
      // later: the load waits out the occupancy and is taken at T+3, so the
      // pulses come one tick apart and never together.
      base = probe->pulses.size();
      t0 = cyc;
      lmem->Inputs.at(1).send(req(MemOp::AMO_ADD, 3, 6, 1, 1), 1);
      step();
      lmem->Inputs.at(2).send(req(MemOp::LD, 3, 7, 2, 2), 1);
      step(10);
      auto v = taken(base);
      CHECK(v.size() == 2);
      if (v.size() == 2) {
        CHECK(v[0].tick == t0 + W + 3 && v[0].wid == 1 && v[0].bank == 3);
        CHECK(v[1].tick == t0 + W + 4 && v[1].wid == 2 && v[1].bank == 3);
      }
      plain_pulses += ops.size() + 2;
    }
    CHECK(probe->pulses.size() == plain_pulses);
  #endif

    // Accesses of an LSU port the translation did not count (the TCU metadata
    // loads share port 0 and bypass it): performed, never pulsed.
    {
      base = probe->pulses.size();
      std::vector<MemOp> ops = {MemOp::LD, MemOp::ST};
    #if VX_CFG_EXT_A_ENABLED
      for (auto op : {MemOp::AMO_ADD, MemOp::AMO_LR, MemOp::AMO_SC})
        ops.push_back(op);
    #endif
    #ifdef VX_CFG_EXT_MBAR_ENABLE
      ops.push_back(MemOp::MBAR_TEST_WAIT);
    #endif
      uint32_t answered = 0;
      for (uint32_t p : {0u, NLSU - 1}) {
        for (MemOp op : ops) {
          uint64_t tag = next_tag;
          lmem->Inputs.at(p).send(req(op, 1, 8, 2, p, false, false), 1);
          step(10);
          answered += (op == MemOp::ST) || rsps.count(tag) != 0;
        }
      }
      CHECK(answered == 2 * ops.size());
      CHECK(probe->pulses.size() == base);
      std::printf("commit: %zu uncounted accesses from LSU ports performed without a pulse\n", 2 * ops.size());
    }
    for (uint32_t b = 0; b < NB; ++b) {
      CHECK(probe->In.at(b).size() == 0);
    }
    std::printf("commit: one pulse per LSU access, taken %u cycles after the request is sent "
                "(bank access +1) for loads and stores%s, none for ingress, DMA: %zu pulses\n",
                uint32_t(W + 1), VX_CFG_EXT_A_ENABLED ? ", +3 for atomics" : "", probe->pulses.size());

  #ifdef VX_CFG_EXT_MBAR_ENABLE
    // ---- 5. mbarrier ops ----
    auto exec = [&](uint32_t port, MemOp op, uint64_t a, uint64_t operand, uint32_t wid, uint32_t tid) {
      uint64_t tag = next_tag++;
      rsp_addr[tag] = a;
      bool is_ld = (op == MemOp::LD);
      MemReq r(op, a, is_ld ? nullptr : word_blk(a, operand), is_ld ? 0 : (0xffull << (a & 63)), tag,
               make_hart_id(5, wid, tid), tag);
      r.flags.lacc = 1;
      lmem->Inputs.at(port).send(r, 1);
      for (uint32_t k = 0; k < 12 && !rsps.count(tag); ++k) {
        step();
      }
      CHECK(rsps.count(tag) == 1);
      uint64_t v = rsps[tag];
      rsps.erase(tag);
      return v;
    };
    auto write = [&](uint64_t a, uint64_t v) {
      uint64_t tag = next_tag++;
      MemReq r(MemOp::ST, a, word_blk(a, v), 0xffull << (a & 63), tag, make_hart_id(5, 0, 0), tag);
      lmem->Inputs.at(INGRESS).send(r, 1);
      step(4);
    };
    using namespace mbar_ref;
    {
      struct Vec { const char* name; uint64_t old; uint32_t op; uint64_t operand; };
      const Vec vecs[] = {
        {"arrive",                 init(3),                          OP_ARRIVE,      1},
        {"arrive-last",            (init(3) & ~COUNT_MAX) | 1,         OP_ARRIVE,      1},
        {"arrive-count",           init(5),                          OP_ARRIVE,      3},
        {"arrive-count-all",       init(5),                          OP_ARRIVE,      5},
        {"arrive-phase1",          pack(1, 1, 0, 1),                 OP_ARRIVE,      1},
        {"arrive-count-max",       init(uint32_t(COUNT_MAX)),        OP_ARRIVE,      COUNT_MAX},
        {"arrive-pending-tx",      pack(1, 1, 64, 0),                OP_ARRIVE,      1},
        {"arrive-reserved",        init(2) | (uint64_t(3) << 61),    OP_ARRIVE,      1},
        {"arrive_tx",              init(1),                          OP_ARRIVE_TX,   64},
        {"arrive_tx-zero",         init(2),                          OP_ARRIVE_TX,   0},
        {"arrive_tx-complete",     pack(1, 2, -32, 0),               OP_ARRIVE_TX,   32},
        {"expect_tx",              init(2),                          OP_EXPECT_TX,   100},
        {"expect_tx-max",          init(2),                          OP_EXPECT_TX,   uint64_t(TX_MAX)},
        {"expect_tx-flip",         pack(0, 1, -32, 0),               OP_EXPECT_TX,   32},
        {"complete_tx",            pack(1, 1, 64, 0),                OP_COMPLETE_TX, 16},
        {"complete_tx-negative",   init(1),                          OP_COMPLETE_TX, 32},
        {"complete_tx-flip",       pack(0, 1, 64, 1),                OP_COMPLETE_TX, 64},
        {"complete_tx-min",        pack(0, 1, TX_MAX, 0),            OP_COMPLETE_TX, uint64_t(2 * TX_MAX)},
        {"test_wait-0",            init(2),                          OP_TEST_WAIT,   init(2)},
        {"test_wait-1",            pack(2, 2, 0, 1),                 OP_TEST_WAIT,   init(2)},
        {"test_wait-1b",           init(2),                          OP_TEST_WAIT,   pack(0, 0, 0, 1)},
        {"test_wait-0b",           pack(2, 2, 0, 1),                 OP_TEST_WAIT,   ~uint64_t(0)},
        {"try_wait-0",             init(4),                          OP_TRY_WAIT,    0},
        {"try_wait-1",             pack(4, 4, 8, 1),                 OP_TRY_WAIT,    0},
      };
      uint32_t n = 0, flips = 0;
      for (auto& v : vecs) {
        uint32_t port = n % NLSU;
        uint64_t a = lmem_addr(n % NB, 20 + n);
        write(a, v.old);
        auto ref = apply(v.op, v.old, v.operand);
        CHECK(ref.err == ERR_NONE);
        size_t ebase = evts->events.size();
        uint64_t ret = exec(port, mbar_memop(v.op), a, v.operand, 1, port);
        step(4);
        {
          bool insert = (v.op == OP_TRY_WAIT) && (ref.ret_word == 0);
          size_t want = (ref.flip || insert) ? 1 : 0;
          CHECK(evts->events.size() - ebase == want);
          if (evts->events.size() - ebase == 1) {
            auto& e = evts->events.back();
            CHECK(e.bank == n % NB && e.evt.row == 20 + n && e.evt.flip == ref.flip);
            CHECK(ref.flip || e.evt.wid == 1);
          }
        }
        uint64_t now = exec(port, MemOp::LD, a, 0, 1, port);
        bool ok = (ret == ref.ret_word) && (now == (ref.writes ? ref.new_word : v.old));
        if (is_wait(v.op)) {
          ok = ok && (ret <= 1) && !ref.writes;
        }
        CHECK(ok);
        if (!ok) {
          std::fprintf(stderr, "mbar vector %s: old=0x%016llx operand=0x%llx ret=0x%llx (want 0x%llx)"
                       " word=0x%016llx (want 0x%016llx)\n", v.name, (unsigned long long)v.old,
                       (unsigned long long)v.operand, (unsigned long long)ret,
                       (unsigned long long)ref.ret_word, (unsigned long long)now,
                       (unsigned long long)(ref.writes ? ref.new_word : v.old));
        }
        flips += ref.flip;
        ++n;
      }
      // Hand-checked anchors for the golden model itself.
      CHECK(apply(OP_ARRIVE, (init(3) & ~COUNT_MAX) | 1, 1).new_word == (init(3) | (uint64_t(1) << 63)));
      CHECK(tx(apply(OP_COMPLETE_TX, init(1), 32).new_word) == -32);
      CHECK(apply(OP_EXPECT_TX, pack(0, 1, -32, 0), 32).flip);
      CHECK(!apply(OP_ARRIVE_TX, init(1), 64).flip);
      std::printf("mbar vectors: %u ops vs mbar_ref, %u phase flips\n", n, flips);
    }

    // Waiter-table events: an op taken in tick T flips at T+2 and a try_wait
    // that reads the phase incomplete inserts at T+1, with its warp; a
    // try_wait that reads it complete and a test_wait send nothing.
    {
      struct Ev { uint32_t op; uint64_t old; uint64_t operand; int32_t at; bool flip; };
      const Ev evs[] = {
        {OP_ARRIVE,      init(4),          4, 2, true},
        {OP_ARRIVE,      init(4),          1, -1, false},
        {OP_COMPLETE_TX, pack(0, 2, 8, 0), 8, 2, true},
        {OP_TRY_WAIT,    init(4),          0, 1, false},
        {OP_TRY_WAIT,    pack(4, 4, 0, 1), 0, -1, false},
        {OP_TEST_WAIT,   init(4),          0, -1, false},
      };
      for (auto& v : evs) {
        uint64_t a = lmem_addr(3, 62);
        write(a, v.old);
        size_t ebase = evts->events.size();
        t0 = cyc;
        uint64_t tag = next_tag++;
        rsp_addr[tag] = a;
        MemReq mr(mbar_memop(v.op), a, word_blk(a, v.operand), 0xffull << (a & 63), tag,
                  make_hart_id(5, 2, 3), tag);
        mr.flags.lacc = 1;
        lmem->Inputs.at(3).send(mr, 1);
        step(10);
        rsps.erase(tag);
        size_t got = evts->events.size() - ebase;
        CHECK(got == (v.at >= 0 ? 1u : 0u));
        if (got == 1 && v.at >= 0) {
          auto& e = evts->events.back();
          CHECK(e.tick == t0 + W + uint64_t(v.at) && e.bank == 3 && e.evt.row == 62 && e.evt.flip == v.flip);
          CHECK(v.flip || e.evt.wid == 2);
        }
      }
      std::printf("mbar events: insert at accept+1 with the warp, flip at accept+2\n");
    }

    // The occupancy and shadow: an op, then an LSU load of the same word sent
    // a tick later. The op pulses at T+3; the load is taken at T+3 behind a
    // wait (no write, no shadow) and at T+4 behind a writing op.
    for (uint32_t op : {uint32_t(OP_TEST_WAIT), uint32_t(OP_ARRIVE), uint32_t(OP_TRY_WAIT), uint32_t(OP_EXPECT_TX)}) {
      uint64_t a = lmem_addr(2, 60);
      write(a, init(4));
      base = probe->pulses.size();
      t0 = cyc;
      uint64_t tag = next_tag++;
      rsp_addr[tag] = a;
      MemReq mr(mbar_memop(op), a, word_blk(a, 1), 0xffull << (a & 63), tag, make_hart_id(5, 3, 1), tag);
      mr.flags.lacc = 1;
      lmem->Inputs.at(1).send(mr, 1);
      step();
      uint64_t ltag = next_tag++;
      rsp_addr[ltag] = a;
      MemReq lr(MemOp::LD, a, nullptr, 0, ltag, make_hart_id(5, 2, 2), ltag);
      lr.flags.lacc = 1;
      lmem->Inputs.at(2).send(lr, 1);
      step(10);
      bool w = !is_wait(op);
      auto v = taken(base);
      CHECK(v.size() == 2);
      if (v.size() == 2) {
        CHECK(v[0].tick == t0 + W + 3 && v[0].wid == 3 && v[0].bank == 2);
        CHECK(v[1].tick == t0 + W + 4 + w && v[1].wid == 2 && v[1].bank == 2);
      }
      CHECK(rsps[ltag] == apply(op, init(4), 1).new_word);
      CHECK(rsps[tag] == apply(op, init(4), 1).ret_word);
    }
    std::printf("mbar occupancy: pulses at accept+3; a load of the word waits 3 cycles behind a wait, "
                "4 behind a writing op\n");

    // A writing op breaks another hart's reservation; a wait does not.
    for (uint32_t op : {uint32_t(OP_ARRIVE), uint32_t(OP_TEST_WAIT), uint32_t(OP_COMPLETE_TX)}) {
      uint64_t a = lmem_addr(1, 61);
      write(a, init(4));
      exec(0, MemOp::AMO_LR, a, 0, 1, 0);
      exec(1, mbar_memop(op), a, 1, 2, 1);
      uint64_t sc = exec(0, MemOp::AMO_SC, a, 0x1234, 1, 0);
      uint64_t now = exec(0, MemOp::LD, a, 0, 1, 0);
      if (is_wait(op)) {
        CHECK(sc == 0 && now == 0x1234);
      } else {
        CHECK(sc == 1 && now == apply(op, init(4), 1).new_word);
      }
    }
    std::printf("mbar reservations: a writing op breaks an LR, a wait does not\n");

    // A random stream on three words (two share a bank) from every LSU port and
    // the ingress, one op in flight per word, against DMA writes and plain
    // traffic to other words of the same banks.
    {
      std::mt19937_64 rng(11);
      const uint64_t words[3] = {lmem_addr(0, 70), lmem_addr(0, 71), lmem_addr(1, 70)};
      const uint64_t plain[2] = {lmem_addr(0, 72), lmem_addr(1, 73)};
      uint64_t model[3];
      struct Pend { uint64_t tag; uint64_t ret; bool check; };
      std::map<uint32_t, Pend> pend;  // per word
      for (uint32_t w = 0; w < 3; ++w) {
        model[w] = init(1 + uint32_t(rng() % 4));
        write(words[w], model[w]);
      }
      base = probe->pulses.size();
      size_t ebase = evts->events.size();
      uint64_t want_pulses = 0, ops = 0, waits = 0, flips = 0, loads = 0, dmas = 0, inserts = 0;
      std::vector<uint64_t> tokens[3];
      for (uint32_t c = 0; c < 4000; ++c) {
        for (uint32_t w = 0; w < 3; ++w) {
          if (pend.count(w) || (rng() % 100) >= 40) {
            continue;
          }
          uint32_t port = uint32_t(rng() % (NLSU + 1));
          if (lmem->Inputs.at(port).full()) {
            continue;
          }
          uint64_t old = model[w];
          int64_t t = tx(old);
          uint32_t pick = uint32_t(rng() % 7);
          uint32_t op = 0;
          uint64_t operand = 0;
          MemOp mop = MemOp::LD;
          if (pick == 0) {
            mop = MemOp::LD;
          } else if (pick == 1) {
            op = (rng() & 1) ? OP_TEST_WAIT : OP_TRY_WAIT;
            operand = tokens[w].empty() ? rng() : tokens[w][rng() % tokens[w].size()];
          } else if (pick == 2 && pending(old) != 0) {
            op = OP_ARRIVE;
            operand = 1 + rng() % pending(old);
          } else if (pick == 3 && pending(old) != 0) {
            op = OP_ARRIVE_TX;
            operand = (t > 0) ? 0 : uint64_t(-t) + rng() % 64;
          } else if (pick == 4) {
            op = OP_EXPECT_TX;
            operand = (t > 2048) ? 0 : rng() % 512;
            if (pending(old) == 0 && t < 0 && (rng() & 1)) {
              operand = uint64_t(-t);
            }
          } else {
            op = OP_COMPLETE_TX;
            operand = (t < -2048) ? 0 : rng() % 512;
            if (pending(old) == 0 && t > 0 && (rng() & 1)) {
              operand = uint64_t(t);
            }
          }
          uint64_t ret;
          if (op == 0) {
            ret = old;
            ++loads;
          } else {
            mop = mbar_memop(op);
            auto r = apply(op, old, operand);
            CHECK(r.err == ERR_NONE);
            ret = r.ret_word;
            if (r.writes) {
              model[w] = r.new_word;
              tokens[w].push_back(old);
            } else {
              ++waits;
            }
            flips += r.flip;
            inserts += (op == OP_TRY_WAIT) && (r.ret_word == 0);
            ++ops;
          }
          uint64_t tag = next_tag++;
          rsp_addr[tag] = words[w];
          uint32_t wid = uint32_t(rng() % VX_CFG_NUM_WARPS);
          bool is_ld = (op == 0);
          MemReq mr(mop, words[w], is_ld ? nullptr : word_blk(words[w], operand),
                    is_ld ? 0 : (0xffull << (words[w] & 63)), tag, make_hart_id(5, wid, port), tag);
          mr.flags.lacc = (rng() % 4) != 0;
          lmem->Inputs.at(port).send(mr, 1);
          want_pulses += (port < NLSU) && mr.flags.lacc;
          pend[w] = {tag, ret, true};
        }
        // Plain stores and loads from LSU ports, and DMA beats, on the other words.
        if ((rng() % 100) < 30) {
          uint32_t port = uint32_t(rng() % NLSU);
          uint64_t a = plain[rng() % 2];
          bool st = rng() & 1;
          if (!lmem->Inputs.at(port).full()) {
            uint64_t tag = next_tag++;
            rsp_addr[tag] = a;
            MemReq mr(st ? MemOp::ST : MemOp::LD, a, st ? word_blk(a, rng()) : nullptr,
                      st ? (0xffull << (a & 63)) : 0, tag, make_hart_id(5, 0, port), tag);
            mr.flags.lacc = (rng() % 4) != 0;
            lmem->Inputs.at(port).send(mr, 1);
            want_pulses += mr.flags.lacc;
          }
        }
        if ((rng() % 100) < 25 && !lmem->Inputs.at(DMA).full()) {
          uint64_t a = plain[rng() % 2];
          MemReq d(MemOp::ST, a, word_blk(a, rng()), 0xffull << (a & 63), next_tag++, 0, 0);
          d.flags.dma = 1;
          lmem->Inputs.at(DMA).send(d, 1);
          ++dmas;
        }
        step();
        for (uint32_t w = 0; w < 3; ++w) {
          auto it = pend.find(w);
          if (it == pend.end() || !rsps.count(it->second.tag)) {
            continue;
          }
          uint64_t got = rsps[it->second.tag];
          if (got != it->second.ret) {
            CHECK(got == it->second.ret);
            std::fprintf(stderr, "mbar random: word %u tag %llu returned 0x%016llx, want 0x%016llx\n", w,
                         (unsigned long long)it->second.tag, (unsigned long long)got,
                         (unsigned long long)it->second.ret);
          }
          rsps.erase(it->second.tag);
          pend.erase(it);
        }
      }
      for (uint32_t k = 0; k < 20 && !pend.empty(); ++k) {
        step();
        for (uint32_t w = 0; w < 3; ++w) {
          auto it = pend.find(w);
          if (it != pend.end() && rsps.count(it->second.tag)) {
            CHECK(rsps[it->second.tag] == it->second.ret);
            pend.erase(it);
          }
        }
      }
      CHECK(pend.empty());
      step(8);
      // One pulse per LSU-port access, never two on one bank in one tick.
      auto v = taken(base);
      CHECK(v.size() == want_pulses);
      std::map<std::pair<uint64_t, uint32_t>, uint32_t> per_tick;
      for (auto& p : v) {
        CHECK(++per_tick[std::make_pair(p.tick, p.bank)] == 1);
      }
      // One event per flip and per incomplete try_wait, never two on one bank in one tick.
      uint64_t got_flips = 0, got_inserts = 0;
      std::map<std::pair<uint64_t, uint32_t>, uint32_t> evt_tick;
      for (size_t k = ebase; k < evts->events.size(); ++k) {
        auto& e = evts->events[k];
        got_flips += e.evt.flip;
        got_inserts += !e.evt.flip;
        CHECK(++evt_tick[std::make_pair(e.tick, e.bank)] == 1);
      }
      CHECK(got_flips == flips && got_inserts == inserts);
      for (uint32_t w = 0; w < 3; ++w) {
        CHECK(exec(w % NLSU, MemOp::LD, words[w], 0, 0, 0) == model[w]);
      }
      std::printf("mbar random: %llu ops (%llu waits, %llu flips, %llu inserts), %llu loads, %llu DMA beats, "
                  "%zu pulses\n", (unsigned long long)ops, (unsigned long long)waits, (unsigned long long)flips,
                  (unsigned long long)inserts, (unsigned long long)loads, (unsigned long long)dmas, v.size());
    }
  #endif
  }

  test_cta_fence();

  std::printf(g_fail ? "FAILED (%d)\n" : "PASSED\n", g_fail);
  sw.reset();
  platform.finalize();
  return g_fail ? 1 : 0;
}
