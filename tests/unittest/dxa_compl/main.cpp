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

// DxaCompl behind a real LocalMem, the way the core wires them. A copy's DMA
// writes enter the local memory's DMA ports; a tap between the local memory
// and the unit records the cycle T each write commits at its bank. Checked:
// a copy completes only once all its writes have committed, its final write
// first or not, a bank held by an atomic delaying one of them; a barrier
// event reaches the barrier unit at T+2 of the last commit; with mbarriers, a
// complete_tx of the box bytes executes at the bank and its phase flip lands
// at T+4, the RTL's write-back; a burst of copies completes in order, one per
// cycle; when the queue is full, the local memory's DXA input holds a copy's
// final write and lets it commit in the cycle after an entry leaves, so the
// queue never holds more than its depth (two copies a cycle against one
// release a cycle, and complete_tx requests held back by a busy bank); and the
// retired-slot and bounds checks abort with the shared texts.

#include "local_mem.h"
#include "dxa_compl.h"
#include "mem_block_pool.h"
#ifdef DXA_MBAR_ENABLE
#include "mbar_ref.h"
#endif
#include <algorithm>
#include <cinttypes>
#include <csignal>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

using namespace vortex;

#ifndef VX_CFG_EXT_DXA_ENABLE
#error "the DXA completion unit needs a DXA build"
#endif

namespace {

constexpr uint32_t NUM_BANKS  = VX_CFG_LMEM_NUM_BANKS;
constexpr uint32_t LMEM_SIZE  = 1u << VX_CFG_LMEM_LOG_SIZE;
constexpr uint32_t BLOCK      = VX_CFG_MEM_BLOCK_SIZE;
constexpr uint32_t WORD       = VX_CFG_XLEN / 8;
constexpr uint32_t DEPTH      = DxaCompl::DEPTH;

// The local memory's ports, as the core numbers them in an mbarrier build.
constexpr uint32_t PORT_LSU   = 0;
constexpr uint32_t PORT_COMPL = 2;
constexpr uint32_t PORT_DMA   = 3;
constexpr uint32_t NUM_DMA    = 2;
constexpr uint32_t NUM_PORTS  = PORT_DMA + NUM_DMA;

constexpr uint32_t SLOT_STRIDE = LMEM_SIZE / 4;

std::string strf(const char* fmt, ...) {
  char buf[512];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  return buf;
}

#ifdef DXA_MBAR_ENABLE
uint32_t bank_of(uint64_t addr) {
  return uint32_t((addr / WORD) % NUM_BANKS);
}
#endif

// Between the local memory and the unit, ticking between them: records the
// commit cycle of every DXA write and forwards it in the same cycle.
class Tap : public SimObject<Tap> {
public:
  struct Rec {
    uint64_t cycle;
    DxaWrCommit c;
  };

  SimChannel<DxaWrCommit> In;
  SimChannel<DxaWrCommit> Out;
  std::vector<Rec> recs;

  Tap(const SimContext& ctx, const char* name)
    : SimObject<Tap>(ctx, name), In(this, 4 * NUM_BANKS), Out(this, 4 * NUM_BANKS) {}

protected:
  void on_reset() { recs.clear(); }
  void on_tick() {
    while (!In.empty()) {
      recs.push_back({SimPlatform::instance().cycles(), In.peek()});
      Out.send(In.peek(), 0);
      In.pop();
    }
  }
  friend class SimObject<Tap>;
};

// The barrier unit's end of the release link.
class BarSink : public SimObject<BarSink> {
public:
  struct Rel {
    uint64_t cycle;
    uint32_t bar;
  };

  SimEventLink<uint32_t> In;
  std::vector<Rel> rels;

  BarSink(const SimContext& ctx, const char* name) : SimObject<BarSink>(ctx, name), In(this) {
    In.bind(this, &BarSink::on_rel);
  }

protected:
  void on_reset() { rels.clear(); }

private:
  void on_rel(const uint32_t& bar) { rels.push_back({SimPlatform::instance().cycles(), bar}); }
  friend class SimObject<BarSink>;
};

// One copy into one block: its writes, the last of them carrying the completion.
struct Copy {
  uint32_t xfer = 0;
  uint32_t raw_bar = 0;
  bool     mbar = false;
  uint32_t mbar_off = 0;
  uint32_t bytes = 0;
  uint32_t slot = 0;
  uint32_t gen = 0;
  std::vector<uint64_t> blocks;   // block addresses, in emission order
};

class Bench {
public:
  Bench() {
    LocalMem::Config cfg{};
    cfg.capacity  = LMEM_SIZE;
    cfg.line_size = WORD;
    cfg.num_reqs  = NUM_PORTS;
    cfg.dma_begin = PORT_DMA;
    cfg.B         = log2ceil(NUM_BANKS);
    cfg.write_reponse = false;
#ifdef VX_CFG_EXT_DLMEM_ENABLE
    cfg.commit_ports = 1;
#endif
#ifdef DXA_MBAR_ENABLE
    cfg.dxa_compl_port = PORT_COMPL;
#endif
    lmem_  = LocalMem::Create("lmem", cfg);
    tap_   = Tap::Create("tap");
    compl_ = DxaCompl::Create("dxa_compl", NUM_BANKS);
    bars_  = BarSink::Create("bars");

    lmem_->DxaWrOut.bind(&tap_->In);
    tap_->Out.bind(&compl_->WrIn);
    compl_->TxbarOut.bind(&bars_->In);
    compl_->FreeOut.bind(&lmem_->DxaFreeIn);
#ifdef DXA_MBAR_ENABLE
    compl_->ReqOut.bind(&lmem_->Inputs.at(PORT_COMPL));
    lmem_->Outputs.at(PORT_COMPL).bind(&compl_->RspIn);
#endif
    pending_.resize(NUM_PORTS);
    SimPlatform::instance().reset();
  }

  uint64_t now() const { return SimPlatform::instance().cycles(); }

  // One cycle; what the unit and the banks delivered is collected after it.
  void tick() {
    for (uint32_t p = 0; p < NUM_PORTS; ++p) {
      auto& q = pending_.at(p);
      if (!q.empty() && !lmem_->Inputs.at(p).full()) {
        lmem_->Inputs.at(p).send(q.front(), 0);
        q.erase(q.begin());
      }
    }
    SimPlatform::instance().tick();
    for (uint32_t p = 0; p < NUM_PORTS; ++p) {
      if (p == PORT_COMPL && compl_bound())
        continue;
      auto& ch = lmem_->Outputs.at(p);
      while (!ch.empty()) {
        if (p == PORT_LSU) {
          lsu_rsp_.push_back(ch.peek());
        }
        ch.pop();
      }
    }
#ifdef VX_CFG_EXT_DLMEM_ENABLE
    for (auto& ch : lmem_->CommitOut) {
      while (!ch.empty()) ch.pop();
    }
#endif
#ifdef DXA_MBAR_ENABLE
    for (uint32_t b = 0; b < NUM_BANKS; ++b) {
      auto& ch = lmem_->MbarEvtOut.at(b);
      while (!ch.empty()) {
        if (ch.peek().flip) {
          flips_.push_back({this->now(), b});
        }
        ch.pop();
      }
    }
#endif
  }

  static constexpr bool compl_bound() {
#ifdef DXA_MBAR_ENABLE
    return true;
#else
    return false;
#endif
  }

  bool idle() const {
    for (auto& q : pending_) {
      if (!q.empty()) return false;
    }
    return true;
  }

  void run(uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
      this->tick();
    }
  }

  // A request into the local memory's port p, taken by its input from the next tick.
  void issue(uint32_t port, const MemReq& req) { pending_.at(port).push_back(req); }

  // A copy's write, a beat of its own; the final write's beat ends the copy.
  void dma(uint32_t port, const Copy& c, uint64_t block, bool final_write, uint32_t writes) {
    MemReq req;
    req.op      = MemOp::ST;
    req.addr    = block;
    req.byteen  = ~uint64_t(0) >> (64 - BLOCK);
    req.data    = make_mem_block();
    for (uint32_t i = 0; i < BLOCK; ++i) {
      (*req.data)[i] = uint8_t(block + i);
    }
    req.uuid      = c.xfer;
    req.flags.dma = 1;
    req.dxa.xfer  = c.xfer;
    req.dxa.beat  = beat_seq_++;
    req.dxa.last  = final_write;
#ifdef DXA_MBAR_ENABLE
    req.dxa.mode     = c.mbar;
    req.dxa.mbar_off = c.mbar_off;
    req.dxa.tx_bytes = c.bytes;
    req.dxa.slot     = c.slot;
    req.dxa.gen      = c.gen;
#endif
    if (final_write) {
      req.flags.dxa_notify_done   = 1;
      req.flags.dxa_notify_bar_id = c.raw_bar;
      req.dxa.writes = writes;
    }
    this->issue(port, req);
  }

  MemReq word_req(MemOp op, uint64_t addr, uint64_t value) {
    MemReq req;
    req.op     = op;
    req.addr   = addr;
    uint32_t off = uint32_t(addr & (BLOCK - 1));
    req.data   = make_mem_block();
    std::memset(req.data->data(), 0, req.data->size());
    std::memcpy(req.data->data() + off, &value, sizeof(value));
    req.byteen = uint64_t(0xff) << off;
    req.hart_id = 1;
    return req;
  }

  // Reads a doubleword through the LSU port.
  uint64_t load(uint64_t addr) {
    lsu_rsp_.clear();
    MemReq req;
    req.op = MemOp::LD;
    req.addr = addr;
    this->issue(PORT_LSU, req);
    for (int i = 0; i < 32 && lsu_rsp_.empty(); ++i) {
      this->tick();
    }
    if (lsu_rsp_.empty()) {
      this->fail(strf("load 0x%" PRIx64 ": no response", addr));
      return 0;
    }
    uint64_t v = 0;
    std::memcpy(&v, lsu_rsp_.front().data->data() + (addr & (BLOCK - 1)), sizeof(v));
    lsu_rsp_.clear();
    return v;
  }

#ifdef DXA_MBAR_ENABLE
  void admit(uint32_t slot, uint32_t gen) {
    CtaSlotEvt e;
    e.kind = CtaSlotEvt::ADMIT;
    e.slot = slot;
    e.gen = gen;
    e.stride = SLOT_STRIDE;
    compl_->SlotIn.send(e, 0);
  }
#endif

  // The commit cycle of every write of copy x; 0 when one is missing.
  uint64_t last_commit(uint32_t xfer, uint32_t writes) const {
    uint64_t last = 0;
    uint32_t n = 0;
    for (auto& r : tap_->recs) {
      if (r.c.tag.xfer == xfer) {
        last = std::max(last, r.cycle);
        ++n;
      }
    }
    return (n == writes) ? last : 0;
  }

  uint64_t first_commit(uint32_t xfer) const {
    for (auto& r : tap_->recs) {
      if (r.c.tag.xfer == xfer) return r.cycle;
    }
    return 0;
  }

  // The commit cycle of every DXA write, oldest first.
  std::vector<uint64_t> commit_cycles() const {
    std::vector<uint64_t> v;
    for (auto& r : tap_->recs) v.push_back(r.cycle);
    return v;
  }

  bool final_committed_first(uint32_t xfer) const {
    for (auto& r : tap_->recs) {
      if (r.c.tag.xfer == xfer) return r.c.notify;
    }
    return false;
  }

  void fail(const std::string& msg) {
    ++failures_;
    if (failures_ <= 50) {
      std::printf("FAIL %s\n", msg.c_str());
    }
  }

  void check(bool ok, const std::string& msg) {
    if (!ok) this->fail(msg);
  }

  uint32_t failures() const { return failures_; }
  const std::vector<BarSink::Rel>& rels() const { return bars_->rels; }
  void clear_rels() { bars_->rels.clear(); }
#ifdef DXA_MBAR_ENABLE
  struct Flip {
    uint64_t cycle;
    uint32_t bank;
  };
  const std::vector<Flip>& flips() const { return flips_; }
  void clear_flips() { flips_.clear(); }
#endif
  uint32_t pending(uint32_t port) const { return uint32_t(pending_.at(port).size()); }

private:
  LocalMem::Ptr lmem_;
  std::shared_ptr<Tap> tap_;
  DxaCompl::Ptr compl_;
  std::shared_ptr<BarSink> bars_;
#ifdef DXA_MBAR_ENABLE
  std::vector<Flip> flips_;
#endif
  std::vector<std::vector<MemReq>> pending_;
  std::vector<MemRsp> lsu_rsp_;
  uint32_t failures_ = 0;
  uint32_t beat_seq_ = 0;
};

uint32_t g_xfer = 1;

Copy make_copy(uint32_t raw_bar, std::vector<uint64_t> blocks) {
  Copy c;
  c.xfer = g_xfer++;
  c.raw_bar = raw_bar;
  c.blocks = std::move(blocks);
  c.bytes = uint32_t(c.blocks.size() * BLOCK);
  c.slot = 0;
  c.gen = 1;
  return c;
}

uint32_t decoded(uint32_t raw) {
  return bar_decode_id(raw, VX_CFG_NUM_BARRIERS);
}

// Every DMA write lands in the bank of its block address; an atomic on that
// bank holds it for the atomic's occupancy.
uint64_t blk(uint32_t i) {
  return uint64_t(i) * BLOCK + 0x400;
}

void legacy_single(Bench& tb) {
  tb.clear_rels();
  Copy c = make_copy((3u << 8) | 1u, {blk(0)});
  tb.dma(PORT_DMA, c, c.blocks[0], true, 1);
  tb.run(12);
  uint64_t t = tb.last_commit(c.xfer, 1);
  tb.check(t != 0, "legacy single: the write did not commit");
  tb.check(tb.rels().size() == 1, strf("legacy single: %zu releases", tb.rels().size()));
  if (!tb.rels().empty()) {
    tb.check(tb.rels()[0].cycle == t + 2,
             strf("legacy single: released at %" PRIu64 ", commit at %" PRIu64 " (want +2)", tb.rels()[0].cycle, t));
    tb.check(tb.rels()[0].bar == decoded(c.raw_bar),
             strf("legacy single: bar %u, want %u", tb.rels()[0].bar, decoded(c.raw_bar)));
  }
  std::printf("  legacy, one write: commit %" PRIu64 ", barrier release +2\n", t);
}

#if VX_CFG_EXT_A_ENABLED
// The final write commits first; the other write waits behind an atomic on its bank.
void legacy_held(Bench& tb) {
  tb.clear_rels();
  Copy c = make_copy(2u << 8, {blk(1), blk(2)});
  const uint64_t other = c.blocks[0];
  // An atomic on the other write's bank, a word away from the copy.
  tb.issue(PORT_LSU, tb.word_req(MemOp::AMO_ADD, (other & ~uint64_t(BLOCK - 1)) + LMEM_SIZE / 2, 1));
  tb.tick();
  tb.dma(PORT_DMA, c, c.blocks[1], true, 2);
  tb.dma(PORT_DMA + 1, c, other, false, 0);
  tb.run(16);
  uint64_t t = tb.last_commit(c.xfer, 2);
  tb.check(t != 0, "legacy held: a write did not commit");
  tb.check(tb.rels().size() == 1, strf("legacy held: %zu releases", tb.rels().size()));
  if (!tb.rels().empty()) {
    tb.check(tb.rels()[0].cycle == t + 2,
             strf("legacy held: released at %" PRIu64 ", last commit at %" PRIu64 " (want +2)", tb.rels()[0].cycle, t));
  }
  std::printf("  legacy, two writes, final first=%d: commits %" PRIu64 "..%" PRIu64 ", release +2 of the last\n",
              tb.final_committed_first(c.xfer), tb.first_commit(c.xfer), t);
}
#endif

// DEPTH single-write copies, one per cycle: in order, one release per cycle.
void legacy_burst(Bench& tb) {
  tb.clear_rels();
  std::vector<Copy> cs;
  for (uint32_t i = 0; i < DEPTH; ++i) {
    cs.push_back(make_copy(i + 1, {blk(8 + i)}));
  }
  for (auto& c : cs) {
    tb.dma(PORT_DMA, c, c.blocks[0], true, 1);
    tb.tick();
  }
  tb.run(16);
  tb.check(tb.rels().size() == cs.size(), strf("legacy burst: %zu releases of %zu", tb.rels().size(), cs.size()));
  for (size_t i = 0; i < cs.size() && i < tb.rels().size(); ++i) {
    uint64_t t = tb.last_commit(cs[i].xfer, 1);
    tb.check(tb.rels()[i].bar == decoded(cs[i].raw_bar), strf("legacy burst: release %zu out of order", i));
    tb.check(tb.rels()[i].cycle == t + 2,
             strf("legacy burst: copy %zu released at %" PRIu64 ", commit %" PRIu64, i, tb.rels()[i].cycle, t));
  }
  std::printf("  legacy, burst of %u: one release per cycle, each +2\n", DEPTH);
}

// Two single-write copies a cycle, on the two DMA ports and two banks,
// against one release a cycle: the queue fills. A final write the crossbar
// forwards in cycle c commits at c+1, the RTL's accept cycle, and the RTL
// accepts a held beat once the queue's full flag, the count at the end of
// the cycle before, drops. With two writes a cycle the free entries bound
// them: the writes committing at c are as many of the waiting ones as the
// queue had free entries at the end of c-1.
void legacy_full(Bench& tb) {
  if (NUM_BANKS < 16) {
    std::printf("  legacy, two copies a cycle: skipped, the DMA blocks share one bank at %u banks\n", NUM_BANKS);
    return;
  }
  tb.clear_rels();
  const uint32_t n = 2 * DEPTH + 6;
  std::vector<Copy> cs;
  std::vector<uint64_t> issued;
  for (uint32_t i = 0; i < n; ++i) {
    cs.push_back(make_copy(i + 1, {blk(64 + i)}));
  }
  for (uint32_t i = 0; i < n; i += 2) {
    for (uint32_t k = 0; k < 2 && i + k < n; ++k) {
      tb.dma(PORT_DMA + k, cs[i + k], cs[i + k].blocks[0], true, 1);
      issued.push_back(tb.now());
    }
    tb.tick();
  }
  tb.run(4 * n);
  tb.check(tb.rels().size() == n, strf("legacy full: %zu releases of %u", tb.rels().size(), n));
  std::vector<uint64_t> commits;
  for (auto& c : cs) commits.push_back(tb.last_commit(c.xfer, 1));
  std::vector<uint64_t> pops;   // an entry leaves the cycle before its release lands
  for (auto& r : tb.rels()) pops.push_back(r.cycle - 1);
  auto count_at_end = [&](uint64_t c) {
    uint32_t in = 0, out = 0;
    for (auto x : commits) in  += (x != 0 && x <= c);
    for (auto p : pops)    out += (p <= c);
    return in - out;
  };
  uint32_t peak = 0, held = 0;
  for (uint64_t c = issued.front() + 1; c <= tb.rels().back().cycle; ++c) {
    const uint32_t before = count_at_end(c - 1);
    peak = std::max(peak, count_at_end(c));
    uint32_t now = 0, waiting = 0;
    for (size_t i = 0; i < n; ++i) {
      now     += (commits[i] == c);
      waiting += (issued[i] + 1 <= c && commits[i] >= c);
    }
    tb.check(count_at_end(c) <= DEPTH, strf("legacy full: %u entries at cycle %" PRIu64, count_at_end(c), c));
    tb.check(now == std::min(waiting, DEPTH - before),
             strf("legacy full: %u final writes committed at %" PRIu64 ", %u waiting, %u entries free",
                  now, c, waiting, DEPTH - before));
    held += (now < waiting);
  }
  tb.check(held != 0, "legacy full: no final write waited for an entry");
  std::printf("  legacy, two copies a cycle: the queue holds up to %u, final writes wait in %u cycles\n",
              peak, held);
}

#ifdef DXA_MBAR_ENABLE
constexpr uint64_t MBAR_ADDR = 0x108;   // in slot 0, off the copies' banks

void mbar_init(Bench& tb, uint32_t tx_bytes) {
  tb.issue(PORT_LSU, tb.word_req(MemOp::ST, MBAR_ADDR, mbar_ref::init(1)));
  tb.run(4);
  tb.issue(PORT_LSU, tb.word_req(MemOp::MBAR_ARRIVE_TX, MBAR_ADDR, tx_bytes));
  tb.run(8);
}

Copy mbar_copy(std::vector<uint64_t> blocks, uint32_t bytes) {
  Copy c = make_copy(0, std::move(blocks));
  c.mbar = true;
  c.mbar_off = uint32_t(MBAR_ADDR >> 3);
  c.bytes = bytes;
  c.slot = 0;
  c.gen = 1;
  return c;
}

// The final write first, the other held by an atomic: the complete_tx runs
// after both and the phase flips at T+4 of the last commit.
void mbar_one(Bench& tb) {
  tb.clear_flips();
  const uint32_t bytes = 2 * BLOCK;
  mbar_init(tb, bytes);
  uint64_t w0 = tb.load(MBAR_ADDR);
  tb.check(mbar_ref::tx(w0) == int64_t(bytes) && mbar_ref::pending(w0) == 0 && mbar_ref::phase(w0) == 0,
           strf("mbar one: object after arrive_tx 0x%" PRIx64, w0));
  Copy c = mbar_copy({blk(3), blk(4)}, bytes);
  tb.issue(PORT_LSU, tb.word_req(MemOp::AMO_ADD, (c.blocks[0] & ~uint64_t(BLOCK - 1)) + LMEM_SIZE / 2, 1));
  tb.tick();
  tb.dma(PORT_DMA, c, c.blocks[1], true, 2);
  tb.dma(PORT_DMA + 1, c, c.blocks[0], false, 0);
  tb.run(16);
  uint64_t t = tb.last_commit(c.xfer, 2);
  tb.check(t != 0, "mbar one: a write did not commit");
  tb.check(tb.flips().size() == 1, strf("mbar one: %zu phase flips", tb.flips().size()));
  if (!tb.flips().empty()) {
    tb.check(tb.flips()[0].cycle == t + 4,
             strf("mbar one: flip at %" PRIu64 ", last commit at %" PRIu64 " (want +4)", tb.flips()[0].cycle, t));
    tb.check(tb.flips()[0].bank == bank_of(MBAR_ADDR), "mbar one: flip on the wrong bank");
  }
  uint64_t w1 = tb.load(MBAR_ADDR);
  uint64_t want = mbar_ref::apply(mbar_ref::OP_COMPLETE_TX, w0, bytes).new_word;
  tb.check(w1 == want, strf("mbar one: object 0x%" PRIx64 ", want 0x%" PRIx64, w1, want));
  tb.check(tb.rels().empty() || tb.rels().back().cycle < t, "mbar one: a barrier release");
  std::printf("  mbarrier, two writes, final first=%d: last commit %" PRIu64 ", flip +4, object 0x%" PRIx64 "\n",
              tb.final_committed_first(c.xfer), t, w1);
}

// Atomics on the mbarrier's bank hold the complete_tx requests back while
// copies keep landing, one a cycle: the queue fills, the DXA input holds the
// next final writes back, and every copy completes in the end, the phase
// flipping once.
void mbar_backpressure(Bench& tb) {
  tb.clear_flips();
  const uint32_t ncopies = DEPTH + 6;
  mbar_init(tb, ncopies * BLOCK);
  uint64_t w0 = tb.load(MBAR_ADDR);
  std::vector<Copy> cs;
  for (uint32_t i = 0; i < ncopies; ++i) {
    cs.push_back(mbar_copy({blk(16 + i)}, BLOCK));
  }
  const uint64_t hot = (MBAR_ADDR & ~uint64_t(BLOCK - 1)) + LMEM_SIZE / 2;
  std::vector<uint64_t> issued(ncopies, 0);
  // The atomics saturate the bank before the copies start.
  constexpr uint32_t kWarm = 24;
  for (uint32_t cyc = 0; cyc < 400; ++cyc) {
    if (cyc < 160 && (cyc % 2) == 0) {
      tb.issue(PORT_LSU, tb.word_req(MemOp::AMO_ADD, hot, 1));
    }
    if (cyc >= kWarm && cyc < kWarm + ncopies) {
      tb.dma(PORT_DMA, cs[cyc - kWarm], cs[cyc - kWarm].blocks[0], true, 1);
      issued[cyc - kWarm] = tb.now();
    }
    tb.tick();
  }
  uint64_t base = UINT64_MAX, worst = 0;
  bool all = true;
  for (uint32_t i = 0; i < ncopies; ++i) {
    uint64_t t = tb.last_commit(cs[i].xfer, 1);
    all &= (t != 0);
    if (t != 0) {
      base  = std::min(base, t - issued[i]);
      worst = std::max(worst, t - issued[i]);
    }
  }
  tb.check(all, "mbar backpressure: a copy never committed");
  tb.check(worst > base + 8, strf("mbar backpressure: no final write was held (delays %" PRIu64 "..%" PRIu64 ")", base, worst));
  tb.check(tb.flips().size() == 1, strf("mbar backpressure: %zu phase flips", tb.flips().size()));
  uint64_t w1 = tb.load(MBAR_ADDR);
  uint64_t want = w0;
  for (uint32_t i = 0; i < ncopies; ++i) {
    want = mbar_ref::apply(mbar_ref::OP_COMPLETE_TX, want, BLOCK).new_word;
  }
  tb.check(w1 == want, strf("mbar backpressure: object 0x%" PRIx64 ", want 0x%" PRIx64, w1, want));
  std::printf("  mbarrier, %u copies against a busy bank: finals held up to %" PRIu64 " cycles, all complete, object 0x%" PRIx64 "\n",
              ncopies, worst - base, w1);
}

// A run that must abort, in a child; its stderr must hold the text.
void negative(Bench& tb, const char* what, const char* text, const std::function<void(Bench&)>& body) {
  std::fflush(stdout);
  std::fflush(stderr);
  int fds[2];
  if (pipe(fds) != 0) {
    tb.fail("pipe failed");
    return;
  }
  pid_t pid = fork();
  if (pid == 0) {
    close(fds[0]);
    dup2(fds[1], STDERR_FILENO);
    body(tb);
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
  bool aborted = WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
  if (!aborted || err.find(text) == std::string::npos) {
    tb.fail(strf("%s: aborted=%d, stderr: %s", what, aborted, err.c_str()));
  } else {
    std::printf("  %s aborts with \"%s\"\n", what, text);
  }
}
#endif

}

int main() {
  std::printf("dxa_compl: banks=%u block=%u depth=%u mbarriers=%d\n", NUM_BANKS, BLOCK, DEPTH,
#ifdef DXA_MBAR_ENABLE
              1
#else
              0
#endif
  );
  Bench tb;
#ifdef DXA_MBAR_ENABLE
  // Slot 0 holds the issuing block, generation 1.
  tb.admit(0, 1);
  tb.run(2);
#endif
  legacy_single(tb);
#if VX_CFG_EXT_A_ENABLED
  legacy_held(tb);
#endif
  legacy_burst(tb);
  legacy_full(tb);
#ifdef DXA_MBAR_ENABLE
  mbar_one(tb);
  mbar_backpressure(tb);
  negative(tb, "a write for a re-admitted slot", "DXA write into a retired CTA slot", [](Bench& b) {
    b.admit(0, 2);
    b.run(2);
    Copy c = mbar_copy({blk(40)}, BLOCK);
    b.dma(PORT_DMA, c, c.blocks[0], true, 1);
    b.run(10);
  });
  negative(tb, "an mbarrier outside the block", "mbarrier offset outside the block", [](Bench& b) {
    Copy c = mbar_copy({blk(41)}, BLOCK);
    c.mbar_off = SLOT_STRIDE / 8;
    b.dma(PORT_DMA, c, c.blocks[0], true, 1);
    b.run(10);
  });
#endif
  if (tb.failures() != 0) {
    std::printf("FAILED: %u check(s) failed\n", tb.failures());
    return 1;
  }
  std::printf("PASSED\n");
  return 0;
}
