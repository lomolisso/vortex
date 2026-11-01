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

// The DXA engine's bulk copies, the twin of hw/unittest/dxa_core's bulk case.
// The bench is the issuer core's local memory: it takes the engine's requests
// under random back-pressure and answers its reads in order per port after a
// random delay. Checked: every read is a whole local-memory word of a source
// line, a word wider than a block arriving as its blocks; reads and writes
// never share a cycle; at most LMEM_RD_DEPTH lines are read ahead of their
// responses; every write stays in one row; the copies leave exactly the
// expected bytes; and each copy's final write carries its completion:
// complete_tx of the size on its mbarrier, its slot and generation.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <vector>

#include "core.h"
#include "dxa_core.h"
#include "mem_block_pool.h"
#include "processor.h"
#include "simobject.h"
#include <VX_types.h>

using namespace vortex;

#ifndef DXA_MBAR_ENABLE
#error "bulk copies need a DXA build with mbarriers"
#endif

namespace {

constexpr uint32_t BLOCK      = VX_CFG_MEM_BLOCK_SIZE;
constexpr uint32_t LINE       = VX_CFG_L1_LINE_SIZE;
constexpr uint32_t ROW        = DxaCore::LMEM_ROW_SIZE;
constexpr uint32_t PORTS      = DxaCore::LMEM_PORTS_PER_CORE;
constexpr uint32_t LMEM_BYTES = 1u << VX_CFG_LMEM_LOG_SIZE;
constexpr uint64_t LMEM_BASE  = VX_MEM_LMEM_BASE_ADDR;

[[noreturn]] void fail(const char* what) {
  std::printf("FAILED: %s\n", what);
  std::exit(1);
}

uint64_t rng_state = 1;
uint32_t rnd() {
  rng_state = rng_state * 6364136223846793005ull + 1442695040888963407ull;
  return uint32_t(rng_state >> 33);
}

uint8_t pattern(uint32_t i) { return uint8_t((i * 37u + (i >> 8) * 11u + 5u) ^ 0xa5u); }

struct BulkCopy {
  uint32_t src, dst, size, mbar_off, slot, gen;
};

struct Pending {
  MemRsp   rsp;
  uint64_t due;
};

struct Final {
  bool     seen = false;
  uint32_t writes = 0;
  DxaWrTag tag;
};

void tick() { SimPlatform::instance().tick(); }

// Runs the copies back to back. ready_pct: the chance a port takes a request
// in a cycle; max_lat: the largest response delay; hold: cycles with no
// response at all, which must stop the reads at the line reader's depth.
void run(DxaCore* dxa, Core* core, const char* name, const std::vector<BulkCopy>& copies,
         uint32_t ready_pct, uint32_t max_lat, uint32_t hold) {
  std::vector<uint8_t> mem(LMEM_BYTES), ref(LMEM_BYTES);
  for (uint32_t i = 0; i < LMEM_BYTES; ++i)
    mem[i] = ref[i] = pattern(i);
  for (auto& c : copies)
    for (uint32_t i = 0; i < c.size; ++i)
      ref[c.dst + i] = pattern(c.src + i);

  static uint64_t uuid = 100;
  std::map<uint64_t, const BulkCopy*> by_uuid;
  std::deque<DxaReq> issue;
  for (auto& c : copies) {
    DxaReq req{};
    req.core      = core;
    req.uuid      = ++uuid;
    req.mode      = true;
    req.bulk      = true;
    req.src_addr  = LMEM_BASE + c.src;
    req.size      = c.size;
    req.smem_addr = LMEM_BASE + c.dst;
    req.mbar_off  = c.mbar_off;
    req.slot      = c.slot;
    req.gen       = c.gen;
    req.win       = true;
    req.win_room  = LMEM_BYTES - c.dst;
    by_uuid[req.uuid] = &c;
    issue.push_back(req);
  }

  std::vector<std::deque<Pending>> rsps(PORTS);
  std::map<uint64_t, Final> finals;
  std::map<uint64_t, uint32_t> writes;
  constexpr uint32_t LINE_READS = (LINE > ROW ? LINE / ROW : 1) * PORTS;  // block reads per line
  uint64_t reads = 0;
  uint64_t lines = 0;
  for (auto& c : copies)
    lines += (((c.src + c.size + LINE - 1) & ~(LINE - 1)) - (c.src & ~(LINE - 1))) / LINE;
  uint64_t first_cycle = SimPlatform::instance().cycles();
  uint64_t now = 0;
  uint64_t last_activity = 0;
  for (; now < 200000; ++now) {
    if (!issue.empty() && !dxa->dxa_req_in.at(0).full()) {
      dxa->dxa_req_in.at(0).send(issue.front());
      issue.pop_front();
    }
    tick();
    bool rd_now = false, wr_now = false;
    for (uint32_t p = 0; p < PORTS; ++p) {
      auto& ch = dxa->lmem_req_out.at(p);
      if (ch.empty() || (rnd() % 100) >= ready_pct)
        continue;
      MemReq req = ch.peek();
      ch.pop();
      last_activity = now;
      const uint64_t a = req.addr - LMEM_BASE;
      if (!req.flags.dma)
        fail("a DXA request without the DMA flag");
      if (!req.is_write()) {
        rd_now = true;
        if (req.op != MemOp::LD)
          fail("a read that is not a load");
        if ((a % std::min(ROW, BLOCK)) != 0)
          fail("a read not aligned to its word");
        if (PORTS > 1 && uint32_t((a / BLOCK) % PORTS) != p)
          fail("a word's block read on another port");
        ++reads;
        Pending r;
        r.rsp = MemRsp(req.tag, req.hart_id, req.uuid, make_mem_block());
        std::memcpy(r.rsp.data->data(), mem.data() + (a & ~uint64_t(BLOCK - 1)), BLOCK);
        uint64_t due = (now < hold ? hold : now) + 1 + rnd() % max_lat;
        if (!rsps[p].empty())
          due = std::max(due, rsps[p].back().due);
        r.due = due;
        rsps[p].push_back(r);
        continue;
      }
      wr_now = true;
      const uint64_t blk = a & ~uint64_t(BLOCK - 1);
      if (req.byteen == 0 || a != blk + (uint64_t(__builtin_ctzll(req.byteen)) & ~uint64_t(VX_CFG_XLEN / 8 - 1)))
        fail("a write not addressed to the first word it writes");
      uint64_t lo = BLOCK, hi = 0;
      for (uint32_t b = 0; b < BLOCK; ++b) {
        if ((req.byteen >> b) & 1) {
          mem[blk + b] = (*req.data)[b];
          lo = std::min<uint64_t>(lo, b);
          hi = b;
        }
      }
      if (lo > hi || ((blk + lo) / ROW) != ((blk + hi) / ROW))
        fail("a write empty or across rows");
      ++writes[req.uuid];
      if (req.dxa.remote || !req.dxa.mode)
        fail("a local bulk write that is remote or not in mbarrier mode");
      if (req.flags.dxa_notify_done) {
        auto& f = finals[req.uuid];
        if (f.seen)
          fail("two final writes for one copy");
        if (!req.dxa.last)
          fail("the final write not marked last");
        f.seen = true;
        f.writes = req.dxa.writes;
        f.tag = req.dxa;
      }
    }
    if (rd_now && wr_now && ready_pct == 100)
      fail("a read and a write in one cycle");
    if (hold != 0 && now == hold && reads != uint64_t(DxaCore::LMEM_RD_DEPTH) * LINE_READS)
      fail("the reads ahead of their responses are not the line reader's depth");
    for (uint32_t p = 0; p < PORTS; ++p) {
      auto& q = rsps[p];
      auto& ch = dxa->lmem_rsp_in.at(p);
      if (!q.empty() && q.front().due <= now && !ch.full()) {
        ch.send(q.front().rsp, 0);
        q.pop_front();
        last_activity = now;
      }
    }
    if (finals.size() == copies.size() && now > last_activity + 50)
      break;
  }
  if (finals.size() != copies.size())
    fail("a copy without its final write");
  for (auto& [u, f] : finals) {
    const BulkCopy& c = *by_uuid.at(u);
    if (f.writes != writes[u])
      fail("the final write's count is not the copy's writes");
    if (f.tag.tx_bytes != c.size || f.tag.mbar_off != c.mbar_off || f.tag.slot != c.slot || f.tag.gen != c.gen)
      fail("the completion fields are not the copy's");
  }
  if (reads != lines * LINE_READS)
    fail("not every word of every source line read once");
  for (uint32_t i = 0; i < LMEM_BYTES; ++i) {
    if (mem[i] != ref[i]) {
      std::printf("  byte 0x%x: 0x%02x, expected 0x%02x\n", i, mem[i], ref[i]);
      fail("wrong local-memory contents");
    }
  }
  std::printf("  %s: %zu copies, %llu lines read, %llu cycles\n", name, copies.size(), (unsigned long long)lines,
              (unsigned long long)(SimPlatform::instance().cycles() - first_cycle));
}

} // namespace

int main() {
  Processor processor;
  auto* core = processor.get_first_core();
  auto dxa = DxaCore::Create("dxa_bulk_test", nullptr);
  SimPlatform::instance().reset();
  std::printf("dxa_bulk: row=%u line=%u block=%u ports=%u inflight=%u depth=%u\n", ROW, LINE, BLOCK, PORTS,
              VX_CFG_DXA_MAX_INFLIGHT, DxaCore::LMEM_RD_DEPTH);

  // Source and destination offsets that differ modulo the row and the line.
  const std::vector<BulkCopy> mix = {
    {  16, 7216,  208, 0x700, 1, 3},
    { 256, 7488,   64, 0x701, 1, 3},
    { 368, 7600,   16, 0x702, 2, 5},
    { 432, 7680,  720, 0x703, 0, 1},
    {4128, 8464, 2048, 0x704, 3, 9},
  };
  run(dxa.get(), core, "mix", mix, 100, 1, 0);
  run(dxa.get(), core, "mix, back-pressure", mix, 60, 6, 0);
  run(dxa.get(), core, "mix, slow memory", mix, 100, 24, 0);
  // A 2 KB copy whose responses are held: the reads stop at the line limit.
  run(dxa.get(), core, "held responses", {{4096, 10752, 2048, 0x710, 1, 2}}, 100, 3, 400);

  std::printf("PASSED\n");
  return 0;
}
