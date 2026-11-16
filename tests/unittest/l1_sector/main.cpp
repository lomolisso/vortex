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

// SimX twin of hw/unittest/l1h100: the SM's L1 (Cache with h100_l1, as the
// socket configures it) through the same cases (l1h100_cases.h) with the same
// cycle contract; --csv=<file> must equal the RTL bench's. A line request
// leaves as one MemReq per touched memory block, every block but the last
// marked by sm_port_more(); the memory answers all blocks of a request in one
// cycle.

#include "mem/cache.h"
#include "mem/sm_port.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>

#include "l1h100_cases.h"

using namespace vortex;
using namespace l1h100_cases;

#ifndef VX_CFG_H100_SM_ENABLE
#error "the l1_sector twin tests the H100 SM path: build with -DVX_CFG_H100_SM_ENABLE"
#endif

static_assert(LINE % BLOCK == 0, "an L1 line is whole memory blocks");

class SimxDriver {
public:
  SimxDriver() {
    Cache::Config cfg{
      false,
      (uint8_t)log2ceil(VX_CFG_DCACHE_SIZE),
      (uint8_t)log2ceil(VX_CFG_DCACHE_LINE_SIZE),
      (uint8_t)log2ceil(VX_CFG_DCACHE_LINE_SIZE),
      (uint8_t)log2ceil(DCACHE_WORD_SIZE),
      (uint8_t)log2ceil(VX_CFG_DCACHE_NUM_WAYS),
      (uint8_t)log2ceil(VX_CFG_DCACHE_NUM_BANKS),
      VX_CFG_XLEN,
      1,
      1,
      VX_CFG_DCACHE_WRITEBACK,
      VX_CFG_DCACHE_DIRTYBYTES,
      false,
      VX_CFG_DCACHE_MSHR_SIZE,
      VX_CFG_DCACHE_LATENCY,
      VX_CFG_DCACHE_REPL_POLICY,
      false,
    };
    cfg.h100_l1 = true;
    l1_ = Cache::Create("l1h100", cfg);
    auto& platform = SimPlatform::instance();
    platform.reset();
    for (uint32_t i = 0; i < CAP_LINES + 64; ++i)
      platform.tick();
    now_ = 0;
  }

  uint64_t now() const { return now_; }

  void push(const Op& op, uint32_t tag) {
    q_.push_back({op, tag});
    ops_[tag] = op;
  }

  std::vector<Event> cycle() {
    auto& platform = SimPlatform::instance();
    std::vector<Event> ev;
    if (!q_.empty() && !sent_ && l1_->core_req_in.at(0).empty()) {
      l1_->core_req_in.at(0).send(this->core_req(q_.front().first, q_.front().second), 0);
      sent_ = true;
    }
    while (!rsp_.empty() && rsp_.front().due <= now_) {
      for (auto& r : rsp_.front().blocks)
        l1_->mem_rsp_in.at(0).send(r, 0);
      rsp_.pop_front();
    }
    platform.tick();
    if (sent_ && l1_->core_req_in.at(0).empty()) {
      ev.push_back({EV_ACCEPT, q_.front().second, true, false, false, 0, {}});
      q_.pop_front();
      sent_ = false;
    }
    while (!l1_->mem_req_out.at(0).empty()) {
      MemReq req = l1_->mem_req_out.at(0).peek();
      l1_->mem_req_out.at(0).pop();
      this->mem_block(req, ev);
    }
    while (!l1_->core_rsp_out.at(0).empty()) {
      const MemRsp& rsp = l1_->core_rsp_out.at(0).peek();
      uint32_t tag = (uint32_t)rsp.tag;
      Event e{EV_RSP, tag, true, false, false, 0, {}};
      auto it = ops_.find(tag);
      if (it != ops_.end() && it->second.kind == K_LD) {
        const Op& op = it->second;
        Bytes& m = mem_.at(op.line);
        for (uint32_t i = 0; i < LINE; ++i) {
          if (!op.byteen[i])
            continue;
          uint8_t got = rsp.data ? (*rsp.data)[i % BLOCK] : 0;
          e.data_ok &= (got == m[i]);
        }
      }
      ev.push_back(e);
      l1_->core_rsp_out.at(0).pop();
    }
    ++now_;
    return ev;
  }

private:
  struct Pending {
    uint64_t due;
    std::vector<MemRsp> blocks;
  };

  // The access's memory block: every op of the cases stays inside one.
  MemReq core_req(const Op& op, uint32_t tag) {
    uint32_t first = LINE;
    for (uint32_t i = 0; i < LINE; ++i) {
      if (op.byteen[i]) {
        first = i;
        break;
      }
    }
    uint32_t blk = (first == LINE) ? 0 : first / BLOCK;
    uint64_t addr = op.line * LINE + blk * BLOCK;
    auto data = std::make_shared<mem_block_t>();
    uint64_t be = 0;
    for (uint32_t i = 0; i < BLOCK; ++i) {
      (*data)[i] = op.data[blk * BLOCK + i];
      if (op.byteen[blk * BLOCK + i])
        be |= 1ull << i;
    }
    MemReq req(op.kind == K_ST ? MemOp::ST : (op.kind == K_FENCE ? MemOp::FLUSH : MemOp::LD), addr, data, be, tag);
    if (op.kind == K_FENCE)
      req.flags.fence = 1;
    return req;
  }

  // Gathers a line request's blocks; a read is answered block by block, all in one cycle.
  void mem_block(const MemReq& req, std::vector<Event>& ev) {
    uint64_t line = req.addr / LINE;
    if (req.op == MemOp::FLUSH || req.flags.fence) {
      // A fence passed below: the memory acknowledges it.
      ev.push_back({EV_MEM, 0, true, false, true, line, {}});
      rsp_.push_back({now_ + MEM_DELAY, {MemRsp(req.tag, req.hart_id, 0, nullptr)}});
      return;
    }
    uint32_t base = (uint32_t)(req.addr % LINE) / BLOCK * BLOCK;
    if (!gather_.active) {
      gather_.active = true;
      gather_.line = line;
      gather_.rw = req.is_write();
      gather_.byteen.fill(0);
      gather_.blocks.clear();
    }
    for (uint32_t i = 0; i < BLOCK; ++i) {
      if (req.byteen >> i & 1) {
        gather_.byteen[base + i] = 1;
        if (gather_.rw)
          mem_.at(line)[base + i] = req.data ? (*req.data)[i] : 0;
      }
    }
    if (!gather_.rw) {
      auto d = std::make_shared<mem_block_t>();
      Bytes& m = mem_.at(line);
      for (uint32_t i = 0; i < BLOCK; ++i)
        (*d)[i] = m[base + i];
      gather_.blocks.push_back(MemRsp(req.tag, req.hart_id, 0, d));
    }
    if (sm_port_has_more(req))
      return;
    ev.push_back({EV_MEM, 0, true, gather_.rw, false, gather_.line, gather_.byteen});
    if (!gather_.rw)
      rsp_.push_back({now_ + MEM_DELAY, gather_.blocks});
    gather_.active = false;
  }

  struct Gather {
    bool active = false;
    bool rw = false;
    uint64_t line = 0;
    Bytes byteen{};
    std::vector<MemRsp> blocks;
  };

  std::shared_ptr<Cache> l1_;
  uint64_t now_ = 0;
  bool sent_ = false;
  std::deque<std::pair<Op, uint32_t>> q_;
  std::map<uint32_t, Op> ops_;
  std::deque<Pending> rsp_;
  Gather gather_;
  MemImage mem_;
};

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  uint32_t seed = 1, nrand = 3000;
  bool quick = false;
  const char* csv_path = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--seed=", 7)) seed = (uint32_t)atoi(argv[i] + 7);
    else if (!strncmp(argv[i], "--random=", 9)) nrand = (uint32_t)atoi(argv[i] + 9);
    else if (!strncmp(argv[i], "--csv=", 6)) csv_path = argv[i] + 6;
    else if (!strcmp(argv[i], "--quick")) quick = true;
    else { std::printf("usage: %s [--seed=N] [--random=N] [--quick] [--csv=file]\n", argv[0]); return 2; }
  }
  FILE* csv = csv_path ? std::fopen(csv_path, "w") : nullptr;
  std::printf("l1h100 (SimX): %u B, %u ways, line %u, sector %u, protected %u, seed %u\n", VX_CFG_DCACHE_SIZE,
              VX_CFG_DCACHE_NUM_WAYS, LINE, SECTOR, VX_CFG_L1_REPL_PROTECTED, seed);
  SimxDriver d;
  Runner<SimxDriver> run(d, csv);
  int rc = run.run(seed, nrand, quick);
  if (csv)
    std::fclose(csv);
  std::printf(rc ? "FAILED!\n" : "PASSED!\n");
  return rc;
}
