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

// SimX twin of hw/unittest/l1h100: SmL1 on SmSram, bound as the core binds
// them, run through the same cases (l1h100_cases.h); --csv=<file> must equal
// the RTL bench's.

#include "sm_sram.h"
#include "sm_l1.h"
#include "cache.h"

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

constexpr uint32_t NB = VX_CFG_NUM_LSU_BLOCKS;
constexpr uint32_t NL = VX_CFG_NUM_LSU_LANES;
constexpr uint32_t BLOCK = VX_CFG_MEM_BLOCK_SIZE;
constexpr uint32_t RD_TCU_PORT = 0;

static_assert(NL <= MAX_LANES, "the bench drives at most 32 lanes");
static_assert(VX_CFG_LSU_PENDING_SIZE >= NUM_TAGS, "the bench keeps up to NUM_TAGS requests in flight");
static_assert(LINE % BLOCK == 0, "a line is whole memory blocks");

class SimxDriver {
public:
  SimxDriver() {
    sram_ = SmSram::Create("sm_sram", SmSram::Config{NB});
    l1_ = SmL1::Create("sm_l1", SmL1::Config{NB});
    l1_->GlReqOut.bind(&sram_->GlReqIn);
    sram_->GlRspOut.bind(&l1_->GlRspIn);
    sram_->WfRetiredOut.bind(&l1_->WfRetiredIn);
    l1_->FillOut.bind(&sram_->WrIn.at(0));
    sram_->FillDoneOut.bind(&l1_->FillDoneIn);
    s_rows_ = VX_CFG_SMEM_PER_SM_MAX / SmSram::ROW_BYTES;
    SimPlatform::instance().reset();
  }

  void push(const Req& r, uint32_t tag) { q_.push_back({r, tag}); }
  void flush() { flush_on_ = true; flush_sent_ = false; }
  void set_mem_delay(uint32_t d) { mem_.delay = d; }
  void set_row_stream(bool on) {
    rows_on_ = on;
    row_phase_ = 0;
  }
  FlatMem& mem() { return mem_; }
  bool busy() const { return l1_->busy() || !q_.empty() || !mrsp_.empty() || flush_on_; }

  std::vector<Event> cycle() {
    auto& platform = SimPlatform::instance();
    std::vector<Event> ev;
    if (!q_.empty() && !sent_) {
      l1_->LsuReqIn.at(0).send(lsu_req(q_.front().first, q_.front().second), 0);
      sent_ = true;
    }
    // every other cycle, so the global path keeps read-port cycles under the
    // TCU's higher priority; a sent row stays queued until taken
    if (rows_on_ && !row_sent_ && (row_phase_ & 1) == 0) {
      sram_->RowRdIn.at(RD_TCU_PORT).send(SmSramRowRd{row_ % s_rows_, 0}, 0);
      row_sent_ = true;
    }
    if (!mrsp_.empty() && mrsp_.front().when <= now_ && l1_->MemRspIn.empty()) {
      l1_->MemRspIn.send(mrsp_.front().rsp, 0);
      mrsp_.pop_front();
    }
    if (flush_on_ && !flush_sent_) {
      l1_->flush_begin();
      flush_sent_ = true;
    }
    platform.tick();

    if (sent_ && l1_->LsuReqIn.at(0).empty()) {
      ev.push_back(Event{EV_ACCEPT, q_.front().second});
      q_.pop_front();
      sent_ = false;
    }
    auto& out = l1_->LsuRspOut.at(0);
    while (!out.empty()) {
      const LsuRsp& rsp = out.peek();
      Event e{EV_RSP};
      e.tag = uint32_t(rsp.tag) & (NUM_TAGS - 1);
      for (uint32_t i = 0; i < NL; ++i) {
        if (!rsp.mask.test(i))
          continue;
        e.mask |= 1u << i;
        const auto it = addr_.find({e.tag, i});
        const uint64_t a = (it == addr_.end()) ? 0 : it->second;
        if (rsp.data.at(i))
          std::memcpy(&e.data[i], rsp.data.at(i)->data() + (a % BLOCK) / WORD * WORD, WORD);
      }
      ev.push_back(e);
      out.pop();
    }
    std::map<uint64_t, uint32_t> rd;
    auto& mreq = l1_->MemReqOut;
    while (!mreq.empty()) {
      const MemReq& m = mreq.peek();
      const uint64_t base = m.addr / BLOCK * BLOCK;
      uint8_t be[BLOCK], data[BLOCK];
      for (uint32_t o = 0; o < BLOCK; ++o) {
        be[o] = (m.byteen >> o) & 1;
        data[o] = m.data ? (*m.data)[o] : 0;
      }
      if (m.op == MemOp::ST) {
        mem_.write(base, data, be, BLOCK);
      } else if (m.op == MemOp::AMO_ADD) {
        this->reply(m, mem_.amo_add(base, data, be, BLOCK));
      } else if (m.op != MemOp::LD || m.flags.fence) {
        this->reply(m, std::vector<uint8_t>(BLOCK, 0));
      } else {
        auto& s = rd[base / LINE];
        for (uint32_t o = 0; o < BLOCK; ++o)
          s |= uint32_t(be[o]) << ((base % LINE + o) / SECTOR);
        this->reply(m, mem_.read(base, BLOCK));
      }
      mreq.pop();
    }
    for (auto& kv : rd) {
      Event e{EV_MEMRD};
      e.line = kv.first;
      e.mask = kv.second;
      ev.push_back(e);
    }
    if (row_sent_ && sram_->RowRdIn.at(RD_TCU_PORT).empty()) {
      ++row_;
      row_sent_ = false;
    }
    if (rows_on_)
      ++row_phase_;
    auto& rout = sram_->RowRspOut.at(RD_TCU_PORT);
    while (!rout.empty())
      rout.pop();
    if (flush_on_ && flush_sent_ && l1_->flush_done()) {
      ev.push_back(Event{EV_FLUSHED});
      flush_on_ = false;
    }
    ++now_;
    return ev;
  }

private:
  struct Pend {
    uint64_t when;
    MemRsp rsp;
  };

  void reply(const MemReq& m, const std::vector<uint8_t>& bytes) {
    auto blk = std::make_shared<mem_block_t>();
    std::memcpy(blk->data(), bytes.data(), BLOCK);
    const uint64_t when = std::max(now_ + mem_.delay, mrsp_.empty() ? 0 : mrsp_.back().when);
    mrsp_.push_back({when, MemRsp(m.tag, m.hart_id, m.uuid, blk)});
  }

  LsuReq lsu_req(const Req& r, uint32_t tag) {
    LsuReq q(NL);
    q.op = (r.kind == K_ST) ? MemOp::ST : (r.kind == K_ADD) ? MemOp::AMO_ADD : MemOp::LD;
    q.tag = tag;
    for (uint32_t i = 0; i < NL; ++i) {
      if (!(r.mask >> i & 1))
        continue;
      const uint32_t woff = uint32_t(r.addr[i] % BLOCK) / WORD * WORD;
      q.mask.set(i);
      q.addrs[i] = r.addr[i];
      q.byteen[i] = uint64_t(lane_be(r, i)) << woff;
      auto b = std::make_shared<mem_block_t>();
      std::memset(b->data(), 0, b->size());
      const uint64_t w = lane_wdata(r, i);
      std::memcpy(b->data() + woff, &w, WORD);
      q.data[i] = b;
      addr_[{tag, i}] = r.addr[i];
    }
    return q;
  }

  SmSram::Ptr sram_;
  SmL1::Ptr l1_;
  uint64_t now_ = 0;
  uint32_t s_rows_ = 0, row_ = 0, row_phase_ = 0;
  bool sent_ = false, row_sent_ = false, rows_on_ = false, flush_on_ = false, flush_sent_ = false;
  std::deque<std::pair<Req, uint32_t>> q_;
  std::deque<Pend> mrsp_;
  std::map<std::pair<uint32_t, uint32_t>, uint64_t> addr_;
  FlatMem mem_;
};

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  uint32_t seed = 1, nrand = 2000;
  const char* csv_path = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--seed=", 7)) seed = (uint32_t)atoi(argv[i] + 7);
    else if (!strncmp(argv[i], "--random=", 9)) nrand = (uint32_t)atoi(argv[i] + 9);
    else if (!strncmp(argv[i], "--csv=", 6)) csv_path = argv[i] + 6;
    else { std::printf("usage: %s [--seed=N] [--random=N] [--csv=file]\n", argv[0]); return 2; }
  }
  FILE* csv = csv_path ? std::fopen(csv_path, "w") : nullptr;
  std::printf("l1h100 (SimX): %u lanes, line %u B, sector %u B, seed %u\n", NL, LINE, SECTOR, seed);
  SimxDriver d;
  Runner<SimxDriver> run(d, NL, csv);
  int rc = run.run(seed, nrand);
  if (csv)
    std::fclose(csv);
  std::printf(rc ? "FAILED!\n" : "PASSED!\n");
  return rc;
}
