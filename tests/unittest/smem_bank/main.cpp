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

// SimX twin of hw/unittest/sm_sram: SmSram, the H100 SM's one array, run
// through the same cases (sm_sram_cases.h) with the same cycle contract;
// --csv=<file> must equal the RTL bench's.

#include "sm_sram.h"
#include "sm_l1.h"
#include "cache.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>

#include "sm_sram_cases.h"

using namespace vortex;
using namespace sm_sram_cases;

#ifndef VX_CFG_H100_SM_ENABLE
#error "the smem_bank twin tests the H100 SM path: build with -DVX_CFG_H100_SM_ENABLE"
#endif

constexpr uint32_t NB = VX_CFG_NUM_LSU_BLOCKS;
constexpr uint32_t NL = VX_CFG_NUM_LSU_LANES;
constexpr uint32_t WORD_SIZE = LSU_WORD_SIZE;
constexpr uint64_t BLOCK = VX_CFG_MEM_BLOCK_SIZE;
constexpr uint32_t WR_DXA_PORT = 1;
constexpr uint32_t WR_DSMEM_PORT = 2;
constexpr uint32_t SLOT_BYTES = VX_CFG_N5_SLOT_BYTES;

static std::shared_ptr<mem_block_t> word_block(uint64_t addr, uint64_t v) {
  auto b = std::make_shared<mem_block_t>();
  std::memset(b->data(), 0, b->size());
  std::memcpy(b->data() + (addr % BLOCK) / WORD_SIZE * WORD_SIZE, &v, sizeof(v));
  return b;
}

static uint64_t word_of(const std::shared_ptr<mem_block_t>& b, uint64_t addr) {
  uint64_t v = 0;
  if (b)
    std::memcpy(&v, b->data() + (addr % BLOCK) / WORD_SIZE * WORD_SIZE, sizeof(v));
  return v;
}

// Host-side driver: inputs sent with no delay before a tick are seen by that
// tick; a request was taken in the tick that emptied its channel. After tick
// k a registered output (sent with a delay) is already visible, but it is the
// RTL's output of cycle k + 1, so its event is reported with the next cycle.
class SimxDriver {
public:
  SimxDriver() {
    sram_ = SmSram::Create("sm_sram", SmSram::Config{NB});
    l1_ = SmL1::Create("sm_l1", SmL1::Config{NB});
    l1_->GlReqOut.bind(&sram_->GlReqIn);
    sram_->GlRspOut.bind(&l1_->GlRspIn);
    sram_->WfRetiredOut.bind(&l1_->WfRetiredIn);
    sram_->FillDoneOut.bind(&l1_->FillDoneIn);
    // FillOut stays unbound: the bench passes each refill row on to the array
    // itself (an offer for the next tick either way) and sees its row.
    SimPlatform::instance().reset();
    now_ = 0;
  }

  uint64_t now() const { return now_; }

  void push(uint32_t b, const Req& r, uint32_t tag) {
    q_[b].push_back({r, tag});
    reqs_[b][tag] = r;
  }

  void push_row(uint32_t client, uint32_t row, uint32_t tag) {
    rowq_[client].push_back({row, tag});
    row_of_[tag] = row;
  }

  void push_wr(uint32_t row, uint32_t tag) { wrq_.push_back({row, tag}); }

  void push_cpa(const Req& r, uint32_t tag) {
    cpaq_.push_back({r, tag});
    cpa_mask_[tag] = r.mask;
  }

  void push_ds(uint32_t row, uint32_t slot, bool last, uint32_t tag) { dsq_.push_back({row, slot, last, tag}); }

  // The split from the next tick, as Core sends it; true if SmSram's own split
  // at that size is the same.
  bool set_cfg(const ArrayCfg& c) {
    sram_->CfgIn.send(SmSramCfg{c.s_rows, c.l1_lines, c.l1_protect}, 0);
    l1_->CfgIn.send(SmSramCfg{c.s_rows, c.l1_lines, c.l1_protect}, 0);
    const SmSramCfg m = SmSram::cfg_of(c.s_rows * SmSram::ROW_BYTES);
    return m.s_rows == c.s_rows && m.l1_lines == c.l1_lines && m.l1_protect == c.l1_protect;
  }

  void push_gl(uint32_t b, const gl::Req& r, uint32_t tag) {
    glq_[b].push_back({r, tag});
    glreq_[b][tag] = r;
  }

  const std::vector<GlBad>& gl_bad() const { return gl_bad_; }

  void gl_flush() {
    flush_on_ = true;
    flush_sent_ = false;
  }

  bool gl_busy() const {
    bool q = false;
    for (uint32_t b = 0; b < NB; ++b)
      q |= !glq_[b].empty();
    return q || l1_->busy() || flush_on_ || !mrsp_.empty() || fill_fwd_;
  }

  void push_amo(uint32_t client, uint64_t addr, uint32_t tag) {
    amoq_[client].push_back({addr, tag});
    amo_of_[tag] = addr;
  }

  std::vector<Event> cycle() {
    auto& platform = SimPlatform::instance();
    std::vector<Event> ev;
    gl_bad_.clear();
    ev.swap(next_);
    for (uint32_t b = 0; b < NB; ++b) {
      if (!q_[b].empty() && !sent_[b]) {
        sram_->LsuReqIn.at(b).send(this->lsu_req(q_[b].front().first, q_[b].front().second), 0);
        sent_[b] = true;
      }
    }
    for (uint32_t k = 0; k < ROW_RD_CLIENTS; ++k) {
      if (!rowq_[k].empty() && !row_sent_[k]) {
        sram_->RowRdIn.at(k).send(SmSramRowRd{rowq_[k].front().first, rowq_[k].front().second}, 0);
        row_sent_[k] = true;
      }
    }
    if (!wrq_.empty() && !wr_sent_) {
      SmSramFill f;
      f.row = wrq_.front().first;
      f.src = SmSramSrc::TMA;
      for (uint32_t w = 0; w < SmSram::ROW_BYTES / WORD_SIZE; ++w) {
        const uint64_t v = word_value((uint64_t)f.row * SmSram::ROW_BYTES / WORD_SIZE + w);
        std::memcpy(f.data.data() + w * WORD_SIZE, &v, WORD_SIZE);
      }
      std::fill(f.byteen.begin(), f.byteen.end(), 1);
      sram_->WrIn.at(WR_DXA_PORT).send(f, 0);
      wr_sent_ = true;
    }
    if (!dsq_.empty() && !ds_sent_) {
      const Ds& d = dsq_.front();
      SmSramFill f;
      f.row = d.row;
      f.src = SmSramSrc::DSMEM;
      f.last = d.last;
      for (uint32_t w = 0; w < SmSram::ROW_BYTES / WORD_SIZE; ++w) {
        const uint64_t v = word_value((uint64_t)f.row * SmSram::ROW_BYTES / WORD_SIZE + w);
        std::memcpy(f.data.data() + w * WORD_SIZE, &v, WORD_SIZE);
      }
      for (uint32_t k = 0; k < SmSram::ROW_BYTES; ++k)
        f.byteen[k] = (k / SLOT_BYTES == d.slot);
      sram_->WrIn.at(WR_DSMEM_PORT).send(f, 0);
      ds_sent_ = true;
    }
    if (!cpaq_.empty() && !cpa_sent_) {
      LsuReq q = lsu_req(cpaq_.front().first, cpaq_.front().second);
      sram_->CpaReqIn.at(0).send(q, 0);
      cpa_sent_ = true;
    }
    for (uint32_t a = 0; a < AMO1_CLIENTS; ++a) {
      if (!amoq_[a].empty() && !amo_sent_[a]) {
        const uint64_t addr = amoq_[a].front().first;
        MemReq q;
        q.op = MemOp::AMO_ADD;
        q.addr = addr;
        q.data = word_block(addr, 0);
        q.byteen = 0xffull << (addr % BLOCK);
        q.tag = amoq_[a].front().second;
        q.hart_id = 0;
        sram_->AmoIn.at(a).send(q, 0);
        amo_sent_[a] = true;
      }
    }
    this->gl_pre();
    platform.tick();
    this->gl_post(ev);
    for (uint32_t b = 0; b < NB; ++b) {
      if (sent_[b] && sram_->LsuReqIn.at(b).empty()) {
        ev.push_back({EV_ACCEPT, b, q_[b].front().second, true});
        q_[b].pop_front();
        sent_[b] = false;
      }
      auto& out = sram_->LsuRspOut.at(b);
      while (!out.empty()) {
        const LsuRsp& rsp = out.peek();
        uint32_t tag = (uint32_t)rsp.tag;
        Event e{EV_RSP, b, tag, true};
        auto it = reqs_[b].find(tag);
        uint32_t mask = 0;
        for (uint32_t i = 0; i < NL; ++i)
          mask |= (uint32_t)rsp.mask.test(i) << i;
        if (it == reqs_[b].end() || mask != it->second.mask) {
          std::printf("  response: block %u tag %u mask 0x%x, expected mask 0x%x\n", b, tag, mask,
                      it == reqs_[b].end() ? 0u : it->second.mask);
          e.data_ok = false;
        } else {
          const Req& r = it->second;
          for (uint32_t i = 0; i < NL; ++i) {
            if (!((mask >> i) & 1))
              continue;
            const uint64_t m = lane_bytes<WORD_SIZE>(r, i);
            const uint64_t got = word_of(rsp.data[i], r.addr[i]);
            const uint64_t exp = lane_expect<WORD_SIZE>(r, i);
            if ((got & m) != (exp & m)) {
              e.data_ok = false;
              report_lane(b, tag, i, r.addr[i], got, exp);
            }
          }
        }
        ev.push_back(e);
        out.pop();
      }
    }
    if (wr_sent_ && sram_->WrIn.at(WR_DXA_PORT).empty()) {
      ev.push_back({EV_WR_GNT, 0, wrq_.front().second, true});
      wrq_.pop_front();
      wr_sent_ = false;
    }
    if (ds_sent_ && sram_->WrIn.at(WR_DSMEM_PORT).empty()) {
      ev.push_back({EV_DS_ACC, 0, dsq_.front().tag, true});
      dsq_.pop_front();
      ds_sent_ = false;
    }
    while (!sram_->DsmemBusyOut.empty()) {
      next_.push_back({EV_DS_BUSY, 0, 0, true, (uint32_t)sram_->DsmemBusyOut.peek()});
      sram_->DsmemBusyOut.pop();
    }
    if (cpa_sent_ && sram_->CpaReqIn.at(0).empty()) {
      ev.push_back({EV_CPA_ACC, 0, cpaq_.front().second, true});
      cpaq_.pop_front();
      cpa_sent_ = false;
    }
    {
      auto& out = sram_->CpaRspOut.at(0);
      while (!out.empty()) {
        const LsuRsp& rsp = out.peek();
        uint32_t mask = 0;
        for (uint32_t i = 0; i < NL; ++i)
          mask |= (uint32_t)rsp.mask.test(i) << i;
        auto it = cpa_mask_.find((uint32_t)rsp.tag);
        ev.push_back({EV_CPA_ACK, 0, (uint32_t)rsp.tag, it != cpa_mask_.end() && it->second == mask});
        out.pop();
      }
    }
    for (uint32_t a = 0; a < AMO1_CLIENTS; ++a) {
      if (amo_sent_[a] && sram_->AmoIn.at(a).empty()) {
        ev.push_back({EV_AMO_GNT, a, amoq_[a].front().second, true});
        amoq_[a].pop_front();
        amo_sent_[a] = false;
      }
      auto& out = sram_->AmoRspOut.at(a);
      while (!out.empty()) {
        const MemRsp& rsp = out.peek();
        auto it = amo_of_.find((uint32_t)rsp.tag);
        const bool ok = it != amo_of_.end() && word_of(rsp.data, it->second) == word_value(it->second / WORD_SIZE);
        ev.push_back({EV_AMO_RSP, a, (uint32_t)rsp.tag, ok});
        out.pop();
      }
    }
    for (uint32_t k = 0; k < ROW_RD_CLIENTS; ++k) {
      if (row_sent_[k] && sram_->RowRdIn.at(k).empty()) {
        ev.push_back({EV_ROW_GNT, k, rowq_[k].front().second, true});
        rowq_[k].pop_front();
        row_sent_[k] = false;
      }
    }
    for (uint32_t k = 0; k < ROW_RD_CLIENTS; ++k) {
      auto& rout = sram_->RowRspOut.at(k);
      while (!rout.empty()) {
        const SmSramRowRsp& rsp = rout.peek();
        Event e{EV_ROW_RSP, k, (uint32_t)rsp.tag, true};
        auto it = row_of_.find((uint32_t)rsp.tag);
        if (it == row_of_.end() || rsp.data.size() != SmSram::ROW_BYTES) {
          e.data_ok = false;
        } else {
          const uint64_t w0 = (uint64_t)it->second * SmSram::ROW_BYTES / WORD_SIZE;
          for (uint32_t w = 0; w < SmSram::ROW_BYTES / WORD_SIZE; ++w) {
            uint64_t v = 0;
            std::memcpy(&v, rsp.data.data() + w * WORD_SIZE, WORD_SIZE);
            e.data_ok &= (v == word_value(w0 + w));
          }
        }
        next_.push_back(e);
        rout.pop();
      }
    }
#ifdef VX_CFG_EXT_MBAR_ENABLE
    for (uint32_t k = 0; k < VX_CFG_LMEM_NUM_BANKS; ++k) {
      auto& m = sram_->MbarEvtOut.at(k);
      while (!m.empty()) {
        ev.push_back({EV_MBAR, k, m.peek().row, true, (uint32_t)m.peek().flip});
        m.pop();
      }
    }
#endif
#ifdef VX_CFG_EXT_DLMEM_ENABLE
    for (uint32_t k = 0; k < 3; ++k) {
      auto& c = sram_->CommitOut.at(k);
      while (!c.empty()) {
        next_.push_back({EV_COMMIT, k, 0, true, c.peek().cnt});
        c.pop();
      }
    }
#endif
    ++now_;
    return ev;
  }

private:
#ifdef VX_CFG_EXT_MBAR_ENABLE
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
#endif

  static LsuReq lsu_req(const Req& r, uint32_t tag) {
    LsuReq q(NL);
    q.op = (r.kind == K_ST) ? MemOp::ST : (r.kind == K_ADD) ? MemOp::AMO_ADD : (r.kind == K_CAS) ? MemOp::AMO_CAS : MemOp::LD;
#ifdef VX_CFG_EXT_MBAR_ENABLE
    if (is_mbar(r))
      q.op = mbar_memop(r.mop);
#endif
    q.tag = tag;
    for (uint32_t i = 0; i < NL; ++i) {
      if (!(r.mask >> i & 1))
        continue;
      q.mask.set(i);
      q.addrs[i] = r.addr[i];
      q.byteen[i] = ((r.bytes >= 8) ? 0xffull : ((1ull << r.bytes) - 1)) << (r.addr[i] % BLOCK);
      if (q.op != MemOp::LD)
        q.data[i] = word_block(r.addr[i], lane_data<WORD_SIZE>(r, i));
      // The comparand at the bits of the lane's bytes, as the LSU sends it from its register.
      if (r.kind == K_CAS)
        q.amo_cmp[i] = (lane_cmp<WORD_SIZE>(r, i) >> ((r.addr[i] % WORD_SIZE) * 8)) &
                       ((r.bytes >= 8) ? ~0ull : ((1ull << (r.bytes * 8)) - 1));
    }
#ifdef VX_CFG_EXT_DLMEM_ENABLE
    q.flags.lacc = r.lacc;
#endif
    return q;
  }

  void gl_pre() {
    for (uint32_t b = 0; b < NB; ++b) {
      if (!glq_[b].empty() && !gl_sent_[b]) {
        l1_->LsuReqIn.at(b).send(this->gl_lsu_req(glq_[b].front().first, glq_[b].front().second), 0);
        gl_sent_[b] = true;
      }
    }
    if (!mrsp_.empty() && mrsp_.front().when <= now_ && l1_->MemRspIn.empty()) {
      l1_->MemRspIn.send(mrsp_.front().rsp, 0);
      mrsp_.pop_front();
    }
    if (flush_on_ && !flush_sent_) {
      l1_->flush_begin();
      flush_sent_ = true;
    }
  }

  void gl_post(std::vector<Event>& ev) {
    for (uint32_t b = 0; b < NB; ++b) {
      if (gl_sent_[b] && l1_->LsuReqIn.at(b).empty()) {
        ev.push_back({EV_GL_ACC, b, glq_[b].front().second, true});
        glq_[b].pop_front();
        gl_sent_[b] = false;
      }
      auto& out = l1_->LsuRspOut.at(b);
      while (!out.empty()) {
        const LsuRsp& rsp = out.peek();
        const uint32_t tag = uint32_t(rsp.tag) & (GL_TAGS - 1);
        Event e{EV_GL_RSP, b, tag, true, 0};
        auto it = glreq_[b].find(tag);
        for (uint32_t i = 0; i < NL; ++i) {
          if (!rsp.mask.test(i))
            continue;
          e.cnt |= 1u << i;
          uint64_t w = 0;
          if (it != glreq_[b].end() && rsp.data.at(i))
            std::memcpy(&w, rsp.data.at(i)->data() + (it->second.addr[i] % BLOCK) / gl::WORD * gl::WORD, gl::WORD);
          if (it == glreq_[b].end() || !gl_lane_ok(it->second, i, w)) {
            e.data_ok = false;
            gl_bad_.push_back({b, tag, i, w});
          }
        }
        ev.push_back(e);
        out.pop();
      }
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
      } else if (m.op != MemOp::LD || m.flags.fence) {
        this->reply(m, std::vector<uint8_t>(BLOCK, 0));
      } else {
        auto& secs = rd[base / gl::LINE];
        for (uint32_t o = 0; o < BLOCK; ++o)
          secs |= uint32_t(be[o]) << ((base % gl::LINE + o) / gl::SECTOR);
        this->reply(m, mem_.read(base, BLOCK));
      }
      mreq.pop();
    }
    for (auto& kv : rd)
      ev.push_back({EV_GL_MEMRD, 0, uint32_t(kv.first), true, kv.second});
    if (fill_fwd_ && sram_->WrIn.at(0).empty()) {
      ev.push_back({EV_GL_FILL, 0, 0, true, fill_row_});
      fill_fwd_ = false;
    }
    if (!fill_fwd_ && !l1_->FillOut.empty()) {
      fill_row_ = l1_->FillOut.peek().row;
      sram_->WrIn.at(0).send(l1_->FillOut.peek(), 0);
      l1_->FillOut.pop();
      fill_fwd_ = true;
    }
    if (flush_on_ && flush_sent_ && l1_->flush_done()) {
      ev.push_back({EV_GL_FLUSHED, 0, 0, true});
      flush_on_ = false;
    }
  }

  void reply(const MemReq& m, const std::vector<uint8_t>& bytes) {
    auto blk = std::make_shared<mem_block_t>();
    std::memcpy(blk->data(), bytes.data(), BLOCK);
    const uint64_t when = std::max(now_ + mem_.delay, mrsp_.empty() ? 0 : mrsp_.back().when);
    mrsp_.push_back({when, MemRsp(m.tag, m.hart_id, m.uuid, blk)});
  }

  LsuReq gl_lsu_req(const gl::Req& r, uint32_t tag) {
    LsuReq q(NL);
    q.op = (r.kind == gl::K_ST) ? MemOp::ST : MemOp::LD;
    q.tag = tag;
    for (uint32_t i = 0; i < NL; ++i) {
      if (!(r.mask >> i & 1))
        continue;
      const uint32_t woff = uint32_t(r.addr[i] % BLOCK) / gl::WORD * gl::WORD;
      q.mask.set(i);
      q.addrs[i] = r.addr[i];
      q.byteen[i] = uint64_t(gl::lane_be(r, i)) << woff;
      auto b = std::make_shared<mem_block_t>();
      std::memset(b->data(), 0, b->size());
      const uint64_t w = gl::lane_wdata(r, i);
      std::memcpy(b->data() + woff, &w, gl::WORD);
      q.data[i] = b;
    }
    return q;
  }

  struct Pend {
    uint64_t when;
    MemRsp rsp;
  };
  SmL1::Ptr l1_;
  bool gl_sent_[NB] = {};
  std::deque<std::pair<gl::Req, uint32_t>> glq_[NB];
  std::map<uint32_t, gl::Req> glreq_[NB];
  std::vector<GlBad> gl_bad_;
  std::deque<Pend> mrsp_;
  gl::FlatMem mem_;
  uint32_t fill_row_ = 0;
  bool fill_fwd_ = false, flush_on_ = false, flush_sent_ = false;
  SmSram::Ptr sram_;
  uint64_t now_ = 0;
  bool sent_[NB] = {};
  bool row_sent_[ROW_RD_CLIENTS] = {};
  bool wr_sent_ = false;
  bool amo_sent_[AMO1_CLIENTS] = {};
  std::deque<std::pair<Req, uint32_t>> q_[NB];
  std::map<uint32_t, Req> reqs_[NB];
  std::deque<std::pair<uint32_t, uint32_t>> rowq_[ROW_RD_CLIENTS];
  std::map<uint32_t, uint32_t> row_of_;
  std::deque<std::pair<uint32_t, uint32_t>> wrq_;
  std::deque<std::pair<uint64_t, uint32_t>> amoq_[AMO1_CLIENTS];
  std::map<uint32_t, uint64_t> amo_of_;
  struct Ds {
    uint32_t row = 0, slot = 0;
    bool last = false;
    uint32_t tag = 0;
  };
  std::deque<Ds> dsq_;
  bool ds_sent_ = false;
  std::deque<std::pair<Req, uint32_t>> cpaq_;
  std::map<uint32_t, uint32_t> cpa_mask_;
  bool cpa_sent_ = false;
  std::vector<Event> next_;
};

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  uint32_t seed = 1, nrand = 2000;
  const char* csv_path = nullptr;
  bool unified = false, directed = false;
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--seed=", 7)) seed = (uint32_t)atoi(argv[i] + 7);
    else if (!strncmp(argv[i], "--random=", 9)) nrand = (uint32_t)atoi(argv[i] + 9);
    else if (!strncmp(argv[i], "--csv=", 6)) csv_path = argv[i] + 6;
    else if (!strcmp(argv[i], "--unified")) unified = true;
    else if (!strcmp(argv[i], "--directed")) directed = true;
    else { std::printf("usage: %s [--seed=N] [--random=N] [--csv=file] [--unified] [--directed]\n", argv[0]); return 2; }
  }
  FILE* csv = csv_path ? std::fopen(csv_path, "w") : nullptr;
  std::printf("sm_sram (SimX): %u blocks x %u lanes, banks %u x %u B, seed %u\n", NB, NL, VX_CFG_LMEM_NUM_BANKS,
              VX_CFG_LMEM_BANK_BYTES, seed);
  SimxDriver d;
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  const bool lacc_ok = true;
#else
  const bool lacc_ok = false;
#endif
  Runner<SimxDriver, WORD_SIZE> run(d, NL, NB, csv, lacc_ok);
  int rc = run.run(seed, nrand, unified, directed);
  if (csv)
    std::fclose(csv);
  std::printf(rc ? "FAILED!\n" : "PASSED!\n");
  return rc;
}
