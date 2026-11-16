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

// SimX twin of hw/unittest/smem_gate: SmemGate, the H100 SM path's
// shared-memory port, run through the same cases (smem_gate_cases.h) with the
// same cycle contract; --csv=<file> must equal the RTL bench's.

#include "smem_gate.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>

#include "smem_gate_cases.h"

using namespace vortex;
using namespace smem_gate_cases;

#ifndef VX_CFG_H100_SM_ENABLE
#error "the smem_bank twin tests the H100 SM path: build with -DVX_CFG_H100_SM_ENABLE"
#endif

constexpr uint32_t NB = VX_CFG_NUM_LSU_BLOCKS;
constexpr uint32_t NL = VX_CFG_NUM_LSU_LANES;
constexpr uint32_t WORD_SIZE = LSU_WORD_SIZE;
constexpr uint64_t BLOCK = VX_CFG_MEM_BLOCK_SIZE;

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
// tick; the other clients' request line of cycle c reaches the gate in c + 1.
class SimxDriver {
public:
  SimxDriver() {
    gate_ = SmemGate::Create("smem_gate", NB);
    SimPlatform::instance().reset();
    now_ = 0;
  }

  uint64_t now() const { return now_; }

  void push(uint32_t b, const Req& r, uint32_t tag) {
    q_[b].push_back({r, tag});
    reqs_[b][tag] = r;
  }

  void set_ext(bool v) { ext_ = v; }

  std::vector<Event> cycle() {
    auto& platform = SimPlatform::instance();
    std::vector<Event> ev;
    if (ext_)
      gate_->ExtIn.send(true, 1);
    for (uint32_t b = 0; b < NB; ++b) {
      if (!q_[b].empty() && !sent_[b] && gate_->ReqIn[b].empty()) {
        gate_->ReqIn[b].send(this->lsu_req(q_[b].front().first, q_[b].front().second), 0);
        sent_[b] = true;
      }
      if (!sink_[b].empty() && sink_[b].front().due <= now_) {
        gate_->RspIn[b].send(sink_[b].front().rsp, 0);
        sink_[b].pop_front();
      }
    }
    platform.tick();
    for (uint32_t b = 0; b < NB; ++b) {
      while (!gate_->ReqOut[b].empty()) {
        const LsuReq& out = gate_->ReqOut[b].peek();
        uint32_t mask = 0;
        for (uint32_t i = 0; i < NL; ++i)
          mask |= (uint32_t)out.mask.test(i) << i;
        ev.push_back({EV_ACCEPT, b, (uint32_t)out.tag, mask, true});
        if (out.op != MemOp::ST) {
          Sink s{now_ + SINK_DELAY, LsuRsp(NL)};
          s.rsp.mask = out.mask;
          s.rsp.tag = out.tag;
          for (uint32_t i = 0; i < NL; ++i) {
            if (out.mask.test(i))
              s.rsp.data[i] = word_block(out.addrs[i], word_value(out.addrs[i] / WORD_SIZE));
          }
          sink_[b].push_back(s);
        }
        gate_->ReqOut[b].pop();
        q_[b].pop_front();
        sent_[b] = false;
      }
      while (!gate_->RspOut[b].empty()) {
        const LsuRsp& rsp = gate_->RspOut[b].peek();
        uint32_t tag = (uint32_t)rsp.tag;
        Event e{EV_RSP, b, tag, 0, true};
        auto it = reqs_[b].find(tag);
        uint32_t mask = 0;
        for (uint32_t i = 0; i < NL; ++i)
          mask |= (uint32_t)rsp.mask.test(i) << i;
        if (it == reqs_[b].end() || mask != it->second.mask) {
          e.data_ok = false;
        } else {
          for (uint32_t i = 0; i < NL; ++i) {
            if (mask >> i & 1)
              e.data_ok &= word_of(rsp.data[i], it->second.addr[i]) == word_value(it->second.addr[i] / WORD_SIZE);
          }
        }
        ev.push_back(e);
        gate_->RspOut[b].pop();
      }
    }
    // A fold arrives with its accept or later, never before it.
#ifdef VX_CFG_EXT_DLMEM_ENABLE
    for (uint32_t b = 0; b < NB; ++b) {
      while (!gate_->FoldOut[b].empty()) {
        ev.push_back({EV_FOLD, b, 0, 0, true, gate_->FoldOut[b].peek().cnt});
        gate_->FoldOut[b].pop();
      }
    }
#endif
    ++now_;
    return ev;
  }

private:
  struct Sink {
    uint64_t due;
    LsuRsp rsp;
  };

  static LsuReq lsu_req(const Req& r, uint32_t tag) {
    LsuReq q(NL);
    q.op = (r.kind == K_ST) ? MemOp::ST : (r.kind == K_ADD) ? MemOp::AMO_ADD : (r.kind == K_CAS) ? MemOp::AMO_CAS : MemOp::LD;
    q.tag = tag;
    for (uint32_t i = 0; i < NL; ++i) {
      if (!(r.mask >> i & 1))
        continue;
      q.mask.set(i);
      q.addrs[i] = r.addr[i];
      q.byteen[i] = ((r.bytes >= 8) ? 0xffull : ((1ull << r.bytes) - 1)) << (r.addr[i] % BLOCK);
      if (q.op != MemOp::LD)
        q.data[i] = word_block(r.addr[i], word_value(r.addr[i] / WORD_SIZE));
    }
#ifdef VX_CFG_EXT_DLMEM_ENABLE
    q.flags.lacc = r.lacc;
#endif
    return q;
  }

  SmemGate::Ptr gate_;
  uint64_t now_ = 0;
  bool ext_ = false;
  bool sent_[NB] = {};
  std::deque<std::pair<Req, uint32_t>> q_[NB];
  std::map<uint32_t, Req> reqs_[NB];
  std::deque<Sink> sink_[NB];
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
  std::printf("smem_gate (SimX): %u blocks x %u lanes, banks %u x %u B, seed %u\n", NB, NL, VX_CFG_LMEM_NUM_BANKS,
              VX_CFG_LMEM_BANK_BYTES, seed);
  SimxDriver d;
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  const bool lacc_ok = true;
#else
  const bool lacc_ok = false;
#endif
  Runner<SimxDriver, WORD_SIZE> run(d, NL, NB, csv, lacc_ok);
  int rc = run.run(seed, nrand);
  if (csv)
    std::fclose(csv);
  std::printf(rc ? "FAILED!\n" : "PASSED!\n");
  return rc;
}
