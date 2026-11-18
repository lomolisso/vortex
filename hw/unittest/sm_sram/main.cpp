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

// VX_sm_sram testbench: the H100 SM's one array driven by the LSU blocks'
// shared-memory legs and its row, write, cp.async and one-lane atomic
// clients (sm_sram_cases.h). --csv=<file>
// writes the per-request timing the SimX twin (tests/unittest/smem_bank) must
// reproduce.

#include "vl_simulator.h"
#include "VVX_sm_sram_top.h"
#include "VX_config.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

#include "sm_sram_cases.h"

using namespace sm_sram_cases;

#ifndef VX_CFG_H100_SM_ENABLE
#error "sm_sram tests the H100 SM path: build with -DVX_CFG_H100_SM_ENABLE"
#endif

constexpr uint32_t clog2(uint32_t n) { return n <= 1 ? 0 : 1 + clog2((n + 1) / 2); }

constexpr uint32_t NB         = VX_CFG_NUM_LSU_BLOCKS;
constexpr uint32_t NL         = VX_CFG_NUM_LSU_LANES;
constexpr uint32_t WORD_SIZE  = VX_CFG_XLEN / 8;
constexpr uint32_t ROW_BYTES  = VX_CFG_SM_DATAPATH_BYTES;
constexpr uint32_t ADDR_BITS  = clog2(ROW_BYTES) - clog2(WORD_SIZE) + clog2(VX_CFG_SM_SRAM_BYTES / ROW_BYTES);
constexpr uint32_t TAG_BITS   = 6;
constexpr uint32_t ROW_TAG_BITS = 8;
constexpr uint32_t ROW_BITS   = clog2(VX_CFG_SM_SRAM_BYTES / ROW_BYTES);
constexpr uint32_t DATA_BITS  = WORD_SIZE * 8;
constexpr uint32_t CNT_BITS   = clog2(NL + 1);
#ifdef VX_CFG_EXT_FPAMO_ENABLE
constexpr uint32_t OP_BITS    = 5;
#else
constexpr uint32_t OP_BITS    = 4;
#endif
constexpr uint32_t OP_ADD     = 2;
constexpr uint32_t OP_CAS     = 9;
constexpr uint32_t BANK_BITS  = clog2(VX_CFG_LMEM_NUM_BANKS);
constexpr uint32_t FILL_SRC_TMA = 1;
constexpr uint32_t FILL_SRC_DSMEM = 2;
constexpr uint32_t FILL_BITS  = 2 + ROW_BYTES * 9 + ROW_BITS;
constexpr uint32_t SLOT_BYTES = VX_CFG_N5_SLOT_BYTES;
constexpr uint32_t log2up(uint32_t n) { return n > 1 ? clog2(n) : 1; }
constexpr uint32_t GL_ADDR_BITS = VX_CFG_MEM_ADDR_WIDTH - clog2(WORD_SIZE);
constexpr uint32_t MEM_ADDR_BITS = VX_CFG_MEM_ADDR_WIDTH - clog2(ROW_BYTES);
// The L1's miss-bus tag: {uuid, special (atomic or fence), index} (VX_gpu_pkg SM_L1_MEM_TAG_WIDTH).
constexpr uint32_t MEM_IDX_BITS = std::max(clog2(VX_CFG_SM_MAX_READ_LINES),
    clog2(2 * VX_CFG_LSU_PENDING_SIZE) + clog2(NB) + clog2(ROW_BYTES / WORD_SIZE) + log2up(NL));

static_assert(NL <= MAX_LANES, "the bench drives at most 32 lanes");
static_assert(DATA_BITS == 64, "the H100 SM path is RV64: one lane word per uint64");

template <typename T>
static void set_bits(T& sig, uint32_t lsb, uint32_t width, uint64_t value) {
  if constexpr (std::is_integral_v<T>) {
    uint64_t mask = (width >= 64) ? ~0ull : ((1ull << width) - 1);
    uint64_t v = static_cast<uint64_t>(sig);
    v = (v & ~(mask << lsb)) | ((value & mask) << lsb);
    sig = static_cast<T>(v);
  } else {
    uint32_t i = 0;
    while (i < width) {
      uint32_t bit = lsb + i;
      uint32_t w = bit / 32;
      uint32_t o = bit % 32;
      uint32_t n = std::min(32 - o, width - i);
      uint32_t m = (n == 32) ? 0xffffffffu : ((1u << n) - 1);
      uint32_t chunk = static_cast<uint32_t>(value >> i) & m;
      sig[w] = (sig[w] & ~(m << o)) | (chunk << o);
      i += n;
    }
  }
}

template <typename T>
static uint64_t get_bits(const T& sig, uint32_t lsb, uint32_t width) {
  if constexpr (std::is_integral_v<T>) {
    uint64_t mask = (width >= 64) ? ~0ull : ((1ull << width) - 1);
    return (static_cast<uint64_t>(sig) >> lsb) & mask;
  } else {
    uint64_t value = 0;
    uint32_t i = 0;
    while (i < width) {
      uint32_t bit = lsb + i;
      uint32_t w = bit / 32;
      uint32_t o = bit % 32;
      uint32_t n = std::min(32 - o, width - i);
      uint32_t m = (n == 32) ? 0xffffffffu : ((1u << n) - 1);
      value |= static_cast<uint64_t>((sig[w] >> o) & m) << i;
      i += n;
    }
    return value;
  }
}

class RtlDriver {
public:
  RtlDriver() {
    auto* t = sim_.operator->();
    this->set_cfg(reset_cfg());
    t->reset = 1;
    for (int i = 0; i < 8; ++i)
      tick_ = sim_.step(tick_, 2);
    t->reset = 0;
    now_ = 0;
    gl_tag_bits_ = t->gl_tag_bits;
  }

  // The package's stage counts against the formulas the cases' goldens use.
  bool pipe_consts_ok() {
    auto* t = sim_.operator->();
#ifdef VX_CFG_EXT_DLMEM_ENABLE
    const bool dlmem = true;
#else
    const bool dlmem = false;
#endif
    const uint32_t lmem = vortex::smem_bank_ref::smem_pipe_stages(VX_CFG_NUM_OPCS, dlmem);
    const uint32_t l1 = vortex::l1_slot_ref::l1_pipe_stages(VX_CFG_NUM_OPCS);
    const uint32_t l1m = vortex::l1_slot_ref::l1_miss_pipe_stages(VX_CFG_NUM_OPCS);
    const bool ok = t->pkg_lmem_pipe == lmem && t->pkg_l1_pipe == l1 && t->pkg_l1_miss_pipe == l1m;
    if (!ok)
      std::printf("  MISMATCH pkg LMEM_PIPE_OVERHEAD %u vs smem_pipe_stages %u, L1_PIPE_OVERHEAD %u vs l1_pipe_stages %u,"
                  " L1_MISS_PIPE_OVERHEAD %u vs l1_miss_pipe_stages %u\n",
                  (uint32_t)t->pkg_lmem_pipe, lmem, (uint32_t)t->pkg_l1_pipe, l1, (uint32_t)t->pkg_l1_miss_pipe, l1m);
    return ok;
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

  // The split the bench applies from the next cycle (the top's cfg inputs).
  bool set_cfg(const ArrayCfg& c) {
    auto* t = sim_.operator->();
    t->cfg_s_rows = c.s_rows;
    t->cfg_l1_lines = c.l1_lines;
    t->cfg_l1_protect = c.l1_protect;
    return true;
  }

  void push_gl(uint32_t b, const gl::Req& r, uint32_t tag) {
    glq_[b].push_back({r, tag});
    glreq_[b][tag] = r;
  }

  const std::vector<GlBad>& gl_bad() const { return gl_bad_; }

  void gl_flush() { flush_on_ = true; }

  bool gl_busy() const {
    bool q = false;
    for (uint32_t b = 0; b < NB; ++b)
      q |= !glq_[b].empty();
    return q || l1_busy_ || flush_on_ || !mrsp_.empty();
  }

  void push_amo(uint32_t client, uint64_t addr, uint32_t tag) {
    amoq_[client].push_back({addr, tag});
    amo_of_[tag] = addr;
  }

  std::vector<Event> cycle() {
    auto* t = sim_.operator->();
    std::vector<Event> ev;
    gl_bad_.clear();
    for (uint32_t b = 0; b < NB; ++b) {
      bool v = !q_[b].empty();
      const Req r = v ? q_[b].front().first : Req{};
      const uint32_t tag = v ? q_[b].front().second : 0;
      set_bits(t->in_valid, b, 1, v);
      set_bits(t->in_rw, b, 1, r.kind == K_ST);
      set_bits(t->in_mask, b * NL, NL, r.mask);
      for (uint32_t i = 0; i < NL; ++i) {
        const bool on = (r.mask >> i) & 1;
        uint64_t be = on ? (((1ull << r.bytes) - 1) << (r.addr[i] % WORD_SIZE)) : 0;
        set_bits(t->in_addr, (b * NL + i) * ADDR_BITS, ADDR_BITS, r.addr[i] / WORD_SIZE);
        set_bits(t->in_byteen, (b * NL + i) * WORD_SIZE, WORD_SIZE, be);
        // A CAS's second beat carries its comparands.
        set_bits(t->in_data, (b * NL + i) * DATA_BITS, DATA_BITS,
                 !on ? 0 : cas2_[b] ? lane_cmp<WORD_SIZE>(r, i) : lane_data<WORD_SIZE>(r, i));
      }
      set_bits(t->in_amo, b, 1, is_amo(r) || is_mbar(r));
      set_bits(t->in_amo_op, b * OP_BITS, OP_BITS, is_mbar(r) ? r.mop : r.kind == K_CAS ? OP_CAS : OP_ADD);
      set_bits(t->in_lacc, b, 1, r.lacc);
      set_bits(t->in_tag, b * TAG_BITS, TAG_BITS, tag);
      set_bits(t->in_rsp_ready, b, 1, 1);
    }
    for (uint32_t k = 0; k < ROW_RD_CLIENTS; ++k) {
      const bool rv = !rowq_[k].empty();
      set_bits(t->rd_valid, k, 1, rv);
      set_bits(t->rd_row, k * 32, 32, rv ? rowq_[k].front().first : 0);
      set_bits(t->rd_tag, k * ROW_TAG_BITS, ROW_TAG_BITS, rv ? rowq_[k].front().second : 0);
    }
    {
      // sm_sram_fill_t {row, byteen, data, src}, src in the low bits: the row's pattern, every byte
      const bool wv = !wrq_.empty();
      const uint32_t row = wv ? wrq_.front().first : 0;
      set_bits(t->wr_valid, 0, 1, wv);
      set_bits(t->wr_data, 0, 2, FILL_SRC_TMA);
      for (uint32_t w = 0; w < ROW_BYTES / WORD_SIZE; ++w)
        set_bits(t->wr_data, 2 + w * DATA_BITS, DATA_BITS, word_value((uint64_t)row * ROW_BYTES / WORD_SIZE + w));
      for (uint32_t b = 0; b < ROW_BYTES; b += 32)
        set_bits(t->wr_data, 2 + ROW_BYTES * 8 + b, std::min<uint32_t>(32, ROW_BYTES - b), ~0ull);
      set_bits(t->wr_data, 2 + ROW_BYTES * 9, ROW_BITS, row);
    }
    {
      // a DSMEM slot: its bytes of the row's pattern
      const bool dv = !dsq_.empty();
      const Ds d = dv ? dsq_.front() : Ds{};
      set_bits(t->wr_valid, 1, 1, dv);
      set_bits(t->wr_data, FILL_BITS, 2, FILL_SRC_DSMEM);
      for (uint32_t w = 0; w < ROW_BYTES / WORD_SIZE; ++w)
        set_bits(t->wr_data, FILL_BITS + 2 + w * DATA_BITS, DATA_BITS,
                 word_value((uint64_t)d.row * ROW_BYTES / WORD_SIZE + w));
      for (uint32_t b = 0; b < ROW_BYTES; b += 32) {
        uint64_t m = 0;
        for (uint32_t k = 0; k < std::min<uint32_t>(32, ROW_BYTES - b); ++k)
          m |= (uint64_t)((b + k) / SLOT_BYTES == d.slot) << k;
        set_bits(t->wr_data, FILL_BITS + 2 + ROW_BYTES * 8 + b, std::min<uint32_t>(32, ROW_BYTES - b), m);
      }
      set_bits(t->wr_data, FILL_BITS + 2 + ROW_BYTES * 9, ROW_BITS, d.row);
      t->wr_dsmem_last = dv && d.last;
    }
    {
      // cp.async on block 0: every touched lane word gets its pattern; an
      // mbarrier op (cp.async.mbarrier.arrive) carries its operands
      const bool cv = !cpaq_.empty();
      const Req r = cv ? cpaq_.front().first : Req{};
      set_bits(t->cpa_valid, 0, 1, cv);
      set_bits(t->cpa_mask, 0, NL, r.mask);
      for (uint32_t i = 0; i < NL; ++i) {
        const bool on = (r.mask >> i) & 1;
        set_bits(t->cpa_addr, i * ADDR_BITS, ADDR_BITS, r.addr[i] / WORD_SIZE);
        set_bits(t->cpa_byteen, i * WORD_SIZE, WORD_SIZE, on ? ((1ull << r.bytes) - 1) << (r.addr[i] % WORD_SIZE) : 0);
        set_bits(t->cpa_data, i * DATA_BITS, DATA_BITS,
                 !on ? 0 : is_mbar(r) ? lane_data<WORD_SIZE>(r, i) : word_value(r.addr[i] / WORD_SIZE));
      }
      set_bits(t->cpa_amo, 0, 1, is_mbar(r));
      set_bits(t->cpa_amo_op, 0, OP_BITS, is_mbar(r) ? r.mop : 0);
      set_bits(t->cpa_tag, 0, TAG_BITS, cv ? cpaq_.front().second : 0);
      set_bits(t->cpa_rsp_ready, 0, 1, 1);
    }
    const uint32_t ab = t->amo_attr_bits;
    for (uint32_t a = 0; a < AMO1_CLIENTS; ++a) {
      const bool av = !amoq_[a].empty();
      const uint64_t addr = av ? amoq_[a].front().first : 0;
      set_bits(t->amo_valid, a, 1, av);
      set_bits(t->amo_row, a * ROW_BITS, ROW_BITS, addr / ROW_BYTES);
      set_bits(t->amo_bank, a * BANK_BITS, BANK_BITS, (addr % ROW_BYTES) / VX_CFG_LMEM_BANK_BYTES);
      set_bits(t->amo_byteen, a * WORD_SIZE, WORD_SIZE, 0xff);
      set_bits(t->amo_wdata, a * DATA_BITS, DATA_BITS, 0);
      for (uint32_t b = 0; b < ab; b += 32)
        set_bits(t->amo_attr, a * ab + b, std::min<uint32_t>(32, ab - b), 0);
      // amo_req_t: amo_valid in bit 0, amo_op above it
      set_bits(t->amo_attr, a * ab, 1 + OP_BITS, 1 | (OP_ADD << 1));
      set_bits(t->amo_tag, a * ROW_TAG_BITS, ROW_TAG_BITS, av ? amoq_[a].front().second : 0);
    }
    for (uint32_t b = 0; b < NB; ++b) {
      const bool v = !glq_[b].empty();
      const gl::Req r = v ? glq_[b].front().first : gl::Req{};
      set_bits(t->gl_in_valid, b, 1, v);
      set_bits(t->gl_in_rw, b, 1, r.kind == gl::K_ST);
      set_bits(t->gl_in_mask, b * NL, NL, r.mask);
      for (uint32_t i = 0; i < NL; ++i) {
        const bool on = (r.mask >> i) & 1;
        set_bits(t->gl_in_addr, (b * NL + i) * GL_ADDR_BITS, GL_ADDR_BITS, r.addr[i] / WORD_SIZE);
        set_bits(t->gl_in_byteen, (b * NL + i) * WORD_SIZE, WORD_SIZE, on ? gl::lane_be(r, i) : 0);
        set_bits(t->gl_in_data, (b * NL + i) * DATA_BITS, DATA_BITS, on ? gl::lane_wdata(r, i) : 0);
      }
      set_bits(t->gl_in_tag, b * gl_tag_bits_, gl_tag_bits_, v ? glq_[b].front().second : 0);
      set_bits(t->gl_in_rsp_ready, b, 1, 1);
    }
    t->mem_req_ready = 1;
    const bool mv = !mrsp_.empty() && mrsp_.front().when <= now_;
    t->mem_rsp_valid = mv;
    if (mv) {
      for (uint32_t i = 0; i < ROW_BYTES; ++i)
        set_bits(t->mem_rsp_data, i * 8, 8, mrsp_.front().data[i]);
      set_bits(t->mem_rsp_tag, 0, 64, mrsp_.front().tag);
    }
    t->flush_req = flush_on_;
    t->eval();
    this->gl_outputs(ev, mv);
    for (uint32_t b = 0; b < NB; ++b) {
      if (!q_[b].empty() && get_bits(t->in_ready, b, 1)) {
        if (q_[b].front().first.kind == K_CAS && !cas2_[b]) {
          cas2_[b] = true;
        } else {
          ev.push_back({EV_ACCEPT, b, q_[b].front().second, true});
          q_[b].pop_front();
          cas2_[b] = false;
        }
      }
      if (get_bits(t->in_rsp_valid, b, 1)) {
        uint32_t tag = (uint32_t)get_bits(t->in_rsp_tag, b * TAG_BITS, TAG_BITS);
        uint32_t mask = (uint32_t)get_bits(t->in_rsp_mask, b * NL, NL);
        Event e{EV_RSP, b, tag, true};
        auto it = reqs_[b].find(tag);
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
            const uint64_t got = get_bits(t->in_rsp_data, (b * NL + i) * DATA_BITS, DATA_BITS);
            const uint64_t exp = lane_expect<WORD_SIZE>(r, i);
            if ((got & m) != (exp & m)) {
              e.data_ok = false;
              report_lane(b, tag, i, r.addr[i], got, exp);
            }
          }
        }
        ev.push_back(e);
      }
    }
    for (uint32_t k = 0; k < ROW_RD_CLIENTS; ++k) {
      if (!rowq_[k].empty() && get_bits(t->rd_ready, k, 1)) {
        ev.push_back({EV_ROW_GNT, k, rowq_[k].front().second, true});
        rowq_[k].pop_front();
      }
    }
    if (!wrq_.empty() && get_bits(t->wr_ready, 0, 1)) {
      ev.push_back({EV_WR_GNT, 0, wrq_.front().second, true});
      wrq_.pop_front();
    }
    if (!dsq_.empty() && get_bits(t->wr_ready, 1, 1)) {
      ev.push_back({EV_DS_ACC, 0, dsq_.front().tag, true});
      dsq_.pop_front();
    }
    if ((bool)t->wr_dsmem_busy != ds_busy_) {
      ds_busy_ = t->wr_dsmem_busy;
      ev.push_back({EV_DS_BUSY, 0, 0, true, (uint32_t)ds_busy_});
    }
    if (!cpaq_.empty() && get_bits(t->cpa_ready, 0, 1)) {
      ev.push_back({EV_CPA_ACC, 0, cpaq_.front().second, true});
      cpaq_.pop_front();
    }
    if (get_bits(t->cpa_rsp_valid, 0, 1)) {
      const uint32_t tag = (uint32_t)get_bits(t->cpa_rsp_tag, 0, TAG_BITS);
      auto it = cpa_mask_.find(tag);
      const bool ok = it != cpa_mask_.end() && it->second == (uint32_t)get_bits(t->cpa_rsp_mask, 0, NL);
      ev.push_back({EV_CPA_ACK, 0, tag, ok});
    }
    for (uint32_t a = 0; a < AMO1_CLIENTS; ++a) {
      if (!amoq_[a].empty() && get_bits(t->amo_ready, a, 1)) {
        ev.push_back({EV_AMO_GNT, a, amoq_[a].front().second, true});
        amoq_[a].pop_front();
      }
      if (get_bits(t->amo_rsp_valid, a, 1)) {
        uint32_t tag = (uint32_t)get_bits(t->amo_rsp_tag, a * ROW_TAG_BITS, ROW_TAG_BITS);
        auto it = amo_of_.find(tag);
        const bool ok = it != amo_of_.end() &&
                        get_bits(t->amo_rsp_data, a * DATA_BITS, DATA_BITS) == word_value(it->second / WORD_SIZE);
        ev.push_back({EV_AMO_RSP, a, tag, ok});
      }
    }
    for (uint32_t k = 0; k < ROW_RD_CLIENTS; ++k) {
      if (!get_bits(t->rd_rsp_valid, k, 1))
        continue;
      uint32_t tag = (uint32_t)get_bits(t->rd_rsp_tag, 0, ROW_TAG_BITS);
      Event e{EV_ROW_RSP, k, tag, true};
      auto it = row_of_.find(tag);
      if (it == row_of_.end()) {
        e.data_ok = false;
      } else {
        const uint64_t w0 = (uint64_t)it->second * ROW_BYTES / WORD_SIZE;
        for (uint32_t w = 0; w < ROW_BYTES / WORD_SIZE; ++w)
          e.data_ok &= get_bits(t->rd_rsp_data, w * DATA_BITS, DATA_BITS) == word_value(w0 + w);
      }
      ev.push_back(e);
    }
#ifdef VX_CFG_EXT_MBAR_ENABLE
    for (uint32_t k = 0; k < VX_CFG_LMEM_NUM_BANKS; ++k) {
      if (get_bits(t->mbar_evt_valid, k, 1))
        ev.push_back({EV_MBAR, k, (uint32_t)get_bits(t->mbar_evt_row, k * ROW_BITS, ROW_BITS), true,
                      (uint32_t)get_bits(t->mbar_evt_flip, k, 1)});
    }
#endif
    for (uint32_t k = 0; k < 3; ++k) {
      if (get_bits(t->acc_commit_valid, k, 1))
        ev.push_back({EV_COMMIT, k, 0, true, (uint32_t)get_bits(t->acc_commit_cnt, k * CNT_BITS, CNT_BITS)});
    }
    tick_ = sim_.step(tick_, 2);
    // Register outputs after the edge, as the SimX twin reads SmL1::busy() and flush_done() after its tick.
    l1_busy_ = t->l1_busy;
    if (flush_on_ && t->flush_done) {
      ev.push_back({EV_GL_FLUSHED, 0, 0, true});
      flush_on_ = false;
    }
    ++now_;
    return ev;
  }

private:
  // The global leg's handshakes, the flat memory on the miss bus, the L1's fills.
  void gl_outputs(std::vector<Event>& ev, bool mv) {
    auto* t = sim_.operator->();
    for (uint32_t b = 0; b < NB; ++b) {
      if (!glq_[b].empty() && get_bits(t->gl_in_ready, b, 1)) {
        ev.push_back({EV_GL_ACC, b, glq_[b].front().second, true});
        glq_[b].pop_front();
      }
      if (!get_bits(t->gl_in_rsp_valid, b, 1))
        continue;
      const uint32_t tag = uint32_t(get_bits(t->gl_in_rsp_tag, b * gl_tag_bits_, gl_tag_bits_)) & (GL_TAGS - 1);
      const uint32_t mask = uint32_t(get_bits(t->gl_in_rsp_mask, b * NL, NL));
      Event e{EV_GL_RSP, b, tag, true, mask};
      auto it = glreq_[b].find(tag);
      for (uint32_t i = 0; i < NL; ++i) {
        if (!((mask >> i) & 1))
          continue;
        const uint64_t w = get_bits(t->gl_in_rsp_data, (b * NL + i) * DATA_BITS, DATA_BITS);
        if (it == glreq_[b].end() || !gl_lane_ok(it->second, i, w)) {
          e.data_ok = false;
          gl_bad_.push_back({b, tag, i, w});
        }
      }
      ev.push_back(e);
    }
    if (t->mem_req_valid) {
      const uint64_t line = get_bits(t->mem_req_addr, 0, MEM_ADDR_BITS);
      const uint64_t mtag = get_bits(t->mem_req_tag, 0, 64);
      uint8_t data[ROW_BYTES], be[ROW_BYTES];
      uint32_t secs = 0;
      for (uint32_t i = 0; i < ROW_BYTES; ++i) {
        data[i] = uint8_t(get_bits(t->mem_req_data, i * 8, 8));
        be[i] = uint8_t(get_bits(t->mem_req_byteen, i, 1));
        secs |= uint32_t(be[i]) << (i / gl::SECTOR);
      }
      if (t->mem_req_rw) {
        mem_.write(line * ROW_BYTES, data, be, ROW_BYTES);
      } else if ((mtag >> MEM_IDX_BITS) & 1) {
        this->reply(mtag, std::vector<uint8_t>(ROW_BYTES, 0));
      } else {
        ev.push_back({EV_GL_MEMRD, 0, uint32_t(line), true, secs});
        this->reply(mtag, mem_.read(line * ROW_BYTES, ROW_BYTES));
      }
    }
    if (mv && t->mem_rsp_ready)
      mrsp_.pop_front();
    if (t->l1_fill_fire)
      ev.push_back({EV_GL_FILL, 0, 0, true, uint32_t(t->l1_fill_row)});
  }

  void reply(uint64_t tag, std::vector<uint8_t> data) {
    const uint64_t when = std::max(now_ + mem_.delay, mrsp_.empty() ? 0 : mrsp_.back().when);
    mrsp_.push_back({when, tag, std::move(data)});
  }

  struct MemRsp {
    uint64_t when, tag;
    std::vector<uint8_t> data;
  };
  std::deque<std::pair<gl::Req, uint32_t>> glq_[NB];
  std::map<uint32_t, gl::Req> glreq_[NB];
  std::vector<GlBad> gl_bad_;
  std::deque<MemRsp> mrsp_;
  gl::FlatMem mem_;
  uint32_t gl_tag_bits_ = 0;
  bool flush_on_ = false, l1_busy_ = false;
  vl_simulator<VVX_sm_sram_top> sim_;
  uint64_t tick_ = 0;
  uint64_t now_ = 0;
  std::deque<std::pair<Req, uint32_t>> q_[NB];
  bool cas2_[NB] = {};
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
  bool ds_busy_ = false;
  std::deque<std::pair<Req, uint32_t>> cpaq_;
  std::map<uint32_t, uint32_t> cpa_mask_;
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
    else if (!strncmp(argv[i], "+", 1)) continue;
    else { std::printf("usage: %s [--seed=N] [--random=N] [--csv=file] [--unified] [--directed]\n", argv[0]); return 2; }
  }
  Verilated::commandArgs(argc, argv);
  FILE* csv = csv_path ? std::fopen(csv_path, "w") : nullptr;
  std::printf("sm_sram (RTL): %u blocks x %u lanes, banks %u x %u B, seed %u\n", NB, NL, VX_CFG_LMEM_NUM_BANKS,
              VX_CFG_LMEM_BANK_BYTES, seed);
  RtlDriver d;
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  const bool lacc_ok = true;
#else
  const bool lacc_ok = false;
#endif
  Runner<RtlDriver, WORD_SIZE> run(d, NL, NB, csv, lacc_ok);
  int rc = run.run(seed, nrand, unified, directed);
  if (!d.pipe_consts_ok())
    rc = 1;
  if (csv)
    std::fclose(csv);
  std::printf(rc ? "FAILED!\n" : "PASSED!\n");
  return rc;
}
