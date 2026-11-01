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

// DLMEM endpoints testbench: VX_dlmem_ep_top (per core an egress, an ingress
// and a store tracker, joined by a request and a response network) driven by
// the bench shared with the SimX unit test (tests/unittest/dlmem_ep).
//
// Modes: random all-to-all (its AMOs include mbarrier arrive and tx ops),
// hotspot (AMO adds on one word of core 0), storeflood (credits bind),
// loadflood (pending entries bind), backpressure (LSU response ready 10%),
// lrsc (remote LR/SC mixed with loads and stores) and latency (a fixed
// schedule with always-ready sinks; --csv writes src,dst,op,seq,issue_cycle,
// done_cycle per lane, compared with the SimX run of the same schedule by
// compare_latency.py). Without --mode every mode runs, with seeds 1 and 2.
//
// Built with LRSC_ASSERT=1, a remote LR or SC must stop the simulation on the
// egress assertion, as it would in the core. --mode=async[-random] runs the
// st.async/red.async bench (dlmem_async_bench.h), --mode=rows* the DXA rows.

#include "vl_simulator.h"
#include "VVX_dlmem_ep_top.h"

#include <VX_config.h>
#include "dlmem_ep_bench.h"
#include "dlmem_ref.h"

#if defined(VX_CFG_EXT_DXA_ENABLE) && defined(VX_CFG_EXT_MBAR_ENABLE)
#define DLMEM_EP_ROWS
#include "dlmem_rows_bench.h"
#endif
#include "dlmem_async_bench.h"

#include <iostream>
#include <type_traits>

namespace {

template <typename T>
void set_bits(T& sig, uint32_t lsb, uint32_t width, uint64_t value) {
  if constexpr (std::is_integral_v<T>) {
    uint64_t mask = (width >= 64) ? ~0ull : ((1ull << width) - 1);
    uint64_t v = static_cast<uint64_t>(sig);
    v = (v & ~(mask << lsb)) | ((value & mask) << lsb);
    sig = static_cast<T>(v);
  } else {
    uint32_t i = 0;
    while (i < width) {
      uint32_t bit = lsb + i, w = bit / 32, o = bit % 32;
      uint32_t n = std::min(32 - o, width - i);
      uint32_t m = (n == 32) ? 0xffffffffu : ((1u << n) - 1);
      uint32_t chunk = static_cast<uint32_t>(value >> i) & m;
      sig[w] = (sig[w] & ~(m << o)) | (chunk << o);
      i += n;
    }
  }
}

template <typename T>
uint64_t get_bits(const T& sig, uint32_t lsb, uint32_t width) {
  if constexpr (std::is_integral_v<T>) {
    uint64_t mask = (width >= 64) ? ~0ull : ((1ull << width) - 1);
    return (static_cast<uint64_t>(sig) >> lsb) & mask;
  } else {
    uint64_t value = 0;
    uint32_t i = 0;
    while (i < width) {
      uint32_t bit = lsb + i, w = bit / 32, o = bit % 32;
      uint32_t n = std::min(32 - o, width - i);
      uint32_t m = (n == 32) ? 0xffffffffu : ((1u << n) - 1);
      value |= static_cast<uint64_t>((sig[w] >> o) & m) << i;
      i += n;
    }
    return value;
  }
}

using ep::NB;
using ep::NCC;
using ep::NL;
using ep::NP;
using ep::NW;

class RtlDut final : public ep::Dut {
public:
  RtlDut() : st_out_(NCC, 0), ld_out_(NCC, 0), slot_out_(NCC, 0) {
    auto* t = sim_.operator->();
    t->reset = 1;
    this->drive(ep::In());
    for (int i = 0; i < 8; ++i) {
      tick_ = sim_.step(tick_, 2);
    }
    t->reset = 0;
    t->eval();
  }

  // An ingress slot frees as its response enters the ingress output buffer
  // (two entries), so requests in flight past the slot table can reach SLOTS+2.
  uint32_t slot_bound() const override {
    return ep::SLOTS + 2;
  }

  void cycle(const ep::In& in, ep::Out& out) override {
    auto* t = sim_.operator->();
    this->drive(in);
    t->eval();

    for (uint32_t p = 0; p < NP; ++p) {
      out.req_fire[p] = in.req[p] && get_bits(t->lsu_req_ready, p, 1);
      auto& r = out.rsp[p];
      r.fire = get_bits(t->lsu_rsp_valid, p, 1) && in.rsp_ready[p];
      r.mask = r.fire ? uint32_t(get_bits(t->lsu_rsp_mask, p * NL, NL)) : 0;
      r.tag = r.fire ? uint32_t(get_bits(t->lsu_rsp_tag, p * 8, 8)) : 0;
      r.data.assign(NL, 0);
      for (uint32_t l = 0; l < NL && r.fire; ++l) {
        r.data[l] = get_bits(t->lsu_rsp_data, (p * NL + l) * 64, 64);
      }
    }
    for (uint32_t c = 0; c < NCC; ++c) {
      auto& q = out.lmem[c];
      q.fire = get_bits(t->lmem_req_valid, c, 1) && in.lmem_ready[c];
      q.rw = get_bits(t->lmem_req_rw, c, 1);
      q.local = get_bits(t->lmem_req_local, c, 1);
      q.amo = get_bits(t->lmem_req_amo, c, 1);
      q.uns = get_bits(t->lmem_req_amo_unsigned, c, 1);
      q.amo_op = uint8_t(get_bits(t->lmem_req_amo_op, c * 4, 4));
      q.byteen = uint8_t(get_bits(t->lmem_req_byteen, c * 8, 8));
      q.waddr = uint32_t(get_bits(t->lmem_req_addr, c * 32, 32));
      q.hart = uint32_t(get_bits(t->lmem_req_hart_id, c * 32, 32));
      q.tag = uint32_t(get_bits(t->lmem_req_tag, c * 8, 8));
      q.data = get_bits(t->lmem_req_data, c * 64, 64);
      q.cmp = get_bits(t->lmem_req_amo_cmp, c * 64, 64);

      out.ack_valid[c] = get_bits(t->ack_valid, c, 1);
      out.ack_wid[c] = uint32_t(get_bits(t->ack_wid, c * 8, 8));
      out.sinks_ready[c] = get_bits(t->rsp_ej_ready, c, 1) && get_bits(t->lmem_rsp_ready, c, 1);

      // The network handshakes give the bounds the bench checks: stores and
      // loads/AMOs in flight per source, requests in flight per ingress.
      if (get_bits(t->req_inj_fire, c, 1)) {
        if (get_bits(t->req_inj_op, c * 2, 2) == vortex::dlmem_ref::DLMEM_OP_ST) {
          ++st_out_[c];
        } else {
          ++ld_out_[c];
        }
      }
      if (get_bits(t->rsp_ej_fire, c, 1)) {
        if (get_bits(t->rsp_ej_ack, c, 1)) {
          --st_out_[c];
        } else {
          --ld_out_[c];
        }
      }
      slot_out_[c] += get_bits(t->req_ej_fire, c, 1);
      slot_out_[c] -= get_bits(t->rsp_inj_fire, c, 1);
      out.st_out[c] = st_out_[c];
      out.ld_out[c] = ld_out_[c];
      out.slot_out[c] = slot_out_[c];
    }

    tick_ = sim_.step(tick_, 2);
    t->eval();

    for (uint32_t c = 0; c < NCC; ++c) {
      out.drained[c] = uint32_t(get_bits(t->drained, c * NW, NW));
      out.eg_empty[c] = get_bits(t->egress_empty, c, 1);
      out.in_empty[c] = get_bits(t->ingress_empty, c, 1);
      out.wt_empty[c] = get_bits(t->wtrack_empty, c, 1);
    }
  }

private:
  void drive(const ep::In& in) {
    auto* t = sim_.operator->();
    for (uint32_t p = 0; p < NP; ++p) {
      const ep::ReqView* r = in.req[p];
      set_bits(t->lsu_req_valid, p, 1, r != nullptr);
      set_bits(t->lsu_req_rw, p, 1, r && r->rw);
      set_bits(t->lsu_req_mask, p * NL, NL, r ? r->mask : 0);
      set_bits(t->lsu_req_tag, p * 8, 8, r ? r->tag : 0);
      for (uint32_t l = 0; l < NL; ++l) {
        uint32_t i = p * NL + l;
        const ep::LaneView* lv = (r && (r->mask >> l & 1)) ? &r->lanes[l] : nullptr;
        uint64_t addr = lv ? ((ep::DLMEM_BASE >> 3) | (uint64_t(lv->dst) << ep::LMEM_AW) | lv->waddr) : 0;
        set_bits(t->lsu_req_addr, i * 64, 64, addr);
        set_bits(t->lsu_req_data, i * 64, 64, lv ? lv->data : 0);
        set_bits(t->lsu_req_byteen, i * 8, 8, lv ? lv->byteen : 0);
        set_bits(t->lsu_req_amo, i, 1, lv && lv->amo);
        set_bits(t->lsu_req_amo_op, i * 4, 4, lv ? lv->amo_op : 0);
        set_bits(t->lsu_req_amo_unsigned, i, 1, lv && lv->uns);
        set_bits(t->lsu_req_amo_cmp, i * 64, 64, lv ? lv->cmp : 0);
        set_bits(t->lsu_req_hart_id, i * 32, 32, lv ? lv->hart : 0);
      }
      set_bits(t->lsu_rsp_ready, p, 1, in.rsp_ready[p]);
      set_bits(t->wt_inc_valid, p, 1, in.inc_valid[p]);
      set_bits(t->wt_inc_wid, p * 8, 8, in.inc_wid[p]);
      set_bits(t->wt_inc_cnt, p * 8, 8, in.inc_cnt[p]);
    }
    for (uint32_t c = 0; c < NCC; ++c) {
      set_bits(t->lmem_req_ready, c, 1, in.lmem_ready[c]);
      set_bits(t->lmem_rsp_valid, c, 1, in.lrsp_valid[c]);
      set_bits(t->lmem_rsp_tag, c * 8, 8, in.lrsp_tag[c]);
      set_bits(t->lmem_rsp_data, c * 64, 64, in.lrsp_data[c]);
    #ifdef DLMEM_EP_ROWS
      // No row flits: every slot occupied, the DMA port always ready.
      set_bits(t->slot_valid, c * NW, NW, ~0ull);
      set_bits(t->dma_ready, c, 1, 1);
    #endif
    }
  #ifdef DLMEM_EP_ROWS
    for (uint32_t s = 0; s < ep::NUM_SOCKETS; ++s) {
      set_bits(t->row_valid, s, 1, 0);
    }
  #endif
  }

  vl_simulator<VVX_dlmem_ep_top> sim_;
  uint64_t tick_ = 0;
  std::vector<uint32_t> st_out_;
  std::vector<uint32_t> ld_out_;
  std::vector<uint32_t> slot_out_;
};

#ifdef DLMEM_EP_ROWS
// The same endpoints driven by the row-flit bench: row ports per socket, the
// DMA and local-memory ports per core; the LSU ports stay idle.
class RowRtlDut final : public rows::Dut {
public:
  RowRtlDut() {
    auto* t = sim_.operator->();
    t->reset = 1;
    this->drive(rows::In());
    for (int i = 0; i < 8; ++i) {
      tick_ = sim_.step(tick_, 2);
    }
    t->reset = 0;
    t->eval();
    amo_ctx_ = t->cfg_amo_complete_tx;
  }

  uint32_t amo_complete_tx() const override {
    return amo_ctx_;
  }

  void cycle(const rows::In& in, rows::Out& out) override {
    auto* t = sim_.operator->();
    this->drive(in);
    t->eval();

    for (uint32_t s = 0; s < rows::NUM_SOCKETS; ++s) {
      out.row_fire[s] = (in.row[s] != nullptr) && get_bits(t->row_ready, s, 1);
    }
    for (uint32_t c = 0; c < NCC; ++c) {
      auto& d = out.dma[c];
      d.fire = get_bits(t->dma_valid, c, 1) && in.dma_ready[c];
      d.addr = uint32_t(get_bits(t->dma_addr, c * 32, 32));
      for (uint32_t b = 0; b < rows::NBANKS; ++b) {
        d.word[b] = get_bits(t->dma_data, (c * rows::NBANKS + b) * 64, 64);
        d.byteen[b] = uint8_t(get_bits(t->dma_byteen, (c * rows::NBANKS + b) * 8, 8));
      }
      d.slot = uint32_t(get_bits(t->dma_slot, c * 8, 8));
      d.remote = get_bits(t->dma_remote, c, 1);
      d.last = get_bits(t->dma_last, c, 1);

      auto& q = out.lmem[c];
      q.fire = get_bits(t->lmem_req_valid, c, 1) && in.lmem_ready[c];
      q.rw = get_bits(t->lmem_req_rw, c, 1);
      q.amo = get_bits(t->lmem_req_amo, c, 1);
      q.uns = get_bits(t->lmem_req_amo_unsigned, c, 1);
      q.amo_op = uint32_t(get_bits(t->lmem_req_amo_op, c * 4, 4));
      q.byteen = uint8_t(get_bits(t->lmem_req_byteen, c * 8, 8));
      q.waddr = uint32_t(get_bits(t->lmem_req_addr, c * 32, 32));
      q.hart = uint32_t(get_bits(t->lmem_req_hart_id, c * 32, 32));
      q.tag = uint32_t(get_bits(t->lmem_req_tag, c * 8, 8));
      q.data = get_bits(t->lmem_req_data, c * 64, 64);

      out.rsp_inj[c] = get_bits(t->rsp_inj_fire, c, 1);
    }

    tick_ = sim_.step(tick_, 2);
    t->eval();

    for (uint32_t c = 0; c < NCC; ++c) {
      out.in_empty[c] = get_bits(t->ingress_empty, c, 1);
    }
  }

private:
  void drive(const rows::In& in) {
    auto* t = sim_.operator->();
    for (uint32_t p = 0; p < NP; ++p) {
      set_bits(t->lsu_req_valid, p, 1, 0);
      set_bits(t->lsu_rsp_ready, p, 1, 1);
      set_bits(t->wt_inc_valid, p, 1, 0);
    }
    for (uint32_t s = 0; s < rows::NUM_SOCKETS; ++s) {
      const rows::Flit* f = in.row[s];
      set_bits(t->row_valid, s, 1, f != nullptr);
      set_bits(t->row_dst, s * 8, 8, f ? f->dst : 0);
      set_bits(t->row_src, s * 8, 8, f ? f->src : 0);
      set_bits(t->row_compl, s, 1, f && f->compl_);
      set_bits(t->row_addr, s * 32, 32, f ? f->addr : 0);
      set_bits(t->row_byteen, s * 8, 8, f ? f->byteen : 0);
      set_bits(t->row_data, s * 64, 64, f ? f->data : 0);
      set_bits(t->row_slot, s * 8, 8, f ? f->slot : 0);
    }
    for (uint32_t c = 0; c < NCC; ++c) {
      set_bits(t->slot_valid, c * NW, NW, in.slot_valid[c]);
      set_bits(t->dma_ready, c, 1, in.dma_ready[c]);
      set_bits(t->lmem_req_ready, c, 1, in.lmem_ready[c]);
      set_bits(t->lmem_rsp_valid, c, 1, in.lrsp_valid[c]);
      set_bits(t->lmem_rsp_tag, c * 8, 8, in.lrsp_tag[c]);
      set_bits(t->lmem_rsp_data, c * 64, 64, 0);
    }
  }

  vl_simulator<VVX_dlmem_ep_top> sim_;
  uint64_t tick_ = 0;
  uint32_t amo_ctx_ = 0;
};

int run_rows(const std::string& mode, const std::vector<uint32_t>& seeds, uint32_t copies, const std::string& csv) {
  for (auto seed : (mode == "rows") ? std::vector<uint32_t>{seeds.front()} : seeds) {
    rows::Options opts;
    opts.mode = mode;
    opts.seed = seed;
    opts.copies = copies;
    RowRtlDut dut;
    rows::Bench bench(dut, opts);
    uint32_t failures = bench.run();
    if (failures != 0) {
      std::cout << "FAILED: mode " << mode << " seed " << seed << ": " << bench.first_failure()
                << " [" << failures << " check(s)]" << std::endl;
      return 1;
    }
    std::cout << "mode " << mode << " seed " << seed << ": cycles=" << bench.cycles() << " flits=" << bench.flits()
              << " completions=" << bench.compls() << " row_stalls=" << bench.stalls() << std::endl;
    if (mode == "rows" && !csv.empty()) {
      bench.write_csv(csv);
      std::cout << "wrote " << csv << std::endl;
    }
  }
  return 0;
}
#endif

// The same endpoints driven by the st.async/red.async bench: each core's
// first LSU port carries translated async lanes; the row ports stay idle.
class AsyncRtlDut final : public async::Dut {
public:
  AsyncRtlDut() {
    auto* t = sim_.operator->();
    t->reset = 1;
    this->drive(async::In());
    for (int i = 0; i < 8; ++i) {
      tick_ = sim_.step(tick_, 2);
    }
    t->reset = 0;
    t->eval();
    async_bit_ = t->cfg_async_bit;
    mb_lsb_ = t->cfg_async_mb_lsb;
    amo_add_ = t->cfg_amo_add;
    amo_ctx_ = t->cfg_amo_complete_tx;
  }

  uint32_t amo_add() const override {
    return amo_add_;
  }

  uint32_t amo_complete_tx() const override {
    return amo_ctx_;
  }

  void cycle(const async::In& in, async::Out& out) override {
    auto* t = sim_.operator->();
    this->drive(in);
    t->eval();

    for (uint32_t c = 0; c < NCC; ++c) {
      uint32_t p = c * NB;
      out.req_fire[c] = (in.req[c] != nullptr) && get_bits(t->lsu_req_ready, p, 1);
      auto& q = out.lmem[c];
      q.fire = get_bits(t->lmem_req_valid, c, 1) && in.lmem_ready[c];
      q.rw = get_bits(t->lmem_req_rw, c, 1);
      q.amo = get_bits(t->lmem_req_amo, c, 1);
      q.uns = get_bits(t->lmem_req_amo_unsigned, c, 1);
      q.amo_op = uint32_t(get_bits(t->lmem_req_amo_op, c * 4, 4));
      q.byteen = uint8_t(get_bits(t->lmem_req_byteen, c * 8, 8));
      q.waddr = uint32_t(get_bits(t->lmem_req_addr, c * 32, 32));
      q.hart = uint32_t(get_bits(t->lmem_req_hart_id, c * 32, 32));
      q.tag = uint32_t(get_bits(t->lmem_req_tag, c * 8, 8));
      q.data = get_bits(t->lmem_req_data, c * 64, 64);
      bool ej = get_bits(t->rsp_ej_fire, c, 1);
      bool ack = get_bits(t->rsp_ej_ack, c, 1);
      out.ack_ej[c] = ej && ack;
      out.data_ej[c] = ej && !ack;
      out.wt_ack[c] = get_bits(t->ack_valid, c, 1);
    }

    tick_ = sim_.step(tick_, 2);
    t->eval();

    for (uint32_t c = 0; c < NCC; ++c) {
      out.eg_empty[c] = get_bits(t->egress_empty, c, 1);
      out.in_empty[c] = get_bits(t->ingress_empty, c, 1);
    }
  }

private:
  void drive(const async::In& in) {
    auto* t = sim_.operator->();
    for (uint32_t p = 0; p < NP; ++p) {
      uint32_t c = p / NB;
      const async::Req* r = (p % NB == 0) ? in.req[c] : nullptr;
      set_bits(t->lsu_req_valid, p, 1, r != nullptr);
      set_bits(t->lsu_req_rw, p, 1, r != nullptr);
      set_bits(t->lsu_req_mask, p * NL, NL, r ? r->mask : 0);
      set_bits(t->lsu_req_tag, p * 8, 8, r ? r->wid : 0);
      for (uint32_t l = 0; l < NL; ++l) {
        uint32_t i = p * NL + l;
        const async::Lane* lv = r ? r->lanes[l] : nullptr;
        uint64_t addr = lv ? ((1ull << async_bit_) | (uint64_t(lv->mbar) << mb_lsb_)
                              | (uint64_t(lv->dst) << async::LMEM_AW) | lv->waddr) : 0;
        set_bits(t->lsu_req_addr, i * 64, 64, addr);
        set_bits(t->lsu_req_data, i * 64, 64, lv ? lv->data : 0);
        set_bits(t->lsu_req_byteen, i * 8, 8, lv ? lv->byteen : 0);
        set_bits(t->lsu_req_amo, i, 1, lv && lv->red);
        set_bits(t->lsu_req_amo_op, i * 4, 4, (lv && lv->red) ? amo_add_ : 0);
        set_bits(t->lsu_req_amo_unsigned, i, 1, 0);
        set_bits(t->lsu_req_amo_cmp, i * 64, 64, 0);
        set_bits(t->lsu_req_hart_id, i * 32, 32, lv ? lv->hart : 0);
      }
      set_bits(t->lsu_rsp_ready, p, 1, 1);
      set_bits(t->wt_inc_valid, p, 1, 0);
    }
    for (uint32_t c = 0; c < NCC; ++c) {
      set_bits(t->lmem_req_ready, c, 1, in.lmem_ready[c]);
      set_bits(t->lmem_rsp_valid, c, 1, in.lrsp_valid[c]);
      set_bits(t->lmem_rsp_tag, c * 8, 8, in.lrsp_tag[c]);
      set_bits(t->lmem_rsp_data, c * 64, 64, 0);
    #ifdef DLMEM_EP_ROWS
      set_bits(t->slot_valid, c * NW, NW, ~0ull);
      set_bits(t->dma_ready, c, 1, 1);
    #endif
    }
  #ifdef DLMEM_EP_ROWS
    for (uint32_t s = 0; s < ep::NUM_SOCKETS; ++s) {
      set_bits(t->row_valid, s, 1, 0);
    }
  #endif
  }

  vl_simulator<VVX_dlmem_ep_top> sim_;
  uint64_t tick_ = 0;
  uint32_t async_bit_ = 0;
  uint32_t mb_lsb_ = 0;
  uint32_t amo_add_ = 0;
  uint32_t amo_ctx_ = 0;
};

int run_async(const std::string& mode, const std::vector<uint32_t>& seeds, uint32_t reqs, const std::string& csv) {
  for (auto seed : (mode == "async") ? std::vector<uint32_t>{seeds.front()} : seeds) {
    async::Options opts;
    opts.mode = mode;
    opts.seed = seed;
    opts.reqs = reqs;
    AsyncRtlDut dut;
    async::Bench bench(dut, opts);
    uint32_t failures = bench.run();
    if (failures != 0) {
      std::cout << "FAILED: mode " << mode << " seed " << seed << ": " << bench.first_failure()
                << " [" << failures << " check(s)]" << std::endl;
      return 1;
    }
    std::cout << "mode " << mode << " seed " << seed << ": cycles=" << bench.cycles() << " lanes=" << bench.lanes()
              << " max_pending=" << bench.max_inflight() << std::endl;
    if (mode == "async" && !csv.empty()) {
      bench.write_csv(csv);
      std::cout << "wrote " << csv << std::endl;
    }
  }
  return 0;
}

void usage(const char* prog) {
  std::cerr << "usage: " << prog << " [--mode=random|hotspot|storeflood|loadflood|backpressure|lrsc|latency|all"
#ifdef DLMEM_EP_ROWS
               "|rows|rows-random|rows-retired"
#endif
               "|async|async-random]\n"
               "       [--seed=N] [--reqs=N] [--csv=FILE] [--trace=FILE]" << std::endl;
}

} // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);

  std::vector<std::string> modes = ep::all_modes();
  std::vector<uint32_t> seeds = {1, 2};
  uint32_t reqs = 0;
  std::string csv;
  std::string trace_path;
  std::string row_mode;
  std::string async_mode;
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    if (a[0] == '+') {
      continue;
    }
    if (std::strncmp(a, "--mode=", 7) == 0) {
      std::string m = a + 7;
      if (m != "all") {
      #ifdef DLMEM_EP_ROWS
        if (m.compare(0, 4, "rows") == 0) {
          row_mode = m;
          continue;
        }
      #endif
        if (m.compare(0, 5, "async") == 0) {
          async_mode = m;
          continue;
        }
        if (std::find(modes.begin(), modes.end(), m) == modes.end()) {
          usage(argv[0]);
          return 2;
        }
        modes = {m};
      }
    } else if (std::strncmp(a, "--seed=", 7) == 0) {
      seeds = {static_cast<uint32_t>(std::strtoul(a + 7, nullptr, 0))};
    } else if (std::strncmp(a, "--reqs=", 7) == 0) {
      reqs = static_cast<uint32_t>(std::strtoul(a + 7, nullptr, 0));
    } else if (std::strncmp(a, "--csv=", 6) == 0) {
      csv = a + 6;
    } else if (std::strncmp(a, "--trace=", 8) == 0) {
      trace_path = a + 8;
    } else {
      usage(argv[0]);
      return 2;
    }
  }
  if (!trace_path.empty() && (modes.size() != 1 || seeds.size() != 1)) {
    std::cerr << "--trace needs a single --mode and --seed" << std::endl;
    return 2;
  }

#ifdef DLMEM_EP_ROWS
  if (!row_mode.empty()) {
    std::cout << "dlmem_ep rows: sockets=" << rows::NUM_SOCKETS << " socket_size=" << rows::SOCKET_SIZE
              << " banks=" << rows::NBANKS << " warps=" << NW << std::endl;
    if (run_rows(row_mode, seeds, reqs, csv) != 0) {
      return 1;
    }
    std::cout << "PASSED" << std::endl;
    return 0;
  }
#endif

  if (!async_mode.empty()) {
    std::cout << "dlmem_ep async: sockets=" << async::NUM_SOCKETS << " socket_size=" << async::SOCKET_SIZE
              << " lanes=" << NL << " warps=" << NW << std::endl;
    if (run_async(async_mode, seeds, reqs, csv) != 0) {
      return 1;
    }
    std::cout << "PASSED" << std::endl;
    return 0;
  }

  std::cout << "dlmem_ep: sockets=" << ep::NUM_SOCKETS << " socket_size=" << ep::SOCKET_SIZE
            << " blocks=" << NB << " lanes=" << NL << " warps=" << NW << " zacas=" << ep::ZACAS
            << " lrsc_assert=" << LRSC_ASSERT << std::endl;

  for (auto& mode : modes) {
    for (auto seed : (mode == "latency") ? std::vector<uint32_t>{seeds.front()} : seeds) {
      ep::Options opts;
      opts.mode = mode;
      opts.seed = seed;
      opts.reqs = reqs;
      std::ofstream trace_ofs;
      if (!trace_path.empty()) {
        trace_ofs.open(trace_path);
      }
      RtlDut dut;
      ep::Bench bench(dut, opts, trace_path.empty() ? nullptr : &trace_ofs);
      uint32_t failures = bench.run();
      if (failures != 0) {
        std::cout << "FAILED: mode " << mode << " seed " << seed << ": " << bench.first_failure()
                  << " [" << failures << " check(s)]" << std::endl;
        return 1;
      }
      std::cout << "mode " << mode << " seed " << seed << ": cycles=" << bench.cycles()
                << " lane_ops=" << bench.lane_ops() << " lmem_ops=" << bench.lmem_ops()
                << " max_st=" << bench.max_st() << " max_ld=" << bench.max_ld()
                << " max_slots=" << bench.max_slot() << " digest=" << std::hex << bench.digest()
                << std::dec << std::endl;
      if (mode == "latency" && !csv.empty()) {
        ep::Bench::write_csv(csv, bench.latency());
        std::cout << "wrote " << csv << std::endl;
      }
    }
  }
  std::cout << "PASSED" << std::endl;
  return 0;
}
