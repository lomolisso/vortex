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

// Members of TcuUnit::Impl for WGMMA as on the H100 (VX_CFG_TCU_H100_ENABLE).
// Textually included inside the class: it uses the unit's channels, the
// wgmma group state and the tile buffers of tcu_h100.h. Each cycle mirrors
// VX_tcu_wgmma: the CTA gate, then every block's operand readiness, then the
// micro-ops that fire; the dot products are tcu_fedp_ref's, bit for bit the
// TFR datapath.

public:
  void h100_reset() {
    h100_slot_.fill(0);
    h100_rank_.fill(0);
    h100_retire_cycle_.fill(~uint64_t(0));
    h100_retire_cnt_.fill(0);
    h100_slot_warps_.fill(0);
    h100_stride_ = 0;
    for (auto& d : h100_desc_) d.fill(0);
    h100_inexp_.fill(false);
    h100_owner_.fill(-1);
  }

  void h100_tick() {
    this->h100_slot_events();
    if (tcu_csv_on()) {
      for (uint32_t b = 0; b < VX_CFG_NUM_TCU_BLOCKS; ++b) {
        const auto& ready = wg_ready_.at(b);
        for (uint32_t w = 0; w < VX_CFG_NUM_WARPS; ++w) {
          if (ready.test(w)) {
            tcu_csv(SimPlatform::instance().cycles(), "grp", int32_t(b), "unlock", "w" + std::to_string(w));
            break;
          }
        }
      }
    }
    this->wg_unlock_step();
    this->h100_wg_recheck();
    std::array<bool, VX_CFG_NUM_TCU_BLOCKS> grp_taken{};
    for (uint32_t b = 0; b < VX_CFG_NUM_TCU_BLOCKS; ++b) {
      grp_taken.at(b) = this->h100_wg_request(b);
    }

    // What each block presents this cycle, as the tile buffers see it.
    std::array<TcuH100Obs, VX_CFG_NUM_TCU_BLOCKS> obs{};
    std::array<bool, VX_CFG_NUM_TCU_BLOCKS> conflict{};
    // Blocks presenting the first micro-op of a WGMMA this cycle, and of which CTA.
    std::array<int32_t, VX_CFG_NUM_TCU_BLOCKS> lead_cta;
    lead_cta.fill(-1);
    for (uint32_t b = 0; b < VX_CFG_NUM_TCU_BLOCKS; ++b) {
      auto& input = simobject_->Inputs.at(b);
      if (input.empty() || grp_taken.at(b))
        continue;
      auto trace = input.peek();
      if (!tcu_is_wgmma(std::get<TcuType>(trace->op_type)))
        continue;
      auto args = std::get<IntrTcuArgs>(trace->instr_ptr->get_args());
      int32_t cta = (int32_t)core_->scheduler().warp(trace->wid).cta_csrs.cta_id;
      // The shared B buffer serves one CTA: a block waits while another is
      // inside a WGMMA of a different CTA.
      for (uint32_t k = 0; k < VX_CFG_NUM_TCU_BLOCKS; ++k) {
        if (k != b && h100_inexp_.at(k) && h100_owner_.at(k) != cta)
          conflict.at(b) = true;
      }
      // Two blocks starting WGMMAs of different CTAs in one cycle would each
      // wait on the other from the next: the lowest block starts first.
      for (uint32_t k = 0; k < b; ++k) {
        if (lead_cta.at(k) >= 0 && lead_cta.at(k) != cta)
          conflict.at(b) = true;
      }
      if (tcu_csv_on() && conflict.at(b) != h100_csv_cta_wait_.at(b)) {
        h100_csv_cta_wait_.at(b) = conflict.at(b);
        tcu_csv(SimPlatform::instance().cycles(), "tcu", int32_t(b), conflict.at(b) ? "cta_wait" : "cta_ok",
                "w" + std::to_string(trace->wid) + " cta" + std::to_string(cta));
      }
      if (conflict.at(b))
        continue;
      if (args.is_first_uop && !h100_inexp_.at(b))
        lead_cta.at(b) = cta;
      auto& o = obs.at(b);
      o.valid    = true;
      o.first    = args.is_first_uop;
      o.a_smem   = args.is_a_smem;
      o.esize16  = (elem_bits(args.fmt_s) == 16);
      o.step_m   = args.step_m;
      o.step_n   = args.step_n;
      o.step_k   = args.step_k;
      o.cd_nregs = args.cd_nregs;
      o.cta      = cta;
      if (o.first) {
        o.desc_a = trace->src_data.at(0).at(0).u64;
        o.desc_b = trace->src_data.at(1).at(0).u64;
      } else {
        o.desc_a = h100_desc_.at(b).at(0);
        o.desc_b = h100_desc_.at(b).at(1);
      }
      o.slot_base = uint64_t(h100_slot_.at(trace->wid)) * h100_stride_;
    }

    auto& tbuf = simobject_->tbuf();
    auto ready = tbuf->h100().observe(obs, SimPlatform::instance().cycles());
    perf_stats_.tbuf_stalls     = tbuf->h100().perf().stalls;
    perf_stats_.tbuf_cache_hits = tbuf->h100().perf().hits;

    for (uint32_t b = 0; b < VX_CFG_NUM_TCU_BLOCKS; ++b) {
      auto& input = simobject_->Inputs.at(b);
      if (input.empty() || grp_taken.at(b))
        continue;
      auto trace = input.peek();
      auto tcu_type = std::get<TcuType>(trace->op_type);
      auto args = std::get<IntrTcuArgs>(trace->instr_ptr->get_args());
      bool is_wg = tcu_is_wgmma(tcu_type);
      if (is_wg && (conflict.at(b) || !ready.at(b)))
        continue;
      if (!exec_done_.at(b)) {
        trace->dst_data.assign(VX_CFG_NUM_THREADS, reg_data_t{});
        if (is_wg) {
          this->h100_wgmma(b, trace, obs.at(b));
        } else if (tcu_type == TcuType::WMMA) {
          this->h100_wmma(trace, args);
        } else {
          std::abort();
        }
        exec_done_.at(b) = true;
      }
      if (!simobject_->Outputs.at(b).try_send(trace, kMmaLatency))
        continue;
      exec_done_.at(b) = false;
      if (is_wg && tcu_csv_on()) {
        tcu_csv(SimPlatform::instance().cycles(), "tcu", int32_t(b), "uop_fire",
                "m" + std::to_string(args.step_m) + " n" + std::to_string(args.step_n) + " k" + std::to_string(args.step_k)
                + " w" + std::to_string(trace->wid));
      }
      if (is_wg) {
        if (args.is_first_uop) {
          ++wg_.at(trace->wid).issued;
          h100_desc_.at(b) = {obs.at(b).desc_a, obs.at(b).desc_b};
          h100_inexp_.at(b) = true;
          h100_owner_.at(b) = obs.at(b).cta;
        }
        if (args.is_last_uop) {
          h100_inexp_.at(b) = false;
        }
      }
      DT(3, simobject_->name() << " execute: op=" << tcu_type << ", " << *trace);
      input.pop();
    }
    for (uint32_t b = 0; b < VX_CFG_NUM_TCU_BLOCKS; ++b) {
      for (uint32_t k = b + 1; k < VX_CFG_NUM_TCU_BLOCKS; ++k) {
        __assert(!(h100_inexp_.at(b) && h100_inexp_.at(k) && h100_owner_.at(b) != h100_owner_.at(k)),
                 "TCU blocks " << b << " and " << k << " inside WGMMAs of different CTAs ("
                 << h100_owner_.at(b) << ", " << h100_owner_.at(k) << ")");
      }
    }
  }

private:
  // The dispatcher's slot events: each warp's slot and the slot pitch.
  void h100_slot_events() {
    auto& in = simobject_->SlotIn;
    while (!in.empty()) {
      const auto& e = in.peek();
      if (e.kind == CtaSlotEvt::ADMIT) {
        h100_stride_ = e.stride;
        h100_slot_warps_.at(e.slot) = 0;
      } else if (e.kind == CtaSlotEvt::WARP) {
        // A block's warps are dispatched in rank order after its admission.
        h100_slot_.at(e.wid) = e.slot;
        h100_rank_.at(e.wid) = h100_slot_warps_.at(e.slot)++;
      }
      in.pop();
    }
  }

  // wgmma groups of a warpgroup (the warps w & ~3 .. +3, one per TCU block):
  // a wgmma is collective, so the warpgroup has completed min(done) of them,
  // and a warp's group is pending until every warp of its warpgroup is past
  // the group's boundary.
  // The warps' done counts are registered: a retire in cycle T counts from
  // T+1, in every block, the warp's own included.
  uint32_t h100_done_reg(uint32_t w) const {
    uint32_t d = wg_.at(w).done;
    if (h100_retire_cycle_.at(w) == SimPlatform::instance().cycles())
      d -= h100_retire_cnt_.at(w);
    return d;
  }

  uint32_t h100_wg_done(uint32_t wid) const {
    uint32_t base = wid & ~3u;
    uint32_t d = this->h100_done_reg(base);
    for (uint32_t w = base + 1; w < base + 4; ++w)
      d = std::min(d, this->h100_done_reg(w));
    return d;
  }

  // Every cycle, a waiting warp is checked against its warpgroup's done count.
  void h100_wg_recheck() {
    for (uint32_t w = 0; w < VX_CFG_NUM_WARPS; ++w) {
      auto& g = wg_.at(w);
      if (g.waiting && wg_pending(g, this->h100_wg_done(w)) <= g.wait_n) {
        g.waiting = false;
        wg_ready_.at(w % VX_CFG_NUM_TCU_BLOCKS).set(w);
        tcu_csv(SimPlatform::instance().cycles(), "grp", int32_t(w % VX_CFG_NUM_TCU_BLOCKS), "wait_sat",
                "w" + std::to_string(w));
        DT(2, simobject_->name() << " wait satisfied: wid=" << w);
      }
    }
  }

  static void h100_wg_drop(wg_groups_t& g, uint32_t done) {
    while (!g.bounds.empty() && g.bounds.front() <= done)
      g.bounds.pop_front();
  }

  // A commit_group or wait_group at the head of block b, as wg_request but
  // with the warpgroup's completed count.
  bool h100_wg_request(uint32_t b) {
    auto& input = simobject_->Inputs.at(b);
    if (input.empty())
      return false;
    auto trace = input.peek();
    auto tcu_type = std::get<TcuType>(trace->op_type);
    if (tcu_type != TcuType::COMMIT && tcu_type != TcuType::WAIT)
      return false;
    uint32_t wid = trace->wid;
    auto& g = wg_.at(wid);
    if (tcu_type == TcuType::WAIT)
      trace->resume_warp = false;
    trace->dst_data.assign(VX_CFG_NUM_THREADS, reg_data_t{});
    if (!simobject_->Outputs.at(b).try_send(trace, 1))
      return true;
    uint32_t done = this->h100_wg_done(wid);
    h100_wg_drop(g, done);
    if (tcu_type == TcuType::COMMIT) {
      if (g.bounds.size() == kGroupDepth) {
        std::cout << "Error: wgmma group overflow: wid=" << wid << std::endl;
        std::abort();
      }
      g.bounds.push_back(g.issued);
      tcu_csv(SimPlatform::instance().cycles(), "grp", int32_t(b), "commit",
              "w" + std::to_string(wid) + " issued" + std::to_string(g.issued) + " wgdone" + std::to_string(done));
      DT(2, simobject_->name() << " commit_group: wid=" << wid << ", issued=" << g.issued
         << ", warpgroup done=" << done << ", pending=" << wg_pending(g, done));
    } else {
      uint32_t n = std::get<IntrTcuArgs>(trace->instr_ptr->get_args()).step_m;
      uint32_t pending = wg_pending(g, done);
      DT(2, simobject_->name() << " wait_group: wid=" << wid << ", n=" << n << ", pending=" << pending);
      __assert(!g.waiting, "second wgmma wait of a warp: wid=" << wid);
      tcu_csv(SimPlatform::instance().cycles(), "grp", int32_t(b), "wait",
              "w" + std::to_string(wid) + " n" + std::to_string(n) + " pending" + std::to_string(pending)
              + " wgdone" + std::to_string(done));
      if (pending <= n) {
        wg_ready_.at(b).set(wid);
        tcu_csv(SimPlatform::instance().cycles(), "grp", int32_t(b), "wait_sat", "w" + std::to_string(wid));
      } else {
        g.waiting = true;
        g.wait_n = n;
      }
    }
    DT(3, simobject_->name() << " execute: op=" << tcu_type << ", " << *trace);
    input.pop();
    return true;
  }

  // A WGMMA of warp wid retired; the waits see it from the next cycle.
  void h100_wg_retired(uint32_t wid) {
    auto& g = wg_.at(wid);
    __assert(g.done != g.issued, "wgmma retire without an issue: wid=" << wid);
    ++g.done;
    uint64_t now = SimPlatform::instance().cycles();
    if (h100_retire_cycle_.at(wid) != now) {
      h100_retire_cycle_.at(wid) = now;
      h100_retire_cnt_.at(wid) = 0;
    }
    ++h100_retire_cnt_.at(wid);
    tcu_csv(now, "grp", int32_t(wid % VX_CFG_NUM_TCU_BLOCKS), "retire",
            "w" + std::to_string(wid) + " done" + std::to_string(g.done));
    DT(2, simobject_->name() << " wgmma-retire: wid=" << wid << ", done=" << g.done);
  }

  // The checks the core makes on a WGMMA's first micro-op.
  void h100_check(uint32_t wid, const IntrTcuArgs& args, const TcuH100Obs& o) const {
    auto fail = [&](const char* msg) {
      std::cout << "Error: " << msg << ": wid=" << wid << std::hex
                << ", desc_a=0x" << o.desc_a << ", desc_b=0x" << o.desc_b << std::dec << std::endl;
      std::abort();
    };
    if (!args.is_a_smem)
      fail("H100 WGMMA needs A in shared memory");
    // The tile buffers take the warp's rank in its warpgroup from its slot.
    if ((h100_rank_.at(wid) % 4) != (wid % 4))
      fail("H100 WGMMA needs each CTA's warps 4-aligned");
    uint32_t fs = args.fmt_s, fd = args.fmt_d;
    bool pair_ok = (fs == vt::fp16::id && (fd == vt::fp16::id || fd == vt::fp32::id))
                || ((fs == vt::bf16::id || fs == vt::fp8::id || fs == vt::bf8::id) && fd == vt::fp32::id);
    if (!pair_ok)
      fail("H100 WGMMA does not support this type pair");
    uint32_t esize = elem_bits(fs) / 8;
    if (const char* msg = tcu_desc_ref::check(o.desc_a, esize))
      fail(msg);
    if (const char* msg = tcu_desc_ref::check(o.desc_b, esize))
      fail(msg);
  }

  // One micro-op: lane 4i + q computes row 8 step_m + i of the block's slice
  // and column 8j + 2q + e over one k-step, one FEDP call per lane. A packed
  // f16 D reads and writes half e of the register.
  void h100_wgmma(uint32_t b, instr_trace_t* trace, const TcuH100Obs& o) {
    auto args = std::get<IntrTcuArgs>(trace->instr_ptr->get_args());
    if (args.is_first_uop)
      this->h100_check(trace->wid, args, o);
    auto& bufs = simobject_->tbuf()->h100();
    uint32_t aw[8][TcuH100Bufs::KW];
    uint32_t bw[4][TcuH100Bufs::KW];
    bufs.a_words(b, args.step_m, aw);
    bufs.b_words(args.step_n, bw);
    const bool pk = tcu_h100::packed(args.fmt_d);
    const uint32_t e = args.step_n & 1;
    const uint32_t fmt_d = pk ? tcu_fedp_ref::FMT_FP16 : tcu_fedp_ref::FMT_FP32;
    const auto& rs3 = trace->src_data.at(2);
    auto& rd = trace->dst_data;
    for (uint32_t i = 0; i < 8; ++i) {
      for (uint32_t q = 0; q < 4; ++q) {
        uint32_t t = i * 4 + q;
        uint32_t c = pk ? (uint32_t(rs3.at(t).u64 >> (16 * e)) & 0xffff) : rs3.at(t).u32;
        uint32_t d = tcu_fedp_ref::fedp(args.fmt_s, fmt_d, aw[i], bw[q], c, TcuH100Bufs::KW);
        rd.at(t).u64 = pk ? (uint64_t(d & 0xffff) << (16 * e)) : nan_box(d);
        DTH(3, simobject_->name() << " FEDP: wid=" << trace->wid << ", b=" << b
            << ", i=" << i << ", q=" << q << ", m=" << args.step_m << ", n=" << args.step_n
            << ", k=" << args.step_k << std::hex << ", a0=0x" << aw[i][0] << ", b0=0x" << bw[q][0]
            << ", c=0x" << c << ", d=0x" << d << std::dec << std::endl);
      }
    }
  }

  // A dense WMMA micro-op: float formats through the TFR model with an f32
  // result, as the RTL computes WMMA; integer formats as before.
  void h100_wmma(instr_trace_t* trace, const IntrTcuArgs& args) {
    const auto& rs1 = trace->src_data.at(0);
    const auto& rs2 = trace->src_data.at(1);
    const auto& rs3 = trace->src_data.at(2);
    auto& rd = trace->dst_data;
    uint32_t a_off = (args.step_m % cfg::a_sub_blocks) * cfg::a_block_size;
    uint32_t b_off = (args.step_n % cfg::b_sub_blocks) * cfg::b_block_size;
    bool is_float = (args.fmt_s >= vt::tf32::id && args.fmt_s <= vt::bf8::id);
    PFN_FEDP_N fedp = is_float ? nullptr : select_FEDP_N(args.fmt_s, args.fmt_d);
    for (uint32_t i = 0; i < cfg::tcM; ++i) {
      for (uint32_t j = 0; j < cfg::tcN; ++j) {
        uint32_t aw[cfg::tcK], bw[cfg::tcK];
        reg_data_t ar[cfg::tcK], br[cfg::tcK];
        for (uint32_t z = 0; z < cfg::tcK; ++z) {
          ar[z] = rs1.at(a_off + i * cfg::tcK + z);
          br[z] = rs2.at(b_off + j * cfg::tcK + z);
          aw[z] = ar[z].u32;
          bw[z] = br[z].u32;
        }
        uint32_t t = i * cfg::tcN + j;
        uint32_t c = rs3.at(t).u32;
        uint32_t d = is_float ? tcu_fedp_ref::fedp(args.fmt_s, tcu_fedp_ref::FMT_FP32, aw, bw, c, cfg::tcK)
                              : fedp(ar, br, c, cfg::tcK);
        rd.at(t).u64 = nan_box(d);
      }
    }
  }

  std::array<uint32_t, VX_CFG_NUM_WARPS> h100_slot_{};
  // Each warp's rank in its block, counted from the slot events.
  std::array<uint32_t, VX_CFG_NUM_WARPS> h100_rank_{};
  std::array<uint32_t, VX_CFG_NUM_WARPS> h100_slot_warps_{};
  // The cycle of each warp's last retires and how many, to read done as registered.
  std::array<uint64_t, VX_CFG_NUM_WARPS> h100_retire_cycle_{};
  std::array<uint32_t, VX_CFG_NUM_WARPS> h100_retire_cnt_{};
  uint32_t h100_stride_ = 0;
  // Per block: the descriptors latched at its WGMMA's first micro-op, and
  // the CTA gate's state (inside a WGMMA, of which CTA).
  std::array<std::array<uint64_t, 2>, VX_CFG_NUM_TCU_BLOCKS> h100_desc_{};
  std::array<bool, VX_CFG_NUM_TCU_BLOCKS> h100_inexp_{};
  std::array<int32_t, VX_CFG_NUM_TCU_BLOCKS> h100_owner_{};
  std::array<bool, VX_CFG_NUM_TCU_BLOCKS> h100_csv_cta_wait_{};
