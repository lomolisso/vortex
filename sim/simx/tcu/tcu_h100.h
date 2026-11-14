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

#pragma once

// Tile buffers of the tensor unit with WGMMA as on the H100
// (VX_CFG_TCU_H100_ENABLE): operands are read through 64-bit shared-memory
// descriptors, one 16-byte chunk per local-memory request, in the order and
// with the handshakes of VX_tcu_abuf / VX_tcu_bbuf, so the two models issue
// the same requests in the same cycles.
//
//   A, one per TCU block: the 16 rows of the block's slice of m64 over one
//      k-step (16 bytes of K). A fill is 16 reads; the next read may leave in
//      the cycle the previous response arrives.
//   B, shared: the 8 columns 8j..8j+7 of one column group over one k-step.
//      A fill is 8 reads; the next read leaves the cycle after the previous
//      response.
// A WGMMA's first micro-op always refills both (a K loop rewrites its slab in
// place and reissues the same descriptors).

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "types.h"
#include "instr.h"
#include "debug.h"
#include "tcu_desc_ref.h"

namespace vortex {

// A WGMMA compute micro-op at a TCU block's input in one cycle, as the tile
// buffers see it (the RTL's tcu_tbuf_req_t plus the block's slot base).
struct TcuH100Obs {
  bool     valid    = false;
  bool     first    = false;
  bool     a_smem   = false;
  bool     esize16  = false;
  uint32_t step_m   = 0;
  uint32_t step_n   = 0;
  uint32_t step_k   = 0;
  uint32_t cd_nregs = 0;
  int32_t  cta      = -1;
  uint64_t desc_a   = 0;
  uint64_t desc_b   = 0;
  uint64_t slot_base = 0;  // the block's byte offset in the local memory
};

// Tile-buffer events as CSV on stderr (VORTEX_TCU_CSV=1), in the format the
// RTL's trace uses, so the two request streams can be diffed line by line:
//   TCSV,cycle,unit,block,event,addr_or_step
inline bool tcu_csv_on() {
  static const bool on = [] {
    const char* p = std::getenv("VORTEX_TCU_CSV");
    return p && p[0] == '1';
  }();
  return on;
}

// block < 0 prints "-" (the shared B buffer).
inline void tcu_csv(uint64_t cycle, const char* unit, int32_t block, const char* event,
                    const std::string& arg) {
  if (!tcu_csv_on())
    return;
  if (block < 0)
    std::fprintf(stderr, "TCSV,%lu,%s,-,%s,%s\n", (unsigned long)cycle, unit, event, arg.c_str());
  else
    std::fprintf(stderr, "TCSV,%lu,%s,%d,%s,%s\n", (unsigned long)cycle, unit, block, event, arg.c_str());
}

class TcuH100Bufs {
public:
  static constexpr uint32_t NB      = VX_CFG_NUM_TCU_BLOCKS;
  static constexpr uint32_t A_ROWS  = 16;
  static constexpr uint32_t B_COLS  = 8;
  static constexpr uint32_t KW      = 4;   // 32-bit k-words per micro-op
  static constexpr uint32_t B_SRC   = NB;  // the arbiter serves A0..A3, then B

  struct Perf {
    uint64_t stalls = 0;     // cycles a micro-op waited on A or B, per buffer
    uint64_t hits   = 0;     // cycles B was resident for a micro-op
  };

  void reset() {
    for (auto& s : src_) s = Src{};
    rep_ = -1;
    reads_ = 0;
    perf_ = Perf{};
  }

  // The buffers' readiness for this cycle's micro-ops, from the state at the
  // start of the cycle, and the fills and refetch flags these micro-ops set.
  // Called once per cycle, after the port's tick of the same cycle: a fill
  // the port completed this cycle is seen from the next one, and a fill
  // started here issues its first read in the next one, as with the RTL's
  // registers.
  std::array<bool, NB> observe(const std::array<TcuH100Obs, NB>& obs, uint64_t now) {
    std::array<bool, NB> ready{};
    std::array<bool, NB> a_ready{};
    for (uint32_t b = 0; b < NB; ++b) {
      auto& s = src_.at(b);
      const auto& o = obs.at(b);
      const bool done_now = (s.done_cycle == now);
      bool resident = s.valid && !done_now && s.step_k == o.step_k && (!o.first || s.refetched);
      a_ready.at(b) = !o.valid || !o.a_smem || resident;
      if (o.valid && o.a_smem && !a_ready.at(b))
        ++perf_.stalls;
      if (done_now) {
        if (o.valid && o.first)
          s.refetched = true;
      } else if (o.valid && a_ready.at(b) && !o.first) {
        s.refetched = false;
      }
      if (o.valid && o.a_smem && !resident && !s.fetching && !done_now)
        this->alloc_a(b, o, now);
    }

    rep_ = -1;
    for (uint32_t b = 0; b < NB; ++b) {
      if (obs.at(b).valid) {
        rep_ = int32_t(b);
        break;
      }
    }
    alloc_rep_ = rep_;
    if (tcu_csv_on() && rep_ != last_rep_) {
      tcu_csv(now, "bbuf", -1, "b_rep", std::to_string(rep_));
      last_rep_ = rep_;
    }
    bool b_ready = true;
    if (rep_ >= 0) {
      const auto& o = obs.at(rep_);
      auto& s = src_.at(B_SRC);
      const bool done_now = (s.done_cycle == now);
      bool resident = s.valid && !done_now && s.desc == o.desc_b && s.cta == o.cta
                   && s.step_k == o.step_k && s.j == (o.step_n >> 1)
                   && (!o.first || s.refetched);
      b_ready = resident;
      if (resident)
        ++perf_.hits;
      else
        ++perf_.stalls;
      if (done_now) {
        if (o.first)
          s.refetched = true;
      } else if (resident && !o.first) {
        s.refetched = false;
      }
      if (!resident && !s.fetching && !done_now)
        this->alloc_b(o, now);
    }

    for (uint32_t b = 0; b < NB; ++b) {
      const auto& o = obs.at(b);
      bool key_match = true;
      if (o.valid && rep_ >= 0) {
        const auto& r = obs.at(rep_);
        key_match = o.desc_b == r.desc_b && o.step_k == r.step_k
                 && (o.step_n >> 1) == (r.step_n >> 1) && o.cd_nregs == r.cd_nregs;
      }
      ready.at(b) = a_ready.at(b) && b_ready && key_match;
      if (tcu_csv_on() && (o.valid && !key_match) != key_wait_.at(b)) {
        key_wait_.at(b) = o.valid && !key_match;
        tcu_csv(now, "tcu", int32_t(b), key_wait_.at(b) ? "key_wait" : "key_ok",
                "m" + std::to_string(o.step_m) + " n" + std::to_string(o.step_n) + " k" + std::to_string(o.step_k));
      }
    }
    return ready;
  }

  // The local-memory port: one response and one request per cycle.
  void tick(SimChannel<MemReq>& req_ch, SimChannel<MemRsp>& rsp_ch, uint64_t now) {
    if (!rsp_ch.empty()) {
      const auto& r = rsp_ch.peek();
      uint32_t sid = r.tag >> TAG_SHIFT;
      if (sid <= B_SRC) {
        auto& s = src_.at(sid);
        if (s.fetching && s.inflight && r.data) {
          uint32_t off = uint32_t(s.addrs.at(s.rsp_ctr) & (VX_CFG_MEM_BLOCK_SIZE - 1));
          std::memcpy(s.data.at(s.rsp_ctr).data(), r.data->data() + off, 16);
          s.inflight  = false;
          s.rsp_cycle = now;
          if (s.rsp_ctr + 1 == s.num_reads) {
            s.fetching   = false;
            s.valid      = true;
            s.done_cycle = now;
            DT(3, "tcu-tbuf fill ready: src=" << sid);
          }
          const bool is_b = (sid == B_SRC);
          tcu_csv(now, is_b ? "bbuf" : "abuf", is_b ? -1 : int32_t(sid), is_b ? "b_rsp" : "a_rsp",
                  (is_b ? "c" : "i") + std::to_string(s.rsp_ctr));
          if (s.done_cycle == now)
            tcu_csv(now, is_b ? "bbuf" : "abuf", is_b ? -1 : int32_t(sid), is_b ? "b_ready" : "a_ready", "");
          ++s.rsp_ctr;
        }
      }
      rsp_ch.pop();
    }

    if (req_ch.full())
      return;
    for (uint32_t sid = 0; sid <= B_SRC; ++sid) {
      auto& s = src_.at(sid);
      if (!s.fetching || s.alloc_cycle >= now || s.req_ctr >= s.num_reads || s.inflight)
        continue;
      if (sid == B_SRC && s.rsp_cycle == now)
        continue;
      uint64_t addr = s.addrs.at(s.req_ctr) & ~uint64_t(VX_CFG_MEM_BLOCK_SIZE - 1);
      MemReq m(MemOp::LD, addr, nullptr, 0, (sid << TAG_SHIFT) | s.req_ctr, 0, 0);
      m.flags.local = 1;
      req_ch.send(m, 1);
      DT(3, "tcu-tbuf rd_req: src=" << sid << ", read=" << s.req_ctr << ", addr=0x" << std::hex
         << s.addrs.at(s.req_ctr) << std::dec);
      if (tcu_csv_on()) {
        char a[32];
        std::snprintf(a, sizeof(a), "0x%lx", (unsigned long)(s.addrs.at(s.req_ctr) - VX_MEM_LMEM_BASE_ADDR));
        tcu_csv(now, sid == B_SRC ? "bbuf" : "abuf", sid == B_SRC ? -1 : int32_t(sid),
                sid == B_SRC ? "b_req" : "a_req", a);
      }
      s.inflight = true;
      ++s.req_ctr;
      ++reads_;
      break;
    }
  }

  // A k-words of rows 8 * step_m + i of block b: words[i][kw].
  void a_words(uint32_t b, uint32_t step_m, uint32_t words[8][KW]) const {
    const auto& s = src_.at(b);
    bool mn = tcu_desc_ref::is_mn_major(s.desc);
    for (uint32_t i = 0; i < 8; ++i) {
      uint32_t r = 8 * step_m + i;
      for (uint32_t kw = 0; kw < KW; ++kw) {
        words[i][kw] = mn ? mn_word(s, r / 8, r % 8, kw) : km_word(s, r, kw);
      }
    }
  }

  // B k-words of the columns 8j + 2q + e of micro-op step n' = 2j + e: words[q][kw].
  void b_words(uint32_t step_n, uint32_t words[4][KW]) const {
    const auto& s = src_.at(B_SRC);
    bool mn = tcu_desc_ref::is_mn_major(s.desc);
    for (uint32_t q = 0; q < 4; ++q) {
      uint32_t c = 2 * q + (step_n & 1);
      for (uint32_t kw = 0; kw < KW; ++kw) {
        words[q][kw] = mn ? mn_word(s, 0, c, kw) : km_word(s, c, kw);
      }
    }
  }

  uint64_t reads() const { return reads_; }
  const Perf& perf() const { return perf_; }

private:
  static constexpr uint32_t TAG_SHIFT = 16;

  struct Src {
    bool     valid     = false;
    bool     fetching  = false;
    bool     refetched = false;
    bool     inflight  = false;
    uint64_t alloc_cycle = 0;
    uint64_t rsp_cycle   = ~uint64_t(0);
    uint64_t done_cycle  = ~uint64_t(0);   // the cycle the last response arrived
    uint32_t num_reads = 0;
    uint32_t req_ctr   = 0;
    uint32_t rsp_ctr   = 0;
    // the fill's key
    uint64_t desc   = 0;
    int32_t  cta    = -1;
    uint32_t step_k = 0;
    uint32_t j      = 0;
    // the local-memory byte address of each read and the 16 bytes it returned
    std::array<uint64_t, A_ROWS> addrs{};
    std::array<std::array<uint8_t, 16>, A_ROWS> data{};
  };

  // The local-memory byte of a chunk: the descriptor gives a CTA-relative
  // address within 256 KB, the slot base is added last, and the sum wraps at
  // the local memory's size as the RTL's address does.
  static uint64_t lmem_byte(uint64_t slot_base, uint32_t cta_rel) {
    constexpr uint64_t lmem_mask = (uint64_t(1) << VX_CFG_LMEM_LOG_SIZE) - 1;
    return uint64_t(VX_MEM_LMEM_BASE_ADDR)
         + ((slot_base + (cta_rel & tcu_desc_ref::FIELD_MASK)) & lmem_mask);
  }

  void start(Src& s, uint64_t now, uint32_t num_reads) {
    s.fetching    = true;
    s.valid       = false;
    s.inflight    = false;
    s.alloc_cycle = now;
    s.num_reads   = num_reads;
    s.req_ctr     = 0;
    s.rsp_ctr     = 0;
  }

  // K-major read i: row 16b + i at the k-step's chunk. MN-major read i: k-row
  // 8 step_k + i/2 of MN chunk 2b + i%2.
  void alloc_a(uint32_t b, const TcuH100Obs& o, uint64_t now) {
    tcu_csv(now, "abuf", int32_t(b), "a_alloc", "k" + std::to_string(o.step_k));
    auto& s = src_.at(b);
    this->start(s, now, A_ROWS);
    s.desc   = o.desc_a;
    s.step_k = o.step_k;
    uint32_t esize = o.esize16 ? 2 : 1;
    bool mn = tcu_desc_ref::is_mn_major(o.desc_a);
    for (uint32_t i = 0; i < A_ROWS; ++i) {
      uint32_t rel = mn ? tcu_desc_ref::chunk_addr(o.desc_a, true, esize, 8 * o.step_k + i / 2, 2 * b + i % 2)
                        : tcu_desc_ref::chunk_addr(o.desc_a, false, esize, A_ROWS * b + i, o.step_k);
      s.addrs.at(i) = lmem_byte(o.slot_base, rel);
    }
  }

  // K-major read c: column 8j + c at the k-step's chunk. MN-major read c:
  // k-row 8 step_k + c of MN chunk j.
  void alloc_b(const TcuH100Obs& o, uint64_t now) {
    tcu_csv(now, "bbuf", -1, "b_alloc",
            "k" + std::to_string(o.step_k) + " j" + std::to_string(o.step_n >> 1)
            + " rep" + std::to_string(alloc_rep_));
    auto& s = src_.at(B_SRC);
    this->start(s, now, B_COLS);
    s.desc   = o.desc_b;
    s.cta    = o.cta;
    s.step_k = o.step_k;
    s.j      = o.step_n >> 1;
    uint32_t esize = o.esize16 ? 2 : 1;
    bool mn = tcu_desc_ref::is_mn_major(o.desc_b);
    for (uint32_t c = 0; c < B_COLS; ++c) {
      uint32_t rel = mn ? tcu_desc_ref::chunk_addr(o.desc_b, true, esize, 8 * o.step_k + c, s.j)
                        : tcu_desc_ref::chunk_addr(o.desc_b, false, esize, 8 * s.j + c, o.step_k);
      s.addrs.at(c) = lmem_byte(o.slot_base, rel);
    }
  }

  // Word kw of a K-major line: bytes 4kw..4kw+3 of its chunk.
  static uint32_t km_word(const Src& s, uint32_t line, uint32_t kw) {
    uint32_t w;
    std::memcpy(&w, s.data.at(line).data() + 4 * kw, 4);
    return w;
  }

  // Word kw of element e of MN chunk c (16-bit): k-rows 2kw and 2kw + 1,
  // which are reads 2kw + c and 2kw + 2 + c of an A fill (two chunks per
  // k-row) or reads 2kw and 2kw + 1 of a B fill (one chunk per k-row).
  static uint32_t mn_word(const Src& s, uint32_t c, uint32_t e, uint32_t kw) {
    uint32_t stride = (s.num_reads == A_ROWS) ? 2 : 1;
    uint16_t lo, hi;
    std::memcpy(&lo, s.data.at((2 * kw) * stride + c).data() + 2 * e, 2);
    std::memcpy(&hi, s.data.at((2 * kw + 1) * stride + c).data() + 2 * e, 2);
    return uint32_t(lo) | (uint32_t(hi) << 16);
  }

  std::array<Src, NB + 1> src_{};
  int32_t  rep_   = -1;
  int32_t  alloc_rep_ = -1;
  int32_t  last_rep_ = -2;
  std::array<bool, NB> key_wait_{};
  uint64_t reads_ = 0;
  Perf     perf_;
};

// Micro-op expansion of a WGMMA: ctr runs over the NRC accumulator registers
// and, outside them, over the k-steps (and for a packed f16 D the two halves),
// so every register is revisited NRC micro-ops later. Micro-op (h, n' = 2j+e,
// k) computes rows 8h..8h+7 of the warp's 16 and the B columns 8j + 2q + e.
namespace tcu_h100 {

constexpr uint32_t FMT_FP16 = 2;

inline uint32_t nrc(uint32_t cd_nregs) {
  return (cd_nregs == 0) ? 8 : (cd_nregs == 1) ? 16 : 32;
}

inline bool packed(uint32_t fmt_d) {
  return fmt_d == FMT_FP16;
}

inline uint32_t uop_count(const IntrTcuArgs& args) {
  return nrc(args.cd_nregs) * (packed(args.fmt_d) ? 4 : 2);
}

inline void uop_fill(Instr& uop, const IntrTcuArgs& args, uint32_t ctr, uint32_t total) {
  const uint32_t n  = nrc(args.cd_nregs);
  const uint32_t lg = log2ceil(n);
  const uint32_t r  = ctr % n;
  uint32_t h, j, e, k;
  if (packed(args.fmt_d)) {
    h = r & 1; j = r >> 1; e = (ctr >> lg) & 1; k = ctr >> (lg + 1);
  } else {
    e = r & 1; h = (r >> 1) & 1; j = r >> 2; k = ctr >> lg;
  }
  const bool first = (ctr == 0);
  const bool last  = (ctr == total - 1);
  uop.set_op_type(TcuType::WGMMA);
  uop.set_args(IntrTcuArgs{args.is_a_smem, args.cd_nregs, args.fmt_s, args.fmt_d,
                           h, 2 * j + e, k, first ? 1u : 0u, last ? 1u : 0u, 0});
  uop.set_dest_reg(r, RegType::Float);
  if (first) {
    uop.set_src_reg(0, 10, RegType::Integer);
    uop.set_src_reg(1, 11, RegType::Integer);
  }
  uop.set_src_reg(2, r, RegType::Float);
  if (packed(args.fmt_d)) {
    // One f16 half of the register; the other half and the NaN box stay.
    uop.set_dst_bytesel(uint8_t(0x3u << (2 * e)));
  }
  uop.set_fu_lock(first);
  uop.set_fu_unlock(last);
}

} // namespace tcu_h100

} // namespace vortex
