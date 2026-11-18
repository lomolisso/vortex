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

// Golden model of the H100-shaped shared memory: banked wavefronts, latency
// and atomic passes of one warp request. Shared by SimX, the testbenches and
// the host tests; shapes come from SmemCfg, never VX_config.h.

#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

namespace vortex {
namespace smem_bank_ref {

struct SmemCfg {
  uint32_t banks;        // power of two
  uint32_t bank_bytes;   // power of two
  uint32_t pipe;         // smem_pipe_stages
};

// Register stages a shared access passes outside the wavefront stage, from its
// issue to the dependent instruction's issue (VX_gpu_pkg LMEM_PIPE_OVERHEAD):
// issue, operand collection (+ routing between collectors), the LSU and its
// response, the local-memory switch, and the DLMEM address translation.
constexpr uint32_t smem_pipe_stages(uint32_t num_opcs, bool dlmem) {
  return 12 + ((num_opcs > 1) ? 1 : 0) + 2 + (dlmem ? 1 : 0);
}

namespace detail {

constexpr uint32_t MAX_LANES = 32;
// Bank words one lane can touch: an 8-B lane on 1-B banks.
constexpr uint32_t MAX_WORDS_PER_LANE = 8;

inline uint32_t lane_words(uint64_t addr, uint32_t access_bytes, uint32_t bank_bytes, uint64_t* out) {
  uint64_t first = addr / bank_bytes;
  uint64_t last  = (addr + access_bytes - 1) / bank_bytes;
  uint32_t n = 0;
  for (uint64_t w = first; w <= last && n < MAX_WORDS_PER_LANE; ++w) {
    out[n++] = w;
  }
  return n;
}

} // namespace detail

// D: the largest number of distinct bank words one bank serves, over every
// byte of every active lane; lanes on the same word count once (broadcast).
inline uint32_t smem_wavefronts(const uint64_t* addr, uint32_t lane_mask, uint32_t nlanes,
                                uint32_t access_bytes, const SmemCfg& cfg) {
  uint64_t words[detail::MAX_LANES * detail::MAX_WORDS_PER_LANE];
  uint32_t nwords = 0;
  for (uint32_t i = 0; i < nlanes && i < detail::MAX_LANES; ++i) {
    if (!((lane_mask >> i) & 1))
      continue;
    uint64_t lw[detail::MAX_WORDS_PER_LANE];
    uint32_t n = detail::lane_words(addr[i], access_bytes, cfg.bank_bytes, lw);
    for (uint32_t k = 0; k < n; ++k) {
      bool seen = false;
      for (uint32_t j = 0; j < nwords; ++j) {
        if (words[j] == lw[k]) { seen = true; break; }
      }
      if (!seen)
        words[nwords++] = lw[k];
    }
  }
  uint32_t d = 0;
  for (uint32_t b = 0; b < cfg.banks; ++b) {
    uint32_t cnt = 0;
    for (uint32_t j = 0; j < nwords; ++j) {
      if ((words[j] & (cfg.banks - 1)) == b)
        ++cnt;
    }
    if (cnt > d)
      d = cnt;
  }
  return d;
}

// The array's response register, after the wavefront stage's last cycle.
constexpr uint32_t SMEM_RSP_STAGES = 1;

// Dependent-chase time of a load served in `wavefronts` wavefronts: the
// stages, one cycle of the wavefront stage per wavefront (each reads its banks),
// and the response register.
inline uint32_t smem_load_latency(uint32_t wavefronts, const SmemCfg& cfg) {
  return cfg.pipe + ((wavefronts == 0) ? 1 : wavefronts) + SMEM_RSP_STAGES;
}

// Passes of a warp atomic on the AMO path: the most lane-updates one bank
// performs, every active lane counted (no broadcast), an 8-B lane in both of
// its banks. A CAS takes the same passes.
inline uint32_t smem_atom_passes(const uint64_t* addr, uint32_t lane_mask, uint32_t nlanes,
                                 uint32_t access_bytes, const SmemCfg& cfg) {
  uint32_t per_bank[256] = {0};
  uint32_t banks = (cfg.banks > 256) ? 256 : cfg.banks;
  for (uint32_t i = 0; i < nlanes && i < detail::MAX_LANES; ++i) {
    if (!((lane_mask >> i) & 1))
      continue;
    uint64_t lw[detail::MAX_WORDS_PER_LANE];
    uint32_t n = detail::lane_words(addr[i], access_bytes, cfg.bank_bytes, lw);
    for (uint32_t k = 0; k < n; ++k) {
      per_bank[lw[k] & (banks - 1)] += 1;
    }
  }
  uint32_t p = 0;
  for (uint32_t b = 0; b < banks; ++b) {
    if (per_bank[b] > p)
      p = per_bank[b];
  }
  return p;
}

// A CAS crosses the LSU path in two beats (swap values, then comparands); its
// passes start with the second beat, which the item takes while it holds the stage.
constexpr uint32_t SMEM_CAS_BEATS = 2;

// Wavefront-stage cycles of a warp atomic: one per AMO pass, and a CAS's
// comparand beat.
inline uint32_t smem_atom_stage_cycles(uint32_t passes, bool cas) {
  return ((passes == 0) ? 1 : passes) + (cas ? SMEM_CAS_BEATS - 1 : 0);
}

// Dependent-chase time of a warp atomic of `passes` AMO passes, from its first
// beat: as a load, the stage's cycles in place of the wavefronts (the old
// values leave with the pass's read).
inline uint32_t smem_atom_latency(uint32_t passes, const SmemCfg& cfg, bool cas = false) {
  return cfg.pipe + smem_atom_stage_cycles(passes, cas) + SMEM_RSP_STAGES;
}

// ---- The unified L1 / shared-memory array: rows, ports, AMO path, mbarrier ops ----

// Wavefronts of one warp access on the array: each lane reads or writes
// nbanks[i] consecutive banks from bank_lo[i] of row[i]; a bank serves one row
// per wavefront, lanes on the same {bank, row} share it. For a shared access
// row = word / banks, so this equals smem_wavefronts.
inline uint32_t sram_row_wavefronts(const uint32_t* row, const uint32_t* bank_lo, const uint32_t* nbanks,
                                    uint32_t lane_mask, uint32_t nlanes) {
  uint32_t bank[detail::MAX_LANES * detail::MAX_WORDS_PER_LANE];
  uint32_t brow[detail::MAX_LANES * detail::MAX_WORDS_PER_LANE];
  uint32_t n = 0;
  for (uint32_t i = 0; i < nlanes && i < detail::MAX_LANES; ++i) {
    if (!((lane_mask >> i) & 1))
      continue;
    for (uint32_t k = 0; k < nbanks[i] && k < detail::MAX_WORDS_PER_LANE; ++k) {
      uint32_t b = bank_lo[i] + k;
      bool seen = false;
      for (uint32_t j = 0; j < n; ++j) {
        if (bank[j] == b && brow[j] == row[i]) { seen = true; break; }
      }
      if (!seen) {
        bank[n] = b;
        brow[n] = row[i];
        ++n;
      }
    }
  }
  uint32_t d = 0;
  for (uint32_t j = 0; j < n; ++j) {
    uint32_t cnt = 0;
    for (uint32_t k = 0; k < n; ++k) {
      if (bank[k] == bank[j])
        ++cnt;
    }
    if (cnt > d)
      d = cnt;
  }
  return d;
}

enum SramRdClient  { RD_TCU, RD_DXA, RD_DSMEM, RD_AMO, RD_LSU, RD_N };
enum SramWrClient  { WR_L1FILL, WR_DXA, WR_DSMEM, WR_CPASYNC, WR_LSU, WR_N };
enum SramAmoClient { AMO_DXA, AMO_DSMEM, AMO_MBAR, AMO_LSU, AMO_N };
enum SramWfClass   { WF_SHARED, WF_GLOBAL, WF_N };
enum SramWfOp      { WF_READ, WF_WRITE, WF_AMO };

// One lane-update of the AMO path: bank `bank`, and bank + 1 for an 8-B lane
// (nbanks = 2, even bank). The AMO units sit on even/odd bank pairs.
struct SramAmoLane {
  uint32_t bank;
  uint32_t nbanks;
};

struct SramPortCfg {
  uint32_t dsmem_wr_bytes;
  uint8_t  rd_pri[RD_N];
  uint8_t  wr_pri[WR_N];
  uint8_t  wf_pri_shared;
  uint8_t  wf_pri_global;
  uint8_t  amo_pri[AMO_N];
  uint32_t bank_pairs;    // AMO units, one per even/odd bank pair
  uint32_t amo_wb_delay;  // cycles from an AMO read grant to its write-back (SM_SRAM_AMO_STAGES)
};

// The one arbitration rule: the highest level wins, equal levels share round
// robin from rr (the input after the last winner). Returns n when nothing requests.
inline uint32_t level_pick(const uint8_t* level, const bool* req, uint32_t n, uint32_t& rr) {
  int best = -1;
  for (uint32_t i = 0; i < n; ++i) {
    if (req[i] && int(level[i]) > best)
      best = level[i];
  }
  if (best < 0)
    return n;
  for (uint32_t k = 0; k < n; ++k) {
    uint32_t i = (rr + k) % n;
    if (req[i] && int(level[i]) == best) {
      rr = (i + 1) % n;
      return i;
    }
  }
  return n;
}

// Write cycles of one DSMEM packet's 32-B slots: consecutive slots of one row
// merge up to dsmem_wr_bytes; a different row, a full merge or the packet end
// closes a write.
inline uint32_t dsmem_write_cycles(const uint32_t* slot_rows, uint32_t nslots, uint32_t slot_bytes,
                                   const SramPortCfg& cfg) {
  uint32_t writes = 0, bytes = 0, cur = 0;
  for (uint32_t i = 0; i < nslots; ++i) {
    if (bytes == 0 || slot_rows[i] != cur || bytes + slot_bytes > cfg.dsmem_wr_bytes) {
      ++writes;
      bytes = 0;
      cur = slot_rows[i];
    }
    bytes += slot_bytes;
  }
  return writes;
}

// Cycle model of the array's ports (both RTL and SimX follow it):
// - the wavefront stage holds one LSU item (shared or global class, picked by
//   wf_pri_* when the stage is empty) until each of its wavefronts has its port:
//   a read the read port (RD_LSU), a write the write port (WR_LSU), an AMO pass
//   until every lane has been updated on the AMO path (AMO_LSU);
// - AMO path: per bank pair the winner among the AMO clients' head ops by
//   amo_pri (kept while not granted); the winners together are one RD_AMO read;
//   granted, each pair's winner updates, per bank of the pair, the first of its
//   remaining lanes on that bank (one lane when the two firsts differ and one
//   is 8 B: the earlier), and writes back amo_wb_delay later;
// - read port: one grant among RD_TCU/DXA/DSMEM/AMO/LSU by rd_pri;
// - write port: an AMO write-back due this cycle first, else one grant among
//   the WR clients by wr_pri.
class SramPortRef {
public:
  struct Grants {
    uint32_t rd = RD_N;        // read-port winner, RD_N = none
    uint32_t wr = WR_N;        // write-port winner, WR_N = none or the AMO write-back
    bool     amo_wb = false;   // the write port went to an AMO write-back
    uint32_t amo_pairs = 0;    // bank pairs updated by this cycle's RD_AMO grant (bit mask)
    uint32_t wf_pick = WF_N;   // class picked into the empty wavefront stage this cycle
    bool     wf_done = false;  // the stage's item finished this cycle
    uint32_t wf_id = 0;        // that item's id
  };

  explicit SramPortRef(const SramPortCfg& cfg) : cfg_(cfg), amo_rr_(cfg.bank_pairs, 0), amo_held_(cfg.bank_pairs, AMO_N) {}

  // A client request of `cycles` port grants (rows for TCU/DXA/DSMEM, write
  // cycles for fills); one FIFO per client.
  void push_read(uint32_t client, uint32_t cycles) { if (cycles) rdq_[client].push_back(cycles); }
  void push_write(uint32_t client, uint32_t cycles) { if (cycles) wrq_[client].push_back(cycles); }
  // One 8-B lane-update on bank_pair (a DXA completion, a DSMEM AMO slot).
  void push_amo(uint32_t client, uint32_t bank_pair) {
    amoq_[client].push_back(std::vector<SramAmoLane>(1, SramAmoLane{2 * bank_pair, 2}));
  }
  // One op of several lane-updates (an mbarrier op over a warp's lanes): lanes
  // on different banks go in the same cycle; the client's next op waits.
  void push_amo_op(uint32_t client, const std::vector<SramAmoLane>& lanes) {
    if (!lanes.empty())
      amoq_[client].push_back(lanes);
  }
  // An LSU warp access into its class queue: `cycles` wavefronts (WF_READ,
  // WF_WRITE) or, for WF_AMO, its lane-updates (a CAS lane listed once per pass).
  void push_wf(uint32_t cls, uint32_t op, uint32_t cycles, uint32_t id,
               const std::vector<SramAmoLane>& lanes = std::vector<SramAmoLane>()) {
    wfq_[cls].push_back(WfItem{op, cycles, id, lanes});
  }

  Grants tick() {
    Grants g;
    // Wavefront stage
    if (!wf_busy_) {
      bool req[WF_N] = {!wfq_[WF_SHARED].empty(), !wfq_[WF_GLOBAL].empty()};
      uint8_t lvl[WF_N] = {cfg_.wf_pri_shared, cfg_.wf_pri_global};
      uint32_t c = level_pick(lvl, req, WF_N, wf_rr_);
      if (c != WF_N) {
        wf_ = wfq_[c].front();
        wfq_[c].pop_front();
        wf_busy_ = true;
        g.wf_pick = c;
        if (wf_.op == WF_AMO) {
          wf_.cycles = wf_.lanes.empty() ? 0 : 1;
          if (!wf_.lanes.empty())
            amoq_[AMO_LSU].push_back(wf_.lanes);
        }
      }
    }
    // AMO path: per pair winner among the clients' head ops with a lane there
    uint32_t pairs_req = 0;
    for (uint32_t p = 0; p < cfg_.bank_pairs; ++p) {
      if (amo_held_[p] == AMO_N) {
        bool req[AMO_N];
        for (uint32_t a = 0; a < AMO_N; ++a)
          req[a] = !amoq_[a].empty() && has_pair(amoq_[a].front(), p);
        amo_held_[p] = level_pick(cfg_.amo_pri, req, AMO_N, amo_rr_[p]);
      }
      if (amo_held_[p] != AMO_N)
        pairs_req |= (1u << p);
    }
    // Read port
    bool lsu_rd = wf_busy_ && wf_.op == WF_READ;
    {
      bool req[RD_N] = {!rdq_[RD_TCU].empty(), !rdq_[RD_DXA].empty(), !rdq_[RD_DSMEM].empty(),
                        pairs_req != 0, lsu_rd};
      g.rd = level_pick(cfg_.rd_pri, req, RD_N, rd_rr_);
    }
    // Write port
    bool lsu_wr = wf_busy_ && wf_.op == WF_WRITE;
    if (!wb_.empty() && wb_.front() == cycle_) {
      g.amo_wb = true;
      wb_.pop_front();
    } else {
      bool req[WR_N] = {!wrq_[WR_L1FILL].empty(), !wrq_[WR_DXA].empty(), !wrq_[WR_DSMEM].empty(),
                        !wrq_[WR_CPASYNC].empty(), lsu_wr};
      g.wr = level_pick(cfg_.wr_pri, req, WR_N, wr_rr_);
    }
    // Apply
    if (g.rd < RD_AMO) {
      take(rdq_[g.rd]);
    } else if (g.rd == RD_AMO) {
      g.amo_pairs = pairs_req;
      for (uint32_t p = 0; p < cfg_.bank_pairs; ++p) {
        if (!((pairs_req >> p) & 1))
          continue;
        update_pair(amoq_[amo_held_[p]].front(), p);
        amo_held_[p] = AMO_N;
      }
      for (uint32_t a = 0; a < AMO_N; ++a) {
        if (!amoq_[a].empty() && amoq_[a].front().empty()) {
          amoq_[a].pop_front();
          if (a == AMO_LSU)
            wf_.cycles = 0;
        }
      }
      wb_.push_back(cycle_ + cfg_.amo_wb_delay);
    } else if (g.rd == RD_LSU) {
      --wf_.cycles;
    }
    if (g.wr < WR_LSU) {
      take(wrq_[g.wr]);
    } else if (g.wr == WR_LSU) {
      --wf_.cycles;
    }
    if (wf_busy_ && wf_.cycles == 0) {
      g.wf_done = true;
      g.wf_id = wf_.id;
      wf_busy_ = false;
    }
    ++cycle_;
    return g;
  }

  uint64_t cycle() const { return cycle_; }
  bool idle() const {
    if (wf_busy_ || !wb_.empty())
      return false;
    for (auto& q : rdq_) if (!q.empty()) return false;
    for (auto& q : wrq_) if (!q.empty()) return false;
    for (auto& q : amoq_) if (!q.empty()) return false;
    for (auto& q : wfq_) if (!q.empty()) return false;
    return true;
  }

private:
  struct WfItem {
    uint32_t op;
    uint32_t cycles;
    uint32_t id;
    std::vector<SramAmoLane> lanes;
  };

  static bool on_bank(const SramAmoLane& l, uint32_t b) { return b >= l.bank && b < l.bank + l.nbanks; }

  static bool has_pair(const std::vector<SramAmoLane>& op, uint32_t p) {
    for (auto& l : op) {
      if (on_bank(l, 2 * p) || on_bank(l, 2 * p + 1))
        return true;
    }
    return false;
  }

  static int first_on(const std::vector<SramAmoLane>& op, uint32_t b) {
    for (size_t k = 0; k < op.size(); ++k) {
      if (on_bank(op[k], b))
        return int(k);
    }
    return -1;
  }

  static void update_pair(std::vector<SramAmoLane>& op, uint32_t p) {
    int e = first_on(op, 2 * p);
    int o = first_on(op, 2 * p + 1);
    if (e >= 0 && o >= 0 && e != o && (op[e].nbanks > 1 || op[o].nbanks > 1)) {
      if (e < o) o = -1; else e = -1;
    }
    int hi = (e > o) ? e : o;
    int lo = (e > o) ? o : e;
    op.erase(op.begin() + hi);
    if (lo >= 0 && lo != hi)
      op.erase(op.begin() + lo);
  }
  static void take(std::deque<uint32_t>& q) {
    if (--q.front() == 0)
      q.pop_front();
  }

  SramPortCfg cfg_;
  std::deque<uint32_t> rdq_[RD_N];
  std::deque<uint32_t> wrq_[WR_N];
  std::deque<std::vector<SramAmoLane>> amoq_[AMO_N];
  std::deque<WfItem> wfq_[WF_N];
  std::deque<uint64_t> wb_;
  std::vector<uint32_t> amo_rr_;
  std::vector<uint32_t> amo_held_;
  uint32_t rd_rr_ = 0, wr_rr_ = 0, wf_rr_ = 0;
  WfItem wf_{};
  bool wf_busy_ = false;
  uint64_t cycle_ = 0;
};

// mbarrier ops issued by instructions; each is one AMO pass on the array,
// taken from the op queue, whose write adds a stage to a shared atomic's path.
constexpr uint32_t MBAR_QUEUE_STAGES = 1;

enum MbarOp {
  MBAR_ARRIVE, MBAR_ARRIVE_TX, MBAR_ARRIVE_DROP, MBAR_TEST_WAIT, MBAR_TRY_WAIT,
  MBAR_EXPECT_TX, MBAR_COMPLETE_TX, MBAR_N
};

} // namespace smem_bank_ref
} // namespace vortex
