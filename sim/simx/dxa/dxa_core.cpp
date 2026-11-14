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

#include "dxa_core.h"
#include "dxa_meta.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <deque>
#include <vector>
#ifdef DXA_MBAR_ENABLE
#include <iostream>
#include <string>
#include "dxa_ref.h"
#include "dxa_tmap.h"
#else
#line 24
#endif
#include "core.h"
#include "socket.h"
#include "socket.h"
#include "mem_block_pool.h"
#include "debug.h"
#include "constants.h"

using namespace vortex;

namespace {

// One GMEM port: the DXA joins its socket's L2-facing arb as a single peer.
constexpr uint32_t kDxaMemPorts = 1;

// LMEM "word" granularity for splitting DXA writes. The LocalMem bank
// model applies byteen relative to a VX_CFG_MEM_BLOCK_SIZE-aligned address,
// so each LineWork's destination must lie within one VX_CFG_MEM_BLOCK_SIZE-aligned
// region. mem_block granularity is the byteen-mask scope.
constexpr uint32_t kLmemWordSize = VX_CFG_MEM_BLOCK_SIZE;

// GMEM line size + mask.
constexpr uint32_t kGmemLineSize = VX_CFG_L1_LINE_SIZE;
constexpr uint64_t kGmemLineMask = ~uint64_t(VX_CFG_L1_LINE_SIZE - 1);

// Cores served by one (socket-resident) DXA engine.
constexpr uint32_t kCoresPerSocket = VX_CFG_SOCKET_SIZE;

} // namespace

// ════════════════════════════════════════════════════════════════════
// DxaCore::Impl
// ════════════════════════════════════════════════════════════════════

class DxaCore::Impl {
public:
  // ── Descriptor ──────────────────────────────────────────────────────
  struct Descriptor {
    uint64_t base_addr = 0;
    std::array<uint32_t, 5> sizes = {};
    std::array<uint32_t, 4> strides = {};
    uint32_t meta = 0;
    std::array<uint32_t, 5> element_strides = {};
    std::array<uint16_t, 5> tile_sizes = {};
    uint32_t cfill = 0;
    uint32_t smem_stride = 0;
  };

  // ── Per-line work item produced by addr_gen, consumed by gmem_req ───
  // Each entry = ONE GMEM cache-line read. Row-major writes contribute it to a
  // single contiguous LMEM word write; a K-major transposing load fans it out
  // to `km_num_elems` strided per-element destinations, which smem_wr coalesces
  // back into full byte-masked block writes (one read → bank-parallel scatter,
  // never re-reading a line per element — TMA-style, matching the hardware
  // address generator, which reads per cache line).
  struct LineWork {
    uint64_t gmem_cl_addr;     // CL-aligned global address
    uint64_t smem_word_addr;   // word-aligned SMEM byte address (element 0)
    uint32_t cl_byte_offset;   // start of valid bytes within the CL (element 0)
    uint32_t smem_byte_offset; // destination offset within the SMEM word (element 0)
    uint32_t valid_length;     // valid bytes per write
    uint32_t cfill;            // OOB fill value (lane-replicated)
    bool     oob;              // skip GMEM read; use cfill
    bool     last;             // last work item of the transfer
    uint32_t km_num_elems;     // K-major scatter fan-out (1 = contiguous write)
    uint32_t km_lane_stride;   // SMEM byte stride between scattered elements
    uint8_t  dest_layout;      // DestLayout (Flat/BlockMajor use tiled_dest_elem)
    uint32_t tile_k_row;       // tiled scatter: K-row (dim1 index) for this span
    uint32_t tile_n_base;      // tiled scatter: N (dim0) index where the span starts
#ifdef DXA_MBAR_ENABLE
    uint32_t src_step;         // a gather transfer's line bytes between elements (one a beat), else 0
#else
#line 92
#endif
  };

  // ── Inflight slot for a GMEM read ──────────────────────────────────
  struct InflightSlot {
    bool      allocated = false;
    bool      rsp_arrived = false;
    LineWork  work;            // captured at allocation
    std::shared_ptr<mem_block_t> rsp_data;
  };

  // ── Worker (one per slice) ───────────────────────────────────────────
  enum class WState { IDLE, RUNNING };

  struct Worker {
    WState                  state = WState::IDLE;
    uint32_t                worker_id = 0;
    DxaReq                  req;
    Descriptor              desc;
    bool                    is_multicast = false;
    std::vector<uint32_t>   cta_indices;
    uint32_t                smem_stride = 0;
    uint64_t                issue_cycle = 0;

    // addr_gen: pre-enumerated work list (one entry per GMEM CL +
    // multicast replays handled at smem_wr stage).
    std::vector<LineWork>   work_list;
    uint32_t                ag_idx = 0;          // next entry to issue (gmem_req)

    // gmem_req → rsp_buf → smem_wr inflight bookkeeping.
    std::array<InflightSlot, VX_CFG_DXA_MAX_INFLIGHT> inflight;
    std::deque<uint32_t>    issued_order;        // outstanding tags in issue order
    uint32_t                drain_slot = UINT32_MAX; // slot currently draining

    // smem_wr multicast replay state.
    uint32_t                mc_cta_idx = 0;      // 0..cta_indices.size()
    uint32_t                km_elem_idx = 0;     // 0..km_num_elems (K-major scatter)
    uint32_t                writes_emitted = 0;  // count for perf

    // Completion: the copy into each receiving block has its own id and
    // write count; its final write carries the count.
    uint32_t                xfer_base = 0;
    std::vector<uint32_t>   rcv_writes;
#ifdef DXA_MBAR_ENABLE
    uint32_t                tx_bytes = 0;        // the box, out-of-bounds fill included
    // One entry per receiving block, in replay order: its offset from the
    // issuer's destination, its barrier index and its completion fields.
    struct Rcv {
      uint64_t off      = 0;
      uint32_t bar_idx  = 0;
      uint32_t mbar_off = 0;
      uint32_t slot     = 0;
      uint32_t gen      = 0;
      uint32_t dst_core = 0;
      bool     remote   = false;
    };
    std::vector<Rcv>        rcvs;
    uint32_t                km_byte_off = 0;     // cluster replay: bytes of element km_elem_idx written
    // A bulk copy's line reader (VX_dxa_lmem_rd): its entries, in order, one
    // per line the address generator took. Their word reads to the issuing
    // core take the worker's bus ahead of its write beats; the responses fill
    // the entries in order, and each filled line leaves, in order, when
    // smem_wr captures it, which frees its entry.
    struct RdLine {
      uint64_t addr      = 0;  // the line, in the issuing core's local memory
      uint32_t slot      = 0;
      uint64_t taken     = 0;  // its reads go from the next cycle
      uint64_t filled_at = 0;  // its last response; it leaves from the next cycle
      uint32_t issued    = 0;  // word reads sent
      // Per block of a word: responses landed. A word wider than a block is
      // read on several ports, which may answer in different cycles.
      std::array<uint32_t, DxaCore::LMEM_PORTS_PER_CORE> filled{};
    };
    std::deque<RdLine>      rd_lines;
    // smem_wr's capture registers, pend and defer (the copy's last line while
    // others are outstanding), and its fill buffer, drain_slot: a line
    // captured in a cycle loads the fill buffer from the next one, whose beats
    // go from the cycle after.
    uint32_t                pend_slot  = UINT32_MAX;
    uint32_t                defer_slot = UINT32_MAX;
    uint64_t                fb_at      = 0;   // the cycle drain_slot was loaded
    // The cycle's start, as smem_wr's registers hold it.
    bool                    st_fb = false, st_pend = false, st_defer = false;
    uint32_t                st_outstanding = 0;
    // The PTX traversal (swizzle, element strides, im2col, a global bulk
    // load): a token per line piece, drained one local-memory word a beat.
    bool                    ptx = false;
    bool                    prefetch = false;    // reads only
    uint32_t                swz_mask = 0;        // the swizzle's M, 0 for none
    uint64_t                blk_base = 0;        // the destination block's base, the swizzle's origin
    uint64_t                first_line = 0;      // the first cycle its address generator emits a line
    uint64_t                tok_start  = 0;      // the cycle smem_wr's current line began to drain
    dxa_ref::Desc           ref;                 // the descriptor, decoded by the golden model
    // A store-family transfer: the walk runs from the issuer's local memory
    // (the request's smem side) to global memory, or to a cluster block.
    // addr_gen sends an in-bounds token's local-memory reads, one a cycle,
    // and moves on at its last; the responses are assembled in order, one a
    // cycle, into the token's line image; a complete token waits in the
    // one-entry output slot, which emits its requests one a cycle.
    struct StRead {
      uint64_t word = 0;                     // the physical word read
      std::array<bool, DxaCore::LMEM_PORTS_PER_CORE> got{};
      std::array<uint8_t, DxaCore::LMEM_ROW_SIZE> data{};
      bool     done = false;
      uint64_t at = 0;                       // the cycle its last block landed
    };
    struct StByte {
      uint32_t line_off;                     // in the token's line
      uint64_t phys;                         // its local-memory source byte
    };
    struct StTok {
      LineWork lw;
      std::vector<StByte> bytes;
      uint32_t reads = 0;
      uint32_t got = 0;
      std::array<uint8_t, VX_CFG_L1_LINE_SIZE> img{};
      uint64_t mask = 0;
    };
    struct StWork {
      uint32_t kind = 0;
      uint32_t op = 0;
      uint32_t type = 0;
      uint64_t dst = 0;                      // a cluster reduction's destination
      std::vector<uint64_t> rd;              // addr_gen's token: its reads
      uint32_t rd_sent = 0;
      std::deque<StRead> reads;              // sent, not yet assembled (the credits)
      std::deque<StTok> toks;                // their tokens, in order
      bool     slot_valid = false;
      bool     slot_freed = false;           // emptied in this cycle
      uint64_t slot_at = 0;                  // its first emission cycle
      StTok    slot;
      std::vector<std::pair<uint32_t, uint64_t>> emits;   // the slot's requests: line offset, byte enables there
      uint32_t emit_idx = 0;
      uint32_t writes = 0;
      uint32_t amo_sent = 0;
      uint32_t amo_rcvd = 0;
      uint64_t last_line = 0;
      uint64_t last_wr = 0;                  // the cycle the last write was accepted
      bool     fence_sent = false;
      bool     rd_done = false;
      bool     wr_done = false;
      bool     wr_new = false;               // write_done in this cycle
      uint64_t end_at = 0;
    };
    bool                    store = false;
    StWork                  st;
#else
#line 137
#endif
  };

  // ── Constructor ──────────────────────────────────────────────────────
  explicit Impl(DxaCore* simobject, MemArbiter* gmem_arb)
    : simobject_(simobject)
    , gmem_arb_(gmem_arb)
    , workers_(VX_CFG_NUM_DXA_CORES)
    , cycle_(0)
  {
    for (uint32_t i = 0; i < VX_CFG_NUM_DXA_CORES; ++i)
      workers_[i].worker_id = i;
  }

  void reset() {
    cycle_ = 0;
    queue_.clear();
#ifdef DXA_MBAR_ENABLE
    stream_.clear();
    row_flits_.clear();
    for (auto& w : workers_) {
      w.rd_lines.clear();
      w.pend_slot  = UINT32_MAX;
      w.defer_slot = UINT32_MAX;
    }
    for (auto& q : cs_q_) {
      q.clear();
    }
    for (auto& q : bg_evq_) {
      q.clear();
    }
    std::fill(bg_rr_.begin(), bg_rr_.end(), 0);
    for (auto& w : workers_) {
      w.store = false;
      w.st = Worker::StWork();
    }
    lmem_rr_ = 0;
    red_buf_.clear();
    red_rr_ = 0;
    for (auto& m : maps_) {
      m.valid = false;
    }
    map_rr_     = 0;
    fetch_      = Fetch();
    fetch_port_ = false;
    head_entry_ = 0;
    head_hit_   = false;
#else
#line 153
#endif
    std::fill(beat_seq_.begin(), beat_seq_.end(), 0);
    perf_stats_ = DxaCore::PerfStats();
    for (auto& w : workers_) {
      w.state = WState::IDLE;
      w.work_list.clear();
      w.cta_indices.clear();
      w.issued_order.clear();
      w.drain_slot = UINT32_MAX;
      for (auto& s : w.inflight) { s.allocated = false; s.rsp_arrived = false; s.rsp_data.reset(); }
      w.ag_idx = 0;
      w.mc_cta_idx = 0;
      w.km_elem_idx = 0;
      w.writes_emitted = 0;
    }
  }

  int dcr_write(uint32_t addr, uint32_t value) {
#ifdef DXA_MBAR_ENABLE
    // A launch's KMU writes drop every cached map and restart the victim
    // order; a fetch in flight still lands, but is not kept. The slots stay.
    if (addr >= VX_DCR_KMU_STATE_BEGIN && addr < VX_DCR_KMU_STATE_END) {
      for (auto& m : maps_) {
        m.valid = false;
      }
      map_rr_ = 0;
      fetch_.killed = fetch_.active;
      return 0;
    }
    return set_desc_word(descriptors_.at(VX_DCR_DXA_DESC_SLOT(addr)), VX_DCR_DXA_DESC_WORD(addr), value);
  }

  // One descriptor word: a slot's DCR write, or a fetched map's word.
  static int set_desc_word(Descriptor& d, uint32_t word, uint32_t value) {
#else
#line 171
    uint32_t slot = VX_DCR_DXA_DESC_SLOT(addr);
    uint32_t word = VX_DCR_DXA_DESC_WORD(addr);
    auto& d = descriptors_.at(slot);
#endif
#line 174
    switch (word) {
    case VX_DCR_DXA_DESC_BASE_LO_OFF:   d.base_addr = (d.base_addr & 0xffffffff00000000ull) | value; break;
    case VX_DCR_DXA_DESC_BASE_HI_OFF:   d.base_addr = (d.base_addr & 0x00000000ffffffffull) | (uint64_t(value) << 32); break;
    case VX_DCR_DXA_DESC_SIZE0_OFF:     d.sizes[0] = value; break;
    case VX_DCR_DXA_DESC_SIZE1_OFF:     d.sizes[1] = value; break;
    case VX_DCR_DXA_DESC_SIZE2_OFF:     d.sizes[2] = value; break;
    case VX_DCR_DXA_DESC_SIZE3_OFF:     d.sizes[3] = value; break;
    case VX_DCR_DXA_DESC_SIZE4_OFF:     d.sizes[4] = value; break;
    case VX_DCR_DXA_DESC_STRIDE0_OFF:   d.strides[0] = value; break;
    case VX_DCR_DXA_DESC_STRIDE1_OFF:   d.strides[1] = value; break;
    case VX_DCR_DXA_DESC_STRIDE2_OFF:   d.strides[2] = value; break;
    case VX_DCR_DXA_DESC_STRIDE3_OFF:   d.strides[3] = value; break;
    case VX_DCR_DXA_DESC_META_OFF:      d.meta = value; break;
    case VX_DCR_DXA_DESC_ESTRIDE0_OFF:  d.element_strides[0] = value; break;
    case VX_DCR_DXA_DESC_ESTRIDE1_OFF:  d.element_strides[1] = value; break;
    case VX_DCR_DXA_DESC_ESTRIDE2_OFF:  d.element_strides[2] = value; break;
    case VX_DCR_DXA_DESC_ESTRIDE3_OFF:  d.element_strides[3] = value; break;
    case VX_DCR_DXA_DESC_ESTRIDE4_OFF:  d.element_strides[4] = value; break;
    case VX_DCR_DXA_DESC_TILESIZE01_OFF:
      d.tile_sizes[0] = uint16_t(value & 0xffff);
      d.tile_sizes[1] = uint16_t(value >> 16);
      break;
    case VX_DCR_DXA_DESC_TILESIZE23_OFF:
      d.tile_sizes[2] = uint16_t(value & 0xffff);
      d.tile_sizes[3] = uint16_t(value >> 16);
      break;
    case VX_DCR_DXA_DESC_TILESIZE4_OFF: d.tile_sizes[4] = uint16_t(value & 0xffff); break;
    case VX_DCR_DXA_DESC_CFILL_OFF:     d.cfill = value; break;
    case VX_DCR_DXA_DESC_SMEM_STRIDE_OFF:
      // Round up to the LMEM word size so multicast destination addresses
      // `issuer_addr + r * smem_stride` stay block-aligned. The LMEM model
      // treats requests as block-aligned with a byteen mask; a non-aligned
      // stride would truncate the address and land writes in the wrong
      // block. The dispatcher applies the same rounding to per-CTA LMEM
      // allocations so the host-visible "same offset in receiver LMEM"
      // contract holds end-to-end.
#if VX_CFG_LMEM_NUM_BANKS * (VX_CFG_XLEN / 8) > VX_CFG_MEM_BLOCK_SIZE
      // The slots start on DMA words there, so the stride rounds to one too.
      d.smem_stride = (value + DxaCore::LMEM_ROW_SIZE - 1u) & ~uint32_t(DxaCore::LMEM_ROW_SIZE - 1u);
#else
      d.smem_stride = (value + kLmemWordSize - 1u) & ~uint32_t(kLmemWordSize - 1u);
#endif
#line 232
      break;
    default: break;
    }
    return 0;
  }

  void tick() {
    ++cycle_;

    // Drain in reverse-pipeline order (smem_wr → rsp_buf → gmem_req →
    // addr_gen → setup → req-arb) so each stage's tick reads what its
    // upstream produced earlier this cycle, mimicking forward stage flow.

    // 1) Drain GMEM responses → mark inflight slots arrived.
    // gmem_arb_ already routes responses back to RspOut[worker_id] (its
    // input index) and restores the user's tag (the per-worker slot id).
    for (uint32_t worker_id = 0; worker_id < workers_.size(); ++worker_id) {
      auto& ch = gmem_arb_->RspOut.at(worker_id);
      // One response per port per cycle (each worker has its own RspOut port).
      if (!ch.empty()) {
        auto& rsp = ch.peek();
        uint32_t slot_id = uint32_t(rsp.tag) & (VX_CFG_DXA_MAX_INFLIGHT - 1);
        auto& w = workers_[worker_id];
#ifdef DXA_MBAR_ENABLE
        if (worker_id == 0 && uint64_t(rsp.tag) >= kFetchTag) {
          this->fetch_rsp(rsp);
        } else
        if (w.store) {
          this->store_gmem_rsp(w);
        } else
#else
#line 254
#endif
        if (slot_id < w.inflight.size() && w.inflight[slot_id].allocated) {
          w.inflight[slot_id].rsp_arrived = true;
          w.inflight[slot_id].rsp_data    = rsp.data;
        }
        ch.pop();
      }
    }

#ifdef DXA_MBAR_ENABLE
    // The map fetch's line reads take the GMEM port ahead of the workers.
    fetch_port_ = false;
    this->tick_fetch_send();
    this->drain_lmem_rsp();
    this->tick_bg_events();
    this->tick_core_switch();
    this->tick_stream();
    // While a bulk copy runs, the workers' bus operations pass the RTL's
    // lmem_arb one a cycle, in round-robin order.
    lmem_arb_ = false;
    for (auto& w : workers_) {
      lmem_arb_ |= (w.state == WState::RUNNING && (w.req.bulk || w.store));
    }
    if (lmem_arb_) {
      this->tick_lmem_arb();
    }
#else
#line 241
#endif
    // 2) Tick workers (smem_wr drain → gmem_req issue).
    for (auto& w : workers_) {
      if (w.state == WState::RUNNING) {
#ifdef DXA_MBAR_ENABLE
        const uint32_t writes = w.writes_emitted;
        if (!lmem_arb_)
#else
#line 244
#endif
        tick_worker_smem_wr(w);
#ifdef DXA_MBAR_ENABLE
        // lmem_arb's grant follows the beats here too.
        if (!lmem_arb_ && w.writes_emitted != writes) {
          lmem_rr_ = (w.worker_id + 1) % uint32_t(workers_.size());
        }
#else
#line 245
#endif
        tick_worker_gmem_req(w);
      }
    }

    // 3) Dispatch from queue to idle workers.
#ifdef DXA_MBAR_ENABLE
    // A map operand at the queue head is looked up first; a miss blocks the
    // head until its map is fetched, and the tensormap ops end there.
    bool ready = this->head_ready();
    for (auto& w : workers_) {
      if (w.state == WState::IDLE && ready) {
        const bool fetched = queue_.front().fetched;
        start_worker(w, queue_.front()), queue_.pop_front();
        this->head_popped(fetched);
        // A map operand behind is looked up from the next cycle.
        ready = !queue_.empty() && !queue_.front().map && !queue_.front().tmop && !queue_.front().held;
      }
    }
#else
#line 251
    for (auto& w : workers_) {
      if (w.state == WState::IDLE && !queue_.empty())
        start_worker(w, queue_.front()), queue_.pop_front();
    }
#line 254
#endif

    // 4) Drain DxaUnit channels → req queue (round-robin).
    drain_req_in();
  }

  const DxaCore::PerfStats& perf_stats() const { return perf_stats_; }

private:
  // ── Descriptor helpers ───────────────────────────────────────────────
  static uint32_t desc_rank(uint32_t meta) {
    uint32_t r = (meta >> DXA_DESC_META_DIM_LSB)
                  & ((1u << DXA_DESC_META_DIM_BITS) - 1u);
    return (r == 0) ? 1u : std::min(r, 5u);
  }

  static uint32_t desc_elem_bytes(uint32_t meta) {
    uint32_t enc = (meta >> DXA_DESC_META_ELEMSZ_LSB)
                    & ((1u << DXA_DESC_META_ELEMSZ_BITS) - 1u);
    return 1u << enc;
  }

  // Destination SMEM layout (2-bit LAYOUT meta field). Mirrors dxa.h::Layout.
  enum class DestLayout : uint32_t { RowMajor = 0, KMajor = 1, Flat = 2, BlockMajor = 3 };

  static DestLayout desc_layout(uint32_t meta) {
    uint32_t v = (meta >> DXA_DESC_META_LAYOUT_LSB)
                 & ((1u << DXA_DESC_META_LAYOUT_BITS) - 1u);
    return DestLayout(v);
  }

  // K-major destination layout (TMA style): SMEM addr per element is
  // base + i1 * elem_bytes + e0 * tile1 * elem_bytes (instead of the default
  // row-major base + i1 * tile0 * elem_bytes + e0 * elem_bytes).
  static bool desc_dest_kmajor(uint32_t meta) { return desc_layout(meta) == DestLayout::KMajor; }

  // Flat (sparse) / BlockMajor (dense) reuse the K-major SCATTER datapath
  // (contiguous GMEM read → permuted SMEM write) with a richer per-element
  // destination index. The tile geometry rides ESTRIDE2 (set_tile_geometry).
  static bool desc_dest_tiled(uint32_t meta) {
    DestLayout l = desc_layout(meta);
    return l == DestLayout::Flat || l == DestLayout::BlockMajor;
  }

  // WGMMA tile geometry conveyed by set_tile_geometry() via ESTRIDE2:
  // tcN in [15:0], the dense B block K extent (elements) in [31:16].
  static uint32_t desc_geo_tcn(const Descriptor& d) {
    return std::max<uint32_t>(1u, d.element_strides[2] & 0xffffu);
  }
  static uint32_t desc_geo_bkk(const Descriptor& d) {
    return std::max<uint32_t>(1u, d.element_strides[2] >> 16);
  }

  // SMEM element-index for B element (k = K-row, n = N-col) under the bbuf's
  // native layouts. Mirrors vx_tensor.h::b_sp_flat_idx / b_blockmajor_idx
  // (tcK == tcN for canonical WGMMA configs; ratio = 32-bit-word / elem).
  static uint32_t tiled_dest_elem(DestLayout lay, uint32_t k, uint32_t n,
                                  uint32_t ratio, uint32_t tcN, uint32_t n_steps,
                                  uint32_t bkK) {
    if (lay == DestLayout::Flat) {
      uint32_t b_tcK_words = tcN * 2;
      uint32_t blk_words   = tcN * b_tcK_words;
      uint32_t k_word = k / ratio;
      uint32_t elem   = k % ratio;
      uint32_t k_blk  = k_word / b_tcK_words;
      uint32_t kw_in  = k_word % b_tcK_words;
      uint32_t n_blk  = n / tcN;
      uint32_t n_in   = n % tcN;
      uint32_t word_off = (k_blk * n_steps + n_blk) * blk_words + (kw_in * tcN + n_in);
      return word_off * ratio + elem;
    }
    // BlockMajor (dense): within-block N-outer, K-inner; the block K extent
    // follows the consumer's FEDP depth and is conveyed by the descriptor.
    uint32_t kw          = bkK;
    uint32_t b_blk_elems = kw * tcN;
    uint32_t k_blk = k / kw;
    uint32_t r_in  = k % kw;
    uint32_t n_blk = n / tcN;
    uint32_t n_in  = n % tcN;
    return (k_blk * n_steps + n_blk) * b_blk_elems + n_in * kw + r_in;
  }

  // ── Pull from per-core dxa_req_in[] into queue_ (round-robin) ────────
  void drain_req_in() {
    auto& chs = simobject_->dxa_req_in;
    for (uint32_t i = 0; i < chs.size(); ++i) {
      uint32_t cid = (rr_req_ + i) % chs.size();
      auto& ch = chs.at(cid);
      if (ch.empty()) continue;
      if (queue_.size() >= VX_CFG_DXA_QUEUE_SIZE) break;
      queue_.push_back(ch.peek());
      ch.pop();
      rr_req_ = (cid + 1) % chs.size();
      break;
    }
  }

  // ── Setup: pre-enumerate the work list for this transfer ─────────────
  void start_worker(Worker& w, const DxaReq& req) {
    w.req = req;
    w.issue_cycle = cycle_;
    w.work_list.clear();
    w.ag_idx = 0;
    w.issued_order.clear();
    w.drain_slot = UINT32_MAX;
    for (auto& s : w.inflight) { s.allocated = false; s.rsp_arrived = false; s.rsp_data.reset(); }
    w.mc_cta_idx = 0;
    w.km_elem_idx = 0;
    w.writes_emitted = 0;

    // Multicast setup.
    w.is_multicast = (__builtin_popcount(req.cta_mask) > 1);
    w.cta_indices.clear();
    if (w.is_multicast) {
      for (uint32_t i = 0; i < 32; ++i) {
        if (req.cta_mask & (1u << i)) w.cta_indices.push_back(i);
      }
    }

#ifdef DXA_MBAR_ENABLE
    w.ptx        = false;
    w.prefetch   = false;
    w.swz_mask   = 0;
    w.first_line = 0;
    w.tok_start  = 0;
    w.store      = false;
    if (req.store) {
      this->start_store(w);
      return;
    }
    this->load_receivers(w, req);
    if (req.bulk) {
      this->start_bulk(w);
      return;
    }
#else
#line 372
#endif
    uint32_t num_rcv = w.is_multicast ? uint32_t(w.cta_indices.size()) : 1u;
    if (xfer_seq_ == 0 || xfer_seq_ + num_rcv < xfer_seq_) {
      xfer_seq_ = 1;
    }
    w.xfer_base = xfer_seq_;
    xfer_seq_ += num_rcv;
    w.rcv_writes.assign(num_rcv, 0);

    if (req.desc_slot >= VX_DCR_DXA_DESC_COUNT) {
      // Invalid descriptor — release barrier(s) directly without any LMEM write.
      release_all_barriers(w);
      finish_worker(w);
      return;
    }

    w.desc = descriptors_.at(req.desc_slot);
    w.smem_stride = w.desc.smem_stride;
#ifdef DXA_MBAR_ENABLE
    if (req.map) {
      w.desc = maps_.at(head_entry_).desc;
      w.smem_stride = w.desc.smem_stride;
    }
    if (req.bload) {
      if (!req.prefetch && req.size > req.win_room) {
        this->dxa_abort(w, "DXA box exceeds the destination block: " + std::to_string(req.size)
                        + " bytes, room " + std::to_string(req.win_room));
      }
      w.desc = this->bulk_desc(req);
      w.smem_stride = 0;
    }
    this->setup_ptx(w);
    if (req.prefetch) {
      this->enumerate_ptx(w);
      if (w.work_list.empty()) {
        finish_worker(w);
        return;
      }
      w.work_list.back().last = true;
      w.state = WState::RUNNING;
      perf_stats_.gmem_reads += std::count_if(w.work_list.begin(), w.work_list.end(),
                                              [](const LineWork& lw) { return !lw.oob; });
      return;
    }
#else
#line 389
#endif

#ifdef DXA_MBAR_ENABLE
    if (req.mode) {
      w.tx_bytes = this->box_bytes(w);
    }
    if (!req.bload) {
      const uint64_t extent = this->box_extent(w);
      if (extent > req.win_room) {
        this->dxa_abort(w, "DXA box exceeds the destination block: " + std::to_string(extent)
                        + " bytes, room " + std::to_string(req.win_room));
      }
    }
#endif

#ifdef DXA_MBAR_ENABLE
    this->enumerate_ptx(w);
    if (req.mode && w.work_list.empty()) {
      this->dxa_abort(w, "invalid DXA rank: no work for the box");
    }
#else
#line 401
    enumerate_work_list(w);
#line 406
#endif
    if (w.work_list.empty()) {
      // No work — same edge case as above.
      release_all_barriers(w);
      finish_worker(w);
      return;
    }

    // Mark the *last* work item — its multicast replay's last write carries
    // notify_done.
    w.work_list.back().last = true;

    w.state = WState::RUNNING;
#ifdef DXA_MBAR_ENABLE
    // Fill tokens read nothing.
    perf_stats_.gmem_reads += std::count_if(w.work_list.begin(), w.work_list.end(),
                                            [](const LineWork& lw) { return !lw.oob; });
#else
#line 419
    perf_stats_.gmem_reads += w.work_list.size();
#line 419
#endif

    DT(3, simobject_->name() << "[" << w.worker_id << "] start: core="
       << req.core->id() << ", wid=" << req.wid
       << ", slot=" << req.desc_slot
       << ", lines=" << w.work_list.size()
       << ", multicast=" << w.is_multicast
       << ", num_ctas=" << w.cta_indices.size());
  }

  // Enumerate (CL, smem-word, byte-offset, length, oob) tuples — one per
  // GMEM CL the transfer needs to read. Row-major iteration over outer
  // dims, contiguous span over dim-0, dedup of consecutive same-CL.
  void enumerate_work_list(Worker& w) {
    const auto& desc = w.desc;
    uint32_t rank = desc_rank(desc.meta);
    if (rank < 1 || rank > 5) return;

    uint32_t   elem_bytes  = desc_elem_bytes(desc.meta);
    DestLayout layout      = desc_layout(desc.meta);
    bool       dest_kmajor = (layout == DestLayout::KMajor);
    bool       dest_tiled  = desc_dest_tiled(desc.meta);
    std::array<uint32_t, 5> tiles = {};
    for (uint32_t d = 0; d < 5; ++d)
      tiles[d] = (d < rank) ? std::max<uint32_t>(1u, desc.tile_sizes[d]) : 1u;

    // K-major / tiled (Flat, BlockMajor) scatter require rank ≤ 2.
    if ((dest_kmajor || dest_tiled) && rank > 2) {
      dest_kmajor = false;
      dest_tiled  = false;
      layout      = DestLayout::RowMajor;
    }
    // Both share the single-element scatter datapath (one read → per-element
    // strided/permuted writes); row-major writes the span contiguously.
    bool scatter = dest_kmajor || dest_tiled;

    uint32_t total_rows = 1;
    for (uint32_t d = 1; d < rank; ++d) total_rows *= tiles[d];
    uint32_t row_elems = tiles[0];

    // K-major SMEM per-lane stride = tile1 * elem_bytes.
    uint32_t per_lane_stride_bytes = tiles[1] * elem_bytes;

    // Tiled (Flat/BlockMajor) geometry: ratio = 32-bit-word / elem, tcN from
    // ESTRIDE2, n_steps = row_elems(=xtileN) / tcN.
    uint32_t geo_ratio  = dest_tiled ? std::max<uint32_t>(1u, 4u / elem_bytes) : 1u;
    uint32_t geo_tcN    = dest_tiled ? desc_geo_tcn(desc) : 1u;
    uint32_t geo_bkK    = dest_tiled ? desc_geo_bkk(desc) : 1u;
    uint32_t geo_nsteps = dest_tiled ? std::max<uint32_t>(1u, row_elems / geo_tcN) : 1u;

    uint32_t cfill = desc.cfill;
    uint64_t global_prev_cl = ~uint64_t(0);

    for (uint32_t row = 0; row < total_rows; ++row) {
      // Decompose row into outer-dim local offsets (dim1 varies fastest).
      uint32_t outer[4] = {};
      uint32_t rem = row;
      for (uint32_t d = 1; d < rank; ++d) {
        outer[d - 1] = rem % tiles[d];
        rem /= tiles[d];
      }

      // OOB check on outer dims.
      bool row_in_bounds = true;
      for (uint32_t d = 1; d < rank; ++d) {
        if (w.req.coords[d] + outer[d - 1] >= desc.sizes[d]) { row_in_bounds = false; break; }
      }

      // First element GMEM addr in this row.
      uint64_t row_gbase = desc.base_addr + uint64_t(w.req.coords[0]) * elem_bytes;
      for (uint32_t d = 1; d < rank; ++d)
        row_gbase += uint64_t(w.req.coords[d] + outer[d - 1]) * uint64_t(desc.strides[d - 1]);

      // SMEM destination base for this row:
      //   Row-major:  base + row_idx * row_elems * elem_bytes  (i1 outer × i0 inner).
      //   K-major:    base + outer[0] * elem_bytes              (lane stride applies per e0 below).
      //   Tiled:      computed per-element below (smem_wr recomputes the permuted dest).
      uint64_t row_smem_base = dest_kmajor
          ? (w.req.smem_addr + uint64_t(outer[0]) * elem_bytes)
          : (w.req.smem_addr + uint64_t(row * row_elems) * elem_bytes);

      // Walk dim-0 elements; each LineWork carries one MemReq write of at
      // most one LMEM word AND at most one GMEM CL (so byteen fits in the
      // 64-byte mem_block payload). K-major scatters each element to its
      // own SMEM destination (lane stride = tile1 * elem_bytes).
      for (uint32_t e0 = 0; e0 < row_elems; ) {
        uint64_t gaddr_e = row_gbase + uint64_t(e0) * elem_bytes;
        uint64_t saddr_e;
        if (dest_tiled) {
          uint32_t de = tiled_dest_elem(layout, outer[0], e0, geo_ratio, geo_tcN, geo_nsteps, geo_bkK);
          saddr_e = w.req.smem_addr + uint64_t(de) * elem_bytes;
        } else if (dest_kmajor) {
          saddr_e = row_smem_base + uint64_t(e0) * uint64_t(per_lane_stride_bytes);
        } else {
          saddr_e = row_smem_base + uint64_t(e0) * elem_bytes;
        }
        uint64_t cl_addr = gaddr_e & kGmemLineMask;
        uint64_t sword   = saddr_e & ~uint64_t(kLmemWordSize - 1);
        uint32_t cl_off  = uint32_t(gaddr_e - cl_addr);
        uint32_t s_off   = uint32_t(saddr_e - sword);
        uint32_t cl_room  = kGmemLineSize - cl_off;
        uint32_t sw_room  = kLmemWordSize - s_off;
        uint32_t row_room = (row_elems - e0) * elem_bytes;
        // The GMEM read span is ALWAYS coalesced to the cache line: dim-0
        // elements are contiguous in global memory, so one cache-line read
        // covers as many as fit. K-major then SCATTERS those elements to
        // strided SMEM destinations (one read → km_num_elems writes); row-major
        // writes them contiguously (bounded by the SMEM word). This mirrors the
        // hardware address generator, which reads per cache line and never
        // re-reads per element.
        uint32_t gspan = scatter
            ? std::min({cl_room, row_room, uint32_t(VX_CFG_MEM_BLOCK_SIZE)})
            : std::min({cl_room, sw_room, row_room, uint32_t(VX_CFG_MEM_BLOCK_SIZE)});
        // Round to a whole-element multiple — ensures e0 advances by an
        // integer count and avoids off-by-one element splits.
        gspan -= gspan % elem_bytes;
        if (gspan == 0) break;
        uint32_t num_elems = gspan / elem_bytes;

        bool elem_oob = !row_in_bounds || (w.req.coords[0] + e0 >= desc.sizes[0]);
        // OOB is determined per-element at the start of the span; tile widths
        // are typically aligned to dim-0 size, so this is exact for the common case.

        LineWork lw{};
        lw.gmem_cl_addr     = cl_addr;
        lw.smem_word_addr   = sword;
        lw.cl_byte_offset   = cl_off;
        lw.smem_byte_offset = s_off;
        // Row-major: one contiguous write of the whole span. Scatter (K-major
        // or tiled Flat/BlockMajor): each element is its own write (elem_bytes),
        // gathered into block writes by smem_wr.
        lw.valid_length     = scatter ? elem_bytes : gspan;
        lw.cfill            = cfill;
        lw.oob              = elem_oob;
        lw.last             = false;
        lw.km_num_elems     = scatter ? num_elems : 1;
        lw.km_lane_stride   = dest_kmajor ? per_lane_stride_bytes : 0;
        lw.dest_layout      = uint8_t(layout);
        lw.tile_k_row       = outer[0];   // tiled scatter: K-row (dim1)
        lw.tile_n_base      = e0;          // tiled scatter: N (dim0) span start

        // Dedup consecutive same-CL (non-OOB only).
        if (!elem_oob && cl_addr == global_prev_cl) {
          ++perf_stats_.gmem_dedup;
        } else if (!elem_oob) {
          global_prev_cl = cl_addr;
        }
        w.work_list.push_back(lw);

        e0 += num_elems;
      }
    }
  }

  // ── gmem_req: issue MemReqs with available inflight slots ────────────
  void tick_worker_gmem_req(Worker& w) {
#ifdef DXA_MBAR_ENABLE
    if (w.req.bulk || w.store) {
      return;
    }
    if (w.first_line != 0 && SimPlatform::instance().cycles() < w.first_line) {
      return;
    }
    // A prefetch's out-of-bounds row: a generator cycle, no token.
    if (w.prefetch && w.ag_idx < w.work_list.size() && w.work_list[w.ag_idx].oob) {
      ++w.ag_idx;
      return;
    }
#else
#line 574
#endif
    // One AGU step per cycle: advance ag_idx by at most one, issuing at most
    // one GMEM read to this worker's ReqIn port (or one OOB skip). A while-loop
    // here would issue many reads to a single port in one cycle.
    if (w.ag_idx >= w.work_list.size())
      return;

    // Find a free slot.
    uint32_t slot = UINT32_MAX;
    for (uint32_t s = 0; s < w.inflight.size(); ++s) {
      if (!w.inflight[s].allocated) {
        slot = s;
        break;
      }
    }

    const LineWork& lw = w.work_list[w.ag_idx];

    if (slot == UINT32_MAX) {
      // All slots in flight. Out-of-bounds lines issue no read, so they are
      // excluded: this counts only reads the slot pool actually held back.
      if (!lw.oob) {
        ++perf_stats_.noslot_stalls;
      }
      return;
    }

    // OOB lines skip the GMEM request entirely; we synthesize an immediate
    // arrival (rsp data left null — smem_wr will use cfill).
    if (lw.oob) {
      w.inflight[slot].allocated   = true;
      w.inflight[slot].rsp_arrived = true;
      w.inflight[slot].rsp_data.reset();
      w.inflight[slot].work        = lw;
      w.issued_order.push_back(slot);
      ++w.ag_idx;
      return;
    }

    // Issue real GMEM read. Tag = per-worker slot id (gmem_arb_ prepends
    // its own input-index bits for response routing — we only see slot
    // bits coming back).
    MemReq mreq;
    mreq.addr  = lw.gmem_cl_addr;
    mreq.tag   = slot;
    mreq.hart_id   = w.req.core->id();
    mreq.uuid  = w.req.uuid;

#ifdef DXA_MBAR_ENABLE
    if (fetch_port_) return;  // the map fetch holds the GMEM port
#else
#line 621
#endif
    auto& ch = gmem_arb_->ReqIn.at(w.worker_id);
#ifdef DXA_MBAR_ENABLE
    if (!ch.try_send(mreq, kGmemReqDelay)) return; // arb backpressure
#else
#line 623
    if (!ch.try_send(mreq)) return; // arb backpressure
#endif
#line 624

    w.inflight[slot].allocated   = true;
    w.inflight[slot].rsp_arrived = false;
    w.inflight[slot].rsp_data.reset();
    w.inflight[slot].work        = lw;
    w.issued_order.push_back(slot);
    ++w.ag_idx;
  }

  // ── smem_wr: drain ready slots, build LMEM MemReqs ───────────────────
  void tick_worker_smem_wr(Worker& w) {
#ifdef DXA_MBAR_ENABLE
    if (w.prefetch) {
      this->drain_prefetch(w);
      return;
    }
#else
#line 634
#endif
    if (w.issued_order.empty()) {
      // Nothing in flight — if all addr_gen done, transfer is finished.
      if (w.ag_idx == w.work_list.size()) finish_worker(w);
      return;
    }

    // Match the RTL's OoO direct-drain behavior: select any ready non-last
    // slot without waiting for older requests. Once selected, keep draining
    // the same slot because the multicast/scatter cursors are worker-local.
    // The last work item carries notify_done and must remain deferred until
    // it is the only outstanding slot.
    uint32_t slot = w.drain_slot;
    if (slot == UINT32_MAX) {
      for (uint32_t i = 0; i < w.inflight.size(); ++i) {
        const auto& candidate = w.inflight[i];
        if (candidate.allocated && candidate.rsp_arrived && !candidate.work.last) {
          slot = i;
          break;
        }
      }

      if (slot == UINT32_MAX && w.issued_order.size() == 1) {
        uint32_t last_slot = w.issued_order.front();
        const auto& candidate = w.inflight[last_slot];
        if (candidate.allocated && candidate.rsp_arrived)
          slot = last_slot;
      }

      if (slot == UINT32_MAX) return;
      w.drain_slot = slot;
    }

    // Determine destination core's LMEM port group.
    uint32_t socket_local_cid = w.req.core->id() % kCoresPerSocket;
#ifdef DXA_MBAR_ENABLE
    // A beat for the row egress, or one behind a beat still in the socket's
    // stream, queues in that stream; any other goes straight to its core.
    const uint32_t beat_rcv = w.is_multicast ? w.mc_cta_idx : 0u;
    const bool to_stream = w.rcvs.at(beat_rcv).remote || !stream_.empty();
#else
#line 664
#endif

    // One drain beat per cycle covers one aligned LMEM row
    // (DxaCore::LMEM_ROW_SIZE bytes — the full banked row the RTL writes in
    // a single cycle). A row wider than the 64B mem_block byteen scope is
    // emitted as LMEM_PORTS_PER_CORE same-cycle block writes, one per row
    // half on its own LocalMem input port. A row narrower than a block
    // (small NT) bounds the per-beat gather instead.

    // A banked-row write is committed atomically, byte enables aside, so a row is
    // never half-written. The row spans LMEM_PORTS_PER_CORE channels that fill
    // independently, so gate the beat on the whole group being ready instead of
    // stranding the half already sent.
#ifdef DXA_MBAR_ENABLE
    // smem_wr captures a line into its skid register a cycle before the fill
    // buffer takes it, so a line's first beat comes at least two cycles after
    // the previous line's.
    const bool tok_first = (w.km_elem_idx == 0 && w.km_byte_off == 0 && w.mc_cta_idx == 0);
    const bool tok_gap   = (w.req.tload || w.req.bload) && tok_first;
    if (tok_gap && cycle_ < w.tok_start + 2)
      return;
    if (lmem_arb_) {
      if (lmem_taken_)
        return;
      lmem_taken_ = true;
    }
    if (to_stream ? stream_.full() : !this->core_inputs_ready(socket_local_cid))
      return;
    if (tok_gap)
      w.tok_start = cycle_;
    if (w.req.cluster || w.ptx) {
      this->emit_cluster_beat(w, socket_local_cid, to_stream);
      return;
    }
#else
#line 676
    for (uint32_t p = 0; p < DxaCore::LMEM_PORTS_PER_CORE; ++p) {
      if (simobject_->lmem_req_out.at(socket_local_cid * DxaCore::LMEM_PORTS_PER_CORE + p).full())
        return;
    }
#endif
    constexpr uint64_t kRowMask = ~uint64_t(DxaCore::LMEM_ROW_SIZE - 1);
    uint64_t beat_row = UINT64_MAX;
    // The beat's writes go out together, so that each carries whether the
    // beat ends a copy: the core's DXA input holds such a beat whole.
    std::array<MemReq, DxaCore::LMEM_PORTS_PER_CORE> beat;
    std::array<uint32_t, DxaCore::LMEM_PORTS_PER_CORE> beat_port;
    uint32_t beat_writes = 0;
    bool beat_last = false;

    for (uint32_t emitted = 0; emitted < DxaCore::LMEM_PORTS_PER_CORE; ++emitted) {
      slot = w.drain_slot;
      if (slot == UINT32_MAX) {
        break; // slot released mid-beat
      }
      auto& s = w.inflight[slot];

      const LineWork& lw = s.work;

      // One cache-line read fans out to its scattered writes. The K-major
      // elements landing in the same SMEM block are written TOGETHER in one
      // byte-masked block write: the per-core LMEM port accepts a full
      // VX_CFG_MEM_BLOCK_SIZE word per cycle (banked), exactly like the warp
      // array's transpose — so the engine drains at SMEM bandwidth, not one
      // element per cycle (TMA-style full-bandwidth scatter). Cursors:
      // km_elem_idx advances one block per beat; mc_cta_idx replays the whole
      // group to each multicast receiver.
      const uint32_t num_elems = lw.km_num_elems ? lw.km_num_elems : 1;
      const uint32_t e0        = w.km_elem_idx;
      const uint32_t wlen      = lw.valid_length;

#ifdef DXA_MBAR_ENABLE
      const auto& rcv_e = w.rcvs.at(w.is_multicast ? w.mc_cta_idx : 0u);
      uint32_t cta_warp_idx = 0;
      uint64_t cta_off = 0;
      if (w.is_multicast) {
        cta_warp_idx = w.cta_indices.at(w.mc_cta_idx);
        cta_off = uint64_t(cta_warp_idx) * w.smem_stride;
      }
#else
#line 711
      uint32_t cta_warp_idx = 0;
      uint64_t cta_off = 0;
      if (w.is_multicast) {
        cta_warp_idx = w.cta_indices.at(w.mc_cta_idx);
        cta_off = uint64_t(cta_warp_idx) * w.smem_stride;
      }
#endif

      // Per-element SMEM byte destination. K-major: uniform lane stride from a
      // span base. Tiled (Flat/BlockMajor): permuted index from the WGMMA
      // B-buffer layout formula (idx = within-span dim-0 offset → N = base+idx).
      const bool     tiled    = desc_dest_tiled(w.desc.meta);
      const uint32_t t_eb     = desc_elem_bytes(w.desc.meta);
      const uint32_t t_ratio  = tiled ? std::max<uint32_t>(1u, 4u / t_eb) : 1u;
      const uint32_t t_tcN    = tiled ? desc_geo_tcn(w.desc) : 1u;
      const uint32_t t_bkK    = tiled ? desc_geo_bkk(w.desc) : 1u;
      const uint32_t t_nsteps = tiled ? std::max<uint32_t>(1u, uint32_t(w.desc.tile_sizes[0]) / t_tcN) : 1u;
      auto dest_byte_of = [&](uint32_t idx) -> uint64_t {
        if (tiled) {
          uint32_t de = tiled_dest_elem(DestLayout(lw.dest_layout), lw.tile_k_row,
                                        lw.tile_n_base + idx, t_ratio, t_tcN, t_nsteps, t_bkK);
          return w.req.smem_addr + uint64_t(de) * t_eb + cta_off;
        }
        return lw.smem_word_addr + lw.smem_byte_offset
             + uint64_t(idx) * lw.km_lane_stride + cta_off;
      };

      // The block targeted this write = the block of scatter element e0.
      uint64_t base_byte0 = dest_byte_of(e0);
      uint64_t dword = base_byte0 & ~uint64_t(kLmemWordSize - 1);

      // Row constraint: every write of this beat stays in one aligned LMEM row.
      uint64_t row = base_byte0 & kRowMask;
      if (beat_row == UINT64_MAX) {
        beat_row = row;
      } else if (row != beat_row) {
        break; // next block lands in a different row — next cycle's beat
      }

      // Route this row half to its own LocalMem input port.
      uint32_t half = (DxaCore::LMEM_PORTS_PER_CORE > 1)
          ? uint32_t((dword / kLmemWordSize) & (DxaCore::LMEM_PORTS_PER_CORE - 1))
          : 0u;
#ifdef DXA_MBAR_ENABLE
      // A stream beat takes one entry whole, checked above.
      if (!to_stream) {
        auto& lmem_ch = simobject_->lmem_req_out.at(
            socket_local_cid * DxaCore::LMEM_PORTS_PER_CORE + half);
        uint32_t port_queued = 0;
        for (uint32_t k = 0; k < beat_writes; ++k) {
          port_queued += (beat_port.at(k) == socket_local_cid * DxaCore::LMEM_PORTS_PER_CORE + half);
        }
        if (lmem_ch.size() + port_queued >= lmem_ch.capacity()) {
          break; // backpressure
        }
      }
#else
#line 753
      auto& lmem_ch = simobject_->lmem_req_out.at(
          socket_local_cid * DxaCore::LMEM_PORTS_PER_CORE + half);
      uint32_t port_queued = 0;
      for (uint32_t k = 0; k < beat_writes; ++k) {
        port_queued += (beat_port.at(k) == socket_local_cid * DxaCore::LMEM_PORTS_PER_CORE + half);
      }
      if (lmem_ch.size() + port_queued >= lmem_ch.capacity()) {
        break; // backpressure
      }
#endif
      // Build LMEM MemReq with TLM payload.
      MemReq req;
      req.addr   = dword;
      req.op = MemOp::ST;
      req.tag    = w.req.core->id();           // routing tag
      req.hart_id = w.req.core->id();
      req.uuid   = w.req.uuid;
      req.flags.dma = 1;
      uint64_t byteen = 0;
      auto blk = make_mem_block();
      const uint32_t pat = lw.cfill;

      // Gather every scatter element that falls in this block (and, for rows
      // narrower than a block, this row) into one write.
      uint32_t ee = e0;
      for (; ee < num_elems; ++ee) {
        uint64_t dest_byte = dest_byte_of(ee);
        if ((dest_byte & ~uint64_t(kLmemWordSize - 1)) != dword) break;
        if ((dest_byte & kRowMask) != row) {
          break;
        }
        uint32_t doff  = uint32_t(dest_byte - dword);
        uint64_t emask = (wlen >= 64) ? ~uint64_t(0) : ((uint64_t(1) << wlen) - 1ull);
        byteen |= emask << doff;
        if (s.rsp_data) {
          std::memcpy(blk->data() + doff,
                      s.rsp_data->data() + lw.cl_byte_offset + ee * wlen,
                      wlen);
        } else {
#ifdef DXA_MBAR_ENABLE
          // The fill's byte at local-memory offset a is byte a & 3 of the pattern.
          for (uint32_t b = 0; b < wlen; ++b)
            (*blk)[doff + b] = uint8_t((pat >> (((doff + b) & 3) * 8)) & 0xff);
#else
#line 792
          for (uint32_t b = 0; b < wlen; ++b)
            (*blk)[doff + b] = uint8_t((pat >> ((b & 3) * 8)) & 0xff);
#line 793
#endif
        }
      }
      req.byteen = byteen;
      req.data = blk;

      // notify_done on the LAST block write of the transfer — when the gather
      // reached the last scatter element of the last work item, per receiver.
      bool is_last_elem   = (ee == num_elems);
      bool is_last_work   = lw.last;
      bool is_last_replay = !w.is_multicast || (w.mc_cta_idx + 1 == w.cta_indices.size());
      const uint32_t rcv  = w.is_multicast ? w.mc_cta_idx : 0u;
      req.dxa.xfer = w.xfer_base + rcv;
#ifdef DXA_MBAR_ENABLE
      req.dxa.mode     = w.req.mode;
      req.dxa.mbar_off = rcv_e.mbar_off;
      req.dxa.tx_bytes = w.tx_bytes;
      req.dxa.slot     = rcv_e.slot;
      req.dxa.gen      = rcv_e.gen;
      req.dxa.remote   = rcv_e.remote;
      req.dxa.src_core = w.req.src_core;
      req.dxa.dst_core = rcv_e.dst_core;
#else
#line 813
#endif
      if (is_last_work && is_last_elem && (w.is_multicast || is_last_replay)) {
        req.flags.dxa_notify_done   = 1;
        req.flags.dxa_notify_bar_id = w.req.bar_id + (w.is_multicast ? cta_warp_idx : 0u);
        req.dxa.writes = w.rcv_writes.at(rcv) + 1;
        beat_last = true;
      }
      ++w.rcv_writes.at(rcv);

      beat.at(beat_writes) = req;
      beat_port.at(beat_writes) = socket_local_cid * DxaCore::LMEM_PORTS_PER_CORE + half;
      ++beat_writes;
      ++w.writes_emitted;
      // Count write transactions, not port requests: one per banked-row beat.
      // A beat spans LMEM_PORTS_PER_CORE ports of the same row, so only its
      // first emission is counted.
      if (0 == emitted) {
        ++perf_stats_.lmem_writes;
      }

      // Advance scatter cursor (to first ungathered element); then multicast
      // cursor; then release the slot.
      if (!is_last_elem) {
        w.km_elem_idx = ee;
      } else {
        w.km_elem_idx = 0;
        if (w.is_multicast && (w.mc_cta_idx + 1) < w.cta_indices.size()) {
          ++w.mc_cta_idx;
        } else {
          w.mc_cta_idx = 0;
          // Slot done — release.
          s.allocated = false;
          s.rsp_arrived = false;
          s.rsp_data.reset();
          auto it = std::find(w.issued_order.begin(), w.issued_order.end(), slot);
          if (it != w.issued_order.end()) w.issued_order.erase(it);
          w.drain_slot = UINT32_MAX;
        }
      }
    } // per-tick row-beat emission loop

#ifdef DXA_MBAR_ENABLE
    if (to_stream && beat_writes != 0) {
      StreamBeat sb;
      sb.remote = w.rcvs.at(beat_rcv).remote;
      sb.core   = socket_local_cid;
      sb.count  = beat_writes;
      sb.ready  = cycle_ + 1;
      const uint32_t id = sb.remote ? 0 : beat_seq_.at(socket_local_cid)++;
      for (uint32_t k = 0; k < beat_writes; ++k) {
        beat.at(k).dxa.beat = id;
        beat.at(k).dxa.last = beat_last;
        sb.writes.at(k) = beat.at(k);
        sb.ports.at(k)  = beat_port.at(k);
      }
      stream_.push(sb);
      return;
    }
#else
#line 853
#endif
    if (beat_writes != 0) {
      const uint32_t id = beat_seq_.at(socket_local_cid)++;
      for (uint32_t k = 0; k < beat_writes; ++k) {
        beat.at(k).dxa.beat = id;
        beat.at(k).dxa.last = beat_last;
        simobject_->lmem_req_out.at(beat_port.at(k)).send(beat.at(k));
      }
    }
  }

#ifdef DXA_MBAR_ENABLE
  [[noreturn]] void dxa_abort(const Worker& w, const std::string& msg) {
    std::cerr << SimPlatform::instance().cycles() << ": *** " << simobject_->name()
              << "[" << w.worker_id << "] " << msg << ": core=" << w.req.core->id()
              << " wid=" << w.req.wid << " desc=" << w.req.desc_slot
              << " (#" << w.req.uuid << ")" << std::endl;
    std::abort();
  }

  // A beat of the socket's stream (below).
  struct StreamBeat {
    std::array<MemReq, DxaCore::LMEM_PORTS_PER_CORE>   writes;
    std::array<uint32_t, DxaCore::LMEM_PORTS_PER_CORE> ports;   // a local beat's core inputs
    uint32_t count  = 0;
    uint32_t core   = 0;
    bool     remote = false;
    uint64_t ready  = 0;   // the cycle it reaches the arbiter output
  };

  bool core_inputs_ready(uint32_t core) const {
    for (uint32_t p = 0; p < DxaCore::LMEM_PORTS_PER_CORE; ++p) {
      if (simobject_->lmem_req_out.at(core * DxaCore::LMEM_PORTS_PER_CORE + p).full())
        return false;
    }
    return true;
  }

  bool core_inputs_empty() const {
    for (const auto& ch : simobject_->lmem_req_out) {
      if (ch.size() != 0)
        return false;
    }
    return true;
  }

  // The RTL engine's beats leave through one arbiter output, two deep, split at
  // its head between the cores' DMA ports and the row egress. While a beat is
  // in this stream every later beat queues behind it; with the stream empty a
  // local beat goes straight to its core, as before. A local head moves to its
  // core's DXA input. A remote head starts once the beats ahead of it have left
  // the core inputs, and the row egress keeps it until its last flit has gone:
  // one flit per cycle, a DATA flit per bank word the beat writes, in bank
  // order, and after a copy's final beat its COMPL. A flit's send is its offer
  // at the request switch's DXA input, which holds two, as the RTL egress's
  // output buffer does.
  void tick_stream() {
    auto& out = simobject_->row_out;
    // A cluster reduction's packet takes the egress when no copy beat is half
    // sent, ahead of a waiting beat.
    this->tick_red_arb();
    const bool red_taken = row_flits_.empty() && this->egress_red();
    if (!row_flits_.empty()) {
      if (out.full())
        return;
      DT(3, simobject_->name() << " row flit: " << row_flits_.front());
      out.send(row_flits_.front(), 0);
      row_flits_.pop_front();
      if (!row_flits_.empty())
        return;
      stream_.pop();
    }
    if (stream_.empty() || stream_.front().ready > cycle_)
      return;
    const StreamBeat& head = stream_.front();
    // At a larger socket a bulk copy's local accesses leaving the stream pass
    // the core switch's register, and its remote beats wait only for the
    // accesses ahead of them to leave the engine's output stage (cs_q_).
    const bool switched = (kCoreSwitchRegs != 0) && head.writes.at(0).dxa.bulk;
    if (!head.remote) {
      if (!this->core_inputs_ready(head.core))
        return;
      for (uint32_t k = 0; k < head.count; ++k) {
        simobject_->lmem_req_out.at(head.ports.at(k)).send(head.writes.at(k), switched ? 1 : 0);
      }
      stream_.pop();
      return;
    }
    if (red_taken)
      return;
    if (switched ? !this->core_switch_empty() : !this->core_inputs_empty())
      return;
    this->load_row_beat(head);
    // A beat with neither data nor a completion leaves at once.
    if (row_flits_.empty()) {
      stream_.pop();
    }
  }

  void load_row_beat(const StreamBeat& head) {
    std::vector<const MemReq*> writes;
    for (uint32_t k = 0; k < head.count; ++k) {
      writes.push_back(&head.writes.at(k));
    }
    std::sort(writes.begin(), writes.end(),
              [](const MemReq* a, const MemReq* b) { return a->addr < b->addr; });
    constexpr uint64_t lmem_mask = (uint64_t(1) << VX_CFG_LMEM_LOG_SIZE) - 1;
    constexpr uint32_t word = VX_CFG_XLEN / 8;
    for (const MemReq* wr : writes) {
      for (uint32_t off = 0; off < VX_CFG_MEM_BLOCK_SIZE; off += word) {
        const uint64_t be = (wr->byteen >> off) & ((uint64_t(1) << word) - 1);
        if (be == 0) {
          continue;
        }
        auto blk = make_mem_block();
        std::memset(blk->data(), 0, blk->size());
        std::memcpy(blk->data() + off, wr->data->data() + off, word);
        DlmemReqPkt f;
        f.dst       = wr->dxa.dst_core;
        f.src       = wr->dxa.src_core;
        f.op        = MemOp::ST;
        f.lmem_addr = (wr->addr + off) & lmem_mask;
        f.data      = blk;
        f.byteen    = be << off;
        f.hart_id   = 0;
        f.tag       = wr->dxa.slot;
        f.uuid      = wr->uuid;
        f.dxa       = true;
        row_flits_.push_back(f);
      }
    }
    const MemReq& last = *writes.back();
    if (last.dxa.last) {
      const uint64_t addr = uint64_t(last.dxa.mbar_off) << 3;
      const uint32_t off  = uint32_t(addr & (VX_CFG_MEM_BLOCK_SIZE - 1));
      auto blk = make_mem_block();
      std::memset(blk->data(), 0, blk->size());
      const uint64_t bytes = last.dxa.tx_bytes;
      std::memcpy(blk->data() + off, &bytes, sizeof(bytes));
      DlmemReqPkt f;
      f.dst       = last.dxa.dst_core;
      f.src       = last.dxa.src_core;
      f.op        = MemOp::MBAR_COMPLETE_TX;
      f.lmem_addr = addr & lmem_mask;
      f.data      = blk;
      f.byteen    = uint64_t(0xff) << off;
      f.hart_id   = 0;
      f.tag       = last.dxa.slot;
      f.uuid      = last.uuid;
      f.dxa       = true;
      f.flags.amo_unsigned = 1;
      row_flits_.push_back(f);
    }
  }

  // A cluster multicast replays each LMEM row of a line to every receiver,
  // in increasing rank, before the next row, as the RTL replays each drained
  // word: a beat is one row of one receiver. A local receiver's beat goes to
  // this core, a remote one's to the row egress through the stream.
  void emit_cluster_beat(Worker& w, uint32_t cid, bool to_stream) {
    constexpr uint64_t kRowMask = ~uint64_t(DxaCore::LMEM_ROW_SIZE - 1);
    const uint32_t slot = w.drain_slot;
    auto& s = w.inflight[slot];
    const LineWork& lw = s.work;
    const uint32_t num_elems = lw.km_num_elems ? lw.km_num_elems : 1;
    const uint32_t wlen      = lw.valid_length;
    const uint32_t rcv_idx   = w.mc_cta_idx;
    const auto& rcv          = w.rcvs.at(rcv_idx);
    const bool     tiled     = desc_dest_tiled(w.desc.meta);
    const uint32_t t_eb      = desc_elem_bytes(w.desc.meta);
    const uint32_t t_ratio   = tiled ? std::max<uint32_t>(1u, 4u / t_eb) : 1u;
    const uint32_t t_tcN     = tiled ? desc_geo_tcn(w.desc) : 1u;
    const uint32_t t_bkK     = tiled ? desc_geo_bkk(w.desc) : 1u;
    const uint32_t t_nsteps  = tiled ? std::max<uint32_t>(1u, uint32_t(w.desc.tile_sizes[0]) / t_tcN) : 1u;
    // The element's dense destination, before the swizzle and the receiver's
    // offset. A swizzle moves 16-byte chunks inside their 128-byte row, so a
    // beat's dense word is one destination word.
    auto dense_of = [&](uint32_t idx) -> uint64_t {
      if (tiled) {
        uint32_t de = tiled_dest_elem(DestLayout(lw.dest_layout), lw.tile_k_row,
                                      lw.tile_n_base + idx, t_ratio, t_tcN, t_nsteps, t_bkK);
        return w.req.smem_addr + uint64_t(de) * t_eb;
      }
      return lw.smem_word_addr + lw.smem_byte_offset + uint64_t(idx) * lw.km_lane_stride;
    };
    const bool     swz      = (w.swz_mask != 0);
    const uint64_t row_mask = swz ? ~uint64_t(kSwzWord - 1) : kRowMask;
    const uint32_t src_step = lw.src_step ? lw.src_step : wlen;

    uint32_t e = w.km_elem_idx;
    uint32_t b = w.km_byte_off;
    const uint32_t e_first = e;
    const uint64_t row = (dense_of(e) + b + rcv.off) & row_mask;
    std::array<MemReq, DxaCore::LMEM_PORTS_PER_CORE> beat;
    std::array<uint32_t, DxaCore::LMEM_PORTS_PER_CORE> beat_port;
    uint32_t n = 0;
    while (e < num_elems) {
      // A gather transfer writes one element a beat.
      if (lw.src_step != 0 && b == 0 && e != e_first)
        break;
      const uint64_t dense = dense_of(e) + b;
      if (((dense + rcv.off) & row_mask) != row)
        break;
      const uint64_t dest  = this->swizzle(w, dense) + rcv.off;
      const uint64_t dword = dest & ~uint64_t(kLmemWordSize - 1);
      uint32_t slot_k = 0;
      while (slot_k < n && beat.at(slot_k).addr != dword) {
        ++slot_k;
      }
      if (slot_k == n) {
        if (n == DxaCore::LMEM_PORTS_PER_CORE)
          break;
        MemReq req;
        req.addr      = dword;
        req.op        = MemOp::ST;
        req.tag       = w.req.core->id();
        req.hart_id   = w.req.core->id();
        req.uuid      = w.req.uuid;
        req.flags.dma = 1;
        req.byteen    = 0;
        req.data      = make_mem_block();
        beat.at(n) = req;
        beat_port.at(n) = cid * DxaCore::LMEM_PORTS_PER_CORE
                        + ((DxaCore::LMEM_PORTS_PER_CORE > 1)
                           ? uint32_t((dword / kLmemWordSize) & (DxaCore::LMEM_PORTS_PER_CORE - 1)) : 0u);
        ++n;
      }
      auto& req = beat.at(slot_k);
      // Without a swizzle the run ends at the word or the block; with one, at
      // the 16-byte chunk.
      const uint64_t lim = swz ? ((dense | 15u) + 1 - dense + dest)
                               : std::min((dest & kRowMask) + DxaCore::LMEM_ROW_SIZE, dword + kLmemWordSize);
      const uint32_t len = uint32_t(std::min<uint64_t>(wlen - b, lim - dest));
      const uint32_t doff = uint32_t(dest - dword);
      for (uint32_t k = 0; k < len; ++k) {
        req.byteen |= uint64_t(1) << (doff + k);
        (*req.data)[doff + k] = s.rsp_data
            ? (*s.rsp_data)[lw.cl_byte_offset + e * src_step + b + k]
            : uint8_t((lw.cfill >> (((doff + k) & 3) * 8)) & 0xff);
      }
      b += len;
      if (b == wlen) {
        ++e;
        b = 0;
      }
    }

    const bool copy_end = lw.last && (e == num_elems);
    for (uint32_t k = 0; k < n; ++k) {
      auto& req = beat.at(k);
      req.dxa.xfer     = w.xfer_base + rcv_idx;
      req.dxa.mode     = w.req.mode;
      req.dxa.mbar_off = rcv.mbar_off;
      req.dxa.tx_bytes = w.tx_bytes;
      req.dxa.slot     = rcv.slot;
      req.dxa.gen      = rcv.gen;
      req.dxa.remote   = rcv.remote;
      req.dxa.src_core = w.req.src_core;
      req.dxa.dst_core = rcv.dst_core;
      req.dxa.last     = copy_end;
      req.dxa.bulk     = w.req.bulk;
      // A bulk beat reaches the bank queue of the first word it writes, one
      // that it holds, as the RTL's DMA port meets only the banks it writes.
      if (w.req.bulk && !rcv.remote) {
        req.addr += uint64_t(__builtin_ctzll(req.byteen)) & ~uint64_t(VX_CFG_XLEN / 8 - 1);
      }
      if (copy_end && k + 1 == n) {
        req.flags.dxa_notify_done   = 1;
        req.flags.dxa_notify_bar_id = w.req.bar_id + rcv.bar_idx;
        req.dxa.writes = w.rcv_writes.at(rcv_idx) + 1;
      }
      ++w.rcv_writes.at(rcv_idx);
      ++w.writes_emitted;
    }
    ++perf_stats_.lmem_writes;
    DT(3, simobject_->name() << "[" << w.worker_id << "] " << (w.req.bulk ? "bulk wr" : "cluster wr") << ": addr=0x"
       << std::hex << beat.at(0).addr << ", byteen=0x" << beat.at(0).byteen << std::dec << ", writes=" << n
       << ", last=" << copy_end << " (#" << w.req.uuid << ")");

    if (to_stream) {
      StreamBeat sb;
      sb.remote = rcv.remote;
      sb.core   = cid;
      sb.count  = n;
      sb.ready  = cycle_ + 1;
      const uint32_t id = sb.remote ? 0 : beat_seq_.at(cid)++;
      for (uint32_t k = 0; k < n; ++k) {
        beat.at(k).dxa.beat = id;
        sb.writes.at(k) = beat.at(k);
        sb.ports.at(k)  = beat_port.at(k);
      }
      stream_.push(sb);
    } else if (w.req.bulk) {
      StreamBeat sb;
      sb.core  = cid;
      sb.count = n;
      sb.ready = cycle_ + 1;
      const uint32_t id = beat_seq_.at(cid)++;
      for (uint32_t k = 0; k < n; ++k) {
        beat.at(k).dxa.beat = id;
        sb.writes.at(k) = beat.at(k);
        sb.ports.at(k)  = beat_port.at(k);
      }
      this->bulk_send(sb);
    } else {
      const uint32_t id = beat_seq_.at(cid)++;
      for (uint32_t k = 0; k < n; ++k) {
        beat.at(k).dxa.beat = id;
        simobject_->lmem_req_out.at(beat_port.at(k)).send(beat.at(k));
      }
    }

    // The same row to the next receiver; after the last one, the next row.
    if (rcv_idx + 1 < w.rcvs.size()) {
      ++w.mc_cta_idx;
      return;
    }
    w.mc_cta_idx = 0;
    w.km_elem_idx = e;
    w.km_byte_off = b;
    if (e < num_elems)
      return;
    w.km_elem_idx = 0;
    w.km_byte_off = 0;
    s.allocated = false;
    s.rsp_arrived = false;
    s.rsp_data.reset();
    auto it = std::find(w.issued_order.begin(), w.issued_order.end(), slot);
    if (it != w.issued_order.end()) w.issued_order.erase(it);
    w.drain_slot = UINT32_MAX;
  }

  // The receivers in replay order: a cluster multicast's masked ranks, in
  // increasing rank; else the issuer's own copy or its barrier multicast
  // replays, which share the request's completion fields.
  void load_receivers(Worker& w, const DxaReq& req) {
    w.rcvs.clear();
    w.km_byte_off = 0;
    if (req.cluster) {
      w.is_multicast = (req.num_rcv > 1);
      w.cta_indices.clear();
      for (uint32_t i = 0; i < req.num_rcv; ++i) {
        const auto& r = req.rcvs.at(i);
        Worker::Rcv e;
        e.off      = r.smem_addr - req.smem_addr;
        e.bar_idx  = i;
        e.mbar_off = r.mbar_off;
        e.slot     = r.slot;
        e.gen      = r.gen;
        e.dst_core = r.core;
        e.remote   = r.remote;
        w.rcvs.push_back(e);
        if (w.is_multicast) {
          w.cta_indices.push_back(i);
        }
      }
      return;
    }
    const uint32_t n = w.is_multicast ? uint32_t(w.cta_indices.size()) : 1u;
    for (uint32_t i = 0; i < n; ++i) {
      Worker::Rcv e;
      e.mbar_off = req.mbar_off;
      e.slot     = req.slot;
      e.gen      = req.gen;
      e.dst_core = req.dst_core;
      e.remote   = req.remote;
      w.rcvs.push_back(e);
    }
  }

  // The bytes an mbarrier copy completes with: the full box, rank-limited,
  // out-of-bounds fill included.
  uint32_t box_bytes(const Worker& w) {
    const uint32_t meta = w.desc.meta;
    const uint32_t rank = (meta >> DXA_DESC_META_DIM_LSB) & ((1u << DXA_DESC_META_DIM_BITS) - 1u);
    if (rank < 1 || rank > 5) {
      this->dxa_abort(w, "invalid DXA rank: " + std::to_string(rank));
    }
    uint64_t bytes = desc_elem_bytes(meta);
    if (w.ref.im2col) {
      bytes *= uint64_t(w.ref.channels) * w.ref.pixels;
    } else {
      for (uint32_t d = 0; d < rank; ++d) {
        bytes *= dxa_ref::n_elems(w.ref, d);
      }
    }
    if (bytes == 0) {
      this->dxa_abort(w, "invalid DXA rank: empty box");
    }
    if (bytes > (uint64_t(1) << VX_CFG_LMEM_LOG_SIZE)) {
      this->dxa_abort(w, "DXA box exceeds local memory: " + std::to_string(bytes) + " bytes");
    }
    return uint32_t(bytes);
  }

  // ── The PTX traversal ───────────────────────────────────────────────
#ifdef VX_CFG_TCU_H100_ENABLE
  // A word wider than 128 bytes holds whole swizzle rows, so a swizzled beat
  // is a whole word, as an unswizzled one.
  static constexpr uint32_t kSwzWord = DxaCore::LMEM_ROW_SIZE;
#else
#line 1351
  static constexpr uint32_t kSwzWord = std::min<uint32_t>(DxaCore::LMEM_ROW_SIZE, 128u);
#endif
#line 1352

  // The descriptor's words, as the golden model decodes them.
  static dxa_ref::Desc ref_desc(const Descriptor& d) {
    uint32_t wd[dxa_ref::DESC_WORDS] = {};
    wd[VX_DCR_DXA_DESC_BASE_LO_OFF]    = uint32_t(d.base_addr);
    wd[VX_DCR_DXA_DESC_BASE_HI_OFF]    = uint32_t(d.base_addr >> 32);
    wd[VX_DCR_DXA_DESC_SIZE0_OFF]      = d.sizes[0];
    wd[VX_DCR_DXA_DESC_SIZE1_OFF]      = d.sizes[1];
    wd[VX_DCR_DXA_DESC_SIZE2_OFF]      = d.sizes[2];
    wd[VX_DCR_DXA_DESC_SIZE3_OFF]      = d.sizes[3];
    wd[VX_DCR_DXA_DESC_SIZE4_OFF]      = d.sizes[4];
    wd[VX_DCR_DXA_DESC_STRIDE0_OFF]    = d.strides[0];
    wd[VX_DCR_DXA_DESC_STRIDE1_OFF]    = d.strides[1];
    wd[VX_DCR_DXA_DESC_STRIDE2_OFF]    = d.strides[2];
    wd[VX_DCR_DXA_DESC_STRIDE3_OFF]    = d.strides[3];
    wd[VX_DCR_DXA_DESC_META_OFF]       = d.meta;
    wd[VX_DCR_DXA_DESC_ESTRIDE0_OFF]   = d.element_strides[0];
    wd[VX_DCR_DXA_DESC_ESTRIDE1_OFF]   = d.element_strides[1];
    wd[VX_DCR_DXA_DESC_ESTRIDE2_OFF]   = d.element_strides[2];
    wd[VX_DCR_DXA_DESC_ESTRIDE3_OFF]   = d.element_strides[3];
    wd[VX_DCR_DXA_DESC_ESTRIDE4_OFF]   = d.element_strides[4];
    wd[VX_DCR_DXA_DESC_TILESIZE01_OFF] = uint32_t(d.tile_sizes[0]) | (uint32_t(d.tile_sizes[1]) << 16);
    wd[VX_DCR_DXA_DESC_TILESIZE23_OFF] = uint32_t(d.tile_sizes[2]) | (uint32_t(d.tile_sizes[3]) << 16);
    wd[VX_DCR_DXA_DESC_TILESIZE4_OFF]  = d.tile_sizes[4];
    wd[VX_DCR_DXA_DESC_CFILL_OFF]      = d.cfill;
    return dxa_ref::decode(wd);
  }

  // The bytes from the destination to the end of what the box writes, the
  // swizzle's moves included.
  uint64_t box_extent(const Worker& w) const {
    const auto& d = w.ref;
    uint64_t dense = d.elem;
    if (d.im2col) {
      dense *= uint64_t(d.channels) * d.pixels;
    } else {
      for (uint32_t i = 0; i < d.rank && i < dxa_ref::MAX_RANK; ++i) {
        dense *= dxa_ref::n_elems(d, i);
      }
    }
    if (d.swizzle == dxa_ref::SWZ_NONE || dense == 0)
      return dense;
    const uint32_t rel = uint32_t(w.req.blk_off);
    uint32_t lo = 0, hi = 0;
    dxa_ref::footprint(w.ref, rel, &lo, &hi);
    return hi - rel;
  }

  // The swizzle of a destination byte, on its block-relative offset.
  uint64_t swizzle(const Worker& w, uint64_t a) const {
    if (w.swz_mask == 0)
      return a;
    return w.blk_base + dxa_ref::swizzle(uint32_t(a - w.blk_base), w.ref.swizzle);
  }

  // The synthetic descriptor of a global bulk load: rank 1, 8-byte elements,
  // the range as its tile. A prefetch is a hint and reads at most a local
  // memory's worth of it, so its tile always fits.
  static Descriptor bulk_desc(const DxaReq& req) {
    const uint64_t bytes = req.prefetch
        ? std::min<uint64_t>(req.size, uint64_t(1) << VX_CFG_LMEM_LOG_SIZE) : req.size;
    Descriptor d;
    d.base_addr = req.gsrc;
    d.meta = (1u << DXA_DESC_META_DIM_LSB) | (3u << DXA_DESC_META_ELEMSZ_LSB);
    d.sizes[0] = uint32_t(bytes / 8);
    d.tile_sizes[0] = uint16_t(bytes / 8);
    d.element_strides[0] = 1;
    return d;
  }

  // VX_dxa_setup's latency for the PTX forms, from the request leaving the
  // unit to the first line: 2 cycles of descriptor fetch, the rank's setup
  // (4 up to rank 2, then 2 a rank), 2 of promotion, 1 to the first line.
  static uint64_t ptx_first_line(uint32_t rank) {
    const uint32_t setup = (rank <= 2) ? 4u : 2u * rank;
    return 2 + setup + 2 + 1;
  }

  // The descriptor's checks and the transfer's traversal state.
  void setup_ptx(Worker& w) {
    const DxaReq& req = w.req;
    w.ref = ref_desc(w.desc);
    const auto& d = w.ref;
    if (const char* msg = dxa_ref::check_tensor(d, uint32_t(req.blk_off), req.prefetch)) {
      this->dxa_abort(w, msg);
    }
    w.ptx      = (d.swizzle != dxa_ref::SWZ_NONE) || (dxa_ref::es(d, 0) > 1) || d.im2col || req.bload;
    w.prefetch = req.prefetch;
    w.swz_mask = dxa_ref::swizzle_mask(d.swizzle);
    w.blk_base = req.smem_addr - req.blk_off;
    if (req.tload || req.bload) {
      w.first_line = req.sent + ptx_first_line(std::max(d.rank, 1u));
    }
    // A barrier multicast replays into consecutive slots of this core.
    if (w.ptx && !req.cluster && w.is_multicast) {
      for (uint32_t i = 0; i < w.rcvs.size(); ++i) {
        w.rcvs.at(i).off     = uint64_t(w.cta_indices.at(i)) * w.smem_stride;
        w.rcvs.at(i).bar_idx = w.cta_indices.at(i);
      }
    }
  }

  // The tokens of a tensor copy in an mbarrier build: signed coordinates,
  // traversal strides, the bounds of every element, im2col; a prefetch keeps
  // only its in-bounds lines. An in-bounds box at the default fields gets
  // exactly enumerate_work_list's tokens.
  void enumerate_ptx(Worker& w) {
    const auto& desc = w.desc;
    const uint32_t rank = desc_rank(desc.meta);
    const uint32_t elem = desc_elem_bytes(desc.meta);
    const bool   im2col = w.ref.im2col;
    DestLayout layout   = desc_layout(desc.meta);
    bool dest_kmajor    = (layout == DestLayout::KMajor);
    bool dest_tiled     = desc_dest_tiled(desc.meta);
    if ((dest_kmajor || dest_tiled) && rank > 2) {
      dest_kmajor = false;
      dest_tiled  = false;
      layout      = DestLayout::RowMajor;
    }
    const bool scatter = dest_kmajor || dest_tiled;

    std::array<uint32_t, 5> n  = {1, 1, 1, 1, 1};
    std::array<uint32_t, 5> es = {1, 1, 1, 1, 1};
    for (uint32_t d = 0; d < rank; ++d) {
      const uint32_t tile = std::max<uint32_t>(1u, desc.tile_sizes[d]);
      es[d] = dxa_ref::es(w.ref, d);
      n[d]  = (tile + es[d] - 1) / es[d];
    }
    if (im2col) {
      n[0]  = std::max<uint32_t>(1u, w.ref.channels);
      es[0] = 1;
    }
    const uint32_t n0   = n[0];
    const uint32_t es0  = es[0];
    const int64_t  c0   = int32_t(w.req.coords[0]);
    const int64_t  size0 = desc.sizes[0];
    const uint32_t per_lane_stride_bytes = n[1] * elem;
    const uint32_t geo_ratio  = dest_tiled ? std::max<uint32_t>(1u, 4u / elem) : 1u;
    const uint32_t geo_tcN    = dest_tiled ? desc_geo_tcn(desc) : 1u;
    const uint32_t geo_bkK    = dest_tiled ? desc_geo_bkk(desc) : 1u;
    const uint32_t geo_nsteps = dest_tiled ? std::max<uint32_t>(1u, n0 / geo_tcN) : 1u;
    const uint32_t cfill = desc.cfill;
    uint64_t global_prev_cl = ~uint64_t(0);

    // K-major, Flat and BlockMajor are Vortex layouts with the walk of the
    // tensor issue: no dim-0 bound, unsigned outer coordinates.
    const bool old_walk = (desc_layout(desc.meta) != DestLayout::RowMajor);
    // Every row has the same dim-0 segments: [0, lo0) and [hi0, n0) are
    // filled, [lo0, hi0) is read unless the row itself is out of bounds.
    uint32_t lo0 = 0, hi0 = n0;
    if (!old_walk) {
      if (c0 < 0) {
        lo0 = uint32_t(std::min<int64_t>(n0, (es0 - 1 - c0) / es0));
      }
      const int64_t room = size0 - c0;
      hi0 = (room <= 0) ? lo0 : uint32_t(std::max<int64_t>(lo0, std::min<int64_t>(n0, (room + es0 - 1) / es0)));
    }
    const bool gather = (es0 > 1);

    // A row's tokens, as VX_dxa_addr_gen walks it. The filled segments split
    // at lines of their local-memory address; the read one at lines of its
    // global address, which an out-of-bounds row still has, though it reads
    // nothing. A gather token holds one line's elements, and every token of
    // a gather transfer drains one element a beat.
    // A prefetch reads only the in-bounds part of in-bounds rows; each
    // out-of-bounds row, or the whole box with no in-bounds dim-0 element,
    // still takes one generator cycle (an entry with no read).
    auto skip_entry = [&]() {
      LineWork lw{};
      lw.oob = true;
      w.work_list.push_back(lw);
    };
    const bool pf_empty = w.prefetch && (lo0 == hi0);
    auto emit_row = [&](int64_t gbase, bool row_in, uint64_t row_smem_base, uint32_t k_row) {
      if (w.prefetch && (pf_empty || !row_in)) {
        if (!pf_empty)
          skip_entry();
        return;
      }
      const uint32_t seg_lo[3] = {0, lo0, hi0};
      const uint32_t seg_hi[3] = {lo0, hi0, n0};
      for (uint32_t sg = 0; sg < 3; ++sg) {
        const bool ib   = (sg == 1);
        const bool fill = !ib || !row_in;
        if (fill && w.prefetch)
          continue;
        const uint32_t seg_end = seg_hi[sg];
        for (uint32_t e0 = seg_lo[sg]; e0 < seg_end; ) {
          const uint64_t gaddr_e = uint64_t(gbase + int64_t(e0) * es0 * elem);
          uint64_t saddr_e;
          if (dest_tiled) {
            uint32_t de = tiled_dest_elem(layout, k_row, e0, geo_ratio, geo_tcN, geo_nsteps, geo_bkK);
            saddr_e = w.req.smem_addr + uint64_t(de) * elem;
          } else if (dest_kmajor) {
            saddr_e = row_smem_base + uint64_t(e0) * uint64_t(per_lane_stride_bytes);
          } else {
            saddr_e = row_smem_base + uint64_t(e0) * elem;
          }
          const uint64_t cl_addr = gaddr_e & kGmemLineMask;
          const uint64_t sword   = saddr_e & ~uint64_t(kLmemWordSize - 1);
          const uint32_t cl_off  = uint32_t(gaddr_e - cl_addr);
          const uint32_t s_off   = uint32_t(saddr_e - sword);
          const uint32_t cl_room = kGmemLineSize - cl_off;
          const uint32_t sw_room = kLmemWordSize - s_off;
          const uint32_t seg_room = (seg_end - e0) * elem;
          LineWork lw{};
          uint32_t num_elems;
          if (!ib) {
            const uint32_t ln_room = kGmemLineSize - uint32_t(saddr_e & (kGmemLineSize - 1));
            uint32_t fspan = std::min({sw_room, ln_room, seg_room});
            fspan -= fspan % elem;
            if (fspan == 0)
              break;
            num_elems = fspan / elem;
            lw.valid_length   = gather ? elem : fspan;
            lw.km_num_elems   = gather ? num_elems : 1;
            lw.km_lane_stride = gather ? elem : 0;
            lw.src_step       = gather ? elem : 0;
          } else if (gather) {
            if (cl_room < elem)
              break;
            num_elems = std::min(seg_end - e0, (cl_room - elem) / (es0 * elem) + 1);
            lw.valid_length   = elem;
            lw.km_num_elems   = num_elems;
            lw.km_lane_stride = elem;
            lw.src_step       = fill ? elem : es0 * elem;
          } else {
            uint32_t gspan = scatter ? std::min({cl_room, seg_room, uint32_t(VX_CFG_MEM_BLOCK_SIZE)})
                           : w.ptx   ? std::min(cl_room, seg_room)
                                     : std::min({cl_room, sw_room, seg_room, uint32_t(VX_CFG_MEM_BLOCK_SIZE)});
            gspan -= gspan % elem;
            if (gspan == 0)
              break;
            num_elems = gspan / elem;
            lw.valid_length   = scatter ? elem : gspan;
            lw.km_num_elems   = scatter ? num_elems : 1;
            lw.km_lane_stride = dest_kmajor ? per_lane_stride_bytes : 0;
          }
          lw.gmem_cl_addr     = cl_addr;
          lw.smem_word_addr   = sword;
          lw.cl_byte_offset   = cl_off;
          lw.smem_byte_offset = s_off;
          lw.cfill            = cfill;
          lw.oob              = fill;
          lw.last             = false;
          lw.dest_layout      = uint8_t(layout);
          lw.tile_k_row       = k_row;
          lw.tile_n_base      = e0;
          if (!fill && cl_addr == global_prev_cl) {
            ++perf_stats_.gmem_dedup;
          } else if (!fill) {
            global_prev_cl = cl_addr;
          }
          w.work_list.push_back(lw);
          e0 += num_elems;
        }
      }
    };

    if (pf_empty) {
      skip_entry();
      return;
    }
    if (im2col) {
      this->enumerate_im2col(w, emit_row);
      return;
    }
    uint32_t total_rows = 1;
    for (uint32_t d = 1; d < rank; ++d) total_rows *= n[d];
    for (uint32_t row = 0; row < total_rows; ++row) {
      uint32_t outer[4] = {};
      uint32_t rem = row;
      bool row_in = true;
      int64_t gbase = int64_t(desc.base_addr) + c0 * elem;
      for (uint32_t d = 1; d < rank; ++d) {
        outer[d - 1] = rem % n[d];
        rem /= n[d];
        const int64_t x = int64_t(int32_t(w.req.coords[d])) + int64_t(outer[d - 1]) * es[d];
        row_in = row_in && (old_walk ? (uint64_t(w.req.coords[d]) + outer[d - 1] < desc.sizes[d])
                                     : (x >= 0 && x < int64_t(desc.sizes[d])));
        gbase += x * int64_t(desc.strides[d - 1]);
      }
      const uint64_t row_smem_base = dest_kmajor
          ? (w.req.smem_addr + uint64_t(outer[0]) * elem)
          : (w.req.smem_addr + uint64_t(row * n0) * elem);
      emit_row(gbase, row_in, row_smem_base, outer[0]);
    }
  }

  // The im2col walk (dxa_ref::im2col_pixels): pixel k's channels are a row.
  template <typename EmitRow>
  void enumerate_im2col(Worker& w, EmitRow& emit_row) {
    const auto& d = w.ref;
    dxa_ref::Req q;
    for (uint32_t i = 0; i < dxa_ref::MAX_RANK; ++i) {
      q.coord[i] = int32_t(w.req.coords[i]);
    }
    for (uint32_t i = 0; i < 3; ++i) {
      q.offs[i] = w.req.offs[i];
    }
    const uint64_t row_bytes = uint64_t(std::max<uint32_t>(1u, d.channels)) * d.elem;
    dxa_ref::im2col_pixels(d, q, [&](uint32_t k, const int64_t* x, bool pix_in) {
      int64_t gbase = int64_t(d.base) + x[0] * int64_t(d.elem);
      for (uint32_t i = 1; i < d.rank; ++i) {
        gbase += x[i] * int64_t(d.stride[i - 1]);
      }
      emit_row(gbase, pix_in, w.req.smem_addr + uint64_t(k) * row_bytes, 0u);
    });
  }

  // A prefetch's smem_wr: one arrived token a cycle leaves without a write,
  // the last one after the others.
  void drain_prefetch(Worker& w) {
    if (w.issued_order.empty()) {
      if (w.ag_idx == w.work_list.size())
        finish_worker(w);
      return;
    }
    for (auto it = w.issued_order.begin(); it != w.issued_order.end(); ++it) {
      auto& sl = w.inflight.at(*it);
      if (sl.rsp_arrived && (!sl.work.last || w.issued_order.size() == 1)) {
        sl.allocated   = false;
        sl.rsp_arrived = false;
        sl.rsp_data.reset();
        w.issued_order.erase(it);
        return;
      }
    }
  }

  // A bulk copy: the lines covering [src_addr, src_addr + size) of the
  // issuing core's local memory, each landing at the same distance from the
  // destination as from the source. Its writes are those of a one-receiver
  // cluster copy, completing with complete_tx(size).
  void start_bulk(Worker& w) {
    const DxaReq& req = w.req;
    if (xfer_seq_ == 0 || xfer_seq_ + 1 < xfer_seq_) {
      xfer_seq_ = 1;
    }
    w.xfer_base = xfer_seq_++;
    w.rcv_writes.assign(1, 0);
    w.desc = Descriptor{};
    w.smem_stride = 0;
    w.tx_bytes = req.size;
    if (w.tx_bytes > req.win_room) {
      this->dxa_abort(w, "DXA box exceeds the destination block: " + std::to_string(w.tx_bytes)
                      + " bytes, room " + std::to_string(req.win_room));
    }
    w.rd_lines.clear();
    w.pend_slot  = UINT32_MAX;
    w.defer_slot = UINT32_MAX;
    const uint64_t end = req.src_addr + req.size;
    for (uint64_t la = req.src_addr & kGmemLineMask; la < end; la += kGmemLineSize) {
      const uint64_t b0  = std::max(la, req.src_addr);
      const uint64_t b1  = std::min(la + kGmemLineSize, end);
      const uint64_t dst = req.smem_addr + (b0 - req.src_addr);
      LineWork lw{};
      lw.gmem_cl_addr     = la;
      lw.smem_word_addr   = dst & ~uint64_t(kLmemWordSize - 1);
      lw.smem_byte_offset = uint32_t(dst - lw.smem_word_addr);
      lw.cl_byte_offset   = uint32_t(b0 - la);
      lw.valid_length     = uint32_t(b1 - b0);
      lw.km_num_elems     = 1;
      w.work_list.push_back(lw);
    }
    w.work_list.back().last = true;
    w.state = WState::RUNNING;
    DT(3, simobject_->name() << "[" << w.worker_id << "] bulk start: core=" << req.core->id()
       << ", wid=" << req.wid << ", src=0x" << std::hex << req.src_addr << ", dst=0x" << req.smem_addr
       << ", size=0x" << req.size << ", mbar_off=0x" << req.mbar_off << std::dec << ", lines="
       << w.work_list.size() << ", remote=" << req.remote << ", dst_core=" << req.dst_core);
  }

  // A line's reads: the local-memory words (DxaCore::LMEM_ROW_SIZE bytes) of
  // the line, or the one word holding it. A word wider than a block is read as
  // its blocks, on their ports, as a row is written.
  static constexpr uint32_t kRowSize    = DxaCore::LMEM_ROW_SIZE;
  static constexpr uint32_t kLineReads  = (kGmemLineSize > kRowSize) ? (kGmemLineSize / kRowSize) : 1u;
  static constexpr uint32_t kWordBlocks = DxaCore::LMEM_PORTS_PER_CORE;
  static_assert(kGmemLineSize <= kLmemWordSize, "a source line fits a local-memory block");

  static uint64_t bulk_word(uint64_t line, uint32_t k) {
    return (kGmemLineSize > kRowSize) ? (line + uint64_t(k) * kRowSize) : (line & ~uint64_t(kRowSize - 1));
  }

  // The RTL's 1D setup ahead of a copy's first line, counted from the cycle
  // the request left the unit: the descriptor fetch hands it to the worker 2
  // cycles later, the rank-1 setup takes 4 and its promotion 2, and the
  // address generator emits the first line the cycle after. Behind a running
  // copy the setup overlaps it, as the RTL stages the next request.
  static constexpr uint64_t kBulkFirstLine = 2 + 4 + 2 + 1;

  // At a socket of several cores, the core switch registers a bulk copy's
  // accesses on their way to a core, and a read's response on its way back
  // (VX_socket dxa_lmem_core_switch); at a socket of one, there is none.
  static constexpr uint32_t kCoreSwitchRegs = (kCoresPerSocket > 1) ? 1u : 0u;
  static constexpr uint32_t kCoreSwitchDepth = 2;

  // Whether a bulk access to core `cid` can leave the worker's bus now.
  bool bulk_bus_ready(uint32_t cid) const {
    return (kCoreSwitchRegs != 0) ? (cs_q_.at(cid).size() < kCoreSwitchDepth) : this->core_inputs_ready(cid);
  }

  void bulk_send(const StreamBeat& sb) {
    if (kCoreSwitchRegs != 0) {
      cs_q_.at(sb.core).push_back(sb);
      return;
    }
    for (uint32_t k = 0; k < sb.count; ++k) {
      simobject_->lmem_req_out.at(sb.ports.at(k)).send(sb.writes.at(k));
    }
  }

  bool core_switch_empty() const {
    for (const auto& q : cs_q_) {
      if (!q.empty())
        return false;
    }
    return true;
  }

  // The core switch's registers: each core's oldest access moves on to the
  // core's DXA input once it has spent its cycle there.
  void tick_core_switch() {
    if (kCoreSwitchRegs == 0)
      return;
    for (uint32_t c = 0; c < kCoresPerSocket; ++c) {
      auto& q = cs_q_.at(c);
      if (q.empty() || q.front().ready > cycle_ || !this->core_inputs_ready(c))
        continue;
      for (uint32_t k = 0; k < q.front().count; ++k) {
        simobject_->lmem_req_out.at(q.front().ports.at(k)).send(q.front().writes.at(k));
      }
      q.pop_front();
    }
  }

  // The address generator: one line a cycle, which takes a slot and an entry
  // of the line reader, as their registers stand at the cycle's start.
  void tick_bulk_agu(Worker& w) {
    if (w.ag_idx >= w.work_list.size())
      return;
    if (SimPlatform::instance().cycles() < w.req.sent + kBulkFirstLine)
      return;
    uint32_t slot = UINT32_MAX;
    for (uint32_t s = 0; s < w.inflight.size(); ++s) {
      if (!w.inflight[s].allocated) {
        slot = s;
        break;
      }
    }
    if (slot == UINT32_MAX) {
      ++perf_stats_.noslot_stalls;
      return;
    }
    if (w.rd_lines.size() >= DxaCore::LMEM_RD_DEPTH)
      return;
    const LineWork& lw = w.work_list[w.ag_idx];
    auto& s = w.inflight[slot];
    s.allocated   = true;
    s.rsp_arrived = false;
    s.rsp_data    = make_mem_block();
    std::memset(s.rsp_data->data(), 0, s.rsp_data->size());
    s.work        = lw;
    w.issued_order.push_back(slot);
    Worker::RdLine e;
    e.addr  = lw.gmem_cl_addr;
    e.slot  = slot;
    e.taken = cycle_;
    w.rd_lines.push_back(e);
    ++w.ag_idx;
    DT(3, simobject_->name() << "[" << w.worker_id << "] bulk line: addr=0x" << std::hex << lw.gmem_cl_addr
       << std::dec << ", slot=" << slot << " (#" << w.req.uuid << ")");
  }

  // A cycle in which a bulk copy runs, as the RTL's lmem_arb sees it: each
  // bulk worker's address generator and smem_wr registers act on the cycle's
  // start; the first worker in round-robin order with a bus operation is
  // granted the bus, even when the bus cannot take it this cycle, and the
  // grant moves past it once it goes; then the bulk workers' smem_wr
  // registers load.
  void tick_lmem_arb() {
    for (auto& w : workers_) {
      if (w.state == WState::RUNNING && w.store) {
        this->store_pre(w);
        continue;
      }
      if (w.state != WState::RUNNING || !w.req.bulk)
        continue;
      if (w.issued_order.empty() && w.ag_idx == w.work_list.size()) {
        this->finish_worker(w);
        continue;
      }
      w.st_fb          = (w.drain_slot != UINT32_MAX);
      w.st_pend        = (w.pend_slot != UINT32_MAX);
      w.st_defer       = (w.defer_slot != UINT32_MAX);
      w.st_outstanding = uint32_t(w.issued_order.size());
      this->tick_bulk_agu(w);
    }
    lmem_taken_ = false;
    const uint32_t n = uint32_t(workers_.size());
    for (uint32_t k = 0; k < n; ++k) {
      auto& w = workers_.at((lmem_rr_ + k) % n);
      if (w.state != WState::RUNNING)
        continue;
      const uint32_t writes = w.writes_emitted;
      lmem_sent_ = false;
      if (w.store) {
        this->store_bus_op(w);
      } else if (w.req.bulk) {
        this->bulk_bus_op(w);
      } else {
        this->tick_worker_smem_wr(w);
      }
      if (lmem_sent_ || w.writes_emitted != writes) {
        lmem_rr_ = (w.worker_id + 1) % n;
      }
    }
    for (auto& w : workers_) {
      if (w.state == WState::RUNNING && w.store) {
        this->store_post(w);
      } else if (w.state == WState::RUNNING && w.req.bulk) {
        this->bulk_capture(w);
      }
    }
  }

  // The worker's bus operation: the next read of the oldest entry with reads
  // left, which holds the writer back while it is not taken
  // (VX_dxa_lmem_rd); else a beat of the fill buffer, loaded in an earlier
  // cycle.
  void bulk_bus_op(Worker& w) {
    const uint32_t cid = w.req.core->id() % kCoresPerSocket;
    Worker::RdLine* e = nullptr;
    for (auto& l : w.rd_lines) {
      if (l.issued < kLineReads) {
        e = &l;
        break;
      }
    }
    if (e != nullptr && e->taken < cycle_) {
      if (lmem_taken_)
        return;
      lmem_taken_ = true;
      this->bulk_read(w, *e, cid);
      return;
    }
    if (w.drain_slot == UINT32_MAX || w.fb_at >= cycle_ || lmem_taken_)
      return;
    lmem_taken_ = true;
    const bool to_stream = w.rcvs.at(0).remote || !stream_.empty();
    if (to_stream ? stream_.full() : !this->bulk_bus_ready(cid))
      return;
    this->emit_cluster_beat(w, cid, to_stream);
    lmem_sent_ = true;
  }

  // One word read; a read behind a beat in the socket's stream queues behind
  // it.
  void bulk_read(Worker& w, Worker::RdLine& e, uint32_t cid) {
    const uint64_t word = bulk_word(e.addr, e.issued);
    const bool to_stream = !stream_.empty();
    if (to_stream ? stream_.full() : !this->bulk_bus_ready(cid))
      return;
    StreamBeat sb;
    sb.core  = cid;
    sb.count = kWordBlocks;
    sb.ready = cycle_ + 1;
    for (uint32_t p = 0; p < kWordBlocks; ++p) {
      MemReq req;
      req.op        = MemOp::LD;
      req.addr      = word + uint64_t(p) * kLmemWordSize;
      req.tag       = w.worker_id * VX_CFG_DXA_MAX_INFLIGHT + e.slot;
      req.hart_id   = w.req.core->id();
      req.uuid      = w.req.uuid;
      req.flags.dma = 1;
      req.dxa.bulk  = true;
      sb.writes.at(p) = req;
      sb.ports.at(p)  = cid * kWordBlocks + p;
    }
    if (to_stream) {
      stream_.push(sb);
    } else {
      this->bulk_send(sb);
    }
    DT(3, simobject_->name() << "[" << w.worker_id << "] bulk rd: addr=0x" << std::hex << word << std::dec
       << ", slot=" << e.slot << ", word=" << e.issued << " (#" << w.req.uuid << ")");
    ++e.issued;
    lmem_sent_ = true;
  }

  // smem_wr's registers at the cycle's end (VX_dxa_smem_wr): the fill buffer,
  // emptied by this cycle's beat or idle, loads pend, else the deferred last
  // line once it is the only one left; and the line reader's oldest filled
  // line is captured into pend, or into defer if it is the copy's last line
  // while others are outstanding, which frees its entry.
  void bulk_capture(Worker& w) {
    const bool emptied  = w.st_fb && w.drain_slot == UINT32_MAX;
    const bool fb_empty = emptied || !w.st_fb;
    if (w.st_pend && fb_empty) {
      w.drain_slot = w.pend_slot;
      w.fb_at      = cycle_;
      w.pend_slot  = UINT32_MAX;
    } else if (w.st_defer && fb_empty && !w.st_pend
               && (w.st_outstanding == 1 || (w.st_outstanding == 2 && emptied))) {
      w.drain_slot = w.defer_slot;
      w.fb_at      = cycle_;
      w.defer_slot = UINT32_MAX;
    }
    if (w.rd_lines.empty())
      return;
    const auto& h = w.rd_lines.front();
    if (!bulk_filled(h) || h.filled_at >= cycle_)
      return;
    const bool to_defer = w.inflight.at(h.slot).work.last && w.st_outstanding > 1;
    if (to_defer ? w.st_defer : w.st_pend)
      return;
    (to_defer ? w.defer_slot : w.pend_slot) = h.slot;
    w.inflight.at(h.slot).rsp_arrived = true;
    DT(3, simobject_->name() << "[" << w.worker_id << "] bulk sw-rx: slot=" << h.slot << ", path="
       << (to_defer ? "defer" : "pend") << " (#" << w.req.uuid << ")");
    w.rd_lines.pop_front();
  }

  static bool bulk_filled(const Worker::RdLine& e) {
    for (uint32_t f : e.filled) {
      if (f != kLineReads)
        return false;
    }
    return true;
  }

  // Each port's responses fill its block of the entries' words in order, one
  // a cycle. A response reaches the worker the cycle after its bank access,
  // which is the cycle the RTL's lands at a socket of one core: SimX's DMA
  // accesses meet their bank a cycle after the RTL's, as its writes do. At a
  // larger socket the core switch holds it a cycle more.
  void drain_lmem_rsp() {
    for (uint32_t p = 0; p < simobject_->lmem_rsp_in.size(); ++p) {
      auto& ch = simobject_->lmem_rsp_in.at(p);
      if (ch.empty())
        continue;
      this->bulk_fill(p, ch.peek(), cycle_ - 1 + kCoreSwitchRegs);
      ch.pop();
    }
  }

  void bulk_fill(uint32_t p, const MemRsp& rsp, uint64_t at) {
    const uint32_t blk = p % kWordBlocks;
    auto& w = workers_.at(uint32_t(rsp.tag) / VX_CFG_DXA_MAX_INFLIGHT);
    if (w.store) {
      this->store_fill(w, blk, rsp, at);
      return;
    }
    Worker::RdLine* e = nullptr;
    for (auto& l : w.rd_lines) {
      if (l.filled.at(blk) < kLineReads) {
        e = &l;
        break;
      }
    }
    __assert(e != nullptr && e->slot == uint32_t(rsp.tag) % VX_CFG_DXA_MAX_INFLIGHT
             && e->filled.at(blk) < e->issued && rsp.data,
             simobject_->name() << " DXA local-memory read response without a read: tag=0x"
             << std::hex << rsp.tag);
    const uint64_t word = bulk_word(e->addr, e->filled.at(blk));
    auto& line = w.inflight.at(e->slot).rsp_data;
    if (kGmemLineSize > kRowSize) {
      std::memcpy(line->data() + (word - e->addr), rsp.data->data() + (word & (kLmemWordSize - 1)), kRowSize);
    } else if (word + uint64_t(blk) * kLmemWordSize == (e->addr & ~uint64_t(kLmemWordSize - 1))) {
      std::memcpy(line->data(), rsp.data->data() + (e->addr & (kLmemWordSize - 1)), kGmemLineSize);
    }
    ++e->filled.at(blk);
    if (bulk_filled(*e)) {
      e->filled_at = at;
      DT(3, simobject_->name() << "[" << w.worker_id << "] bulk filled: slot=" << e->slot << ", cycles ago="
         << (cycle_ - at) << " (#" << rsp.uuid << ")");
    }
  }

  // ── Tensor maps in memory (dxa_tmap.h) ──────────────────────────────
  // The fetch's line reads carry tags no worker produces, steered back from
  // worker 0's response port.
  static constexpr uint64_t kFetchTag = uint64_t(VX_CFG_DXA_MAX_INFLIGHT) * VX_CFG_NUM_DXA_CORES;
  static constexpr uint32_t kMapBytes = TMAP_WORDS * 4;
  // A GMEM request reaches the socket's memory arbiter in the cycle it leaves
  // the engine, as in VX_dxa_core, whose worker arbiter registers it only
  // with more than one worker; the map fetch joins after that arbiter.
  static constexpr uint64_t kGmemReqDelay = (VX_CFG_NUM_DXA_CORES > 1) ? 1 : 0;
#ifdef VX_CFG_EXT_FPAMO_ENABLE
  static constexpr bool kHasFpamo = true;
#else
  static constexpr bool kHasFpamo = false;
#endif
  // The fetch timeline (VX_dxa_core MAP state): the cycles from the head's
  // miss to its first line read, and from the last line's response to the
  // head's lookup that reads the filled entry as a hit.
  static constexpr uint64_t kFetchFirst = 1;
  static constexpr uint64_t kFetchHit   = 1;
  // The cycles from a request leaving its unit to its first lookup at an
  // empty queue's head.
  static constexpr uint64_t kSentToHead = 2;

  static uint64_t map_tag(uint64_t addr) { return addr >> 6; }

  // Words 0-21 are a slot's words, decoded as a slot's DCR writes.
  static_assert(TMAP_W_META == VX_DCR_DXA_DESC_META_OFF && TMAP_W_SMEM_STRIDE == VX_DCR_DXA_DESC_SMEM_STRIDE_OFF,
                "a tensor map's words 0-21 are a slot's");
  static Descriptor map_desc(const uint32_t* w) {
    Descriptor d;
    for (uint32_t k = 0; k <= TMAP_W_SMEM_STRIDE; ++k) {
      set_desc_word(d, k, w[k]);
    }
    return d;
  }

  int map_lookup(uint64_t addr) const {
    for (uint32_t e = 0; e < maps_.size(); ++e) {
      if (maps_[e].valid && maps_[e].tag == map_tag(addr))
        return int(e);
    }
    return -1;
  }

  [[noreturn]] void head_abort(const DxaReq& req, const char* msg) {
    std::cerr << SimPlatform::instance().cycles() << ": *** " << simobject_->name() << " " << msg
              << ": core=" << req.core->id() << " wid=" << req.wid << " map=0x" << std::hex << req.map_addr
              << std::dec << " (#" << req.uuid << ")" << std::endl;
    std::abort();
  }

  // The lines covering the map: two at 64-byte lines.
  void fetch_start(const DxaReq& req) {
    fetch_ = Fetch();
    fetch_.active  = true;
    fetch_.addr    = req.map_addr;
    const uint64_t first = req.map_addr & kGmemLineMask;
    fetch_.lines   = uint32_t((req.map_addr + kMapBytes - first + kGmemLineSize - 1) / kGmemLineSize);
    fetch_.send_at = cycle_ + kFetchFirst;
    fetch_.hart_id = req.core->id();
    fetch_.uuid    = req.uuid;
    DT(3, simobject_->name() << " tmap miss: map=0x" << std::hex << req.map_addr << std::dec << ", lines="
       << fetch_.lines << " (#" << req.uuid << ")");
  }

  // Ahead of the workers' requests: a line read a cycle while any is left.
  void tick_fetch_send() {
    if (!fetch_.active || fetch_.sent == fetch_.lines || cycle_ < fetch_.send_at)
      return;
    fetch_port_ = true;
    MemReq mreq;
    mreq.addr    = (fetch_.addr & kGmemLineMask) + uint64_t(fetch_.sent) * kGmemLineSize;
    mreq.tag     = (kFetchTag + fetch_.sent) << log2ceil(VX_CFG_NUM_DXA_CORES);
    mreq.hart_id = fetch_.hart_id;
    mreq.uuid    = fetch_.uuid;
    if (simobject_->gmem_req_out.at(0).try_send(mreq, 0)) {
      DT(3, simobject_->name() << " tmap rd: addr=0x" << std::hex << mreq.addr << std::dec << " (#"
         << fetch_.uuid << ")");
      ++fetch_.sent;
    }
  }

  // The map's bytes of a line; the last response fills the victim entry,
  // unless a launch killed the fetch.
  void fetch_rsp(const MemRsp& rsp) {
    __assert(fetch_.active && rsp.data, simobject_->name() << " DXA map line response without a fetch");
    const uint32_t i = uint32_t(uint64_t(rsp.tag) - kFetchTag);
    const uint64_t line = (fetch_.addr & kGmemLineMask) + uint64_t(i) * kGmemLineSize;
    for (uint32_t b = 0; b < kGmemLineSize; ++b) {
      const uint64_t a = line + b;
      if (a >= fetch_.addr && a < fetch_.addr + kMapBytes) {
        const uint32_t off = uint32_t(a - fetch_.addr);
        fetch_.w[off / 4] &= ~(0xffu << (8 * (off % 4)));
        fetch_.w[off / 4] |= uint32_t(rsp.data->data()[b]) << (8 * (off % 4));
      }
    }
    if (++fetch_.got < fetch_.lines)
      return;
    fetch_.hit_at = cycle_ + kFetchHit;
    if (fetch_.killed) {
      DT(3, simobject_->name() << " tmap fill killed: map=0x" << std::hex << fetch_.addr << std::dec << " (#"
         << fetch_.uuid << ")");
      return;
    }
    uint32_t e = 0;
    while (e < maps_.size() && maps_[e].valid) {
      ++e;
    }
    if (e == maps_.size()) {
      e = map_rr_;
      map_rr_ = (map_rr_ + 1) % uint32_t(maps_.size());
    }
    auto& m = maps_[e];
    m.valid = true;
    m.tag   = map_tag(fetch_.addr);
    std::copy(fetch_.w.begin(), fetch_.w.begin() + TMAP_FETCH_WORDS, m.w.begin());
    m.desc  = map_desc(m.w.data());
    DT(3, simobject_->name() << " tmap fill: map=0x" << std::hex << fetch_.addr << std::dec << ", entry=" << e
       << " (#" << fetch_.uuid << ")");
  }

  // The head's lookup, once a cycle before dispatch: a slot or bulk request
  // goes on as before; a map request goes on once its map hits; PREFETCH and
  // INVAL end at the head. Returns whether the head may start on a worker.
  bool head_ready() {
    if (!queue_.empty()) {
      auto& h = queue_.front();
      if (h.held) {
        // A map operation ahead kept it from the head: its timeline starts here.
        h.held = false;
        h.sent = std::max(h.sent, cycle_ - std::min(cycle_, kSentToHead));
      }
      if (!h.map && !h.tmop)
        return true;
      if (head_hit_)
        return true;  // presented: dispatch takes the entry it hit
      if (h.head_at == 0) {
        h.head_at = cycle_;
        if (h.tmop && h.tmap_op == DXA_TMAP_INVAL) {
          const int e = this->map_lookup(h.map_addr);
          if (e >= 0) {
            maps_[e].valid = false;
          }
          DT(3, simobject_->name() << " tmap inval: map=0x" << std::hex << h.map_addr << std::dec << ", entry="
             << e << " (#" << h.uuid << ")");
          this->head_pop();
          return false;
        }
        if (this->map_lookup(h.map_addr) >= 0) {
          ++perf_stats_.tmap_hits;
        } else {
          ++perf_stats_.tmap_misses;
          h.fetched = true;
          this->fetch_start(h);
          return false;
        }
      } else if (fetch_.active) {
        if (fetch_.got < fetch_.lines || cycle_ < fetch_.hit_at)
          return false;
        fetch_.active = false;
        // The fill is read as a hit; the miss's cycles delay the request's timeline.
        h.sent += cycle_ - h.head_at;
        h.head_at = cycle_;
      }
      const int e = this->map_lookup(h.map_addr);
      if (e < 0) {
        // A launch kept the fill out: a PREFETCH ends, a transfer misses again.
        if (h.tmop) {
          DT(3, simobject_->name() << " tmap prefetch dropped: map=0x" << std::hex << h.map_addr << std::dec
             << " (#" << h.uuid << ")");
          this->head_pop();
          return false;
        }
        ++perf_stats_.tmap_misses;
        this->fetch_start(h);
        return false;
      }
      if (h.tmop) {
        DT(3, simobject_->name() << " tmap prefetch: map=0x" << std::hex << h.map_addr << std::dec << ", entry="
           << e << " (#" << h.uuid << ")");
        this->head_pop();
        return false;
      }
      if (const char* msg = dxa_ref::check_map_words(maps_[e].w.data())) {
        this->head_abort(h, msg);
      }
      head_entry_ = uint32_t(e);
      head_hit_   = true;
      return true;
    }
    return false;
  }

  // A tensormap op leaves the head in its lookup cycle; the next request is
  // looked up from the next cycle.
  void head_pop() {
    queue_.pop_front();
    if (!queue_.empty()) {
      queue_.front().held = true;
    }
  }

  // A request dispatched after its map's fetch held the one behind it.
  void head_popped(bool fetched) {
    head_hit_ = false;
    if (fetched && !queue_.empty()) {
      queue_.front().held = true;
    }
  }

  // ── The store family (LMEM → global memory, or a cluster block) ──────
  static MemOp red_memop(uint32_t code) {
    switch (code) {
    case dxa_ref::AMO_ADD: return MemOp::AMO_ADD;
    case dxa_ref::AMO_XOR: return MemOp::AMO_XOR;
    case dxa_ref::AMO_OR:  return MemOp::AMO_OR;
    case dxa_ref::AMO_AND: return MemOp::AMO_AND;
    case dxa_ref::AMO_MIN: return MemOp::AMO_MIN;
    case dxa_ref::AMO_MAX: return MemOp::AMO_MAX;
#ifdef VX_CFG_EXT_FPAMO_ENABLE
    case dxa_ref::AMO_INC:      return MemOp::AMO_INC;
    case dxa_ref::AMO_DEC:      return MemOp::AMO_DEC;
    case dxa_ref::AMO_FADD:     return MemOp::AMO_FADD;
    case dxa_ref::AMO_FADD2:    return MemOp::AMO_FADD2;
    case dxa_ref::AMO_FADD2_BF: return MemOp::AMO_FADD2_BF;
    case dxa_ref::AMO_FMIN2:    return MemOp::AMO_FMIN2;
    case dxa_ref::AMO_FMAX2:    return MemOp::AMO_FMAX2;
    case dxa_ref::AMO_FMIN2_BF: return MemOp::AMO_FMIN2_BF;
    case dxa_ref::AMO_FMAX2_BF: return MemOp::AMO_FMAX2_BF;
#endif
    default:
      std::cerr << "*** DXA reduction without an atomic: op=0x" << std::hex << code << std::dec << std::endl;
      std::abort();
    }
  }

  // The walk is a load's of the same box: a tensor store's descriptor and
  // coordinates, or a bulk operation's synthetic rank-1 descriptor over its
  // destination (global, or a cluster block's local memory). Its local-memory
  // side is the source in the issuer's block.
  void start_store(Worker& w) {
    DxaReq& req = w.req;
    w.store = true;
    w.st = Worker::StWork();
    auto& st = w.st;
    st.kind = req.st_kind;
    st.op   = req.red_op;
    st.type = req.red_type;
    w.rd_lines.clear();
    w.pend_slot  = UINT32_MAX;
    w.defer_slot = UINT32_MAX;
    w.xfer_base  = 0;
    w.rcv_writes.clear();
    w.rcvs.clear();
    w.smem_stride = 0;
    const bool tensor = dxa_ref::is_tensor(st.kind);
    if (tensor) {
      w.desc = req.map ? maps_.at(head_entry_).desc : descriptors_.at(req.desc_slot);
      if (req.map && st.kind == dxa_ref::ST_RED_TENSOR && st.type == dxa_ref::RT_FROM_MAP) {
        // The map's element type is the reduction's.
        uint32_t type = 0;
        if (const char* msg = dxa_ref::map_red_type(st.op, maps_.at(head_entry_).w.data(), &type)) {
          this->dxa_abort(w, msg);
        }
        if (dxa_ref::red_needs_fpamo(st.op, type) && !kHasFpamo) {
          this->dxa_abort(w, dxa_ref::MSG_RED_FPAMO);
        }
        st.type = type;
        req.red_type = type;
      }
      // VX_dxa_setup's order: the reduction type's size, then the layout,
      // then the load's descriptor rules, then the coordinates.
      const dxa_ref::Desc d = ref_desc(w.desc);
      if (st.kind == dxa_ref::ST_RED_TENSOR && dxa_ref::type_bytes(st.type) != d.elem) {
        this->dxa_abort(w, dxa_ref::MSG_RED_SIZE);
      }
      if (d.layout != dxa_ref::LAYOUT_ROW) {
        this->dxa_abort(w, dxa_ref::MSG_ST_LAYOUT);
      }
    } else {
      DxaReq b = req;
      st.dst    = req.smem_addr;
      b.gsrc    = (st.kind == dxa_ref::ST_RED_CLUSTER) ? req.smem_addr : req.gdst;
      b.prefetch = false;
      w.desc = bulk_desc(b);
      req.smem_addr = req.src_addr;
      req.blk_off   = 0;
    }
    this->setup_ptx(w);
    w.first_line = req.sent + ptx_first_line(std::max(w.ref.rank, 1u));
    if (tensor) {
      const uint64_t extent = this->box_extent(w);
      if (extent > req.win_room) {
        this->dxa_abort(w, "DXA box exceeds the destination block: " + std::to_string(extent)
                        + " bytes, room " + std::to_string(req.win_room));
      }
      dxa_ref::Req q;
      for (uint32_t i = 0; i < dxa_ref::MAX_RANK; ++i) {
        q.coord[i] = int32_t(req.coords[i]);
      }
      if (const char* msg = dxa_ref::check_tensor_store(w.ref, q, uint32_t(req.blk_off))) {
        this->dxa_abort(w, msg);
      }
    } else {
      w.ptx = true;
    }
    this->enumerate_ptx(w);
    if (w.work_list.empty()) {
      this->dxa_abort(w, "invalid DXA rank: no work for the box");
    }
    w.state = WState::RUNNING;
    DT(3, simobject_->name() << "[" << w.worker_id << "] store start: core=" << req.core->id() << ", wid="
       << req.wid << ", kind=" << st.kind << ", src=0x" << std::hex << req.src_addr << ", gdst=0x" << req.gdst
       << ", dst=0x" << st.dst << std::dec << ", size=" << req.size << ", slot=" << req.desc_slot
       << ", tokens=" << w.work_list.size() << ", seq=" << req.bg_seq << " (#" << req.uuid << ")");
  }

  // A token's bytes, in line order: (line offset, physical source byte).
  void store_bytes(const Worker& w, const LineWork& lw, Worker::StTok& t) const {
    const uint32_t num  = lw.km_num_elems ? lw.km_num_elems : 1;
    const uint32_t wlen = lw.valid_length;
    const uint32_t step = lw.src_step ? lw.src_step : wlen;
    const bool tiled = desc_dest_tiled(w.desc.meta);
    const uint32_t t_eb = desc_elem_bytes(w.desc.meta);
    const uint32_t t_ratio  = tiled ? std::max<uint32_t>(1u, 4u / t_eb) : 1u;
    const uint32_t t_tcN    = tiled ? desc_geo_tcn(w.desc) : 1u;
    const uint32_t t_bkK    = tiled ? desc_geo_bkk(w.desc) : 1u;
    const uint32_t t_nsteps = tiled ? std::max<uint32_t>(1u, uint32_t(w.desc.tile_sizes[0]) / t_tcN) : 1u;
    for (uint32_t e = 0; e < num; ++e) {
      uint64_t dense;
      if (tiled) {
        const uint32_t de = tiled_dest_elem(DestLayout(lw.dest_layout), lw.tile_k_row, lw.tile_n_base + e,
                                            t_ratio, t_tcN, t_nsteps, t_bkK);
        dense = w.req.smem_addr + uint64_t(de) * t_eb;
      } else {
        dense = lw.smem_word_addr + lw.smem_byte_offset + uint64_t(e) * lw.km_lane_stride;
      }
      for (uint32_t b = 0; b < wlen; ++b) {
        t.bytes.push_back({lw.cl_byte_offset + e * step + b, this->swizzle(w, dense + b)});
      }
    }
  }

  // Its reads: one per local-memory word its source spans, ascending, each
  // at the physical word the swizzle moves that word to; one per element for
  // a gather (or a scattering layout).
  std::vector<uint64_t> store_reads(const Worker& w, const LineWork& lw) const {
    constexpr uint64_t kRowMask = ~uint64_t(kRowSize - 1);
    std::vector<uint64_t> rd;
    const uint32_t num = lw.km_num_elems ? lw.km_num_elems : 1;
    const uint64_t s = lw.smem_word_addr + lw.smem_byte_offset;
    if (lw.src_step != 0 || num > 1) {
      Worker::StTok t;
      this->store_bytes(w, lw, t);
      for (uint32_t e = 0; e < num; ++e) {
        rd.push_back(t.bytes.at(e * lw.valid_length).phys & kRowMask);
      }
      return rd;
    }
    for (uint64_t a = s & kRowMask; a < s + lw.valid_length; a += kRowSize) {
      rd.push_back(this->swizzle(w, a) & kRowMask);
    }
    return rd;
  }

  // addr_gen's cycle: an out-of-bounds token is taken without a request; an
  // in-bounds token sends its next read when a credit and the bus allow, and
  // leaves with its last one.
  void store_bus_op(Worker& w) {
    auto& st = w.st;
    if (w.ag_idx >= w.work_list.size() || SimPlatform::instance().cycles() < w.first_line)
      return;
    const LineWork& lw = w.work_list.at(w.ag_idx);
    if (lw.oob) {
      ++w.ag_idx;
      return;
    }
    if (st.rd.empty()) {
      st.rd = this->store_reads(w, lw);
      st.rd_sent = 0;
    }
    if (st.reads.size() >= kStCredits || lmem_taken_)
      return;
    lmem_taken_ = true;
    const uint32_t cid = w.req.core->id() % kCoresPerSocket;
    const bool to_stream = !stream_.empty();
    if (to_stream ? stream_.full() : !this->bulk_bus_ready(cid))
      return;
    const uint64_t word = st.rd.at(st.rd_sent);
    StreamBeat sb;
    sb.core  = cid;
    sb.count = kWordBlocks;
    sb.ready = cycle_ + 1;
    for (uint32_t p = 0; p < kWordBlocks; ++p) {
      MemReq req;
      req.op        = MemOp::LD;
      req.addr      = word + uint64_t(p) * kLmemWordSize;
      req.tag       = w.worker_id * VX_CFG_DXA_MAX_INFLIGHT;
      req.hart_id   = w.req.core->id();
      req.uuid      = w.req.uuid;
      req.flags.dma = 1;
      req.dxa.bulk  = true;
      sb.writes.at(p) = req;
      sb.ports.at(p)  = cid * kWordBlocks + p;
    }
    if (to_stream) {
      stream_.push(sb);
    } else {
      this->bulk_send(sb);
    }
    if (st.rd_sent == 0) {
      // The token joins the assembly with its first read, whose response may
      // land before its last read goes.
      Worker::StTok t;
      t.lw    = lw;
      t.reads = uint32_t(st.rd.size());
      this->store_bytes(w, lw, t);
      st.toks.push_back(std::move(t));
    }
    Worker::StRead r;
    r.word = word;
    st.reads.push_back(r);
    DT(3, simobject_->name() << "[" << w.worker_id << "] store rd: addr=0x" << std::hex << word << std::dec
       << ", token=" << w.ag_idx << ", read=" << st.rd_sent << " (#" << w.req.uuid << ")");
    lmem_sent_ = true;
    if (++st.rd_sent < st.rd.size())
      return;
    st.rd.clear();
    st.rd_sent = 0;
    ++w.ag_idx;
  }

  // A read's blocks land in order per port.
  void store_fill(Worker& w, uint32_t blk, const MemRsp& rsp, uint64_t at) {
    Worker::StRead* r = nullptr;
    for (auto& e : w.st.reads) {
      if (!e.got.at(blk)) {
        r = &e;
        break;
      }
    }
    __assert(r != nullptr && rsp.data, simobject_->name() << " DXA store read response without a read");
    r->got.at(blk) = true;
    if (kRowSize > kLmemWordSize) {
      std::memcpy(r->data.data() + blk * kLmemWordSize, rsp.data->data(), kLmemWordSize);
    } else {
      std::memcpy(r->data.data(), rsp.data->data() + (r->word & (kLmemWordSize - 1)), kRowSize);
    }
    for (bool g : r->got) {
      if (!g)
        return;
    }
    r->done = true;
    r->at   = at;
  }

  // The output slot's requests: the store's one write; a global reduction's
  // words holding token bytes, ascending; a cluster reduction's elements.
  void store_load_slot(Worker& w, Worker::StTok&& t) {
    auto& st = w.st;
    st.slot = std::move(t);
    st.slot_valid = true;
    st.slot_at = cycle_ + 1;
    st.emit_idx = 0;
    st.emits.clear();
    auto& sl = st.slot;
    if (st.kind == dxa_ref::ST_BULK || st.kind == dxa_ref::ST_TENSOR) {
      st.emits.push_back({0, sl.mask});
      return;
    }
    const uint32_t wb = (st.kind == dxa_ref::ST_RED_CLUSTER) ? dxa_ref::type_bytes(st.type)
                                                             : dxa_ref::red_word_bytes(st.type);
    const uint64_t wmask = (wb == 8) ? 0xffull : 0xfull;
    const bool half = (dxa_ref::type_bytes(st.type) == 2);
    for (uint32_t off = 0; off < VX_CFG_L1_LINE_SIZE; off += wb) {
      const uint64_t m = (sl.mask >> off) & wmask;
      if (m == 0)
        continue;
      // A 16-bit pair cut by the box edge sends only its half's bytes.
      __assert(m == wmask || (half && (m == 0x3 || m == 0xc)),
               simobject_->name() << " DXA reduction word partly covered: off=" << off);
      st.emits.push_back({off, m});
    }
  }

  // The cycle's start for a store worker: its end, then the output slot's
  // request toward global memory, then the store's trailing fence.
  void store_pre(Worker& w) {
    auto& st = w.st;
    if (st.end_at != 0 && cycle_ >= st.end_at) {
      DT(3, simobject_->name() << "[" << w.worker_id << "] store end (#" << w.req.uuid << ")");
      this->finish_worker(w);
      return;
    }
    if (fetch_port_) {
      return;  // the map fetch holds the GMEM port
    }
    auto& ch = gmem_arb_->ReqIn.at(w.worker_id);
    const bool st_amo = (st.kind != dxa_ref::ST_BULK && st.kind != dxa_ref::ST_TENSOR);
    if (st_amo && st.slot_valid
        && (st.amo_sent - st.amo_rcvd >= kStAmoWindow
         || (st.amo_sent != 0 && st.slot.lw.gmem_cl_addr != st.last_line && st.amo_sent - st.amo_rcvd > 1))) {
      return;
    }
    if (st.slot_valid && st.slot_at <= cycle_ && st.kind != dxa_ref::ST_RED_CLUSTER) {
      const auto& sl = st.slot;
      const uint64_t line = sl.lw.gmem_cl_addr;
      auto blk = make_mem_block();
      std::memcpy(blk->data(), sl.img.data(), VX_CFG_L1_LINE_SIZE);
      MemReq mreq;
      mreq.tag  = 0;
      mreq.uuid = w.req.uuid;
      mreq.data = blk;
      if (st.kind == dxa_ref::ST_BULK || st.kind == dxa_ref::ST_TENSOR) {
        mreq.op      = MemOp::ST;
        mreq.addr    = line;
        mreq.byteen  = sl.mask;
        mreq.hart_id = w.req.core->id();
      } else {
        const uint32_t off = st.emits.at(st.emit_idx).first;
        const auto amo = dxa_ref::red_amo(st.op, st.type);
        mreq.op      = red_memop(amo.amo_op);
        mreq.addr    = line + off;
        mreq.byteen  = st.emits.at(st.emit_idx).second << off;
        mreq.hart_id = w.req.hart_id;
        mreq.flags.amo_unsigned = amo.is_unsigned;
      }
      if (ch.try_send(mreq, kGmemReqDelay)) {
        DT(3, simobject_->name() << "[" << w.worker_id << "] store " << (mreq.op == MemOp::ST ? "wr" : "amo")
           << ": addr=0x" << std::hex << mreq.addr << ", byteen=0x" << mreq.byteen << ", word=0x"
           << *reinterpret_cast<const uint64_t*>(blk->data() + (uint32_t(mreq.addr & (VX_CFG_L1_LINE_SIZE - 1)) & ~7u)) << std::dec
           << " (#" << w.req.uuid << ")");
        if (mreq.op == MemOp::ST) {
          ++st.writes;
          st.last_line = line;
          st.last_wr = cycle_;
        } else {
          ++st.amo_sent;
          st.last_line = line;
        }
        if (++st.emit_idx == st.emits.size()) {
          st.slot_valid = false;
          st.slot_freed = true;
        }
      }
      return;
    }
    const bool stores = (st.kind == dxa_ref::ST_BULK || st.kind == dxa_ref::ST_TENSOR);
    if (stores && st.writes != 0 && !st.fence_sent && this->store_drained(w) && cycle_ > st.last_wr) {
      MemReq f;
      f.op          = MemOp::LD;
      f.addr        = st.last_line;
      f.tag         = 0;
      f.hart_id     = w.req.core->id();
      f.uuid        = w.req.uuid;
      f.flags.fence = 1;
      if (ch.try_send(f, kGmemReqDelay)) {
        st.fence_sent = true;
        DT(3, simobject_->name() << "[" << w.worker_id << "] store fence: line=0x" << std::hex << f.addr
           << std::dec << " (#" << w.req.uuid << ")");
      }
    }
  }

  // Every token taken, assembled and emitted.
  bool store_drained(const Worker& w) const {
    const auto& st = w.st;
    return w.ag_idx >= w.work_list.size() && st.rd.empty() && st.reads.empty() && st.toks.empty()
        && !st.slot_valid;
  }

  // An AMO response, or the fence's.
  void store_gmem_rsp(Worker& w) {
    auto& st = w.st;
    if (st.kind == dxa_ref::ST_BULK || st.kind == dxa_ref::ST_TENSOR) {
      __assert(st.fence_sent && !st.wr_done, simobject_->name() << " DXA store response without its fence");
      st.wr_done = true;
      st.wr_new  = true;
      return;
    }
    __assert(st.amo_rcvd < st.amo_sent, simobject_->name() << " DXA reduction response without a request");
    ++st.amo_rcvd;
    if (st.amo_rcvd == st.amo_sent && this->store_drained(w)) {
      st.wr_done = true;
      st.wr_new  = true;
    }
  }

  // The cycle's end for a store worker: assembly of the oldest landed
  // response, then the operation's completions.
  void store_post(Worker& w) {
    this->store_step(w);
    w.st.slot_freed = false;
    w.st.wr_new = false;
  }

  void store_step(Worker& w) {
    auto& st = w.st;
    // A token's last response waits while the output slot stays full.
    if (!st.reads.empty() && st.reads.front().done && st.reads.front().at < cycle_
        && (st.toks.front().got + 1 < st.toks.front().reads || !st.slot_valid || st.slot_freed)) {
      const auto& r = st.reads.front();
      auto& t = st.toks.front();
      for (const auto& b : t.bytes) {
        if (b.phys >= r.word && b.phys < r.word + kRowSize) {
          t.img.at(b.line_off) = r.data.at(b.phys - r.word);
          t.mask |= uint64_t(1) << b.line_off;
        }
      }
      st.reads.pop_front();
      if (++t.got == t.reads) {
        Worker::StTok done = std::move(t);
        st.toks.pop_front();
        this->store_load_slot(w, std::move(done));
      }
    }
    if (st.end_at != 0)
      return;
    const bool reads_over = w.ag_idx >= w.work_list.size() && st.rd.empty() && st.reads.empty()
                         && st.toks.empty();
    if (st.kind == dxa_ref::ST_RED_CLUSTER) {
      if (reads_over && !st.slot_valid) {
        st.end_at = cycle_ + kStEndDelay;
      }
      return;
    }
    const bool rd_new = !st.rd_done && reads_over;
    if (rd_new) {
      st.rd_done = true;
      const bool stores = (st.kind == dxa_ref::ST_BULK || st.kind == dxa_ref::ST_TENSOR);
      const bool none = stores ? (st.writes == 0 && !st.slot_valid) : (st.amo_sent == 0 && !st.slot_valid);
      if (none) {
        st.wr_done = true;
        st.wr_new  = true;
      }
    }
    if (!st.wr_done && !st.slot_valid && st.amo_sent != 0 && st.amo_rcvd == st.amo_sent
        && this->store_drained(w)) {
      st.wr_done = true;
      st.wr_new  = true;
    }
    if (rd_new || st.wr_new) {
      DxaBgCompl c;
      c.wid        = w.req.wid;
      c.seq        = w.req.bg_seq;
      c.read_done  = rd_new;
      c.write_done = st.wr_new;
      bg_evq_.at(w.worker_id).push_back({c, cycle_ + kBgEvtDelay, w.req.core->id() % kCoresPerSocket});
      DT(3, simobject_->name() << "[" << w.worker_id << "] bg-compl " << c << " (#" << w.req.uuid << ")");
    }
    if (st.wr_new) {
      st.end_at = cycle_ + kStEndDelay;
    }
  }

  // Whether worker w's output slot offers a cluster reduction's packet now.
  bool red_offer(const Worker& w) const {
    const auto& st = w.st;
    return w.state == WState::RUNNING && w.store && st.kind == dxa_ref::ST_RED_CLUSTER && st.slot_valid
        && st.slot_at <= cycle_;
  }

  // Worker w's next element packet; its slot frees with the last one.
  DlmemReqPkt red_take(Worker& w) {
    constexpr uint64_t lmem_mask = (uint64_t(1) << VX_CFG_LMEM_LOG_SIZE) - 1;
    auto& st = w.st;
    const auto& sl = st.slot;
    const uint32_t eb  = dxa_ref::type_bytes(st.type);
    const uint32_t off = st.emits.at(st.emit_idx).first;
    const uint64_t addr = sl.lw.gmem_cl_addr + off;
    const uint32_t boff = uint32_t(addr & (VX_CFG_MEM_BLOCK_SIZE - 1));
    const auto amo = dxa_ref::red_amo(st.op, st.type);
    auto blk = make_mem_block();
    std::memset(blk->data(), 0, blk->size());
    std::memcpy(blk->data() + boff, sl.img.data() + off, eb);
    DlmemReqPkt f;
    f.dst        = w.req.dst_core;
    f.src        = w.req.src_core;
    f.op         = red_memop(amo.amo_op);
    f.lmem_addr  = addr & lmem_mask;
    f.data       = blk;
    f.byteen     = ((eb == 8) ? 0xffull : 0xfull) << boff;
    f.hart_id    = w.req.hart_id;
    f.tag        = w.req.slot;
    f.uuid       = w.req.uuid;
    f.dxa        = true;
    f.mbar_valid = true;
    f.mbar_addr  = (uint64_t(w.req.mbar_off) << 3) & lmem_mask;
    f.flags.amo_unsigned = amo.is_unsigned;
    if (++st.emit_idx == st.emits.size()) {
      st.slot_valid = false;
      st.slot_freed = true;
    }
    return f;
  }

  // With several workers their packets meet in a round-robin arbiter whose
  // two-entry output buffer is registered: a packet it takes in a cycle
  // reaches the row egress the next one.
  void tick_red_arb() {
    const uint32_t n = uint32_t(workers_.size());
    if (n < 2 || red_buf_.size() >= 2)
      return;
    for (uint32_t k = 0; k < n; ++k) {
      const uint32_t i = (red_rr_ + k) % n;
      if (!this->red_offer(workers_.at(i)))
        continue;
      red_buf_.push_back({this->red_take(workers_.at(i)), cycle_ + 1});
      red_rr_ = (i + 1) % n;
      return;
    }
  }

  // A cluster reduction's next element packet to the row egress: the
  // arbiter's head, or with one worker its output slot's.
  bool egress_red() {
    auto& out = simobject_->row_out;
    if (out.full())
      return false;
    if (workers_.size() > 1) {
      if (red_buf_.empty() || red_buf_.front().second > cycle_)
        return false;
      const DlmemReqPkt f = red_buf_.front().first;
      red_buf_.pop_front();
      out.send(f, 0);
      DT(3, simobject_->name() << " red pkt: " << f);
      return true;
    }
    for (auto& w : workers_) {
      if (!this->red_offer(w))
        continue;
      const DlmemReqPkt f = this->red_take(w);
      out.send(f, 0);
      DT(3, simobject_->name() << "[" << w.worker_id << "] red pkt: " << f);
      return true;
    }
    return false;
  }

  // Each core's arbiter passes one ready event a cycle, in round-robin worker
  // order; the core sees it the cycle after.
  void tick_bg_events() {
    const uint32_t n = uint32_t(workers_.size());
    for (uint32_t c = 0; c < kCoresPerSocket; ++c) {
      for (uint32_t k = 0; k < n; ++k) {
        const uint32_t i = (bg_rr_.at(c) + k) % n;
        auto& q = bg_evq_.at(i);
        if (q.empty() || q.front().at > cycle_ || q.front().core != c)
          continue;
        simobject_->BgComplOut.at(c).send(q.front().c, 1);
        q.pop_front();
        bg_rr_.at(c) = (i + 1) % n;
        break;
      }
    }
  }

  // A worker's reads in flight or waiting for assembly.
  static constexpr uint32_t kStCredits = 8;
  // Atomics a worker has toward the last-level cache before their responses:
  // the hardware path (the engine's output slice, the socket arbiter's output
  // buffer and the cache bank) holds this many while the bank holds admission.
  // The path is in order, so the first atomic of the next line, which may go
  // to another bank, waits until at most one of the line before is unanswered.
  static constexpr uint32_t kStAmoWindow = 5;
  // The engine registers what is seen here (the reads assembled, the last
  // response arrived) before it is a completion condition. The core sees the
  // event three cycles after that, through the worker's event register, its
  // elastic buffer and the core arbiter's output register; the worker ends
  // the cycle after the condition.
  static constexpr uint64_t kBgEvtDelay = 3;
  static constexpr uint64_t kStEndDelay = 2;
#else
#line 1194
#endif

  // The barriers go raw to the issuing core's completion unit.
  void release_all_barriers(Worker& w) {
    auto& ch = simobject_->compl_release_out.at(w.req.core->id() % kCoresPerSocket);
    if (w.is_multicast) {
      for (uint32_t cta : w.cta_indices) {
        ch.send(w.req.bar_id + cta);
      }
    } else {
      ch.send(w.req.bar_id);
    }
  }

  void finish_worker(Worker& w) {
    if (w.state == WState::RUNNING || w.state == WState::IDLE) {
      uint64_t latency = cycle_ - w.issue_cycle;
      ++perf_stats_.transfers;
      perf_stats_.total_latency += latency;
      DT(3, simobject_->name() << "[" << w.worker_id << "] complete: core="
         << w.req.core->id() << ", bar=" << w.req.bar_id
         << ", writes=" << w.writes_emitted << ", latency=" << latency);
    }
    w.state = WState::IDLE;
    w.work_list.clear();
    w.cta_indices.clear();
    w.issued_order.clear();
    w.drain_slot = UINT32_MAX;
    w.ag_idx = 0;
    w.mc_cta_idx = 0;
    w.km_elem_idx = 0;
    w.writes_emitted = 0;
  }

  // ── Members ──────────────────────────────────────────────────────────
  DxaCore*    simobject_;
  MemArbiter* gmem_arb_;
  std::array<Descriptor, VX_DCR_DXA_DESC_COUNT> descriptors_;
  std::deque<DxaReq>     queue_;
  std::vector<Worker>    workers_;
  uint32_t               rr_req_ = 0;
  uint32_t               xfer_seq_ = 1;
  std::vector<uint32_t>  beat_seq_ = std::vector<uint32_t>(kCoresPerSocket, 0);  // per core
#ifdef DXA_MBAR_ENABLE
  RingQueue<StreamBeat>   stream_{2};
  std::deque<DlmemReqPkt> row_flits_;   // the row egress's flits of its head beat
  // lmem_arb while a bulk copy runs: the worker its scan starts from, whether
  // a worker was granted the bus this cycle, and whether its operation went.
  uint32_t                lmem_rr_    = 0;
  bool                    lmem_arb_   = false;
  bool                    lmem_taken_ = false;
  bool                    lmem_sent_  = false;
  // Per core: a bulk copy's accesses in the engine's output stage at a
  // larger socket, a cycle each before the core switch takes them.
  std::vector<std::deque<StreamBeat>> cs_q_ = std::vector<std::deque<StreamBeat>>(kCoresPerSocket);
  // Per worker: its bulk-group completion events and the cycle each reaches
  // the per-core arbiter; per core: that arbiter's round-robin start.
  struct BgEv {
    DxaBgCompl c;
    uint64_t   at;     // the cycle it reaches its core's arbiter
    uint32_t   core;   // socket-relative
  };
  std::vector<std::deque<BgEv>> bg_evq_ = std::vector<std::deque<BgEv>>(VX_CFG_NUM_DXA_CORES);
  std::vector<uint32_t> bg_rr_ = std::vector<uint32_t>(kCoresPerSocket, 0);
  // The workers' cluster-reduction arbiter: its registered output buffer
  // (packet, the cycle it reaches the row egress) and round-robin start.
  std::deque<std::pair<DlmemReqPkt, uint64_t>> red_buf_;
  uint32_t red_rr_ = 0;
  // The descriptor cache's tensor-map entries, beside the slots: tagged by the
  // map's address above its 64-byte offset, with the map's first words.
  struct MapEntry {
    bool       valid = false;
    uint64_t   tag   = 0;
    Descriptor desc;
    std::array<uint32_t, TMAP_FETCH_WORDS> w{};
  };
  std::array<MapEntry, DXA_TMAP_ENTRIES> maps_;
  uint32_t map_rr_     = 0;     // the next victim once every entry is valid
  uint32_t head_entry_ = 0;     // the queue head's entry, once it hits
  bool     head_hit_   = false; // the queue head hit and waits for a worker
  // The queue head's map fetch: its line reads, one a cycle from send_at,
  // their responses in any order, and the cycle the head is looked up again;
  // killed by a launch's KMU write, its fill is not kept.
  struct Fetch {
    bool     active  = false;
    bool     killed  = false;
    uint64_t addr    = 0;
    uint32_t lines   = 0;
    uint32_t sent    = 0;
    uint32_t got     = 0;
    uint64_t send_at = 0;
    uint64_t hit_at  = 0;
    uint32_t hart_id = 0;
    uint64_t uuid    = 0;
    std::array<uint32_t, TMAP_WORDS> w{};
  };
  Fetch fetch_;
  bool  fetch_port_ = false;    // the fetch holds the GMEM port this cycle
#else
#line 935
#endif
  uint64_t               cycle_;
  DxaCore::PerfStats     perf_stats_;
};

// ════════════════════════════════════════════════════════════════════
// DxaCore — wrappers
// ════════════════════════════════════════════════════════════════════

DxaCore::DxaCore(const SimContext& ctx, const char* name, Socket* socket)
  : SimObject<DxaCore>(ctx, name)
  , dxa_req_in(kCoresPerSocket, this)
  , gmem_req_out(kDxaMemPorts, this)
  , gmem_rsp_in(kDxaMemPorts, this)
  , lmem_req_out(kCoresPerSocket * LMEM_PORTS_PER_CORE, this)
  , compl_release_out(kCoresPerSocket, this)
#ifdef DXA_MBAR_ENABLE
  , row_out(this)
  , lmem_rsp_in(kCoresPerSocket * LMEM_PORTS_PER_CORE, this)
  , BgComplOut(kCoresPerSocket, this)
#else
#line 950
#endif
{
  __unused(socket);

  // Build the GMEM arbiter (VX_CFG_NUM_DXA_CORES workers → kDxaMemPorts L2-facing).
  // Tag layout used by workers: high bit packs worker_id, low bits the
  // per-worker inflight slot. We pass TAG_SEL_IDX so the arb can route
  // responses back to the right input.
  char sname[100];
  snprintf(sname, 100, "%s-gmem-arb", name);
  gmem_arb_ = MemArbiter::Create(sname, ArbiterType::RoundRobin, VX_CFG_NUM_DXA_CORES, kDxaMemPorts);
  for (uint32_t i = 0; i < kDxaMemPorts; ++i) {
    gmem_arb_->ReqOut.at(i).bind(&gmem_req_out.at(i));
    gmem_rsp_in.at(i).bind(&gmem_arb_->RspIn.at(i));
  }

  impl_ = new Impl(this, gmem_arb_.get());
}

DxaCore::~DxaCore() {
  delete impl_;
}

void DxaCore::on_reset() { impl_->reset(); }
void DxaCore::on_tick()  { impl_->tick(); }

int DxaCore::dcr_write(uint32_t addr, uint32_t value) {
  return impl_->dcr_write(addr, value);
}

const DxaCore::PerfStats& DxaCore::perf_stats() const {
  return impl_->perf_stats();
}
