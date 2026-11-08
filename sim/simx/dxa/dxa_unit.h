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

#include "types.h"
#include "instr_trace.h"
#ifdef DXA_MBAR_ENABLE
#include "dxa_bg.h"
#else
#line 17
#endif

namespace vortex {

class Core;

// DxaReq — per-SFU dispatch packet. The SFU owns the outbound SimChannel;
// DxaUnit is a plain helper sub-class of SfuUnit that decodes lanes and
// pushes onto that channel.
struct DxaReq {
  Core*    core;          // routes barrier-release back at completion
  uint64_t uuid;
  uint32_t wid;
  uint32_t desc_slot;     // descriptor table index
  uint32_t bar_id;        // RAW (pre bar_decode_id) — kept raw so that
                          // multicast offset arithmetic (bar_id + cta_idx)
                          // increments cta_no in the encoded low byte.
                          // Decoded only at the release call site.
  uint32_t cta_mask;      // multicast warp mask (>1 bit ⇒ multicast)
  uint64_t smem_addr;
  uint32_t coords[5];
#ifdef DXA_MBAR_ENABLE
  bool     mode;          // meta[30]: complete_tx on an mbarrier
  uint32_t mbar_off;      // the mbarrier's doubleword offset in the destination's local memory
  uint32_t slot;          // the destination block's CTA slot and its generation
  uint32_t gen;
  // A destination block on another core of the cluster: the rows travel over
  // the DLMEM network, and smem_addr is in that core's local memory.
  bool     remote;
  uint32_t src_core;      // the issuing and the destination core, cluster-relative
  uint32_t dst_core;
  bool     win;           // a DLMEM window destination
  uint32_t win_room;      // and its bytes up to its block's end
  // A cluster multicast (meta[30] with a rank mask): one copy into each masked
  // rank's block, in increasing rank, each at the issuer's block-relative
  // offset and completing on the mbarrier at the issuer's mbarrier offset.
  struct Rcv {
    uint32_t core;        // cluster-relative
    uint32_t slot;
    uint32_t gen;
    uint64_t smem_addr;
    uint32_t mbar_off;
    bool     remote;
  };
  bool     cluster;
  uint32_t num_rcv;
  std::array<Rcv, VX_CFG_DLMEM_MAX_CLUSTER> rcvs;
  // A bulk copy (funct3 = 1), which reads no descriptor: `size` bytes from
  // src_addr, a local-memory address of the issuing core, to smem_addr.
  bool     bulk;
  uint64_t src_addr;
  uint32_t size;
  uint64_t sent;          // the cycle it left the unit
  // The PTX forms: a tensor load (funct3 = 2) or a global bulk load
  // (funct3 = 3), either of which may be a prefetch (no local-memory write,
  // no completion; funct3 = 4 is the bulk one). coords are signed.
  bool     tload;
  bool     bload;
  bool     prefetch;
  uint32_t offs[3];       // the im2col offsets along W, H, D
  uint64_t gsrc;          // a bulk load's global source, and size its bytes
  uint64_t blk_off;       // smem_addr's offset in its block, the swizzle's base
  // A store-family operation (funct3 = 5): `kind` is the DXA_ST_* form, with
  // its reduction op and type; src_addr is the source in the issuing core's
  // local memory, gdst a global destination. A cluster reduction's
  // destination and mbarrier are smem_addr, mbar_off and the receiver fields.
  bool     store    = false;
  uint32_t st_kind  = 0;
  uint32_t red_op   = 0;
  uint32_t red_type = 0;
  uint64_t gdst     = 0;
  // A bulk_group operation (kinds 0-2 and 4) and its group.
  bool     bg       = false;
  uint32_t bg_seq   = 0;
  uint32_t hart_id  = 0;  // the issuing warp's thread 0
  // A map-form tensor operation, whose descriptor is the 128-byte tensor map
  // at map_addr in global memory; or a tensormap op (funct3 7), tmop.
  bool     map      = false;
  uint64_t map_addr = 0;
  bool     tmop     = false;
  uint32_t tmap_op  = 0;   // DXA_TMAP_PREFETCH or DXA_TMAP_INVAL
  // The engine's queue-head bookkeeping: the cycle it was first looked up,
  // whether its map was fetched, and whether a map operation ahead held it.
  uint64_t head_at  = 0;
  bool     fetched  = false;
  bool     held     = false;
#else
#line 43
#endif
};

// DXA sub-unit of the SFU. Plain (non-SimObject) class owned by SfuUnit;
// lane-decodes a DxaType trace and forwards it onto the SFU's outbound
// DxaReq channel.
class DxaUnit {
public:
  // The outbound channel lives on SfuUnit (which is the SimObject).
  DxaUnit(Core* core, SimChannel<DxaReq>& req_out)
    : core_(core), req_out_(req_out) {}

  // Decode lanes 0..3 from the trace's source operands and try to push a
  // DxaReq on the SFU's outbound channel. Returns the trace on success
  // (caller falls through to writeback) or nullptr on backpressure
  // (caller retries next cycle without side effects).
  instr_trace_t* process(instr_trace_t* trace);

#ifdef DXA_MBAR_ENABLE
  // Mirrors the dispatcher's slot generations and the warps' slots.
  void slot_evt(const CtaSlotEvt& e);
#endif

private:
  Core*               core_;
  SimChannel<DxaReq>& req_out_;
#ifdef DXA_MBAR_ENABLE
public:
  // A bulk-group event per COMMIT, WAIT and bulk_group operation, to the
  // core's bulk-group unit, and the warps' group state back from it. The
  // SFU owns them, as it owns req_out.
  SimChannel<DxaBgEvt>   BgEvtOut{req_out_.module()};
  SimChannel<DxaBgState> BgStateIn{req_out_.module(), 4 * VX_CFG_NUM_WARPS + 8};

  void bg_state(const DxaBgState& s);
  void bg_reset();

private:
  bool issue_store(instr_trace_t* trace);
  bool issue_bgop(instr_trace_t* trace);
  bool issue_tmap(instr_trace_t* trace);
  void check_map(const instr_trace_t* trace, uint64_t map) const;
  void decode_store(const instr_trace_t* trace, DxaReq& req) const;

  std::vector<DxaBgState> bg_mirror_ = std::vector<DxaBgState>(VX_CFG_NUM_WARPS);
  // The open group's sequence, per warp. Every COMMIT starts here, so it
  // equals the bulk-group unit's without waiting for its state.
  std::vector<uint32_t>   bg_seq_    = std::vector<uint32_t>(VX_CFG_NUM_WARPS, 0);

  // Resolves a DLMEM window destination and mbarrier through the issuing
  // block's peer row, as the RTL reads it from the core's peer table.
  void translate(const instr_trace_t* trace, uint32_t meta, uint32_t cta_mask, DxaReq& req) const;
  void cluster_receivers(const instr_trace_t* trace, uint32_t meta, uint32_t cta_mask, DxaReq& req) const;
  void translate_bulk(const instr_trace_t* trace, DxaReq& req) const;
  void translate_bulk(const instr_trace_t* trace, DxaReq& req, uint64_t dst, uint64_t mbar,
                      uint64_t src, uint64_t size) const;
  void translate_red_cluster(const instr_trace_t* trace, DxaReq& req, uint64_t dst, uint64_t mbar,
                             uint64_t src, uint64_t size) const;
  bool issue_ptx(instr_trace_t* trace);
  void decode_ptx(const instr_trace_t* trace, DxaType type, DxaReq& req) const;

  [[noreturn]] void issue_abort(const instr_trace_t* trace, const char* msg) const;

  std::vector<uint32_t> warp_slot_ = std::vector<uint32_t>(VX_CFG_NUM_WARPS, 0);
  std::vector<uint32_t> slot_gen_  = std::vector<uint32_t>(VX_CFG_NUM_WARPS, 0);
  std::vector<dlmem_ref::Row> slot_row_ = std::vector<dlmem_ref::Row>(VX_CFG_NUM_WARPS);
  uint32_t stride_ = 0;
#else
#line 72
#endif
};

} // namespace vortex
