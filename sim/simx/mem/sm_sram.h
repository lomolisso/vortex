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

#include <VX_config.h>

#ifdef VX_CFG_H100_SM_ENABLE

#include <algorithm>
#include "types.h"
#include "cache.h"
#include <smem_bank_ref.h>

namespace vortex {

// The launch's split of the array (smem_cfg_t).
struct SmSramCfg {
  uint32_t s_rows;
  uint32_t l1_lines;
  uint32_t l1_protect;
};

enum class SmSramOp : uint8_t { READ, WRITE, AMO, MBAR };

// One lane of a global-class access: its slot row, byte offset in the row,
// the lane word's byte enables and data.
struct SmSramLane {
  uint32_t row = 0;
  uint32_t boff = 0;
  uint32_t byteen = 0;
  uint64_t data = 0;
};

// One global-class warp access from the L1 (sm_sram_req_t).
struct SmSramReq {
  SmSramOp op = SmSramOp::READ;
  BitVector<> mask;
  std::vector<SmSramLane> lanes;
  uint64_t tag = 0;
  uint32_t seq = 0;

  SmSramReq(uint32_t size = LSU_CHANNELS) : mask(size), lanes(size) {}
};

// Per lane the lane word read (sm_sram_rsp_t).
struct SmSramRsp {
  uint64_t tag = 0;
  BitVector<> mask;
  std::vector<uint64_t> data;

  SmSramRsp(uint32_t size = LSU_CHANNELS) : mask(size), data(size, 0) {}
};

enum class SmSramSrc : uint8_t { L1, TMA, DSMEM, CPA };

// One row write (sm_sram_fill_t); `last` closes a DSMEM packet.
struct SmSramFill {
  uint32_t row = 0;
  std::vector<uint8_t> byteen;
  std::vector<uint8_t> data;
  SmSramSrc src = SmSramSrc::L1;
  bool last = false;

  SmSramFill() : byteen(VX_CFG_SM_DATAPATH_BYTES, 0), data(VX_CFG_SM_DATAPATH_BYTES, 0) {}
};

struct SmSramRowRd {
  uint32_t row = 0;
  uint64_t tag = 0;
};

struct SmSramRowRsp {
  uint64_t tag = 0;
  std::vector<uint8_t> data;
};

// LocalMem's ports toward the array (rows read: TCU, DXA, DSMEM; rows written:
// DXA, DSMEM; one-lane atomics: DXA completion, DSMEM).
struct SmSramClientPorts {
  std::vector<SimChannel<SmSramRowRd>>  RowRdOut;
  std::vector<SimChannel<SmSramRowRsp>> RowRspIn;
  std::vector<SimChannel<SmSramFill>>   WrOut;
  std::vector<SimChannel<MemReq>>       AmoOut;
  std::vector<SimChannel<MemRsp>>       AmoRspIn;
  SimChannel<bool>                      DsmemBusyIn;
  explicit SmSramClientPorts(SimObjectBase* m)
    : RowRdOut(3, m), RowRspIn(3, m), WrOut(2, m), AmoOut(2, m), AmoRspIn(2, m), DsmemBusyIn(m) {}
};

}

#include "local_mem.h"

namespace vortex {

// One SRAM array per SM holding shared memory (rows from 0 up) and the L1's
// data slots (rows from the top down), with one read port and one write port
// over all banks (VX_sm_sram.sv, sw/common/smem_bank_ref.h SramPortRef).
class SmSram : public SimObject<SmSram> {
public:
  using Ptr = std::shared_ptr<SmSram>;

  static constexpr uint32_t BANKS      = VX_CFG_LMEM_NUM_BANKS;
  static constexpr uint32_t BANK_BYTES = VX_CFG_LMEM_BANK_BYTES;
  static constexpr uint32_t ROW_BYTES  = VX_CFG_SM_DATAPATH_BYTES;
  static constexpr uint32_t ROWS       = VX_CFG_SM_SRAM_BYTES / ROW_BYTES;
  static constexpr uint32_t ROW_MASK   = (1u << log2ceil(ROWS)) - 1;
  static constexpr uint32_t LANES      = LSU_CHANNELS;
  static constexpr uint32_t WORD       = LSU_WORD_SIZE;
  static constexpr uint32_t HPW        = WORD / BANK_BYTES;
  static constexpr uint32_t NH         = LANES * HPW;
  static constexpr uint32_t PAIRS      = BANKS / 2;
  static constexpr uint32_t RD_ROWS    = 3;  // TCU, DXA, DSMEM
  static constexpr uint32_t WR_ROWS    = 3;  // L1 refill, DXA, DSMEM
  static constexpr uint32_t MRG_BYTES  = VX_CFG_SM_SRAM_DSMEM_WR_BYTES;
  // Output skid: the stage and the capture register each owe an entry when
  // a pick is decided; two more keep one response per cycle.
  static constexpr uint32_t RSP_DEPTH  = 4;
  // mbarrier ops leave the wavefront stage in one cycle and wait for the AMO path.
  static constexpr uint32_t MBQ_DEPTH  = VX_CFG_MBAR_OP_QUEUE;
  // The queue's head takes the AMO path only while it and the op in the finish register fit.
  static constexpr uint32_t MRQ_DEPTH  = 4;
  // LR/SC stations per bank, direct mapped by row: room for every hart's one
  // reservation when the reserved words spread over the banks. A live station
  // refuses other harts' load-reserveds until it has refused one from each.
  static constexpr uint32_t HARTS      = VX_CFG_NUM_WARPS * VX_CFG_NUM_THREADS;
  static constexpr uint32_t RS         = 1u << log2ceil((HARTS + BANKS - 1) / BANKS);
  static constexpr uint32_t RS_HOLD    = HARTS - 1;

  // The pkg's stage counts (VX_gpu_pkg.sv).
  static constexpr uint32_t LMEM_CORE_STAGES    = 12 + OPC_ROUTE_STAGES;
  static constexpr uint32_t LMEM_SWITCH_STAGES  = 2;
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  static constexpr uint32_t LMEM_XLATE_STAGES   = 1;
#else
  static constexpr uint32_t LMEM_XLATE_STAGES   = 0;
#endif
  static constexpr uint32_t LMEM_PIPE_OVERHEAD  = LMEM_CORE_STAGES + LMEM_SWITCH_STAGES + LMEM_XLATE_STAGES;
  static_assert(LMEM_PIPE_OVERHEAD == vortex::smem_bank_ref::smem_pipe_stages(VX_CFG_NUM_OPCS, LMEM_XLATE_STAGES != 0),
                "smem_pipe_stages() must count the stages the model builds");
  static constexpr uint32_t SM_SRAM_BANK_STAGES = 1;
  static constexpr uint32_t SM_SRAM_RSP_STAGES  = 1;
  static constexpr uint32_t SM_SRAM_AMO_STAGES  = 2;

  struct Config {
    uint32_t num_blocks;
  };

  struct PerfStats {
    uint64_t reads = 0;
    uint64_t writes = 0;
    uint64_t wavefronts = 0;
    uint64_t amo_passes = 0;

    PerfStats& operator+=(const PerfStats& rhs) {
      reads += rhs.reads;
      writes += rhs.writes;
      wavefronts += rhs.wavefronts;
      amo_passes += rhs.amo_passes;
      return *this;
    }
  };

  // The launch's split, held until the next one.
  SimChannel<SmSramCfg> CfgIn;

  // Shared class: the LSU blocks' local legs.
  std::vector<SimChannel<LsuReq>> LsuReqIn;
  std::vector<SimChannel<LsuRsp>> LsuRspOut;

  // Global class: the L1's one in-order stream; each access's sequence
  // number leaves on WfRetiredOut as it leaves the wavefront stage.
  SimChannel<SmSramReq> GlReqIn;
  SimChannel<SmSramRsp> GlRspOut;
  SimChannel<uint32_t>  WfRetiredOut;

  // Row reads (TCU, DXA, DSMEM), answered SM_SRAM_BANK_STAGES +
  // SM_SRAM_RSP_STAGES after the grant.
  std::vector<SimChannel<SmSramRowRd>>  RowRdIn;
  std::vector<SimChannel<SmSramRowRsp>> RowRspOut;

  // Row writes (L1 refill, DXA, DSMEM slot); a refill's row leaves on
  // FillDoneOut in its write cycle.
  std::vector<SimChannel<SmSramFill>> WrIn;
  SimChannel<uint32_t> FillDoneOut;

  // cp.async lane writes, acknowledged once written; the copy unit's
  // mbarrier ops, answered from the mbarrier op queue.
  std::vector<SimChannel<LsuReq>> CpaReqIn;
  std::vector<SimChannel<LsuRsp>> CpaRspOut;

  // One-lane atomics (DXA completion, DSMEM), answered the cycle after
  // their AMO-path grant.
  std::vector<SimChannel<MemReq>> AmoIn;
  std::vector<SimChannel<MemRsp>> AmoRspOut;

#ifdef VX_CFG_EXT_MBAR_ENABLE
  // Per bank: a try_wait that found its phase open, or an mbarrier op that
  // completed the phase of the word at row.
  std::vector<SimChannel<MbarBankEvt>> MbarEvtOut;
#endif

#ifdef VX_CFG_EXT_DLMEM_ENABLE
  // The counted lanes of an access as it commits: a shared wavefront access,
  // an LSU atomic, an mbarrier op.
  std::vector<SimChannel<WtInc>> CommitOut;
#endif

  // The DSMEM merge register holds slots not yet in the array; sent as it
  // changes, for the next cycle.
  SimChannel<bool> DsmemBusyOut;

  SmSram(const SimContext& ctx, const char* name, const Config& config);
  virtual ~SmSram();

  const PerfStats& perf_stats() const;

  // The split of the array at a launch's shared-memory size S.
  static SmSramCfg cfg_of(uint32_t s_bytes);

  // Host-side access to the array, by array byte address.
  void read(void* data, uint64_t addr, uint32_t size) const;
  void write(const void* data, uint64_t addr, uint32_t size);

protected:
  void on_reset();
  void on_tick();

  class Impl;
  Impl* impl_;

  friend class SimObject<SmSram>;
};

}

#endif
