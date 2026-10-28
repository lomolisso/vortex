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

#include <array>
#include <mempool.h>
#include <ringqueue.h>
#include "func_unit.h"
#include "instr.h"
#include "VX_config.h"
#ifdef VX_CFG_EXT_DLMEM_ENABLE
#include <deque>
#endif

// The TCU metadata SRAM is present when any metadata-consuming mode (MX or
// sparse) is enabled. Internal derived macro — not a VX_CFG_* knob; same
// derivation as tcu_unit.h.
#if defined(VX_CFG_TCU_MX_ENABLE) || defined(VX_CFG_TCU_SPARSE_ENABLE)
#define TCU_META_ENABLE
#endif

namespace vortex {

// Micro-op generator for packed-load macro instructions (PACKLB.F / PACKLH.F).
class LsuUopGen {
public:
  LsuUopGen(PoolAllocator<Instr, 64>& pool) : pool_(pool) {}

  static uint32_t uop_count(const Instr& instr);

  Instr::Ptr get(const Instr& macro_instr, uint32_t uop_index);

private:
  PoolAllocator<Instr, 64>& pool_;
};

// Fence controller: the block issues nothing from sending its fence token
// into the memory hierarchy until the cycle after the token's response.
class FenceController {
public:
  bool locked() const { return locked_; }

  void engage(instr_trace_t* trace) {
    trace_ = trace;
    locked_ = true;
    returned_ = false;
  }

  void token_returned() { returned_ = true; }

  // Lifts the lock once the token has returned; the caller still skips this
  // cycle's dispatch.
  bool try_release() {
    if (!locked_ || !returned_) return false;
    trace_ = nullptr;
    locked_ = false;
    return true;
  }

  instr_trace_t* trace() const { return trace_; }

  void reset() { trace_ = nullptr; locked_ = false; returned_ = false; }

private:
  instr_trace_t* trace_ = nullptr;
  bool locked_ = false;
  bool returned_ = false;
};

class LsuUnit : public FuncUnit<VX_CFG_NUM_LSU_BLOCKS> {
public:
	LsuUnit(const SimContext& ctx, const char* name, Core*);
	~LsuUnit();

#ifdef TCU_META_ENABLE
	// TCU metadata client port on block 0: TCU_LD load requests arbitrate
	// with the LSU's own traffic into lmem_switch; response fragments return
	// here with the client's own tag restored.
	SimChannel<LsuReq> TcuReqIn;
	SimChannel<LsuRsp> TcuRspOut;
#endif

#ifdef VX_CFG_EXT_DLMEM_ENABLE
	// Per-block requests and fences toward the aperture translation stage.
	std::vector<SimChannel<LsuReq>> XlateReqOut;

	// Per block: the warp of an acquire AMO, sent as the commit takes its last result.
	std::vector<SimChannel<uint32_t>> UnlockOut;

	// Warps the scheduler activated: a new thread's release state starts clean.
	SimChannel<WarpMask> SpawnIn;

	// UnlockOut's delay: the scheduler's stall register, so the warp can be picked
	// this many cycles after the commit takes the result.
	static constexpr uint32_t ACQ_UNLOCK_DELAY = 1;
#endif
#ifdef VX_CFG_EXT_MBAR_ENABLE
	// Per block: the warp of a try_wait, sent as the commit takes its last result.
	std::vector<SimChannel<uint32_t>> MbarRetireOut;

	// The waiter table ticks before the commit that takes the result, so it
	// sees the retire one cycle later and applies it as the earlier cycle's.
	static constexpr uint32_t MBAR_RETIRE_DELAY = 1;
#endif

	// Returns true when all LSU blocks have no in-flight requests.
	bool drained() const;

protected:
	void on_reset() override;
	void on_tick() override;

private:

	void compute_addrs(uint32_t b, instr_trace_t* trace);

	void issue_fence(uint32_t b, instr_trace_t* trace);

	SimChannel<LsuReq>& req_out(uint32_t b);

	// Per-cycle handlers.
	void process_response_step(uint32_t b);
	void process_request_step(uint32_t b);

	// Drain Inputs[b] into req_queue.
	void ingest_inputs(uint32_t b);

#ifdef VX_CFG_EXT_DLMEM_ENABLE
	static bool is_acquire_amo(const instr_trace_t* trace);
#ifdef VX_CFG_EXT_MBAR_ENABLE
	static bool is_try_wait(const instr_trace_t* trace);
#endif

	// Called for every result beat sent to Outputs[b].
	void mark_unlock(uint32_t b, instr_trace_t* beat, bool is_final);

	// Neither local memory, the DLMEM window (the translation's predicate) nor the stacks.
	static bool lane_is_global(uint64_t addr);

	// Runs a CFENCE packet at the head of block b.
	void process_cfence(uint32_t b, instr_trace_t* trace);
#endif

  struct mem_addr_size_t {
		uint64_t addr;
		uint32_t size;
		uint64_t data;
		uint64_t amo_cmp;
		uint32_t tid;
	};

 	struct pending_req_t {
		instr_trace_t* trace;
		uint32_t count;
		bool eop;
		std::vector<mem_addr_size_t> lanes;
		IntrLsuArgs lsu_args;
		bool        is_load;
		// TCU metadata client entry: response fragments are forwarded to
		// TcuRspOut with client_tag restored instead of written to registers.
		bool        is_tcu     = false;
		uint32_t    client_tag = 0;
		bool        is_fence   = false;
	#ifdef VX_CFG_EXT_DLMEM_ENABLE
		bool        gload      = false;  // a global read packet, counted in gload_
	#endif
	};

	// Per-block LSU state. Each member is a named hardware sub-block.
	struct lsu_state_t {
		RingQueue<instr_trace_t*> req_queue{VX_CFG_LSU_QUEUE_IN_SIZE};
		// In-flight LSU requests, keyed by the tag the LSU allocates on
		// issue. Stores the originating trace + per-lane (tid, addr, size)
		// info needed to write the response back to the right registers.
		// (Not a miss-status table — the cache has its own MSHR for that.)
		// Sized by the outstanding pool (MLP depth), decoupled from the
		// input staging queue above.
		HashTable<pending_req_t>  pending_reqs{VX_CFG_LSU_PENDING_SIZE};
		FenceController           fence;
		std::vector<mem_addr_size_t> addr_list;
		uint32_t                  remain_addrs = 0;

		void reset() {
			this->req_queue.clear();
			this->pending_reqs.clear();
			this->fence.reset();
			this->addr_list.clear();
			this->remain_addrs = 0;
		}
	};

	std::array<lsu_state_t, VX_CFG_NUM_LSU_BLOCKS> states_;
	uint64_t pending_loads_;

#ifdef VX_CFG_EXT_DLMEM_ENABLE
	// Packets of each warp's acquire AMO answered so far; a warp has at most one,
	// since it stays stalled until the last is released.
	std::vector<uint32_t> aq_done_;
	// Per block: last results of acquire AMOs sent to Outputs, not yet delivered.
	std::array<std::vector<instr_trace_t*>, VX_CFG_NUM_LSU_BLOCKS> unlock_beats_;
	// Per block, in Outputs' queue order: the warp each result releases, or -1.
	std::array<std::deque<int32_t>, VX_CFG_NUM_LSU_BLOCKS> out_unlock_;
#ifdef VX_CFG_EXT_MBAR_ENABLE
	// The same for try_waits: last results not yet delivered, then the warp each
	// result in Outputs retires, or -1. A warp has at most one try_wait or
	// acquire AMO in flight, so both share aq_done_.
	std::array<std::vector<instr_trace_t*>, VX_CFG_NUM_LSU_BLOCKS> retire_beats_;
	std::array<std::deque<int32_t>, VX_CFG_NUM_LSU_BLOCKS> out_retire_;
#endif
	// Per warp: a global write was sent since its last fence answered, and its
	// global read packets (loads, LR, AMOs) still waiting for their response.
	// Both are registers: a CFENCE sees this cycle's updates the next cycle.
	// The RTL counter's range: every packet the warp's slice can hold, all in flight.
	static constexpr uint32_t GLOAD_MAX = (1u << (log2ceil(VX_CFG_LSU_QUEUE_IN_SIZE + VX_CFG_LSU_PENDING_SIZE + 3)
	                                             + log2ceil(VX_CFG_NUM_THREADS / VX_CFG_NUM_LSU_LANES))) - 1;
	std::vector<bool>     gdirty_;
	std::vector<uint32_t> gload_;
	std::vector<uint32_t> gload_dec_;
#endif
};

}
