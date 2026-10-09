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
};

}
