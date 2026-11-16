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
#include "memory.h"

namespace vortex {

class Cache : public SimObject<Cache> {
public:
	enum ReplPolicy : uint8_t {
		RANDOM = 0,
		FIFO   = 1,
		PLRU   = 2
	#ifdef VX_CFG_H100_SM_ENABLE
		, H100 = 3
	#else
	#line 26
	#endif
	};

	struct Config {
		bool    bypass;         // cache bypass
		uint8_t C;              // log2 cache size
		uint8_t L;              // log2 line size
		uint8_t S;              // log2 sector size (fill/eviction/mem granule); = L => 1 sector/line
		uint8_t W;              // log2 word size
		uint8_t A;              // log2 associativity
		uint8_t B;              // log2 number of banks
		uint8_t addr_width;     // word address bits
		uint8_t num_inputs;     // number of inputs
		uint8_t mem_ports;      // memory ports
		bool    write_back;     // is write-back
		bool    dirty_bytes;    // a writeback carries only the dirty bytes, else its whole sector
		bool    write_reponse;  // enable write response
		uint16_t mshr_size;     // MSHR buffer size
		uint8_t latency;        // pipeline latency
		uint8_t repl_policy;    // replacement policy (ReplPolicy)
		bool    is_llc;					// is last-level cache
		// Every memory agent of the device reaches this cache: a fence drains
		// here instead of locking the inputs and walking the lines.
		bool    fence_shared = false;
	#ifdef VX_CFG_EXT_DLMEM_ENABLE
		// The first level the LSU's fences reach: a CTA fence drains here, then
		// goes on as a plain read of its line.
		bool    fence_cta = false;
	#endif
	#ifdef VX_CFG_H100_SM_ENABLE
		// The H100 L1: per-byte valid with store allocate, an XOR-folded set index,
		// newest-protected random replacement and one line served per cycle.
		bool    h100_l1 = false;
	#else
	#line 53
	#endif
	};

	struct PerfStats {
		uint64_t reads = 0;
		uint64_t writes = 0;
		uint64_t read_misses = 0;
		uint64_t write_misses = 0;
		uint64_t evictions = 0;
		uint64_t bank_stalls = 0;
		uint64_t mshr_stalls = 0;
		uint64_t mem_latency = 0;

		PerfStats& operator+=(const PerfStats& rhs) {
			this->reads += rhs.reads;
			this->writes += rhs.writes;
			this->read_misses += rhs.read_misses;
			this->write_misses += rhs.write_misses;
			this->evictions += rhs.evictions;
			this->bank_stalls += rhs.bank_stalls;
			this->mshr_stalls += rhs.mshr_stalls;
			this->mem_latency += rhs.mem_latency;
			return *this;
		}
	};

	std::vector<SimChannel<MemReq>> core_req_in;
	std::vector<SimChannel<MemRsp>> core_rsp_out;
	std::vector<SimChannel<MemReq>> mem_req_out;
	std::vector<SimChannel<MemRsp>> mem_rsp_in;

	Cache(const SimContext& ctx, const char* name, const Config& config);
	~Cache();

	PerfStats perf_stats() const;

	// Flush API: arm a write-back walk that emits dirty lines via mem_req_out.
	// flush_done() returns true once the walk completes; the caller is expected
	// to tick the simulator until then. Write-through caches no-op.
	void flush_begin();
	bool flush_done() const;

protected:
	void on_reset();
	void on_tick();

private:
	class Impl;
	Impl* impl_;

	friend class SimObject<Cache>;
};
#ifdef VX_CFG_H100_SM_ENABLE

// Stage counts of the H100 L1 path outside the latency the model inserts: a
// global load's response is released no earlier than l1_hit_latency(P) minus
// L1_CORE_STAGES after its request leaves the LSU, so the time from its issue
// to the issue of the instruction that uses it is the measured hit time.
// L1_CORE_PRE_STAGES: the load's issue to its request leaving the LSU.
// L1_CORE_POST_STAGES: its response at the LSU to the dependent issue.
// L1_HIT_STAGES: the request leaving the LSU to its hit response there.
// L1_MISS_STAGES: the request leaving the LSU to the L1's memory request,
// plus the L1's memory response to the response at the LSU.
// OPC_ROUTE_STAGES: the operand collectors' routing stage when there is more
// than one collector (VX_gpu_pkg OPC_ROUTE_STAGES).
constexpr uint32_t OPC_ROUTE_STAGES    = (VX_CFG_NUM_OPCS > 1) ? 1 : 0;
constexpr uint32_t L1_CORE_PRE_STAGES  = 7 + OPC_ROUTE_STAGES;   // issue 2, operand collect 2 (+ routing), dispatch to the LSU's request 3
constexpr uint32_t L1_CORE_POST_STAGES = 3;   // the LSU's response to the commit 3, the dependent issue there
constexpr uint32_t L1_HIT_STAGES       = 14;
constexpr uint32_t L1_MISS_STAGES      = 25;
constexpr uint32_t L1_CORE_STAGES        = L1_CORE_PRE_STAGES + L1_CORE_POST_STAGES;
constexpr uint32_t L1_PIPE_OVERHEAD      = L1_CORE_STAGES + L1_HIT_STAGES;
constexpr uint32_t L1_MISS_PIPE_OVERHEAD = L1_CORE_STAGES + L1_MISS_STAGES;
static_assert(L1_PIPE_OVERHEAD <= VX_CFG_L1_HIT_LATENCY, "L1_HIT_LATENCY is below the pipeline's own hit time");

#else
#line 104
#endif

}