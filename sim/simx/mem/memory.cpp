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

#include <VX_types.h>
#include "memory.h"
#include <vector>
#include <queue>
#include <sstream>
#include <unordered_map>
#include <iostream>
#include <stdlib.h>
#include <dram_sim.h>
#include <memory>

#include "mem_block_pool.h"
#include "constants.h"
#include "types.h"
#include "debug.h"
#include "VX_config.h"

using namespace vortex;

class Memory::Impl {
private:
	struct Pending {
		MemReq request;
		uint32_t channel;
		std::shared_ptr<mem_block_t> rsp_data;  // captured at request time for reads
	};

	Memory*   simobject_;
	Config    config_;
	MemCrossBar::Ptr mem_xbar_;
	std::vector<std::unique_ptr<DramSim>> dram_sims_;
	std::unordered_map<uint64_t, Pending> pending_;
	uint64_t  next_id_;
	RAM*      ram_;
	Memory::PreSendHook pre_send_hook_;
	mutable PerfStats perf_stats_;

	DramSim& dram_of(uint32_t channel) {
		return *dram_sims_.at(channel / DramSim::channels_per_system());
	}

public:
	Impl(Memory* simobject, const Config& config)
		: simobject_(simobject)
		, config_(config)
		, next_id_(0)
		, ram_(nullptr)
	{
		if (config.num_channels != VX_CFG_DRAM_CHANNELS
		 || DramSim::num_systems() * DramSim::channels_per_system() != config.num_channels) {
			std::cerr << "Memory: " << config.num_channels << " channels, the DRAM model has "
			          << DramSim::num_systems() << " x " << DramSim::channels_per_system() << std::endl;
			std::abort();
		}
		for (uint32_t s = 0; s < DramSim::num_systems(); ++s) {
			dram_sims_.emplace_back(new DramSim(DramSim::Config{
				DramSim::channels_per_system(), s * DramSim::channels_per_system(), false}));
		}
		char sname[100];
		snprintf(sname, 100, "%s-xbar", simobject->name().c_str());
		mem_xbar_ = MemCrossBar::Create(sname, ArbiterType::RoundRobin, ArbiterType::RoundRobin,
			config.num_ports, config.num_channels,
			[](const MemCrossBar::ReqType& req) {
				return DramSim::channel_of(req.addr);
		});
		for (uint32_t i = 0; i < config.num_ports; ++i) {
			simobject->mem_req_in.at(i).bind(&mem_xbar_->ReqIn.at(i));
			mem_xbar_->RspOut.at(i).bind(&simobject->mem_rsp_out.at(i));
		}
	}

	~Impl() {
		std::vector<const DramSim*> systems;
		for (auto& d : dram_sims_) {
			systems.push_back(d.get());
		}
		DramSim::print_perf(systems.data(), systems.size());
	}

	const PerfStats& perf_stats() const {
		perf_stats_.bank_stalls = mem_xbar_->collisions();
		return perf_stats_;
	}

	void reset() {
		for (auto& d : dram_sims_) {
			d->reset();
		}
		pending_.clear();
	}

	void tick() {
		for (auto& d : dram_sims_) {
			d->tick();
		}

		for (uint32_t c = 0; c < config_.num_channels; ++c) {
			auto& dram = this->dram_of(c);
			uint32_t local = c % DramSim::channels_per_system();
			while (dram.rsp_valid(local)) {
				auto& p = pending_.at(dram.rsp_peek(local).tag);
				MemRsp mem_rsp{p.request.tag, p.request.hart_id, p.request.uuid};
				mem_rsp.data = p.rsp_data;
				if (!mem_xbar_->RspIn.at(c).try_send(mem_rsp, VX_CFG_DRAM_RSP_BOUNDARY_DELAY))
					break;
				DT(3, simobject_->name() << " mem-rsp" << c << ": " << mem_rsp);
				pending_.erase(dram.rsp_peek(local).tag);
				dram.rsp_pop(local);
			}
		}

		for (uint32_t c = 0; c < config_.num_channels; ++c) {
			if (mem_xbar_->ReqOut.at(c).empty())
				continue;

			auto& mem_req = mem_xbar_->ReqOut.at(c).peek();
			auto& dram = this->dram_of(c);
			uint32_t local = c % DramSim::channels_per_system();
			bool is_write = mem_req.is_write();
			if (!dram.can_accept(local, is_write))
				continue;

			std::shared_ptr<mem_block_t> rsp_data;
			if (ram_) {
				uint64_t line_addr = mem_req.addr & ~uint64_t(VX_CFG_MEM_BLOCK_SIZE - 1);
				// Cache fills/writebacks are simulator-internal traffic and
				// don't carry the kernel's intent (e.g. a write-back cache
				// reads a write-only buffer to fill the line on write-miss,
				// since memory buses lack per-region read/write permissions).
				// Suppress ACL for the duration; ACL still guards upload/download.
				ram_->enable_acl(false);
				if (is_write) {
					// Apply byte-enabled write to RAM at request arrival.
					// IO_COUT-range bytes are tapped to the print buffer and
					// not stored in RAM.
					if (mem_req.data) {
						for (uint32_t b = 0; b < VX_CFG_MEM_BLOCK_SIZE; ++b) {
							if (mem_req.byteen & (1ull << b)) {
								uint8_t value = (*mem_req.data)[b];
								ram_->write(&value, line_addr + b, 1);
							}
						}
					}
				} else {
					// Capture the line at request time; response carries it back.
					rsp_data = make_mem_block();
					ram_->read(rsp_data->data(), line_addr, VX_CFG_MEM_BLOCK_SIZE);
				}
				ram_->enable_acl(true);
			}

			if (pre_send_hook_) {
				pre_send_hook_(mem_req);
			}

			uint64_t block_addr = mem_req.addr & ~uint64_t(VX_CFG_MEM_BLOCK_SIZE - 1);
			uint64_t byteen = is_write ? mem_req.byteen : ~uint64_t(0);
			uint32_t sectors = DramSim::sector_mask(DramSim::local_addr(block_addr), VX_CFG_MEM_BLOCK_SIZE, byteen);
			if (sectors == 0) {
				// A write without enabled bytes moves no data.
				DT(3, simobject_->name() << " mem-req" << c << ": " << mem_req);
				mem_xbar_->ReqOut.at(c).pop();
				continue;
			}
			uint64_t id = next_id_++;
			if (!is_write) {
				pending_.emplace(id, Pending{mem_req, c, rsp_data});
			}
			dram.send(DramSim::Req{local, DramSim::local_addr(block_addr), sectors, is_write, id});

			DT(3, simobject_->name() << " mem-req" << c << ": " << mem_req);
			mem_xbar_->ReqOut.at(c).pop();
		}
	}

	void attach_ram(RAM* ram) {
		ram_ = ram;
	}

	void set_pre_send_hook(Memory::PreSendHook hook) {
		pre_send_hook_ = std::move(hook);
	}
};

///////////////////////////////////////////////////////////////////////////////

Memory::Memory(const SimContext& ctx, const char* name, const Config& config)
	: SimObject<Memory>(ctx, name)
	, mem_req_in(config.num_ports, this)
	, mem_rsp_out(config.num_ports, this)
	, impl_(new Impl(this, config))
{}

Memory::~Memory() {
  delete impl_;
}

void Memory::on_reset() {
  impl_->reset();
}

void Memory::on_tick() {
  impl_->tick();
}

void Memory::attach_ram(RAM* ram) {
  impl_->attach_ram(ram);
}

void Memory::set_pre_send_hook(PreSendHook hook) {
  impl_->set_pre_send_hook(std::move(hook));
}

const Memory::PerfStats &Memory::perf_stats() const {
	return impl_->perf_stats();
}