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

#include "local_mem.h"
#include "mem_block_pool.h"
#include <mem.h>
#include <bitmanip.h>
#include <vector>
#include <cstring>
#include <algorithm>
#include "types.h"
#if VX_CFG_EXT_A_ENABLED
#include "amo_ops.h"
#include "amo_unit.h"
#endif
#ifdef VX_CFG_EXT_MBAR_ENABLE
#include <iostream>
#include "mbar_ref.h"
#endif

using namespace vortex;

class LocalMem::Impl {
protected:
	LocalMem* simobject_;
	Config    config_;
	RAM       ram_;
	uint32_t 	addr_bits_;
#if VX_CFG_EXT_A_ENABLED
	AmoUnit   amo_unit_;
	// Per-bank atomic occupancy. A bank has a single port and the value read out
	// is registered before the update is computed, so an atomic owns its bank
	// for three cycles -- read, capture, write back -- and it accepts nothing
	// else meanwhile.
	static constexpr uint32_t AMO_BANK_STALL_CYCLES = 2;
	std::vector<uint32_t> amo_busy_;
	// A read issued the cycle after a write to the same word returns the word's
	// pre-update value, so the bank holds such a read off for one cycle. The
	// write-back is the third cycle of the occupancy, so this shadow covers the
	// fourth -- the cycle the next contending atomic arrives, which is why
	// back-to-back atomics on one word cost four cycles and not three.
	std::vector<uint64_t> amo_rdw_addr_;
	std::vector<bool>     amo_rdw_valid_;
	// Whether the atomic now occupying the bank ends in a write at all. A
	// load-reserved never writes and a failed store-conditional must not, so
	// neither leaves a shadow behind.
	std::vector<bool>     amo_rdw_pending_;
#endif
	MemCrossBar::Ptr mem_xbar_;
	mutable PerfStats perf_stats_;
#ifdef VX_CFG_EXT_DLMEM_ENABLE
	// Per bank, the cycle the last commit pulse lands.
	std::vector<uint64_t> commit_due_;
#endif
#ifdef VX_CFG_EXT_MBAR_ENABLE
	// Per bank, the cycle the last mbarrier event lands.
	std::vector<uint64_t> evt_due_;
#else
#line 64
#endif

	uint64_t to_local_addr(uint64_t addr) {
		return bit_getw(addr, 0, addr_bits_-1);
	}

#if VX_CFG_EXT_A_ENABLED
	// The word a bank is addressed by. Two byte addresses falling in one word
	// are the same access as far as the bank is concerned.
	uint64_t bank_word(uint64_t addr) {
		return to_local_addr(addr) >> log2ceil(config_.line_size);
	}
#endif

#ifdef VX_CFG_EXT_MBAR_ENABLE
	[[noreturn]] void mbar_abort(uint32_t bank, const MemReq& req, const char* msg,
	                             uint64_t word = 0, uint64_t operand = 0) {
		std::cerr << SimPlatform::instance().cycles() << ": " << simobject_->name() << "-bank" << bank
		          << " " << msg << ": op=" << req.op << " addr=0x" << std::hex << req.addr
		          << " byteen=0x" << req.byteen << " word=0x" << word << " operand=0x" << operand
		          << std::dec << " (#" << req.uuid << ")" << std::endl;
		std::abort();
	}
#endif

public:
	Impl(LocalMem* simobject, const Config& config)
		: simobject_(simobject)
		, config_(config)
		, ram_(config.capacity)
		, addr_bits_(log2ceil(config.capacity))
#if VX_CFG_EXT_A_ENABLED
		, amo_unit_(VX_CFG_AMO_RS_SIZE < 2 ? 2u : (uint32_t)VX_CFG_AMO_RS_SIZE)
		, amo_busy_(1 << config.B, 0)
		, amo_rdw_addr_(1 << config.B, 0)
		, amo_rdw_valid_(1 << config.B, false)
		, amo_rdw_pending_(1 << config.B, false)
#endif
#ifdef VX_CFG_EXT_DLMEM_ENABLE
		, commit_due_(1 << config.B, 0)
#endif
#ifdef VX_CFG_EXT_MBAR_ENABLE
		, evt_due_(1 << config.B, 0)
#else
#line 104
#endif
	{
		char sname[100];
		snprintf(sname, 100, "%s-xbar", simobject->name().c_str());
		uint32_t lg2_line_size = log2ceil(config_.line_size);
		uint32_t num_banks = 1 << config.B;
		// Round-robin across the LSU ports: a fixed priority lets a lower LSU
		// block hold a contended bank forever and starve the others.
		mem_xbar_ = MemCrossBar::Create(sname, ArbiterType::RoundRobin, ArbiterType::Priority,
		 config.num_reqs, num_banks,
		 [lg2_line_size, num_banks](const MemCrossBar::ReqType& req) {
    	// Custom logic to calculate the output index using bank interleaving
			return (uint32_t)((req.addr >> lg2_line_size) & (num_banks-1));
		});
		// The RTL serves DMA traffic (DXA writes, TCU tile-buffer reads) through
		// its DMA port ahead of the LSU at every bank.
		mem_xbar_->set_priority_inputs(config.dma_begin);
		for (uint32_t i = 0; i < config.num_reqs; ++i) {
			simobject->Inputs.at(i).bind(&mem_xbar_->ReqIn.at(i));
			mem_xbar_->RspOut.at(i).bind(&simobject->Outputs.at(i));
		}
#ifdef VX_CFG_EXT_DXA_ENABLE
		this->dxa_gate_init(simobject, num_banks);
#elif defined(VX_CFG_EXT_MBAR_ENABLE)
#line 133
#else
#line 124
#endif
	}

	virtual ~Impl() {}

	void reset() {
		perf_stats_ = PerfStats();
#if VX_CFG_EXT_A_ENABLED
		amo_unit_.reset();
		std::fill(amo_busy_.begin(), amo_busy_.end(), 0);
		std::fill(amo_rdw_valid_.begin(), amo_rdw_valid_.end(), false);
		std::fill(amo_rdw_pending_.begin(), amo_rdw_pending_.end(), false);
#endif
#ifdef VX_CFG_EXT_DLMEM_ENABLE
		std::fill(commit_due_.begin(), commit_due_.end(), 0);
#endif
#ifdef VX_CFG_EXT_MBAR_ENABLE
		std::fill(evt_due_.begin(), evt_due_.end(), 0);
#else
#line 139
#endif
#ifdef VX_CFG_EXT_DXA_ENABLE
		this->dxa_gate_reset();
#elif defined(VX_CFG_EXT_MBAR_ENABLE)
#line 153
#else
#line 139
#endif
	}

#ifdef VX_CFG_EXT_DLMEM_ENABLE
	// An access from an LSU port that the translation counted (flags.lacc; the
	// TCU metadata loads share port 0 and bypass it) is committed `delay`
	// cycles after the bank takes it: a load or a store at once, registered
	// (1); an atomic at its write-back, the third cycle of its occupancy,
	// registered (3). The crossbar appended the input port to the tag.
	void commit_pulse(uint32_t bank, const MemReq& req, uint32_t delay) {
		if (!req.flags.lacc || (req.tag & ((1u << log2ceil(config_.num_reqs)) - 1)) >= config_.commit_ports)
			return;
		constexpr uint32_t LOG_THREADS = log2ceil(VX_CFG_NUM_THREADS);
		constexpr uint32_t WID_MASK = (1u << log2ceil(VX_CFG_NUM_WARPS)) - 1;
		uint64_t due = SimPlatform::instance().cycles() + delay;
		__assert(due > commit_due_.at(bank), simobject_->name() << "-bank" << bank
		         << ": two commit pulses in one cycle");
		commit_due_.at(bank) = due;
		simobject_->CommitOut.at(bank).send((req.hart_id >> LOG_THREADS) & WID_MASK, delay);
	}
#endif

#ifdef VX_CFG_EXT_MBAR_ENABLE
	// The waiter table's view of an mbarrier op the bank just took: a try_wait
	// that read an incomplete phase parks its warp from the read stage, the
	// second cycle of the occupancy; a completed phase is seen at the
	// write-back, the third.
	void mbar_event(uint32_t bank, const MemReq& req, const mbar_ref::Result& r) {
		constexpr uint32_t LOG_THREADS = log2ceil(VX_CFG_NUM_THREADS);
		constexpr uint32_t WID_MASK = (1u << log2ceil(VX_CFG_NUM_WARPS)) - 1;
		MbarBankEvt evt{r.flip, 0, uint32_t(this->bank_word(req.addr) >> config_.B)};
		uint32_t delay = 2;
		if (!r.flip) {
			if (req.op != MemOp::MBAR_TRY_WAIT || r.ret_word != 0)
				return;
			evt.wid = (req.hart_id >> LOG_THREADS) & WID_MASK;
			delay = 1;
		}
		uint64_t due = SimPlatform::instance().cycles() + delay;
		__assert(due > evt_due_.at(bank), simobject_->name() << "-bank" << bank
		         << ": two mbarrier waiter events in one cycle");
		evt_due_.at(bank) = due;
		simobject_->MbarEvtOut.at(bank).send(evt, delay);
	}
#else
#line 159
#endif

#ifdef VX_CFG_EXT_DXA_ENABLE
	// A DXA write reaches the completion unit in the cycle its bank commits it.
	void dxa_commit(const MemReq& req) {
#ifdef DXA_MBAR_ENABLE
		// A peer's word: its copy completes through the DLMEM ingress.
		if (req.dxa.remote) {
			--row_words_;
			return;
		}
#else
#line 163
#endif
		DxaWrCommit c;
		c.tag    = req.dxa;
		c.notify = req.flags.dxa_notify_done;
		c.bar_id = req.flags.dxa_notify_bar_id;
		c.uuid   = req.uuid;
		simobject_->DxaWrOut.send(c, 0);
		if (c.notify) {
			++dxa_done_finals_;
		}
	}

	void dxa_free(uint32_t free) {
		dxa_free_ = free;
	}

private:
	static bool is_dxa_write(const MemReq& req) {
#ifdef DXA_MBAR_ENABLE
		if (req.dxa.remote)
			return false;
#else
#line 180
#endif
		return req.flags.dma && req.is_write();
	}

	static bool beat_before(uint32_t a, uint32_t b) {
		return int32_t(a - b) < 0;
	}

	// The RTL holds a beat that ends a copy at its DMA input while the
	// completion queue is full: the whole beat, and the beats behind it. A
	// write the crossbar forwards in cycle c commits at c+1, the RTL's accept
	// cycle, so the gate counts the entries the queue will hold in c+1.
	void dxa_gate_init(LocalMem* simobject, uint32_t num_banks) {
		simobject->DxaFreeIn.bind(simobject, &LocalMem::on_dxa_free);
		mem_xbar_->set_input_gate(
			[this](const std::vector<SimChannel<MemReq>>& inputs) { this->dxa_gate_begin(inputs); },
			[this](uint32_t input, const MemReq& req) { return this->dxa_gate_admit(input, req); });
		for (uint32_t b = 0; b < num_banks; ++b) {
			mem_xbar_->ReqOut.at(b).tx_callback([this](const MemReq& req, uint64_t) {
				if (is_dxa_write(req) && req.flags.dxa_notify_done) {
					++dxa_fwd_finals_;
				}
#ifdef DXA_MBAR_ENABLE
				if (req.flags.dma && req.dxa.remote) {
					++row_words_;
				}
#else
#line 201
#endif
			});
		}
	}

	void dxa_gate_reset() {
		dxa_free_        = UINT32_MAX;
		dxa_fwd_finals_  = 0;
		dxa_done_finals_ = 0;
		dxa_hold_        = false;
		dxa_hold_beat_   = 0;
#ifdef DXA_MBAR_ENABLE
		row_words_       = 0;
#else
#line 211
#endif
	}

	void dxa_gate_begin(const std::vector<SimChannel<MemReq>>& inputs) {
#ifdef DXA_MBAR_ENABLE
		dxa_inputs_ = &inputs;
#else
#line 214
#endif
		dxa_fwd_finals_ -= dxa_done_finals_;
		dxa_done_finals_ = 0;
		dxa_hold_ = false;
		if (dxa_free_ == UINT32_MAX)
			return;
		std::vector<uint32_t> finals;   // the final beats at the inputs, oldest first
		for (uint32_t i = config_.dma_begin; i < config_.num_reqs; ++i) {
			const auto& ch = inputs.at(i);
			if (ch.empty() || !is_dxa_write(ch.peek()) || !ch.peek().dxa.last)
				continue;
			const uint32_t beat = ch.peek().dxa.beat;
			auto it = std::find_if(finals.begin(), finals.end(),
			                       [beat](uint32_t f) { return !beat_before(f, beat); });
			if (it == finals.end() || *it != beat) {
				finals.insert(it, beat);
			}
		}
		const int64_t room = int64_t(dxa_free_) - int64_t(dxa_fwd_finals_);
		auto first_held = [&](int64_t n) -> const uint32_t* {
			const size_t k = size_t(std::max<int64_t>(n, 0));
			return (k < finals.size()) ? &finals[k] : nullptr;
		};
		const uint32_t* held = first_held(room);
#ifdef DXA_MBAR_ENABLE
		// The complete_tx at the queue head frees an entry if it wins its bank
		// this cycle, which a beat released for that entry must not take first.
		if (held != nullptr) {
			const uint32_t* next = first_held(room + 1);
			auto absent = [&](uint32_t i) {
				if (i < config_.dma_begin || next == nullptr || inputs.at(i).empty())
					return false;
				const auto& req = inputs.at(i).peek();
				return is_dxa_write(req) && !beat_before(req.dxa.beat, *next);
			};
			if (mem_xbar_->would_win(config_.dxa_compl_port, absent)) {
				held = next;
			}
		}
#endif
		if (held != nullptr) {
			dxa_hold_      = true;
			dxa_hold_beat_ = *held;
			DT(2, simobject_->name() << " dxa hold: beat=" << *held << ", free=" << dxa_free_
			   << ", forwarded finals=" << dxa_fwd_finals_);
		}
	}

	bool dxa_gate_admit(uint32_t input, const MemReq& req) const {
#ifdef DXA_MBAR_ENABLE
		// The row port is a DMA port; a unit without one leaves it at 0.
		if (input >= config_.dma_begin && input == config_.dxa_row_port
		 && (this->dxa_row_wait() || this->dxa_row_amo_wait(req)))
			return false;
		// A copy's completion follows its words into the banks.
		if (row_words_ != 0 && req.dxa.remote && req.op == MemOp::MBAR_COMPLETE_TX)
			return false;
#else
#line 262
#endif
		return !dxa_hold_ || input < config_.dma_begin || !is_dxa_write(req)
		    || beat_before(req.dxa.beat, dxa_hold_beat_);
	}

	uint32_t dxa_free_;         // the completion queue's free entries in the next cycle
	uint32_t dxa_fwd_finals_;   // final writes forwarded to a bank that dxa_free_ does not count yet
	uint32_t dxa_done_finals_;  // final writes committed since the gate last counted
	bool     dxa_hold_;
	uint32_t dxa_hold_beat_;    // the oldest beat held back this cycle
#ifdef DXA_MBAR_ENABLE

	// The RTL's DMA port takes one request a cycle, a peer's DXA word only when
	// no other DMA input presents one: the arbiter's last, priority input.
	bool dxa_row_wait() const {
		for (uint32_t i = config_.dma_begin; i < config_.dxa_row_port; ++i) {
			const auto& ch = dxa_inputs_->at(i);
			if (!ch.empty() && this->dxa_gate_admit(i, ch.peek()))
				return true;
		}
		return false;
	}

	// The RTL's DMA port refuses a write while the bank has an atomic in flight,
	// which keeps a peer's word at the ingress and the flits behind it waiting.
	// A word forwarded now meets its bank next cycle: the in-flight atomic must
	// have written back by then, and none may start this cycle.
	bool dxa_row_amo_wait(const MemReq& req) const {
		const uint32_t bank = uint32_t(req.addr >> log2ceil(config_.line_size)) & ((1u << config_.B) - 1);
		if (amo_busy_[bank] != 0)
			return amo_busy_[bank] > 1;
		const auto& queue = mem_xbar_->ReqOut.at(bank);
		return !queue.empty() && (memop_is_atomic(queue.peek().op) || memop_is_mbar(queue.peek().op));
	}

	const std::vector<SimChannel<MemReq>>* dxa_inputs_ = nullptr;
	uint32_t row_words_ = 0;    // a peer's words forwarded to a bank and not yet committed
#else
#line 271
#endif

public:
#elif defined(VX_CFG_EXT_MBAR_ENABLE)
#line 200
#else
#line 160
#endif
	void tick() {
		// process bank requets from xbar
		uint32_t num_banks = (1 << config_.B);
		for (uint32_t i = 0; i < num_banks; ++i) {
#if VX_CFG_EXT_A_ENABLED
			// Cycles an in-flight atomic owns; the bank serves nothing else.
			if (amo_busy_[i] != 0) {
				if (0 == --amo_busy_[i]) {
					amo_rdw_valid_[i] = amo_rdw_pending_[i]; // write-back just issued
				}
				continue;
			}
			// The shadow is one cycle wide: consume it here whether or not
			// anything is waiting on the bank.
			const bool     rdw_shadow = amo_rdw_valid_[i];
			const uint64_t rdw_addr   = amo_rdw_addr_[i];
			amo_rdw_valid_[i] = false;
#endif
			auto& xbar_req_out = mem_xbar_->ReqOut.at(i);
			if (xbar_req_out.empty())
				continue;

			auto& bank_req = xbar_req_out.peek();

#if VX_CFG_EXT_A_ENABLED
			// A read of the word the write-back just stored waits a cycle. An
			// atomic reads the bank for its old value, so it counts as a reader
			// here even though the request carries a write flag; that is what
			// makes contending atomics on one word cost four cycles.
			const bool reads_bank = !bank_req.is_write() || memop_is_atomic(bank_req.op);
			if (rdw_shadow
			 && reads_bank
			 && this->bank_word(bank_req.addr) == rdw_addr) {
				continue;
			}

			// Shared-memory atomics: read-modify-write in place and ALWAYS
			// return the old word. An AMO has a destination register, so the
			// LSU reserved a response slot; a missing response hangs the warp
			// (this was the prior behaviour — LMEM had no AMO path). LR/SC
			// reservations are tracked per hart for correct contended
			// compare-swap, mirroring the dcache AMO path used for global/SSBO.
#ifdef VX_CFG_EXT_MBAR_ENABLE
			if (memop_is_atomic(bank_req.op) || memop_is_mbar(bank_req.op)) {
#else
			if (memop_is_atomic(bank_req.op)) {
#endif
				const uint32_t bmask   = VX_CFG_MEM_BLOCK_SIZE - 1;
				const uint64_t la      = to_local_addr(bank_req.addr);
				const uint32_t byte_off = (uint32_t)(la & bmask);
				const uint32_t wbytes  = bank_req.byteen
					? (uint32_t)__builtin_popcountll(bank_req.byteen) : 4u;
				const uint8_t  width   = (wbytes >= 8) ? 3 : 2;
				const MemOp    op      = bank_req.op;
				const bool     is_lr   = (op == MemOp::AMO_LR);
				const bool     is_sc   = (op == MemOp::AMO_SC);
				const bool     sc_fail = is_sc && !amo_unit_.check(bank_req.hart_id, la);
#ifdef VX_CFG_EXT_MBAR_ENABLE
				// A wait writes nothing, so it is no commit and leaves no shadow, like a load-reserved.
				const bool     is_mbar = memop_is_mbar(op);
				const bool     is_wait = is_mbar && mbar_ref::is_wait(mbar_op_code(op));
				const bool     do_store = !is_lr && !sc_fail && !is_wait;
				if (is_mbar && ((byte_off & 7) != 0 || bank_req.byteen != (0xffull << byte_off))) {
					this->mbar_abort(i, bank_req, mbar_ref::MSG_ALIGN);
				}
#else
				const bool     do_store = !is_lr && !sc_fail;
#endif

				uint8_t obuf[8] = {0};
				ram_.read(obuf, la, wbytes);
				const uint64_t old_word = amo_load_word(obuf, 0, width);
				const uint64_t rhs = bank_req.data
					? amo_load_word(bank_req.data->data(), byte_off, width) : 0;
				auto rmw = amo_compute(op, width, old_word, rhs,
				                       bank_req.flags.amo_unsigned,
				                       bank_req.amo_cmp);
#ifdef VX_CFG_EXT_MBAR_ENABLE
				if (is_mbar) {
					const uint32_t err = mbar_ref::apply(mbar_op_code(op), old_word, rhs).err;
					if (err & mbar_ref::ERR_UNDERFLOW) {
						this->mbar_abort(i, bank_req, mbar_ref::MSG_UNDERFLOW, old_word, rhs);
					}
					if (err & mbar_ref::ERR_TX_RANGE) {
						this->mbar_abort(i, bank_req, mbar_ref::MSG_TX_RANGE, old_word, rhs);
					}
				}
#endif

				// Build the response before mutating any state so a full RspIn
				// can stall-retry cleanly.
				MemRsp bank_rsp{bank_req.tag, bank_req.hart_id, bank_req.uuid};
				auto rsp_data = make_mem_block();
				std::memset(rsp_data->data(), 0, rsp_data->size());
				const uint64_t ret_word = is_sc ? (sc_fail ? 1ull : 0ull) : rmw.ret_word;
				amo_store_word(rsp_data->data(), byte_off, width, ret_word);
				bank_rsp.data = rsp_data;
				if (!mem_xbar_->RspIn.at(i).try_send(bank_rsp))
					continue; // stall; no state mutated yet

				// An SC gives up its own reservation whether it succeeds or
				// fails; an LR claims one. Everything else is a plain RMW,
				// whose commit breaks the other harts' reservations below.
				if (is_lr)      amo_unit_.reserve(bank_req.hart_id, la);
				else if (is_sc) amo_unit_.clear(bank_req.hart_id, la);
				if (do_store) {
					uint8_t sbuf[8];
					amo_store_word(sbuf, 0, width, rmw.new_word);
					ram_.write(sbuf, la, wbytes);
					// Any committed store breaks every other hart's reservation
					// on the word. Without this a second hart's stale
					// reservation survives the first hart's successful SC, so
					// both report success and one update is lost.
					amo_unit_.invalidate(la, bank_req.hart_id);
				}
				perf_stats_.reads  += !do_store;
				perf_stats_.writes += do_store;
				DT(4, simobject_->name() << "-bank" << i << " amo : " << bank_req);
				amo_rdw_addr_[i]    = this->bank_word(bank_req.addr);
				amo_rdw_pending_[i] = do_store;
#ifdef VX_CFG_EXT_DLMEM_ENABLE
				this->commit_pulse(i, bank_req, AMO_BANK_STALL_CYCLES + 1);
#endif
#ifdef VX_CFG_EXT_MBAR_ENABLE
				if (is_mbar) {
					this->mbar_event(i, bank_req, mbar_ref::apply(mbar_op_code(op), old_word, rhs));
				}
#else
#line 283
#endif
				xbar_req_out.pop();
				amo_busy_[i] = AMO_BANK_STALL_CYCLES;   // capture, then write back
				continue;
			}
#endif

			// Apply byte-enabled writes from TLM payload to local RAM.
			if (bank_req.is_write() && bank_req.data) {
				uint64_t line_addr = to_local_addr(bank_req.addr) & ~uint64_t(VX_CFG_MEM_BLOCK_SIZE - 1);
				for (uint32_t b = 0; b < VX_CFG_MEM_BLOCK_SIZE; ++b) {
					if (bank_req.byteen & (1ull << b)) {
						uint8_t value = (*bank_req.data)[b];
						ram_.write(&value, line_addr + b, 1);
					}
				}
#if VX_CFG_EXT_A_ENABLED
				if (bank_req.flags.dma) {
					// A DMA write is another device's write, so it breaks every
					// hart's reservation on each bank word it writes, as the RTL does.
					const uint64_t word_mask = config_.line_size - 1;
					const uint64_t word_en   = (uint64_t(1) << config_.line_size) - 1;
					const uint64_t byteen    = bank_req.byteen;
					amo_unit_.invalidate_if([&](uint64_t addr) {
						if ((addr & ~uint64_t(VX_CFG_MEM_BLOCK_SIZE - 1)) != line_addr) {
							return false;
						}
						const uint64_t word_off = (addr & ~word_mask) - line_addr;
						return ((byteen >> word_off) & word_en) != 0;
					});
				} else {
					// A plain store breaks a reservation on the word it writes, the
					// same as an atomic's commit does; otherwise a store between a
					// hart's load-reserved and its store-conditional would go
					// unnoticed and the conditional would wrongly succeed.
					amo_unit_.invalidate(to_local_addr(bank_req.addr), bank_req.hart_id);
				}
#endif
			}

			// Loads always respond. Stores respond when configured globally OR
			// the request opts in via MEM_FLAG_STRSP.
			if (!bank_req.is_write() || config_.write_reponse || bank_req.flags.strsp) {
				// send xbar response — for reads, capture the line payload.
				MemRsp bank_rsp{bank_req.tag, bank_req.hart_id, bank_req.uuid};
				if (!bank_req.is_write()) {
					auto rsp_data = make_mem_block();
					uint64_t line_addr = to_local_addr(bank_req.addr) & ~uint64_t(VX_CFG_MEM_BLOCK_SIZE - 1);
					ram_.read(rsp_data->data(), line_addr, VX_CFG_MEM_BLOCK_SIZE);
					bank_rsp.data = rsp_data;
				}
				if (!mem_xbar_->RspIn.at(i).try_send(bank_rsp))
					continue; // stall
			}

			DT(4, simobject_->name() << "-bank" << i << " req : " << bank_req);

			// update perf counters
			perf_stats_.reads += !bank_req.is_write();
			perf_stats_.writes += bank_req.is_write();
#ifdef VX_CFG_EXT_DXA_ENABLE
			if (bank_req.flags.dma && bank_req.is_write()) {
				this->dxa_commit(bank_req);
			}
#elif defined(VX_CFG_EXT_MBAR_ENABLE)
#line 387
#elif VX_CFG_EXT_A_ENABLED
#line 340
#else
#line 349
#endif

#ifdef VX_CFG_EXT_DLMEM_ENABLE
			// The store is written and the load has read its data now.
			if (bank_req.op == MemOp::ST || bank_req.op == MemOp::LD) {
				this->commit_pulse(i, bank_req, 1);
			}
#endif

			// remove input
			xbar_req_out.pop();
		}
	}

	const PerfStats& perf_stats() const {
		perf_stats_.bank_stalls = mem_xbar_->collisions();
		return perf_stats_;
	}

};

///////////////////////////////////////////////////////////////////////////////

LocalMem::LocalMem(const SimContext& ctx, const char* name, const Config& config)
	: SimObject<LocalMem>(ctx, name)
	, Inputs(config.num_reqs, this)
	, Outputs(config.num_reqs, this)
#ifdef VX_CFG_EXT_DLMEM_ENABLE
	, CommitOut(1 << config.B, this)
#endif
#ifdef VX_CFG_EXT_MBAR_ENABLE
	, MbarEvtOut(1 << config.B, this)
#else
#line 371
#endif
#ifdef VX_CFG_EXT_DXA_ENABLE
	, DxaWrOut(this, 2u << config.B)
	, DxaFreeIn(this)
#elif defined(VX_CFG_EXT_MBAR_ENABLE)
#line 423
#else
#line 371
#endif
	, impl_(new Impl(this, config))
{}

LocalMem::~LocalMem() {
  delete impl_;
}

void LocalMem::on_reset() {
  impl_->reset();
}

void LocalMem::on_tick() {
  impl_->tick();
}

const LocalMem::PerfStats& LocalMem::perf_stats() const {
  return impl_->perf_stats();
}

#ifdef VX_CFG_EXT_DXA_ENABLE
void LocalMem::on_dxa_free(const uint32_t& free) {
  impl_->dxa_free(free);
}
#endif
