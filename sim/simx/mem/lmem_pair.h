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

#include <cstdint>
#include <vector>

namespace vortex {

// A lane word spans PAIR banks of one row, as VX_local_mem pairs its SRAMs:
// an access of the even bank that reaches the upper half holds the odd bank
// in its cycle, and an atomic's write-back to the upper half holds the other
// bank's stores there in the write-back cycle.
class LmemPairHold {
public:
	static constexpr uint32_t PAIR = (VX_CFG_XLEN / 8) / VX_CFG_LMEM_BANK_BYTES;
	static constexpr uint64_t HI_MASK = (PAIR > 1)
		? (((uint64_t(1) << VX_CFG_LMEM_BANK_BYTES) - 1) << VX_CFG_LMEM_BANK_BYTES) : 0;
	static_assert(PAIR == 1 || PAIR == 2, "a lane word spans one or two banks");

	// The byte enables cover the whole block, so the upper half is that of the
	// lane word the address falls in.
	static bool reaches_hi(uint64_t addr, uint64_t byteen) {
		const uint32_t word_off = uint32_t(addr & (VX_CFG_MEM_BLOCK_SIZE - 1)) & ~uint32_t(VX_CFG_XLEN / 8 - 1);
		return PAIR > 1 && ((byteen >> word_off) & HI_MASK) != 0;
	}

	void reset(uint32_t num_banks) {
		taken_hi_at_.assign(num_banks, ~uint64_t(0));
		wb_hi_at_.assign(num_banks, ~uint64_t(0));
	}

	void take(uint32_t bank, uint64_t addr, uint64_t byteen, uint64_t now) {
		if ((bank & 1) == 0 && reaches_hi(addr, byteen))
			taken_hi_at_[bank] = now;
	}

	void write_back(uint32_t bank, bool store, uint64_t addr, uint64_t byteen, uint64_t at) {
		wb_hi_at_[bank] = (store && reaches_hi(addr, byteen)) ? at : ~uint64_t(0);
	}

	bool held(uint32_t bank, bool store, uint64_t addr, uint64_t byteen, uint64_t now) const {
		if (PAIR == 1)
			return false;
		if (bank & 1)
			return taken_hi_at_[bank - 1] == now || (store && wb_hi_at_[bank - 1] == now);
		return store && reaches_hi(addr, byteen) && wb_hi_at_[bank + 1] == now;
	}

private:
	std::vector<uint64_t> taken_hi_at_;
	std::vector<uint64_t> wb_hi_at_;
};

}
