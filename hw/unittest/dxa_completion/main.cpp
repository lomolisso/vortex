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

// VX_dxa_completion testbench. The unit queues the last DMA beat of each DXA
// transfer and turns it into a barrier release (legacy mode) or an mbarrier
// complete_tx request into the local memory (mbarrier mode), in arrival order.
// Directed scenarios pin the cycle of every output (a last beat accepted in
// cycle T is at the queue head in T+1): one transfer of each mode, beats that
// are not the last, a burst of last beats in consecutive cycles (multicast
// releases), the two modes interleaved behind a stalled head, and a full queue,
// which raises `full` so the DMA port holds the next last beat until an entry
// leaves. A random run then checks every cycle against a model of the queue,
// with the gate applied as VX_mem_unit applies it. --neg=retired|gen|compl|bounds
// sends a beat or a complete_tx for a freed slot, a re-admitted slot, a slot
// freed while its complete_tx waits, and an mbarrier outside the issuing
// block; the unit's assert must abort the run.

#include "vl_simulator.h"
#include "VVX_dxa_completion_top.h"
#include "VX_config.h"

#include <cinttypes>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <random>
#include <string>

constexpr uint32_t NUM_WARPS = VX_CFG_NUM_WARPS;
constexpr uint32_t NUM_BANKS = VX_CFG_LMEM_NUM_BANKS;
constexpr uint32_t LMEM_SIZE = 1u << VX_CFG_LMEM_LOG_SIZE;
constexpr uint32_t DEPTH     = (NUM_WARPS < 2) ? 2 : NUM_WARPS;
constexpr uint32_t GEN_MASK  = 0xf;              // DXA_CTA_GEN_W
constexpr uint32_t OP_COMPLETE_TX = 0xd;         // AMO_OP_MBAR_COMPLETE_TX
static_assert(NUM_WARPS >= 2, "the scenarios use two slots");
constexpr uint32_t clog2(uint32_t n) { return (n <= 1) ? 0 : 1 + clog2((n + 1) / 2); }
// BAR_ADDR_W: the barrier address of a legacy release is {warp, barrier}.
constexpr uint32_t BAR_BITS  = clog2(NUM_WARPS) + clog2(VX_CFG_NUM_BARRIERS);
constexpr uint32_t BAR_MASK  = (BAR_BITS == 0) ? 1 : ((1u << BAR_BITS) - 1);

static_assert(NUM_WARPS <= 8, "slot generations are packed in 64 bits");
static_assert(NUM_BANKS <= 32, "bank fires are packed in 32 bits");

static std::string strf(const char* fmt, ...) {
  char buf[512];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  return buf;
}

struct Beat {
  bool     valid   = false;
  bool     last    = false;
  bool     mode    = false;   // 1: mbarrier
  uint32_t payload = 0;       // legacy barrier address, or mbarrier word offset
  uint32_t bytes   = 0;
  uint32_t slot    = 0;
  uint32_t gen     = 0;
  uint32_t banks   = 1;       // banks the beat writes
};

struct Inputs {
  Beat beat;
  bool txbar_ready = true;
  bool compl_ready = true;
};

class Bench {
public:
  Bench() {
    auto* t = sim_.operator->();
    for (uint32_t s = 0; s < NUM_WARPS; ++s) {
      valid_ |= 1u << s;
    }
    t->reset = 1;
    this->drive(Inputs{});
    for (int i = 0; i < 4; ++i) {
      tick_ = sim_.step(tick_, 2);
    }
    t->reset = 0;
  }

  uint32_t stride() const { return LMEM_SIZE / NUM_WARPS; }

  // A beat of the issuing slot, valid for the current slot state.
  Beat beat(uint32_t slot, bool last, bool mode, uint32_t payload, uint32_t bytes) const {
    Beat b;
    b.valid   = true;
    b.last    = last;
    b.mode    = mode;
    b.payload = payload;
    b.bytes   = bytes;
    b.slot    = slot;
    b.gen     = gen_[slot];
    return b;
  }

  // An mbarrier word offset inside the slot's block.
  uint32_t mbar_off(uint32_t slot, uint32_t word) const {
    return (slot * this->stride()) / 8 + (word % (this->stride() / 8));
  }

  bool full() {
    auto* t = sim_.operator->();
    t->eval();
    return t->full;
  }

  // One cycle. The beat's last flag must already respect `full` (the gate).
  void cycle(const Inputs& in) {
    auto* t = sim_.operator->();
    this->drive(in);
    t->eval();

    const bool head_legacy = !q_.empty() && !q_.front().mode;
    const bool head_mbar   = !q_.empty() && q_.front().mode;
    if (bool(t->full) != (q_.size() == DEPTH)) {
      this->fail(strf("cycle %" PRIu64 ": full=%d with %zu queued", cycle_, t->full, q_.size()));
    }
    if (bool(t->txbar_valid) != head_legacy) {
      this->fail(strf("cycle %" PRIu64 ": txbar_valid=%d, model head %s", cycle_, t->txbar_valid, this->head_str().c_str()));
    } else if (head_legacy) {
      if (t->txbar_addr != q_.front().payload || !t->txbar_is_done) {
        this->fail(strf("cycle %" PRIu64 ": txbar addr=0x%x done=%d, expected 0x%x", cycle_,
                        t->txbar_addr, t->txbar_is_done, q_.front().payload));
      }
    }
    if (bool(t->compl_valid) != head_mbar) {
      this->fail(strf("cycle %" PRIu64 ": compl_valid=%d, model head %s", cycle_, t->compl_valid, this->head_str().c_str()));
    } else if (head_mbar) {
      const Beat& h = q_.front();
      if (t->compl_rw != 0 || t->compl_addr != h.payload || t->compl_data != h.bytes
       || t->compl_byteen != 0xff || !t->compl_amo_valid || t->compl_amo_op != OP_COMPLETE_TX
       || t->compl_amo_unsigned || t->compl_hart_id != 0 || t->compl_is_lacc) {
        this->fail(strf("cycle %" PRIu64 ": complete_tx rw=%d addr=0x%" PRIx64 " data=0x%" PRIx64
                        " byteen=0x%x amo=%d op=%u uns=%d hart=%u lacc=%d, expected addr=0x%x bytes=%u",
                        cycle_, t->compl_rw, (uint64_t)t->compl_addr, (uint64_t)t->compl_data,
                        t->compl_byteen, t->compl_amo_valid, t->compl_amo_op, t->compl_amo_unsigned,
                        t->compl_hart_id, t->compl_is_lacc, h.payload, h.bytes));
      }
    }
    if (!t->compl_rsp_ready) {
      this->fail(strf("cycle %" PRIu64 ": the complete_tx response is not swallowed", cycle_));
    }

    // Clock edge.
    if ((head_legacy && in.txbar_ready) || (head_mbar && in.compl_ready)) {
      q_.pop_front();
      ++pops_;
      if (head_mbar) {
        ++completes_;
      } else {
        ++releases_;
      }
    }
    if (in.beat.valid && in.beat.last) {
      if (q_.size() == DEPTH) {
        this->fail(strf("cycle %" PRIu64 ": bench pushed into a full queue", cycle_));
      }
      q_.push_back(in.beat);
    }
    tick_ = sim_.step(tick_, 2);
    ++cycle_;
  }

  void idle(uint32_t n, bool txbar_ready = true, bool compl_ready = true) {
    for (uint32_t i = 0; i < n; ++i) {
      Inputs in;
      in.txbar_ready = txbar_ready;
      in.compl_ready = compl_ready;
      this->cycle(in);
    }
  }

  // Outputs at the start of the next cycle (no edge).
  bool txbar_valid()  { auto* t = sim_.operator->(); t->eval(); return t->txbar_valid; }
  bool compl_valid()  { auto* t = sim_.operator->(); t->eval(); return t->compl_valid; }

  void set_slot(uint32_t slot, bool valid) {
    valid_ = valid ? (valid_ | (1u << slot)) : (valid_ & ~(1u << slot));
  }
  void readmit(uint32_t slot) {
    gen_[slot] = (gen_[slot] + 1) & GEN_MASK;
  }

  void fail(const std::string& msg) {
    ++failures_;
    if (failures_ <= 50) {
      std::printf("FAIL %s\n", msg.c_str());
    }
  }

  size_t queued() const { return q_.size(); }
  uint32_t failures() const { return failures_; }
  uint64_t now() const { return cycle_; }
  uint64_t releases() const { return releases_; }
  uint64_t completes() const { return completes_; }

private:
  std::string head_str() const {
    if (q_.empty()) return "empty";
    return strf("%s 0x%x", q_.front().mode ? "mbar" : "legacy", q_.front().payload);
  }

  void drive(const Inputs& in) {
    auto* t = sim_.operator->();
    uint32_t fire = 0;
    if (in.beat.valid) {
      fire = in.beat.banks & ((NUM_BANKS >= 32) ? ~0u : ((1u << NUM_BANKS) - 1));
      if (fire == 0) fire = 1;
    }
    t->bank_wr_fire = fire;
    t->in_last      = in.beat.valid && in.beat.last;
    t->in_mode      = in.beat.mode;
    t->in_payload   = in.beat.payload;
    t->in_tx_bytes  = in.beat.bytes;
    t->in_slot      = in.beat.slot;
    t->in_gen       = in.beat.gen;
    uint64_t gens = 0;
    for (uint32_t s = 0; s < NUM_WARPS; ++s) {
      gens |= uint64_t(gen_[s]) << (8 * s);
    }
    t->slot_valid  = valid_;
    t->slot_gen    = gens;
    t->slot_stride = this->stride();
    t->txbar_ready = in.txbar_ready;
    t->compl_ready = in.compl_ready;
    t->compl_rsp_valid = 0;
  }

  vl_simulator<VVX_dxa_completion_top> sim_;
  std::deque<Beat> q_;
  uint32_t valid_ = 0;
  uint32_t gen_[NUM_WARPS] = {};
  uint64_t tick_ = 0;
  uint64_t cycle_ = 0;
  uint64_t pops_ = 0;
  uint64_t releases_ = 0;
  uint64_t completes_ = 0;
  uint32_t failures_ = 0;
};

///////////////////////////////////////////////////////////////////////////////

static void expect(Bench& tb, bool cond, const char* what) {
  if (!cond) {
    tb.fail(strf("cycle %" PRIu64 ": %s", tb.now(), what));
  }
}

static uint32_t directed(Bench& tb) {
  uint32_t n = 0;

  // A legacy transfer: the release is at the queue head one cycle after the last beat.
  {
    Inputs in;
    in.beat = tb.beat(0, false, false, 3, 0);
    tb.cycle(in);
    expect(tb, !tb.txbar_valid() && !tb.compl_valid(), "a beat that is not the last queued an entry");
    in.beat = tb.beat(0, true, false, 3, 0);
    tb.cycle(in);
    expect(tb, tb.txbar_valid() && !tb.compl_valid(), "legacy release not at the head one cycle after the last beat");
    tb.idle(1);
    expect(tb, !tb.txbar_valid(), "legacy release not taken");
    ++n;
  }

  // An mbarrier transfer: complete_tx at the head one cycle after the last beat.
  {
    Inputs in;
    in.beat = tb.beat(1, false, true, tb.mbar_off(1, 5), 4096);
    tb.cycle(in);
    in.beat = tb.beat(1, true, true, tb.mbar_off(1, 5), 4096);
    tb.cycle(in);
    expect(tb, tb.compl_valid() && !tb.txbar_valid(), "complete_tx not at the head one cycle after the last beat");
    tb.idle(3, true, false);
    expect(tb, tb.compl_valid(), "complete_tx dropped while the local memory stalls");
    tb.idle(1);
    expect(tb, !tb.compl_valid(), "complete_tx not taken");
    ++n;
  }

  // A burst of last beats in consecutive cycles, as multicast receivers get them.
  {
    for (uint32_t r = 0; r < DEPTH; ++r) {
      Inputs in;
      in.txbar_ready = (r % 2) == 0;
      in.beat = tb.beat(0, true, false, (r * 8 + 1) & BAR_MASK, 0);
      tb.cycle(in);
    }
    tb.idle(2 * DEPTH, false);
    tb.idle(2 * DEPTH);
    expect(tb, tb.queued() == 0, "burst not drained");
    ++n;
  }

  // The two modes interleaved: a stalled legacy head holds the complete_tx behind it.
  {
    Inputs in;
    in.txbar_ready = false;
    in.beat = tb.beat(0, true, false, 2, 0);
    tb.cycle(in);
    in.beat = tb.beat(2 % NUM_WARPS, true, true, tb.mbar_off(2 % NUM_WARPS, 0), 64);
    tb.cycle(in);
    const size_t n_in = (DEPTH > 2) ? 3 : 2;
    if (n_in == 3) {
      in.beat = tb.beat(0, true, false, 6, 0);
      tb.cycle(in);
    }
    tb.idle(3, false, true);
    expect(tb, tb.queued() == n_in && !tb.compl_valid(), "complete_tx passed a stalled release");
    tb.idle(1, true, false);
    expect(tb, tb.compl_valid(), "complete_tx not at the head after the release");
    tb.idle(2, true, false);
    tb.idle(3);
    expect(tb, tb.queued() == 0, "interleaved entries not drained");
    ++n;
  }

  // A full queue: `full` holds the next last beat at the DMA port until an entry leaves.
  {
    for (uint32_t r = 0; r < DEPTH; ++r) {
      Inputs in;
      in.compl_ready = false;
      in.beat = tb.beat(r % NUM_WARPS, true, true, tb.mbar_off(r % NUM_WARPS, r), 128 + r);
      tb.cycle(in);
    }
    expect(tb, tb.full(), "queue of DEPTH entries not full");
    Beat held = tb.beat(0, true, false, 7, 0);
    uint32_t waited = 0;
    for (uint32_t c = 0; c < 4 * DEPTH && tb.full(); ++c) {
      Inputs in;
      in.compl_ready = (c >= 3);
      tb.cycle(in);   // the DMA port holds `held`
      ++waited;
    }
    expect(tb, waited == 4, "full did not drop on the cycle after the first complete_tx left");
    Inputs in;
    in.beat = held;
    tb.cycle(in);
    tb.idle(2 * DEPTH + 2);
    expect(tb, tb.queued() == 0, "held beat lost or queue not drained");
    ++n;
  }
  return n;
}

static void random_run(Bench& tb, uint32_t seed, uint32_t cycles) {
  std::mt19937 rng(seed);
  uint32_t held = 0;
  for (uint32_t c = 0; c < cycles; ++c) {
    Inputs in;
    // Bursts of back-pressure on each output.
    in.txbar_ready = (rng() % 8) != 0 && ((c / 64) % 5 != 1);
    in.compl_ready = (rng() % 4) != 0 && ((c / 64) % 7 != 3);
    if (rng() % 3 != 0) {
      uint32_t slot = rng() % NUM_WARPS;
      bool mode = rng() % 2;
      bool last = (rng() % 3) == 0;
      uint32_t payload = mode ? tb.mbar_off(slot, rng()) : (rng() & BAR_MASK);
      Beat b = tb.beat(slot, last, mode, payload, 1 + rng() % LMEM_SIZE);
      b.banks = rng();
      if (last && tb.full()) {
        ++held;           // the DMA port holds it; the engine retries later
      } else {
        in.beat = b;
      }
    }
    tb.cycle(in);
  }
  tb.idle(4 * DEPTH);
  if (tb.queued() != 0) {
    tb.fail(strf("random run (seed %u): %zu entries left", seed, tb.queued()));
  }
  std::printf("  random seed=%u: %u cycles, %u last beats held by a full queue\n", seed, cycles, held);
}

static int negative(Bench& tb, const std::string& neg) {
  Inputs in;
  if (neg == "retired") {
    // The issuing block left its slot before the transfer's data landed.
    in.beat = tb.beat(1, false, true, tb.mbar_off(1, 0), 64);
    tb.set_slot(1, false);
  } else if (neg == "gen") {
    // The slot was freed and taken again by another block since the issue.
    in.beat = tb.beat(NUM_WARPS - 1, false, false, 1, 0);
    tb.readmit(NUM_WARPS - 1);
  } else if (neg == "compl") {
    // The last beat landed while the block was live; it left before its complete_tx.
    in.compl_ready = false;
    in.beat = tb.beat(1, true, true, tb.mbar_off(1, 0), 64);
    tb.cycle(in);
    tb.set_slot(1, false);
    in = Inputs{};
  } else if (neg == "bounds") {
    // The mbarrier is in the next block's slot.
    in.beat = tb.beat(0, true, true, tb.mbar_off(1, 0), 64);
    tb.cycle(in);
    in = Inputs{};
  } else {
    std::printf("unknown --neg=%s\n", neg.c_str());
    return 2;
  }
  for (int i = 0; i < 4; ++i) {
    tb.cycle(in);
    in = Inputs{};
  }
  std::printf("the unit took --neg=%s without asserting\n", neg.c_str());
  return 1;
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  uint32_t seed = 1;
  uint32_t cycles = 20000;
  std::string neg;
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    if (std::strncmp(a, "--seed=", 7) == 0) {
      seed = static_cast<uint32_t>(std::strtoul(a + 7, nullptr, 0));
    } else if (std::strncmp(a, "--cycles=", 9) == 0) {
      cycles = static_cast<uint32_t>(std::strtoul(a + 9, nullptr, 0));
    } else if (std::strncmp(a, "--neg=", 6) == 0) {
      neg = a + 6;
    } else if (a[0] == '+') {
      continue;
    } else {
      std::printf("usage: %s [--seed=N] [--cycles=N] [--neg=retired|gen|compl|bounds]\n", argv[0]);
      return 2;
    }
  }

  std::printf("dxa_completion: warps=%u banks=%u lmem=%u depth=%u seed=%u\n",
              NUM_WARPS, NUM_BANKS, LMEM_SIZE, DEPTH, seed);
  Bench tb;
  if (!neg.empty()) {
    return negative(tb, neg);
  }
  uint32_t n = directed(tb);
  for (uint32_t s = 0; s < 3; ++s) {
    random_run(tb, seed * 7919u + s, cycles);
  }
  std::printf("%u scenarios, %" PRIu64 " releases, %" PRIu64 " complete_tx, %" PRIu64 " cycles\n",
              n, tb.releases(), tb.completes(), tb.now());
  if (tb.failures() != 0) {
    std::printf("FAILED: %u check(s) failed\n", tb.failures());
    return 1;
  }
  std::printf("PASSED\n");
  return 0;
}
