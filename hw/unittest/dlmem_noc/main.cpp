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

// DLMEM network testbench: one cluster's socket switches plus the cluster
// crossbar, driven from every core's injection port and drained at every
// ejection port.
//
// --mode=random: uniform random traffic (self included) with random gaps,
// hot spots and permutations, under ejection back-pressure phases (long stalls,
// random, always ready). Every packet must arrive exactly once, at its
// destination, with its payload intact and in injection order per
// (source, destination); a hot spot under always-ready sinks must not starve
// any source; a no-progress watchdog catches hangs.
//
// --mode=latency: the fixed schedule below with always-ready sinks; writes
// latency.csv (src,dst,seq,inject_cycle,deliver_cycle) for comparison with the
// SimX model's run of the same schedule (compare_latency.py).
//
// Latency schedule. Cycle 0 is the first cycle after reset. Each source keeps a
// FIFO of (dst, release) packets and offers its head from cycle `release` on,
// holding it until the network accepts it; seq numbers a source's packets from
// 0 in schedule order. inject_cycle is the cycle the injection handshake fires,
// deliver_cycle the cycle the ejection handshake fires.
//   0. reset:       every source sends one packet to core 0, then one to core
//                   NCC-1, released at 0, or at 2 when the destination is in
//                   its own socket, so that local and remote packets meet at
//                   the destination's switch: the first grants after reset are
//                   contended ones.
//   1. isolated:    T1 = 8*NCC + 64; for k in [0, NCC*NCC): source k / NCC
//                   sends one packet to core k % NCC, released at T1 + 8*k.
//   2. permutation: T2 = T1 + 8*NCC*NCC + 64; every source s sends 16 packets,
//                   the j-th to core (s + j + 1) % NCC, all released at T2.
//   3. hot spot:    T3 = T2 + 64*NCC + 64; every source sends 8 packets to core
//                   0, then 8 to core NCC-1, all released at T3.

#include "vl_simulator.h"
#include "VVX_dlmem_noc_top.h"
#include "VX_config.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#ifndef NOC_DATAW
#error "NOC_DATAW must come from the Makefile's DATAW"
#endif

static constexpr uint32_t clog2(uint32_t n) {
  uint32_t r = 0;
  while ((1u << r) < n) {
    ++r;
  }
  return r;
}

// Must match VX_dlmem_pkg (checked in VX_dlmem_noc_top).
constexpr uint32_t SOCKET_SIZE = VX_CFG_SOCKET_SIZE;
constexpr uint32_t NUM_SOCKETS = (VX_CFG_NUM_CORES / VX_CFG_SOCKET_SIZE) > 0 ? (VX_CFG_NUM_CORES / VX_CFG_SOCKET_SIZE) : 1;
constexpr uint32_t NCC         = NUM_SOCKETS * SOCKET_SIZE;
constexpr uint32_t DST_W       = (NCC > 1) ? clog2(NCC) : 1;
constexpr uint32_t DATAW       = NOC_DATAW;

// Payload layout: source core, per-source sequence number, then filler.
constexpr uint32_t SRC_BITS = 8;
constexpr uint32_t SEQ_BITS = 24;
constexpr uint32_t NW32     = (DATAW + 31) / 32;

constexpr uint64_t LOCAL_LATENCY  = 1;
constexpr uint64_t REMOTE_LATENCY = 3;

static_assert(NCC >= 1 && NCC <= (1u << SRC_BITS), "unsupported core count");
static_assert(DATAW >= SRC_BITS + SEQ_BITS, "DATAW too small for the source and sequence fields");

///////////////////////////////////////////////////////////////////////////////

// Bit-field access that works on every Verilator port type (CData..QData,
// VlWide) and on the testbench's 32-bit word arrays.
template <typename T>
static void set_bits(T& sig, uint32_t lsb, uint32_t width, uint64_t value) {
  if constexpr (std::is_integral_v<T>) {
    uint64_t mask = (width >= 64) ? ~0ull : ((1ull << width) - 1);
    uint64_t v = static_cast<uint64_t>(sig);
    v = (v & ~(mask << lsb)) | ((value & mask) << lsb);
    sig = static_cast<T>(v);
  } else {
    uint32_t i = 0;
    while (i < width) {
      uint32_t bit = lsb + i;
      uint32_t w = bit / 32;
      uint32_t o = bit % 32;
      uint32_t n = std::min(32 - o, width - i);
      uint32_t m = (n == 32) ? 0xffffffffu : ((1u << n) - 1);
      uint32_t chunk = static_cast<uint32_t>(value >> i) & m;
      sig[w] = (sig[w] & ~(m << o)) | (chunk << o);
      i += n;
    }
  }
}

template <typename T>
static uint64_t get_bits(const T& sig, uint32_t lsb, uint32_t width) {
  if constexpr (std::is_integral_v<T>) {
    uint64_t mask = (width >= 64) ? ~0ull : ((1ull << width) - 1);
    return (static_cast<uint64_t>(sig) >> lsb) & mask;
  } else {
    uint64_t value = 0;
    uint32_t i = 0;
    while (i < width) {
      uint32_t bit = lsb + i;
      uint32_t w = bit / 32;
      uint32_t o = bit % 32;
      uint32_t n = std::min(32 - o, width - i);
      uint32_t m = (n == 32) ? 0xffffffffu : ((1u << n) - 1);
      value |= static_cast<uint64_t>((sig[w] >> o) & m) << i;
      i += n;
    }
    return value;
  }
}

static std::string strf(const char* fmt, ...) {
  char buf[1024];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  return buf;
}

static uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

static uint32_t socket_of(uint32_t core) {
  return core / SOCKET_SIZE;
}

///////////////////////////////////////////////////////////////////////////////

using Payload = std::array<uint32_t, NW32>;

struct Packet {
  uint32_t src = 0;
  uint32_t dst = 0;
  uint32_t seq = 0;
  Payload  data{};
  uint64_t release = 0;       // first cycle the source may offer it
  uint64_t offer_cycle = 0;   // first cycle it was offered
  uint64_t inject_cycle = 0;
  uint64_t deliver_cycle = 0;
};

struct Options {
  std::string mode = "random";
  std::string csv  = "latency.csv";
  uint32_t seed     = 1;
  uint32_t packets  = 3000;   // per source, uniform test
  uint32_t burst    = 64;     // per source, hot-spot and permutation tests
  uint32_t watchdog = 2000;
};

enum class ReadyKind { ALWAYS, RANDOM, STALL };

class Testbench {
public:
  explicit Testbench(const Options& opts)
    : opts_(opts)
    , rng_(opts.seed)
    , src_(NCC)
    , next_seq_(NCC, 0)
    , inflight_(NCC * NCC)
    , ready_pct_(NCC, 100)
    , sink_ready_(NCC, true)
    , stalled_out_(NCC)
    , max_wait_(NCC, 0) {
    auto* t = sim_.operator->();
    t->reset = 1;
    this->drive();
    for (int i = 0; i < 8; ++i) {
      tick_ = sim_.step(tick_, 2);
    }
    t->reset = 0;
  }

  int run() {
    std::printf("dlmem_noc: sockets=%u socket_size=%u cores=%u dataw=%u dst_w=%u mode=%s seed=%u\n",
                NUM_SOCKETS, SOCKET_SIZE, NCC, DATAW, DST_W, opts_.mode.c_str(), opts_.seed);
    if (opts_.mode == "latency") {
      this->run_latency();
    } else {
      this->run_random();
    }
    this->final_check();
    std::printf("total cycles: %" PRIu64 ", packets: injected %" PRIu64 ", delivered %" PRIu64 "\n",
                cycle_, n_injected_, n_delivered_);
    if (failures_ != 0) {
      std::printf("FAILED: %u check(s) failed\n", failures_);
      return 1;
    }
    std::printf("PASSED\n");
    return 0;
  }

private:
  ///////////////////////////////////////////////////////////////////////////
  // stimulus

  uint32_t rand_below(uint32_t n) {
    return static_cast<uint32_t>(rng_() % n);
  }

  bool chance(uint32_t pct) {
    return rand_below(100) < pct;
  }

  void enqueue(uint32_t src, uint32_t dst, uint64_t release) {
    Packet p;
    p.src = src;
    p.dst = dst;
    p.seq = next_seq_[src]++;
    p.release = release;
    for (uint32_t w = 0; w < NW32; ++w) {
      uint64_t key = (static_cast<uint64_t>(opts_.seed) << 48) ^ (static_cast<uint64_t>(src) << 40)
                   ^ (static_cast<uint64_t>(p.seq) << 8) ^ w;
      p.data[w] = static_cast<uint32_t>(splitmix64(key));
    }
    if (DATAW % 32 != 0) {
      p.data[NW32 - 1] &= (1u << (DATAW % 32)) - 1;
    }
    set_bits(p.data, 0, SRC_BITS, src);
    set_bits(p.data, SRC_BITS, SEQ_BITS, p.seq);
    src_[src].queue.push_back(p);
  }

  ///////////////////////////////////////////////////////////////////////////
  // DUT access

  void drive() {
    auto* t = sim_.operator->();
    for (uint32_t s = 0; s < NCC; ++s) {
      const auto& offer = src_[s].offer;
      set_bits(t->in_valid, s, 1, offer.has_value());
      set_bits(t->in_dst, s * DST_W, DST_W, offer ? offer->dst : 0);
      for (uint32_t w = 0; w < NW32; ++w) {
        uint32_t n = std::min(32u, DATAW - 32 * w);
        set_bits(t->in_data, s * DATAW + 32 * w, n, offer ? offer->data[w] : 0);
      }
    }
    for (uint32_t d = 0; d < NCC; ++d) {
      set_bits(t->out_ready, d, 1, sink_ready_[d]);
    }
  }

  Payload sample_out(uint32_t d) {
    auto* t = sim_.operator->();
    Payload p{};
    for (uint32_t w = 0; w < NW32; ++w) {
      uint32_t n = std::min(32u, DATAW - 32 * w);
      p[w] = static_cast<uint32_t>(get_bits(t->out_data, d * DATAW + 32 * w, n));
    }
    return p;
  }

  void fail(const std::string& msg) {
    ++failures_;
    if (failures_ <= 20) {
      std::printf("FAIL [cycle %" PRIu64 "] %s\n", cycle_, msg.c_str());
    } else if (failures_ == 21) {
      std::printf("... further failure messages suppressed\n");
    }
  }

  void deliver(uint32_t d, const Payload& got, uint32_t dst) {
    if (dst != d) {
      this->fail(strf("core %u ejected a packet addressed to core %u", d, dst));
    }
    uint32_t src = static_cast<uint32_t>(get_bits(got, 0, SRC_BITS));
    uint32_t seq = static_cast<uint32_t>(get_bits(got, SRC_BITS, SEQ_BITS));
    if (src >= NCC) {
      this->fail(strf("core %u ejected a packet with a corrupted source field %u", d, src));
      return;
    }
    auto& q = inflight_[src * NCC + d];
    if (q.empty() || q.front().seq != seq) {
      auto it = std::find_if(q.begin(), q.end(), [&](const Packet& p) { return p.seq == seq; });
      if (it == q.end()) {
        this->fail(strf("core %u ejected src %u seq %u, which is not in flight to it (duplicate or misrouted)",
                        d, src, seq));
      } else {
        this->fail(strf("core %u ejected src %u seq %u ahead of seq %u from the same source (reordered, or that one lost)",
                        d, src, seq, q.front().seq));
        q.erase(it);
      }
      return;
    }
    Packet p = q.front();
    q.pop_front();
    if (p.data != got) {
      std::string e, g;
      for (uint32_t w = NW32; w-- > 0;) {
        e += strf("%08x", p.data[w]);
        g += strf("%08x", got[w]);
      }
      this->fail(strf("src %u seq %u to core %u: payload corrupted\n  expected %s\n  got      %s",
                      src, seq, d, e.c_str(), g.c_str()));
    }
    p.deliver_cycle = cycle_;
    if (record_) {
      delivered_.push_back(p);
    }
    ++n_delivered_;
  }

  // One cycle: offer, drive, sample every handshake before the edge, clock.
  void clock() {
    auto* t = sim_.operator->();
    for (uint32_t d = 0; d < NCC; ++d) {
      sink_ready_[d] = (ready_pct_[d] >= 100) || this->chance(ready_pct_[d]);
    }
    for (uint32_t s = 0; s < NCC; ++s) {
      auto& src = src_[s];
      if (!src.offer && !src.queue.empty() && src.queue.front().release <= cycle_) {
        src.offer = src.queue.front();
        src.offer->offer_cycle = cycle_;
        src.queue.pop_front();
      }
    }
    this->drive();
    t->eval();

    bool progress = false;

    for (uint32_t s = 0; s < NCC; ++s) {
      auto& offer = src_[s].offer;
      if (offer && get_bits(t->in_ready, s, 1)) {
        offer->inject_cycle = cycle_;
        max_wait_[s] = std::max(max_wait_[s], cycle_ - offer->offer_cycle);
        inflight_[offer->src * NCC + offer->dst].push_back(*offer);
        offer.reset();
        ++n_injected_;
        progress = true;
      }
    }

    for (uint32_t d = 0; d < NCC; ++d) {
      bool valid = get_bits(t->out_valid, d, 1);
      uint32_t dst = static_cast<uint32_t>(get_bits(t->out_dst, d * DST_W, DST_W));
      if (stalled_out_[d]) {
        // A registered output must hold a stalled packet unchanged.
        if (!valid || this->sample_out(d) != stalled_out_[d]->first || dst != stalled_out_[d]->second) {
          this->fail(strf("core %u: ejection output changed while stalled", d));
        }
        stalled_out_[d].reset();
      }
      if (!valid) {
        continue;
      }
      Payload got = this->sample_out(d);
      if (sink_ready_[d]) {
        this->deliver(d, got, dst);
        progress = true;
      } else {
        stalled_out_[d] = std::make_pair(got, dst);
      }
    }

    tick_ = sim_.step(tick_, 2);
    ++cycle_;
    if (progress) {
      last_progress_ = cycle_;
    }
  }

  bool outstanding() const {
    for (const auto& s : src_) {
      if (s.offer) {
        return true;
      }
    }
    for (const auto& q : inflight_) {
      if (!q.empty()) {
        return true;
      }
    }
    return false;
  }

  bool queued() const {
    for (const auto& s : src_) {
      if (!s.queue.empty()) {
        return true;
      }
    }
    return false;
  }

  bool watchdog() {
    if (!this->outstanding()) {
      last_progress_ = cycle_;
      return true;
    }
    if (cycle_ - last_progress_ <= opts_.watchdog) {
      return true;
    }
    uint32_t offers = 0;
    size_t in_flight = 0;
    const Packet* oldest = nullptr;
    for (const auto& s : src_) {
      offers += s.offer.has_value();
    }
    for (const auto& q : inflight_) {
      in_flight += q.size();
      if (!q.empty() && (!oldest || q.front().inject_cycle < oldest->inject_cycle)) {
        oldest = &q.front();
      }
    }
    std::string first = oldest ? strf(", oldest src %u seq %u to core %u injected at cycle %" PRIu64,
                                      oldest->src, oldest->seq, oldest->dst, oldest->inject_cycle)
                               : std::string();
    this->fail(strf("no progress for %u cycles: %u offer(s) pending, %zu packet(s) in flight%s",
                    opts_.watchdog, offers, in_flight, first.c_str()));
    return false;
  }

  // Runs until every queued packet is delivered; false on a hang or too many failures.
  bool drain(const char* ctx, bool random_phases) {
    last_progress_ = cycle_;
    while (this->queued() || this->outstanding()) {
      if (random_phases && cycle_ >= phase_end_) {
        this->new_ready_phase();
      }
      this->clock();
      if (!this->watchdog() || failures_ > 20) {
        std::printf("%s: aborted\n", ctx);
        return false;
      }
    }
    for (int i = 0; i < 4; ++i) {
      this->clock();
    }
    return true;
  }

  void set_always_ready() {
    std::fill(ready_pct_.begin(), ready_pct_.end(), 100);
  }

  // Ejection back-pressure phases; a stall is always followed by a phase that
  // drains, so a stall alone never trips the watchdog.
  void new_ready_phase() {
    static const uint32_t levels[] = {10, 30, 50, 80, 100};
    ReadyKind kind;
    uint32_t p = this->rand_below(100);
    if (last_kind_ == ReadyKind::STALL) {
      kind = (p < 40) ? ReadyKind::ALWAYS : ReadyKind::RANDOM;
    } else {
      kind = (p < 25) ? ReadyKind::ALWAYS : (p < 75) ? ReadyKind::RANDOM : ReadyKind::STALL;
    }
    uint32_t len = 16 + this->rand_below(240);
    switch (kind) {
    case ReadyKind::ALWAYS:
      this->set_always_ready();
      break;
    case ReadyKind::RANDOM:
      for (auto& r : ready_pct_) {
        r = levels[this->rand_below(5)];
      }
      break;
    case ReadyKind::STALL: {
      bool all = this->chance(30);
      for (auto& r : ready_pct_) {
        r = (all || this->chance(50)) ? 0 : levels[this->rand_below(5)];
      }
      len = 50 + this->rand_below(251);
      ++n_stalls_;
      break;
    }
    }
    last_kind_ = kind;
    gap_pct_ = std::array<uint32_t, 4>{0, 10, 50, 90}[this->rand_below(4)];
    phase_end_ = cycle_ + len;
  }

  ///////////////////////////////////////////////////////////////////////////
  // random mode

  void run_uniform() {
    uint32_t before = failures_;
    std::vector<uint32_t> remaining(NCC, opts_.packets);
    uint32_t left = NCC * opts_.packets;
    this->new_ready_phase();
    last_progress_ = cycle_;
    uint64_t start = cycle_;
    while (left != 0) {
      if (cycle_ >= phase_end_) {
        this->new_ready_phase();
      }
      for (uint32_t s = 0; s < NCC; ++s) {
        if (remaining[s] != 0 && !src_[s].offer && src_[s].queue.empty() && !this->chance(gap_pct_)) {
          this->enqueue(s, this->rand_below(NCC), cycle_);
          --remaining[s];
          --left;
        }
      }
      this->clock();
      if (!this->watchdog() || failures_ > 20) {
        std::printf("uniform: aborted\n");
        return;
      }
    }
    bool ok = this->drain("uniform", true);
    std::printf("uniform: %u packets per source, %u stall phases, %" PRIu64 " cycles: %s\n",
                opts_.packets, n_stalls_, cycle_ - start, (ok && failures_ == before) ? "ok" : "FAILED");
  }

  // Generous starvation bound for a hot spot under always-ready sinks: a remote
  // source waits for its socket's turn at the crossbar, the crossbar's turn at
  // the target's switch, and its own turn among its socket's cores.
  static uint64_t hotspot_bound() {
    return 2ull * SOCKET_SIZE * NUM_SOCKETS * (SOCKET_SIZE + 1) + 32;
  }

  void run_hotspot(uint32_t target, bool always_ready) {
    uint32_t before = failures_;
    std::fill(max_wait_.begin(), max_wait_.end(), 0);
    for (uint32_t s = 0; s < NCC; ++s) {
      for (uint32_t j = 0; j < opts_.burst; ++j) {
        this->enqueue(s, target, cycle_);
      }
    }
    if (always_ready) {
      this->set_always_ready();
    } else {
      this->new_ready_phase();
    }
    uint64_t start = cycle_;
    bool ok = this->drain("hotspot", !always_ready);
    std::string waits;
    if (always_ready && ok) {
      uint64_t bound = hotspot_bound();
      for (uint32_t s = 0; s < NCC; ++s) {
        waits += strf(" %" PRIu64, max_wait_[s]);
        if (max_wait_[s] > bound) {
          this->fail(strf("hot spot on core %u: source %u waited %" PRIu64 " cycles to inject, bound %" PRIu64
                          " (starvation)", target, s, max_wait_[s], bound));
        }
      }
      // Always-ready sinks drain one packet per cycle at the target.
      uint64_t cycles = cycle_ - start;
      uint64_t floor = static_cast<uint64_t>(NCC) * opts_.burst;
      if (cycles < floor) {
        this->fail(strf("hot spot on core %u: %" PRIu64 " packets delivered in %" PRIu64 " cycles", target, floor, cycles));
      }
    }
    std::printf("hotspot: target %u, %s sinks, %" PRIu64 " cycles%s%s: %s\n", target,
                always_ready ? "always-ready" : "random", cycle_ - start,
                waits.empty() ? "" : ", max inject wait per source:", waits.c_str(),
                (ok && failures_ == before) ? "ok" : "FAILED");
  }

  void run_permutation(const std::vector<uint32_t>& perm, const char* name) {
    uint32_t before = failures_;
    for (uint32_t j = 0; j < opts_.burst; ++j) {
      for (uint32_t s = 0; s < NCC; ++s) {
        this->enqueue(s, perm[s], cycle_);
      }
    }
    this->new_ready_phase();
    uint64_t start = cycle_;
    bool ok = this->drain("permutation", true);
    std::printf("permutation (%s): %" PRIu64 " cycles: %s\n", name, cycle_ - start,
                (ok && failures_ == before) ? "ok" : "FAILED");
  }

  void run_random() {
    this->run_uniform();
    if (failures_ > 20) {
      return;
    }
    std::vector<uint32_t> targets = {0, NCC - 1, this->rand_below(NCC)};
    for (uint32_t target : targets) {
      this->run_hotspot(target, true);
      this->run_hotspot(target, false);
      if (failures_ > 20) {
        return;
      }
    }
    std::vector<uint32_t> perm(NCC);
    std::iota(perm.begin(), perm.end(), 0);
    this->run_permutation(perm, "self");
    for (uint32_t s = 0; s < NCC; ++s) {
      perm[s] = (s + SOCKET_SIZE) % NCC;
    }
    this->run_permutation(perm, "next socket");
    for (int i = 0; i < 2 && failures_ <= 20; ++i) {
      std::shuffle(perm.begin(), perm.end(), rng_);
      this->run_permutation(perm, "random");
    }
  }

  ///////////////////////////////////////////////////////////////////////////
  // latency mode

  void run_latency() {
    uint64_t t1 = 8ull * NCC + 64;
    uint64_t t2 = t1 + 8ull * NCC * NCC + 64;
    uint64_t t3 = t2 + 64ull * NCC + 64;
    auto t0 = [](uint32_t src, uint32_t dst) -> uint64_t {
      return (socket_of(src) == socket_of(dst)) ? 2 : 0;
    };
    for (uint32_t s = 0; s < NCC; ++s) {
      this->enqueue(s, 0, t0(s, 0));
      this->enqueue(s, NCC - 1, t0(s, NCC - 1));
    }
    for (uint32_t k = 0; k < NCC * NCC; ++k) {
      this->enqueue(k / NCC, k % NCC, t1 + 8ull * k);
    }
    for (uint32_t s = 0; s < NCC; ++s) {
      for (uint32_t j = 0; j < 16; ++j) {
        this->enqueue(s, (s + j + 1) % NCC, t2);
      }
    }
    for (uint32_t s = 0; s < NCC; ++s) {
      for (uint32_t j = 0; j < 8; ++j) {
        this->enqueue(s, 0, t3);
      }
      for (uint32_t j = 0; j < 8; ++j) {
        this->enqueue(s, NCC - 1, t3);
      }
    }
    this->set_always_ready();
    record_ = true;
    bool ok = this->drain("latency", false);

    // Zero-load latency: one registered hop per switch or crossbar stage.
    uint32_t iso = 0;
    for (const Packet& p : delivered_) {
      if (p.release < t1 || p.release >= t2) {
        continue;
      }
      ++iso;
      uint64_t lat = p.deliver_cycle - p.inject_cycle;
      uint64_t want = (socket_of(p.src) == socket_of(p.dst)) ? LOCAL_LATENCY : REMOTE_LATENCY;
      if (p.inject_cycle != p.release) {
        this->fail(strf("isolated src %u to core %u: injected at cycle %" PRIu64 ", released at %" PRIu64,
                        p.src, p.dst, p.inject_cycle, p.release));
      }
      if (lat != want) {
        this->fail(strf("isolated src %u to core %u: latency %" PRIu64 ", expected %" PRIu64,
                        p.src, p.dst, lat, want));
      }
    }
    if (ok && iso != NCC * NCC) {
      this->fail(strf("isolated phase delivered %u packets, expected %u", iso, NCC * NCC));
    }

    std::sort(delivered_.begin(), delivered_.end(), [](const Packet& a, const Packet& b) {
      return (a.src != b.src) ? (a.src < b.src) : (a.seq < b.seq);
    });
    FILE* f = std::fopen(opts_.csv.c_str(), "w");
    if (!f) {
      this->fail(strf("cannot write %s", opts_.csv.c_str()));
      return;
    }
    std::fprintf(f, "src,dst,seq,inject_cycle,deliver_cycle\n");
    uint64_t sum[2] = {0, 0}, cnt[2] = {0, 0}, worst = 0;
    for (const Packet& p : delivered_) {
      std::fprintf(f, "%u,%u,%u,%" PRIu64 ",%" PRIu64 "\n", p.src, p.dst, p.seq, p.inject_cycle, p.deliver_cycle);
      int remote = socket_of(p.src) != socket_of(p.dst);
      sum[remote] += p.deliver_cycle - p.inject_cycle;
      ++cnt[remote];
      worst = std::max(worst, p.deliver_cycle - p.inject_cycle);
    }
    std::fclose(f);
    std::printf("latency: %zu packets written to %s; mean latency local %.2f (%" PRIu64 "), remote %.2f (%" PRIu64
                "), max %" PRIu64 ": %s\n", delivered_.size(), opts_.csv.c_str(),
                cnt[0] ? double(sum[0]) / cnt[0] : 0.0, cnt[0], cnt[1] ? double(sum[1]) / cnt[1] : 0.0, cnt[1],
                worst, (ok && failures_ == 0) ? "ok" : "FAILED");
  }

  ///////////////////////////////////////////////////////////////////////////

  void final_check() {
    uint32_t bad = 0;
    for (uint32_t i = 0; i < NCC * NCC; ++i) {
      const auto& q = inflight_[i];
      if (!q.empty() && ++bad <= 10) {
        this->fail(strf("src %u to core %u: %zu packet(s) never delivered (first seq %u)",
                        i / NCC, i % NCC, q.size(), q.front().seq));
      }
    }
    if (n_injected_ != n_delivered_ && failures_ == 0) {
      this->fail(strf("%" PRIu64 " packets injected, %" PRIu64 " delivered", n_injected_, n_delivered_));
    }
  }

  struct Source {
    std::deque<Packet> queue;
    std::optional<Packet> offer;
  };

  Options opts_;
  std::mt19937_64 rng_;
  vl_simulator<VVX_dlmem_noc_top> sim_;
  uint64_t tick_  = 0;
  uint64_t cycle_ = 0;
  uint64_t last_progress_ = 0;

  std::vector<Source> src_;
  std::vector<uint32_t> next_seq_;
  std::vector<std::deque<Packet>> inflight_;   // [src * NCC + dst], injection order
  std::vector<uint32_t> ready_pct_;
  std::vector<bool> sink_ready_;
  std::vector<std::optional<std::pair<Payload, uint32_t>>> stalled_out_;
  std::vector<uint64_t> max_wait_;
  std::vector<Packet> delivered_;
  bool record_ = false;

  ReadyKind last_kind_ = ReadyKind::ALWAYS;
  uint32_t gap_pct_ = 0;
  uint64_t phase_end_ = 0;
  uint32_t n_stalls_ = 0;

  uint64_t n_injected_ = 0;
  uint64_t n_delivered_ = 0;
  uint32_t failures_ = 0;
};

static void usage(const char* prog) {
  std::printf("usage: %s [--mode=random|latency] [--csv=FILE] [--seed=N] [--packets=N] [--burst=N]"
              " [--watchdog=N]\n", prog);
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);

  Options opts;
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    if (a[0] == '+') {
      continue;
    }
    if (std::strncmp(a, "--mode=", 7) == 0) {
      opts.mode = a + 7;
    } else if (std::strncmp(a, "--csv=", 6) == 0) {
      opts.csv = a + 6;
    } else if (std::strncmp(a, "--seed=", 7) == 0) {
      opts.seed = static_cast<uint32_t>(std::strtoul(a + 7, nullptr, 0));
    } else if (std::strncmp(a, "--packets=", 10) == 0) {
      opts.packets = static_cast<uint32_t>(std::strtoul(a + 10, nullptr, 0));
    } else if (std::strncmp(a, "--burst=", 8) == 0) {
      opts.burst = static_cast<uint32_t>(std::strtoul(a + 8, nullptr, 0));
    } else if (std::strncmp(a, "--watchdog=", 11) == 0) {
      opts.watchdog = static_cast<uint32_t>(std::strtoul(a + 11, nullptr, 0));
    } else {
      usage(argv[0]);
      return 2;
    }
  }
  if (opts.mode != "random" && opts.mode != "latency") {
    usage(argv[0]);
    return 2;
  }

  Testbench tb(opts);
  return tb.run();
}
