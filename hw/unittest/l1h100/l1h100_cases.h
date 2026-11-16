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

// The cases and checks of the H100 L1 bench, run unchanged by the RTL
// testbench (VX_cache with H100_L1) and its SimX twin (Cache with h100_l1)
// through a driver with the same cycle contract. Accesses go one at a time, so
// every outcome is l1_ref.h's L1Ref on the same stream from reset: hit or
// miss, the sectors a miss requests (one line request per miss), the bytes a
// store writes through, whole-L1 invalidation on a fence, and the victim
// sequence (a wrong victim shows as a wrong later hit). One case runs two
// misses to one line at once: the second waits for the first's fill and then
// requests its own sectors. Accesses stay inside one memory block, as a SimX
// request does. Each run writes a CSV of every access's outcome
// and timing, which must be identical between the two models.

#include <l1_ref.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace l1h100_cases {

using namespace vortex::l1_ref;

constexpr uint32_t LINE      = VX_CFG_DCACHE_LINE_SIZE;
constexpr uint32_t SECTOR    = VX_CFG_DCACHE_SECTOR_SIZE;
constexpr uint32_t CAP_LINES = VX_CFG_DCACHE_SIZE / VX_CFG_DCACHE_LINE_SIZE;
constexpr uint32_t MEM_DELAY = 20;   // the memory's reply time
constexpr uint32_t NUM_TAGS  = 256;  // the bench's core tag field: 8 bits
// Accesses stay inside one memory block, the most a SimX request carries.
constexpr uint32_t BLOCK     = VX_CFG_MEM_BLOCK_SIZE;

enum Kind : uint32_t { K_LD = 0, K_ST = 1, K_FENCE = 2 };

using Bytes = std::array<uint8_t, LINE>;

struct Op {
  uint32_t kind = K_LD;
  uint64_t line = 0;        // line address
  Bytes    byteen{};        // 0/1 per byte
  Bytes    data{};          // stores
  std::string name;
};

enum EvType : uint32_t { EV_ACCEPT = 0, EV_RSP = 1, EV_MEM = 2 };

struct Event {
  uint32_t type;
  uint32_t tag;          // ACCEPT / RSP
  bool     data_ok;      // RSP of a load: its bytes equal the memory's
  bool     rw;           // MEM
  bool     fence;        // MEM: a fence or flush passed below (answered, carries no data)
  uint64_t line;         // MEM
  Bytes    byteen;       // MEM
};

inline uint8_t init_byte(uint64_t addr) {
  return uint8_t((addr * 0x9E3779B1u) >> 13);
}

// The memory below the L1; both drivers keep one and answer reads from it.
struct MemImage {
  std::map<uint64_t, Bytes> lines;
  Bytes& at(uint64_t line) {
    auto it = lines.find(line);
    if (it == lines.end()) {
      Bytes b;
      for (uint32_t i = 0; i < LINE; ++i)
        b[i] = init_byte(line * LINE + i);
      it = lines.emplace(line, b).first;
    }
    return it->second;
  }
};

inline Op access(uint32_t kind, uint64_t addr, uint32_t bytes, const std::string& name, uint8_t seed = 0) {
  Op o;
  o.kind = kind;
  o.line = addr / LINE;
  o.name = name;
  for (uint32_t i = 0; i < bytes; ++i) {
    o.byteen[(addr % LINE) + i] = 1;
    o.data[(addr % LINE) + i] = uint8_t(seed + 17 * i + 1);
  }
  return o;
}

inline Op fence(const std::string& name) {
  Op o;
  o.kind = K_FENCE;
  o.name = name;
  return o;
}

inline uint32_t mask_of(const Bytes& be) {
  uint32_t m = 0;
  for (uint32_t i = 0; i < LINE; ++i)
    if (be[i])
      m |= 1u << (i / SECTOR);
  return m;
}

inline std::vector<uint8_t> vec(const Bytes& b) {
  return std::vector<uint8_t>(b.begin(), b.end());
}

inline L1Ref make_ref() {
  return L1Ref(VX_CFG_DCACHE_SIZE, VX_CFG_DCACHE_NUM_WAYS, LINE, SECTOR, VX_CFG_DCACHE_NUM_BANKS,
               VX_CFG_L1_REPL_PROTECTED);
}

// Driver contract (both models): push() queues an access at the core input,
// held valid from the next cycle() until accepted; cycle() applies the
// inputs, samples the outputs into events and advances one clock. The memory
// answers each read MEM_DELAY cycles after the request with the line's bytes
// from the driver's MemImage, and applies each write to it when sent.
template <typename Driver>
class Runner {
public:
  Runner(Driver& d, FILE* csv) : d_(d), csv_(csv), ref_(make_ref()) {}

  int run(uint32_t seed, uint32_t nrand, bool quick) {
    this->fill_rows();
    this->wpolicy_rows();
    this->assoc(quick);
    this->repl(quick);
    this->inval();
    this->pending_line();
    this->random(seed, nrand);
    std::printf("l1h100: %u accesses, %u checks, %u errors\n", n_ops_, checks_, errors_);
    return errors_ ? 1 : 0;
  }

private:
  struct Result {
    bool hit = false;
    uint32_t reads = 0, writes = 0, fences = 0, read_mask = 0, write_mask = 0;
    bool data_ok = true;
    uint64_t accept = 0, done = 0, present = 0;
  };

  void check(bool ok, const std::string& what) {
    ++checks_;
    if (!ok) {
      ++errors_;
      if (errors_ <= 40)
        std::printf("  MISMATCH %s\n", what.c_str());
    }
    if (errors_ >= 200) {
      std::printf("l1h100: stopping after %u errors\nFAILED!\n", errors_);
      std::exit(1);
    }
  }

  // One access alone: run until it is answered (a load or fence) or its write
  // went out (a store), then let the cache settle.
  Result one(const Op& op, const char* phase) {
    Result r;
    uint32_t tag = next_tag_++ % NUM_TAGS;
    r.present = d_.now();
    d_.push(op, tag);
    bool accepted = false, done = false;
    for (uint32_t n = 0; !done; ++n) {
      if (n > 20000) {
        this->check(false, op.name + ": no progress");
        break;
      }
      for (auto& e : d_.cycle()) {
        const uint64_t c = d_.now() - 1;
        if (e.type == EV_ACCEPT && e.tag == tag) {
          accepted = true;
          r.accept = c;
          if (op.kind == K_ST && r.writes)
            done = true;
        } else if (e.type == EV_MEM && e.fence) {
          ++r.fences;
        } else if (e.type == EV_MEM) {
          if (e.rw) {
            ++r.writes;
            r.write_mask = mask_of(e.byteen);
            this->check(e.line == op.line && e.byteen == op.byteen, op.name + ": the write-through carries other bytes");
            if (op.kind == K_ST && accepted)
              done = true;
          } else {
            ++r.reads;
            r.read_mask |= mask_of(e.byteen);
            this->check(e.line == op.line, op.name + ": a read request for another line");
          }
          r.done = c;
        } else if (e.type == EV_RSP && e.tag == tag) {
          r.data_ok = e.data_ok;
          r.done = c;
          done = true;
        }
      }
    }
    for (int i = 0; i < 4; ++i)
      d_.cycle();
    ++n_ops_;
    r.hit = (r.reads == 0);
    this->log(phase, op, r);
    return r;
  }

  void log(const char* phase, const Op& op, const Result& r) {
    if (csv_)
      std::fprintf(csv_, "%s,%u,%u,%llu,%u,%u,%u,%u,%llu,%llu\n", phase, n_ops_, op.kind, (unsigned long long)op.line,
                   r.reads, r.read_mask, r.writes, r.write_mask, (unsigned long long)(r.accept - r.present),
                   (unsigned long long)(r.done - r.accept));
  }

  // Runs op on the DUT and the golden; checks every outcome.
  bool step(const Op& op, const char* phase) {
    if (op.kind == K_FENCE) {
      ref_.invalidate_all();
      Result r = this->one(op, phase);
      this->check(r.reads == 0 && r.writes == 0 && r.fences <= 1, op.name + ": a fence read or wrote memory");
      return true;
    }
    if (op.kind == K_ST) {
      ref_.store(op.line, vec(op.byteen));
      Result r = this->one(op, phase);
      this->check(r.writes == 1 && r.reads == 0, op.name + ": a store must write through once and read nothing ("
                  + std::to_string(r.writes) + " writes, " + std::to_string(r.reads) + " reads)");
      return true;
    }
    auto g = ref_.load(op.line, vec(op.byteen));
    Result r = this->one(op, phase);
    this->check(r.data_ok, op.name + ": wrong load data");
    this->check(r.writes == 0, op.name + ": a load wrote memory");
    if (g.hit) {
      this->check(r.reads == 0, op.name + ": expected a hit, the L1 read memory");
    } else {
      this->check(r.reads == 1, op.name + ": expected one line request, saw " + std::to_string(r.reads));
      this->check(r.read_mask == g.sector_mask, op.name + ": requested sectors " + std::to_string(r.read_mask)
                  + ", expected " + std::to_string(g.sector_mask));
    }
    return r.reads == 0;
  }

  void fill_rows() {
    std::vector<uint32_t> xs = {4, 8, 12, 16, 20, 24, 28, 32, 36, 60, 64, 96, 124};
    for (size_t i = 0; i < xs.size(); ++i) {
      uint64_t a = (0x10000 + i * 64) * LINE;
      this->step(access(K_LD, a, 4, "fill miss +0"), "fill");
      this->step(access(K_LD, a + xs[i], 4, "fill then +" + std::to_string(xs[i])), "fill");
    }
  }

  // 05 sec 4.5 rows Vortex expresses (byte-valid store allocate).
  void wpolicy_rows() {
    struct Row { const char* name; std::vector<std::pair<uint32_t, uint32_t>> steps; };
    std::vector<Row> rows = {
      {"ld A; st A -> ld A", {{K_LD, 0}, {K_ST, 0}, {K_LD, 0}}},
      {"ld A; st A+4 -> ld A+4", {{K_LD, 0}, {K_ST, 4}, {K_LD, 4}}},
      {"st A -> ld A", {{K_ST, 0}, {K_LD, 0}}},
      {"st A -> ld A+4", {{K_ST, 0}, {K_LD, 4}}},
      {"whole sector stored -> ld A", {{K_ST, 0}, {K_ST, 4}, {K_ST, 8}, {K_ST, 12}, {K_ST, 16}, {K_ST, 20}, {K_ST, 24}, {K_ST, 28}, {K_LD, 0}}},
      {"7 of 8 words stored -> ld the 8th", {{K_ST, 0}, {K_ST, 4}, {K_ST, 8}, {K_ST, 12}, {K_ST, 16}, {K_ST, 20}, {K_ST, 24}, {K_LD, 28}}},
      {"st A; ld A+4 -> ld A", {{K_ST, 0}, {K_LD, 4}, {K_LD, 0}}},
      {"st 8 B -> ld 4 B inside", {{K_ST, 0}, {K_LD, 4}}},
      {"st A+32 -> ld A (other sector)", {{K_ST, 32}, {K_LD, 0}}},
      {"st 64 B -> ld its second sector", {{K_ST, 64}, {K_LD, 96}}}};
    for (size_t r = 0; r < rows.size(); ++r) {
      uint64_t a = (0x20000 + r * 64) * LINE;
      for (auto& s : rows[r].steps) {
        uint32_t bytes = 4;
        if (std::string(rows[r].name).find("8 B") != std::string::npos && s.first == K_ST) bytes = 8;
        if (std::string(rows[r].name).find("st 64 B") != std::string::npos && s.first == K_ST) bytes = 64;
        this->step(access(s.first, a + s.second, bytes, rows[r].name, uint8_t(r)), "wpol");
      }
    }
  }

  // Capacity lines at a power-of-two stride, twice: no miss in the second pass.
  void assoc(bool quick) {
    const uint32_t n = quick ? CAP_LINES / 4 : CAP_LINES;
    for (uint64_t stride = LINE; stride <= 65536; stride *= quick ? 8 : 2) {
      this->step(fence("assoc fence"), "assoc");
      for (int pass = 0; pass < 2; ++pass) {
        uint32_t misses = 0;
        for (uint32_t i = 0; i < n; ++i)
          misses += !this->step(access(K_LD, 0x40000000ull + i * stride, 4, "assoc"), "assoc");
        if (pass == 1)
          this->check(misses == 0, "assoc stride " + std::to_string(stride) + ": " + std::to_string(misses) + " misses in the second pass");
      }
    }
  }

  // Cyclic passes over more lines than fit: every outcome is the golden's.
  void repl(bool quick) {
    for (uint32_t pct : {110u, 140u}) {
      this->step(fence("repl fence"), "repl");
      const uint32_t n = (quick ? CAP_LINES / 4 : CAP_LINES) * pct / 100;
      for (int lap = 0; lap < 3; ++lap)
        for (uint32_t i = 0; i < n; ++i)
          this->step(access(K_LD, 0x80000000ull + uint64_t(i) * LINE, 4, "repl " + std::to_string(pct)), "repl");
    }
  }

  void inval() {
    uint64_t a = 0x30000ull * LINE;
    this->step(access(K_LD, a, 4, "inval warm"), "inval");
    this->step(access(K_LD, a, 4, "inval hit"), "inval");
    this->step(fence("inval fence"), "inval");
    this->step(access(K_LD, a, 4, "inval after the fence"), "inval");
    this->step(access(K_ST, a + 64, 4, "inval store"), "inval");
    this->step(fence("inval fence 2"), "inval");
    this->step(access(K_LD, a + 64, 4, "inval stored byte after the fence"), "inval");
  }

  // Two loads of one line, other sectors, presented back to back: one line
  // request each, the second only after the first's fill.
  void pending_line() {
    this->step(fence("pending fence"), "pend");
    uint64_t a = 0x31000ull * LINE;
    Op x = access(K_LD, a, 4, "pending first");
    Op y = access(K_LD, a + 2 * SECTOR, 4, "pending second");
    ref_.load(x.line, vec(x.byteen));
    ref_.load(y.line, vec(y.byteen));
    uint32_t tx = next_tag_++ % NUM_TAGS, ty = next_tag_++ % NUM_TAGS;
    uint64_t t0 = d_.now();
    d_.push(x, tx);
    d_.push(y, ty);
    std::vector<std::pair<uint64_t, uint32_t>> reads;
    uint64_t rx = 0, ry = 0;
    for (uint32_t n = 0; (rx == 0 || ry == 0) && n < 10000; ++n) {
      for (auto& e : d_.cycle()) {
        uint64_t c = d_.now() - 1;
        if (e.type == EV_MEM && !e.rw && !e.fence)
          reads.push_back({c, mask_of(e.byteen)});
        if (e.type == EV_RSP && e.tag == tx) rx = c;
        if (e.type == EV_RSP && e.tag == ty) ry = c;
      }
    }
    for (int i = 0; i < 4; ++i)
      d_.cycle();
    this->check(reads.size() == 2, "pending: " + std::to_string(reads.size()) + " line requests, expected 2");
    if (reads.size() == 2) {
      this->check(reads[0].second == 1u && reads[1].second == (1u << 2), "pending: requested sectors "
                  + std::to_string(reads[0].second) + ", " + std::to_string(reads[1].second) + ", expected 1, 4");
      this->check(reads[1].first >= reads[0].first + MEM_DELAY, "pending: the second request left before the first fill");
    }
    if (csv_)
      std::fprintf(csv_, "pend,%llu,%llu,%llu,%llu\n", (unsigned long long)(reads.size() > 0 ? reads[0].first - t0 : 0),
                   (unsigned long long)(reads.size() > 1 ? reads[1].first - t0 : 0), (unsigned long long)(rx - t0),
                   (unsigned long long)(ry - t0));
  }

  void random(uint32_t seed, uint32_t n) {
    std::mt19937 rng(seed);
    this->step(fence("random fence"), "rnd");
    for (uint32_t i = 0; i < n; ++i) {
      // 24 lines on 4 sets' worth of addresses: refills, evictions, partial sectors.
      uint64_t line = 0x50000ull + (rng() % 24) * (uint64_t(ref_.sets()) / 4 + 1);
      uint32_t k = rng() % 10;
      if (k == 0 && i % 97 == 0) {
        this->step(fence("random fence"), "rnd");
        continue;
      }
      uint32_t bytes = 1u << (rng() % 4);       // 1, 2, 4, 8
      if (rng() % 16 == 0) bytes = SECTOR;
      if (rng() % 64 == 0) bytes = BLOCK;
      uint32_t off = (rng() % (LINE / bytes)) * bytes;
      this->step(access(k < 6 ? K_LD : K_ST, line * LINE + off, bytes, "random", uint8_t(i)), "rnd");
    }
  }

  Driver& d_;
  FILE* csv_;
  L1Ref ref_;
  uint32_t next_tag_ = 0, n_ops_ = 0, checks_ = 0, errors_ = 0;
};

} // namespace l1h100_cases
