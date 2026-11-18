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

// The SM's L1 on the unified array, driven through one LSU block's global leg
// (shared by the RTL bench and its SimX twin). Sequential accesses are checked
// against L1SlotDir (hit or miss, the line reads with their sector masks);
// every access, overlapped or not, against a flat memory in program order.

#pragma once

#include <VX_types.h>
#include <l1_slot_ref.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace l1h100_cases {

using namespace vortex::l1_slot_ref;

constexpr uint32_t LINE      = VX_CFG_SM_DATAPATH_BYTES;
constexpr uint32_t SECTOR    = VX_CFG_DCACHE_SECTOR_SIZE;
constexpr uint32_t WORD      = VX_CFG_XLEN / 8;
constexpr uint32_t MAX_LANES = 32;
constexpr uint32_t NUM_TAGS  = 32;
constexpr uint64_t GL_LINE0  = 0x40000000ull / LINE;
// Lines in the I/O window are non-cacheable.
constexpr uint64_t NC_LINE0  = 0x2000ull / LINE;
constexpr uint32_t NC_LINES  = 4;
constexpr uint32_t MEM_DELAY = 20;

static_assert(VX_CFG_DCACHE_LINE_SIZE == LINE, "one L1 line per array row");
static_assert(NC_LINE0 * LINE >= VX_MEM_IO_BASE_ADDR && (NC_LINE0 + NC_LINES) * LINE <= VX_MEM_IO_END_ADDR,
              "the non-cacheable lines must sit in the I/O window");

enum Kind : uint32_t { K_LD = 0, K_ST = 1, K_ADD = 2 };
enum EvType : uint32_t { EV_ACCEPT, EV_RSP, EV_MEMRD, EV_FLUSHED };

struct Req {
  uint32_t kind  = K_LD;
  uint32_t mask  = 0;
  uint32_t bytes = WORD;
  uint64_t addr[MAX_LANES] = {};
  uint64_t val[MAX_LANES]  = {};  // store value or atomic operand
  uint64_t exp[MAX_LANES]  = {};  // set at submission: the value a load or atomic returns
  std::string name;
};

inline uint64_t value_mask(uint32_t bytes) { return bytes >= 8 ? ~0ull : ((1ull << (bytes * 8)) - 1); }
inline bool nc_line(uint64_t l) { return l >= NC_LINE0 && l < NC_LINE0 + NC_LINES; }
// Byte enables and data of a lane inside its word.
inline uint32_t lane_be(const Req& r, uint32_t i) { return ((1u << r.bytes) - 1) << (r.addr[i] % WORD); }
inline uint64_t lane_wdata(const Req& r, uint32_t i) { return (r.val[i] & value_mask(r.bytes)) << (8 * (r.addr[i] % WORD)); }

struct Event {
  uint32_t type;
  uint32_t tag = 0;
  uint32_t mask = 0;                       // EV_RSP: lanes; EV_MEMRD: sectors
  uint64_t line = 0;                       // EV_MEMRD
  std::array<uint64_t, MAX_LANES> data{};  // EV_RSP: lane words
};

inline uint8_t init_byte(uint64_t a) { return uint8_t((a * 0x9E3779B97F4A7C15ull) >> 56); }

struct MemImage {
  std::map<uint64_t, std::array<uint8_t, LINE>> lines;
  std::array<uint8_t, LINE>& line(uint64_t l) {
    auto it = lines.find(l);
    if (it == lines.end()) {
      std::array<uint8_t, LINE> a;
      for (uint32_t i = 0; i < LINE; ++i)
        a[i] = init_byte(l * LINE + i);
      it = lines.emplace(l, a).first;
    }
    return it->second;
  }
  uint64_t load(uint64_t a, uint32_t n) {
    uint64_t v = 0;
    for (uint32_t i = 0; i < n; ++i)
      v |= uint64_t(line((a + i) / LINE)[(a + i) % LINE]) << (8 * i);
    return v;
  }
  void store(uint64_t a, uint32_t n, uint64_t v) {
    for (uint32_t i = 0; i < n; ++i)
      line((a + i) / LINE)[(a + i) % LINE] = uint8_t(v >> (8 * i));
  }
};

// The memory below the L1: requests apply in arrival order, replies leave in
// order after a fixed delay, at most one per cycle.
struct FlatMem {
  MemImage img;
  uint32_t delay = MEM_DELAY;

  std::vector<uint8_t> read(uint64_t base, uint32_t n) {
    std::vector<uint8_t> d(n);
    for (uint32_t i = 0; i < n; ++i)
      d[i] = uint8_t(img.load(base + i, 1));
    return d;
  }
  void write(uint64_t base, const uint8_t* data, const uint8_t* be, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
      if (be[i])
        img.store(base + i, 1, data[i]);
    }
  }
  // An add over the enabled bytes of [base, base + n); returns the bytes before it.
  std::vector<uint8_t> amo_add(uint64_t base, const uint8_t* data, const uint8_t* be, uint32_t n) {
    std::vector<uint8_t> old = this->read(base, n);
    uint32_t first = n, cnt = 0;
    for (uint32_t i = 0; i < n; ++i) {
      if (be[i]) {
        first = std::min(first, i);
        ++cnt;
      }
    }
    if (cnt == 0)
      return old;
    uint64_t op = 0;
    for (uint32_t i = 0; i < cnt; ++i)
      op |= uint64_t(data[first + i]) << (8 * i);
    const uint64_t a = base + first;
    img.store(a, cnt, (img.load(a, cnt) + op) & value_mask(cnt));
    return old;
  }
};

inline L1SlotCfg knob_l1_cfg() {
  L1SlotCfg c{};
  c.s_bytes = VX_CFG_SMEM_PER_SM_MAX;
  c.sram_bytes = VX_CFG_SM_SRAM_BYTES;
  c.reserve_bytes = VX_CFG_L1_RESERVE_BYTES;
  c.row_bytes = VX_CFG_SM_DATAPATH_BYTES;
  c.valid_bytes = VX_CFG_L1_VALID_BYTES;
  c.sector_bytes = VX_CFG_DCACHE_SECTOR_SIZE;
  c.tag_sets = VX_CFG_L1_TAG_SETS;
  c.tag_ways = VX_CFG_L1_TAG_WAYS;
  c.tag_banks = VX_CFG_L1_TAG_BANKS;
  c.index_hash = VX_CFG_L1_INDEX_HASH;
  c.upd_depth = 6;
  c.protect_pct = VX_CFG_L1_REPL_PROTECT_PCT;
  c.mshr_lines = VX_CFG_SM_MAX_READ_LINES;
  c.max_lines = VX_CFG_L1_MAX_LINES;
  return c;
}

template <typename Driver>
class Runner {
public:
  Runner(Driver& d, uint32_t nlanes, FILE* csv) : d_(d), nl_(std::min(nlanes, MAX_LANES)), csv_(csv) {
    dir_.configure(knob_l1_cfg());
    for (uint32_t t = 0; t < NUM_TAGS; ++t)
      free_tags_.push_back(t);
  }

  int run(uint32_t seed, uint32_t nrand) {
    rng_.seed(seed);
    std::printf("L1 on the array: %u lines at the reset carve-out (protect %u), %u sets x %u ways, hash %u\n",
                dir_.lines(), dir_.protect(), VX_CFG_L1_TAG_SETS, VX_CFG_L1_TAG_WAYS, VX_CFG_L1_INDEX_HASH);
    this->outcomes();
    this->directed();
    this->random(nrand);
    this->final_check();
    std::printf("%u accesses, %u checks, %u errors\n", n_acc_, checks_, errors_);
    return errors_ ? 1 : 0;
  }

private:
  struct Pending {
    Req r;
    uint32_t id = 0, got = 0;
    bool ok = true;
  };
  struct Rd {
    uint64_t line;
    uint32_t secs;
    bool operator==(const Rd& o) const { return line == o.line && secs == o.secs; }
  };

  void check(bool ok, const std::string& what) {
    ++checks_;
    if (!ok) {
      ++errors_;
      if (errors_ <= 20)
        std::printf("  MISMATCH %s\n", what.c_str());
    }
  }

  // ---- requests ----

  Req one(uint32_t kind, uint64_t line, uint32_t off, uint32_t bytes, const std::string& name) {
    Req r;
    r.kind = kind;
    r.bytes = bytes;
    r.mask = 1;
    r.addr[0] = line * LINE + off;
    r.val[0] = rng_();
    r.name = name;
    return r;
  }

  // Lanes on distinct words of the given lines, round robin over them.
  Req lanes(uint32_t kind, const std::vector<uint64_t>& ls, uint32_t n, uint32_t bytes, const std::string& name) {
    Req r;
    r.kind = kind;
    r.bytes = bytes;
    r.name = name;
    std::vector<uint32_t> words(LINE / WORD);
    for (uint32_t w = 0; w < words.size(); ++w)
      words[w] = w;
    std::vector<std::vector<uint32_t>> pick(ls.size(), words);
    for (auto& p : pick)
      std::shuffle(p.begin(), p.end(), rng_);
    n = std::min<uint32_t>(std::min(n, nl_), uint32_t(ls.size() * words.size()));
    for (uint32_t i = 0; i < n; ++i) {
      const uint32_t k = i % ls.size();
      const uint32_t w = pick[k][i / ls.size()];
      const uint32_t sub = (bytes < WORD) ? (rng_() % (WORD / bytes)) * bytes : 0;
      r.mask |= 1u << i;
      r.addr[i] = ls[k] * LINE + w * WORD + sub;
      r.val[i] = rng_();
    }
    return r;
  }

  uint32_t submit(Req r) {
    for (uint32_t i = 0; i < nl_; ++i) {
      if (!(r.mask >> i & 1))
        continue;
      const uint64_t old = prog_.load(r.addr[i], r.bytes);
      r.exp[i] = old;
      if (r.kind == K_ST)
        prog_.store(r.addr[i], r.bytes, r.val[i]);
      else if (r.kind == K_ADD)
        prog_.store(r.addr[i], r.bytes, (old + r.val[i]) & value_mask(r.bytes));
    }
    while (free_tags_.empty())
      this->step();
    const uint32_t tag = free_tags_.front();
    free_tags_.pop_front();
    Pending p;
    p.r = r;
    p.id = n_acc_++;
    pend_[tag] = p;
    d_.push(r, tag);
    return tag;
  }

  void step() {
    auto ev = d_.cycle();
    bool progress = false;
    for (auto& e : ev) {
      progress = true;
      switch (e.type) {
      case EV_ACCEPT: {
        auto it = pend_.find(e.tag);
        if (it != pend_.end() && it->second.r.kind == K_ST)
          this->retire(it);
        break;
      }
      case EV_RSP: {
        auto it = pend_.find(e.tag);
        if (it == pend_.end()) {
          this->check(false, "response with no request, tag " + std::to_string(e.tag));
          break;
        }
        auto& p = it->second;
        for (uint32_t i = 0; i < nl_; ++i) {
          if (!(e.mask >> i & 1))
            continue;
          const uint32_t sh = 8 * (p.r.addr[i] % WORD);
          const uint64_t got = (e.data[i] >> sh) & value_mask(p.r.bytes);
          if (!(p.r.mask >> i & 1) || (p.got >> i & 1) || got != p.r.exp[i]) {
            p.ok = false;
            char b[200];
            std::snprintf(b, sizeof(b), "%s #%u lane %u addr 0x%llx: got 0x%llx expected 0x%llx", p.r.name.c_str(), p.id,
                          i, (unsigned long long)p.r.addr[i], (unsigned long long)got, (unsigned long long)p.r.exp[i]);
            this->check(false, b);
          }
        }
        p.got |= e.mask;
        if ((p.got & p.r.mask) == p.r.mask)
          this->retire(it);
        break;
      }
      case EV_MEMRD:
        reads_.push_back({e.line, e.mask});
        break;
      case EV_FLUSHED:
        flushed_ = true;
        break;
      }
    }
    idle_ = progress ? 0 : idle_ + 1;
    if (idle_ == 1000000) {
      std::printf("  HANG: no progress for %u cycles, %zu requests in flight\n", idle_, pend_.size());
      std::exit(1);
    }
  }

  void retire(typename std::map<uint32_t, Pending>::iterator it) {
    auto& p = it->second;
    ++checks_;
    if (!p.ok)
      ++errors_;
    if (!seq_mode_ && csv_)
      std::fprintf(csv_, "B,%u,%s,%u\n", p.id, p.r.name.c_str(), p.ok ? 1 : 0);
    free_tags_.push_back(it->first);
    pend_.erase(it);
  }

  void drain() {
    while (!pend_.empty() || d_.busy())
      this->step();
  }

  // ---- the golden for one sequential access ----

  std::vector<Rd> golden(const Req& r) {
    std::vector<uint64_t> ls;
    std::map<uint64_t, std::pair<uint64_t, uint64_t>> wm;  // line -> (touched words, full words)
    std::map<uint64_t, uint32_t> secs;
    const auto& c = dir_.cfg();
    for (uint32_t i = 0; i < nl_; ++i) {
      if (!(r.mask >> i & 1))
        continue;
      const uint64_t l = r.addr[i] / LINE;
      const uint32_t off = uint32_t(r.addr[i] % LINE);
      if (!wm.count(l))
        ls.push_back(l);
      wm[l].first |= l1_touched_words(off, r.bytes, c);
      wm[l].second |= l1_full_words(off, r.bytes, c);
      for (uint32_t b = off; b < off + r.bytes; ++b)
        secs[l] |= 1u << (b / SECTOR);
    }
    std::vector<Rd> rds;
    for (auto l : ls) {
      const bool nc = nc_line(l);
      if (r.kind == K_ADD) {
        dir_.clear(l);
        continue;
      }
      if (r.kind == K_ST) {
        if (nc)
          continue;
        if (!dir_.lookup(l, 0).tag)
          this->alloc(l, true, 0);
        dir_.store(l, wm[l].second);
        continue;
      }
      if (!nc && dir_.lookup(l, wm[l].first).hit)
        continue;
      const uint32_t slot = this->alloc(l, !nc, secs[l]);
      rds.push_back({l, secs[l]});
      dir_.fill(slot, secs[l]);
      dir_.fill_done(slot);
      dir_.replay_done(slot);
    }
    return rds;
  }

  uint32_t alloc(uint64_t l, bool cacheable, uint32_t secs) {
    for (int k = 0; k < 1000; ++k) {
      auto a = dir_.allocate(l, cacheable, 0, secs);
      if (a.ok)
        return a.slot;
      dir_.tick();
    }
    std::printf("  golden: no allocation for line 0x%llx\n", (unsigned long long)l);
    std::abort();
  }

  std::string fmt(const std::vector<Rd>& v) {
    std::string s;
    char b[48];
    for (auto& x : v) {
      std::snprintf(b, sizeof(b), "%s%llx:%x", s.empty() ? "" : "|", (unsigned long long)(x.line - GL_LINE0), x.secs);
      s += b;
    }
    return s.empty() ? "hit" : s;
  }

  void seq(const Req& r) {
    const auto exp = this->golden(r);
    reads_.clear();
    this->submit(r);
    this->drain();
    const auto got = reads_;
    this->check(got == exp, r.name + ": line reads " + this->fmt(got) + ", golden " + this->fmt(exp));
    for (int k = 0; k < 8; ++k)
      dir_.tick();
    const int inv = dir_.check();
    this->check(inv == 0, r.name + ": golden pool invariant " + std::to_string(inv));
    if (csv_)
      std::fprintf(csv_, "A,%u,%s,%s\n", n_acc_ - 1, r.name.c_str(), this->fmt(got).c_str());
  }

  void flush() {
    this->drain();
    flushed_ = false;
    d_.flush();
    while (!flushed_)
      this->step();
    this->drain();
    dir_.invalidate_all();
    if (csv_)
      std::fprintf(csv_, "F\n");
  }

  // Lines sharing one tag set, from `from` on.
  std::vector<uint64_t> same_set(uint64_t from, uint32_t n) {
    std::vector<uint64_t> v;
    const uint32_t s = dir_.set_of(from);
    for (uint64_t l = from; v.size() < n; ++l) {
      if (dir_.set_of(l) == s)
        v.push_back(l);
    }
    return v;
  }

  // ---- phases ----

  // Sequential: every outcome against L1SlotDir.
  void outcomes() {
    seq_mode_ = true;
    const uint64_t L = GL_LINE0;
    // non-cacheable: a borrowed slot each time, never a hit
    for (int k = 0; k < 2; ++k)
      this->seq(this->lanes(K_LD, {NC_LINE0}, 2, WORD, "nc-load"));
    // sector fill: only the requested sectors, a hit once they are valid
    this->seq(this->one(K_LD, L + 1, 0, WORD, "sector-miss"));
    this->seq(this->one(K_LD, L + 1, 0, WORD, "sector-hit"));
    this->seq(this->one(K_LD, L + 1, 2 * SECTOR, WORD, "sector-2"));
    this->seq(this->lanes(K_LD, {L + 1}, LINE / WORD, WORD, "line-all"));
    this->seq(this->lanes(K_LD, {L + 1}, LINE / WORD, WORD, "line-all-hit"));
    // store allocate: the fully written words valid, the rest missing
    this->seq(this->one(K_ST, L + 2, 8, WORD, "st-alloc"));
    this->seq(this->one(K_LD, L + 2, 8, WORD, "st-alloc-hit"));
    this->seq(this->one(K_LD, L + 2, 0, WORD, "st-alloc-word-miss"));
    this->seq(this->one(K_ST, L + 3, 5, 1, "st-alloc-partial"));
    this->seq(this->one(K_LD, L + 3, 0, WORD, "st-alloc-partial-miss"));
    // a global atomic clears the line's valid bits
    this->seq(this->one(K_LD, L + 4, 16, WORD, "amo-pre"));
    this->seq(this->one(K_LD, L + 4, 16, WORD, "amo-pre-hit"));
    this->seq(this->one(K_ADD, L + 4, 16, WORD, "amo"));
    this->seq(this->one(K_LD, L + 4, 16, WORD, "amo-cleared"));
    // in-set victims: one set over-filled, then reread
    auto set = this->same_set(L + 64, VX_CFG_L1_TAG_WAYS + 1);
    for (auto l : set)
      this->seq(this->one(K_LD, l, 0, WORD, "inset-fill"));
    for (auto l : set)
      this->seq(this->one(K_LD, l, 0, WORD, "inset-reread"));
    // age-list victims: more lines than the L1 holds, mixed with stores and
    // multi-lane line requests with sector masks
    const uint32_t pool = dir_.lines() * 2;
    for (uint32_t k = 0; k < dir_.lines() * 5; ++k) {
      const uint64_t l = L + 4096 + rng_() % pool;
      const uint32_t x = rng_() % 10;
      if (x == 0)
        this->seq(this->one(K_ST, l, (rng_() % (LINE / WORD)) * WORD, WORD, "age-st"));
      else if (x < 3)
        this->seq(this->lanes(K_LD, {l}, 1 + rng_() % (LINE / WORD), (rng_() & 1) ? WORD : 4, "age-line"));
      else
        this->seq(this->one(K_LD, l, (rng_() % (LINE / WORD)) * WORD, WORD, "age-ld"));
    }
    // whole-L1 invalidation
    this->flush();
    for (uint32_t k = 0; k < 16; ++k)
      this->seq(this->one(K_LD, L + 4096 + k, 0, WORD, "after-flush"));
    seq_mode_ = false;
  }

  void burst(const std::vector<Req>& v) {
    for (auto& r : v)
      this->submit(r);
    this->drain();
  }

  // Overlapped: data in program order.
  void directed() {
    const uint64_t L = GL_LINE0 + 65536;
    // (i) store hits, then the line evicted in-set and its way/slot reused, then a load of the new line
    auto set = this->same_set(L, VX_CFG_L1_TAG_WAYS + 1);
    std::vector<Req> b;
    for (uint32_t k = 0; k < VX_CFG_L1_TAG_WAYS; ++k)
      b.push_back(this->one(K_LD, set[k], 0, WORD, "i-inset-fill"));
    this->burst(b);
    b.clear();
    for (uint32_t k = 0; k < VX_CFG_L1_TAG_WAYS; ++k)
      b.push_back(this->one(K_ST, set[k], 0, WORD, "i-inset-st"));
    b.push_back(this->one(K_LD, set.back(), 0, WORD, "i-inset-new"));
    b.push_back(this->one(K_LD, set.back(), 0, WORD, "i-inset-new2"));
    for (uint32_t k = 0; k < VX_CFG_L1_TAG_WAYS; ++k)
      b.push_back(this->one(K_LD, set[k], 0, WORD, "i-inset-old"));
    this->burst(b);
    // (i) through age-list victims: a full L1, stores to resident lines, misses that draw them
    const uint32_t n = dir_.lines() + dir_.lines() / 4;
    b.clear();
    for (uint32_t k = 0; k < n; ++k)
      b.push_back(this->one(K_LD, L + 256 + k, 0, WORD, "i-age-fill"));
    this->burst(b);
    b.clear();
    for (uint32_t k = 0; k < 64; ++k)
      b.push_back(this->one(K_ST, L + 256 + n - 1 - k * 2, 0, WORD, "i-age-st"));
    for (uint32_t k = 0; k < 64; ++k)
      b.push_back(this->one(K_LD, L + 8192 + k, 0, WORD, "i-age-new"));
    for (uint32_t k = 0; k < 64; ++k) {
      b.push_back(this->one(K_LD, L + 8192 + k, 0, WORD, "i-age-new2"));
      b.push_back(this->one(K_LD, L + 256 + n - 1 - k * 2, 0, WORD, "i-age-old"));
    }
    this->burst(b);
    // (ii) full- and partial-word stores to a pending sector, then loads; a load
    // miss then a store to the same word before the fill
    const uint64_t Z = L + 16384;
    b.clear();
    b.push_back(this->one(K_LD, Z, 0, WORD, "ii-miss"));
    b.push_back(this->one(K_ST, Z, WORD, WORD, "ii-st-full"));
    b.push_back(this->one(K_ST, Z, 2 * WORD + 3, 1, "ii-st-part"));
    b.push_back(this->lanes(K_LD, {Z}, LINE / WORD, WORD, "ii-ld-line"));
    b.push_back(this->one(K_LD, Z + 1, 8, WORD, "ii-old"));
    b.push_back(this->one(K_ST, Z + 1, 8, WORD, "ii-st-same"));
    b.push_back(this->one(K_LD, Z + 1, 8, WORD, "ii-new"));
    this->burst(b);
    // (iii) an atomic to a line with an open entry
    b.clear();
    b.push_back(this->one(K_LD, Z + 2, 24, WORD, "iii-miss"));
    b.push_back(this->one(K_ADD, Z + 2, 24, WORD, "iii-amo"));
    b.push_back(this->one(K_LD, Z + 2, 24, WORD, "iii-ld"));
    b.push_back(this->one(K_ADD, Z + 2, 32, WORD, "iii-amo2"));
    b.push_back(this->lanes(K_LD, {Z + 2}, LINE / WORD, WORD, "iii-ld-line"));
    this->burst(b);
    // (iv) a slow wavefront stage (row reads ahead on the read port) with a
    // short fill latency: a filled slot is not reused before its replays read it
    d_.set_mem_delay(2);
    d_.set_row_stream(true);
    b.clear();
    for (uint32_t k = 0; k < 400; ++k)
      b.push_back(this->rand_req(L + 20000, dir_.lines() + 32, "iv"));
    this->burst(b);
    d_.set_row_stream(false);
    d_.set_mem_delay(MEM_DELAY);
    // (v) one warp access on more lines of one set than the set has ways: cold loads, hits in the
    // first lanes and misses after them, stores, and loads and stores of those lines in flight together
    auto vs = this->same_set(L + 32768, nl_);
    std::vector<uint64_t> vr(vs.begin() + vs.size() / 2, vs.end());
    vr.insert(vr.end(), vs.begin(), vs.begin() + vs.size() / 2);
    b.clear();
    b.push_back(this->lanes(K_LD, vs, nl_, WORD, "v-ld"));
    this->burst(b);
    b.clear();
    b.push_back(this->lanes(K_LD, vr, nl_, WORD, "v-ld-hitmiss"));
    this->burst(b);
    b.clear();
    b.push_back(this->lanes(K_ST, vs, nl_, WORD, "v-st"));
    this->burst(b);
    b.clear();
    b.push_back(this->lanes(K_LD, vr, nl_, WORD, "v-mix-ld"));
    b.push_back(this->lanes(K_ST, vs, nl_, 4, "v-mix-st"));
    b.push_back(this->lanes(K_LD, vs, nl_, WORD, "v-mix-ld2"));
    b.push_back(this->lanes(K_ST, vr, nl_, WORD, "v-mix-st2"));
    b.push_back(this->lanes(K_LD, vr, nl_, WORD, "v-mix-ld3"));
    this->burst(b);
  }

  Req rand_req(uint64_t base, uint32_t span, const std::string& name) {
    const uint32_t x = rng_() % 20;
    const uint32_t nls = 1 + rng_() % 3;
    std::vector<uint64_t> ls;
    for (uint32_t k = 0; k < nls; ++k)
      ls.push_back(base + rng_() % span);
    std::sort(ls.begin(), ls.end());
    ls.erase(std::unique(ls.begin(), ls.end()), ls.end());
    const uint32_t bytes = (rng_() % 4 == 0) ? 4 : WORD;
    if (x == 0)
      return this->lanes(K_LD, {NC_LINE0 + rng_() % NC_LINES}, 1 + rng_() % 4, WORD, name + "-nc");
    if (x < 3)
      return this->lanes(K_ADD, ls, 1 + rng_() % 4, WORD, name + "-amo");
    if (x < 8)
      return this->lanes(K_ST, ls, 1 + rng_() % nl_, (rng_() % 4 == 0) ? 1 : bytes, name + "-st");
    return this->lanes(K_LD, ls, 1 + rng_() % nl_, bytes, name + "-ld");
  }

  void random(uint32_t nrand) {
    std::vector<Req> b;
    for (uint32_t k = 0; k < nrand; ++k)
      b.push_back(this->rand_req(GL_LINE0 + 131072, dir_.lines() * 3 / 2, "rand"));
    this->burst(b);
  }

  // Write-through: memory equals the program image once the L1 is flushed.
  void final_check() {
    this->flush();
    uint32_t bad = 0;
    for (auto& kv : prog_.lines) {
      if (d_.mem().img.line(kv.first) != kv.second)
        ++bad;
    }
    this->check(bad == 0, "memory image: " + std::to_string(bad) + " lines differ from program order");
  }

  Driver& d_;
  uint32_t nl_;
  FILE* csv_;
  L1SlotDir dir_;
  MemImage prog_;
  std::mt19937 rng_;
  std::map<uint32_t, Pending> pend_;
  std::deque<uint32_t> free_tags_;
  std::vector<Rd> reads_;
  bool seq_mode_ = false, flushed_ = false;
  uint32_t idle_ = 0, n_acc_ = 0, checks_ = 0, errors_ = 0;
};

}  // namespace l1h100_cases
