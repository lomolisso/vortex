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

// VX_cache testbench in the last-level role: every core port is driven
// directly and the memory side is a model whose fill latency the testbench
// sets cycle by cycle. Directed sweeps release a held line fill at every cycle
// offset around a racing request: a core atomic on another line of the bank
// (race), a store behind an atomic in one MSHR chain (chain, and chainbp with
// memory refusing requests while the chain replays), atomics on several
// missing lines whose fills return together (wbqburst), a request reaching the
// filled line while an older same-word access still waits in its chain
// (bypass), a load behind an atomic in one chain (chainload), and a load
// replaying ahead of an atomic's writeback while the response path stalls
// (settle). A randomized run mixes loads, stores, atomics and LR/SC under
// random latency and back-pressure. Fence and flush rounds check where each
// one walks the cache and where a fence only drains (fence), and that a
// stream of either on one port does not starve the others (fstorm). With
// CTA_FENCE (DLMEM builds), a CTA-scope fence drains at any level, keeps the
// lines, waits for an older write from another port, and sends no fence or
// flush to memory (fencecta).
// Responses and final contents are checked against a reference memory that
// applies each word's requests in the order the cache accepted them.

#include "vl_simulator.h"
#include "VVX_cache_top.h"
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
#include <map>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// VX_cache_top parameters a build overrides arrive twice: as Verilator
// -G<NAME> for the model and as -DTB_<NAME> here. Defaults mirror the wrapper.
#ifndef TB_NUM_REQS
#define TB_NUM_REQS 4
#endif
#ifndef TB_MEM_PORTS
#define TB_MEM_PORTS 1
#endif
#ifndef TB_CACHE_SIZE
#define TB_CACHE_SIZE VX_CFG_DCACHE_SIZE
#endif
#ifndef TB_LINE_SIZE
#define TB_LINE_SIZE VX_CFG_DCACHE_LINE_SIZE
#endif
#ifndef TB_SECTOR_SIZE
#define TB_SECTOR_SIZE VX_CFG_DCACHE_SECTOR_SIZE
#endif
#ifndef TB_NUM_BANKS
#define TB_NUM_BANKS 4
#endif
#ifndef TB_NUM_WAYS
#define TB_NUM_WAYS VX_CFG_DCACHE_NUM_WAYS
#endif
#ifndef TB_WORD_SIZE
#define TB_WORD_SIZE VX_CFG_DCACHE_WORD_SIZE
#endif
#ifndef TB_MSHR_SIZE
#define TB_MSHR_SIZE 16
#endif
#ifndef TB_WRITEBACK
#define TB_WRITEBACK VX_CFG_DCACHE_WRITEBACK
#endif
#ifndef TB_LATENCY
#define TB_LATENCY VX_CFG_DCACHE_LATENCY
#endif
#ifndef TB_AMO_ENABLE
#define TB_AMO_ENABLE 1
#endif

#ifndef TB_IS_LLC
#define TB_IS_LLC 1
#endif
#ifndef TB_CTA_FENCE
#define TB_CTA_FENCE 0
#endif

// Without atomics a bank's load/store logic does not depend on IS_LLC, which
// then only selects the cache's level: a last-level instance is the shared
// level by default, where a fence drains instead of walking.
#if (!TB_IS_LLC && TB_AMO_ENABLE) || (defined(TB_WRITE_ENABLE) && !TB_WRITE_ENABLE)
#error "the testbench models a writable cache, and atomics only at the last level"
#endif

constexpr uint32_t clog2(uint64_t x) {
  uint32_t n = 0;
  while ((uint64_t(1) << n) < x) {
    ++n;
  }
  return n;
}

constexpr uint32_t XLEN        = VX_CFG_XLEN;
constexpr uint32_t ADDR_BITS   = VX_CFG_MEM_ADDR_WIDTH;
constexpr uint32_t NUM_REQS    = TB_NUM_REQS;
constexpr uint32_t MEM_PORTS   = TB_MEM_PORTS;
constexpr uint32_t CACHE_SIZE  = TB_CACHE_SIZE;
constexpr uint32_t LINE_SIZE   = TB_LINE_SIZE;
constexpr uint32_t SECTOR_SIZE = TB_SECTOR_SIZE;
constexpr uint32_t NUM_BANKS   = TB_NUM_BANKS;
constexpr uint32_t NUM_WAYS    = TB_NUM_WAYS;
constexpr uint32_t WORD_SIZE   = TB_WORD_SIZE;
constexpr uint32_t MSHR_SIZE   = TB_MSHR_SIZE;
constexpr uint32_t LATENCY     = TB_LATENCY;
constexpr bool     WRITEBACK   = (TB_WRITEBACK != 0);
constexpr bool     AMO_EN      = (TB_AMO_ENABLE != 0);
constexpr bool     SHARED      = (TB_IS_LLC != 0);
constexpr bool     CTA_FENCE   = (TB_CTA_FENCE != 0);
constexpr bool     CAS_EN      = (VX_CFG_EXT_ZACAS_ENABLED != 0);

constexpr uint32_t WORDS_PER_LINE   = LINE_SIZE / WORD_SIZE;
constexpr uint32_t WORDS_PER_SECTOR = SECTOR_SIZE / WORD_SIZE;
constexpr uint32_t SETS_PER_BANK    = CACHE_SIZE / (NUM_BANKS * NUM_WAYS * LINE_SIZE);
constexpr uint32_t WORD_ADDR_BITS   = ADDR_BITS - clog2(WORD_SIZE);
constexpr uint32_t SECTOR_ADDR_BITS = ADDR_BITS - clog2(SECTOR_SIZE);
// An atomic is at most one register wide (the LSU issues no 64-bit atomic on RV32).
constexpr uint32_t MAX_AMO_SIZE     = std::min<uint32_t>(std::min<uint32_t>(XLEN / 8, WORD_SIZE), 8);

#if !defined(NDEBUG) || defined(SCOPE)
constexpr uint32_t UUID_BITS = 44;
#else
constexpr uint32_t UUID_BITS = 1;
#endif
constexpr uint32_t TAG_ID_BITS  = 16;
constexpr uint32_t TAG_BITS     = TAG_ID_BITS + UUID_BITS;
constexpr uint32_t MEM_TAG_BITS = UUID_BITS + clog2(MSHR_SIZE) + clog2((NUM_BANKS + MEM_PORTS - 1) / MEM_PORTS);

// VX_gpu_pkg hart_id = {global core, warp, thread} of an instantiated core.
constexpr uint32_t NUM_SOCKETS  = std::max<uint32_t>(VX_CFG_NUM_CORES / VX_CFG_SOCKET_SIZE, 1);
constexpr uint32_t TOTAL_CORES  = VX_CFG_NUM_CLUSTERS * NUM_SOCKETS * VX_CFG_SOCKET_SIZE;
constexpr uint32_t HART_ID_BITS = std::max<uint32_t>(clog2(TOTAL_CORES) + clog2(VX_CFG_NUM_WARPS) + clog2(VX_CFG_NUM_THREADS), 1);
constexpr uint32_t NUM_HARTS    = (HART_ID_BITS >= 4) ? 16 : (1u << HART_ID_BITS);

// Bit offsets in the attr field of mem_bus_attr_t {is_fence, amo,
// is_addr_om, is_addr_local, is_addr_io, is_flush} with amo_req_t {hart_id,
// [amo_cmp], amo_unsigned, amo_op, amo_valid} (VX_gpu_pkg.sv, packed MSB
// first).
constexpr uint32_t ATTR_FLUSH     = 0;
constexpr uint32_t ATTR_AMO_VALID = 4;
constexpr uint32_t ATTR_AMO_OP    = ATTR_AMO_VALID + 1;
constexpr uint32_t ATTR_AMO_UNS   = ATTR_AMO_OP + 4;
constexpr uint32_t ATTR_AMO_CMP   = ATTR_AMO_UNS + 1;
constexpr uint32_t ATTR_HART      = ATTR_AMO_CMP + (CAS_EN ? XLEN : 0);
constexpr uint32_t ATTR_FENCE     = ATTR_HART + HART_ID_BITS;
#ifdef VX_CFG_EXT_DLMEM_ENABLE
// {is_fence_cta, is_lacc, is_addr_remote} above is_fence.
constexpr uint32_t ATTR_FENCE_CTA = ATTR_FENCE + 3;
constexpr uint32_t ATTR_BITS      = ATTR_FENCE_CTA + 1;
#else
static_assert(!CTA_FENCE, "CTA fences exist in DLMEM builds only");
constexpr uint32_t ATTR_BITS      = ATTR_FENCE + 1;
#endif

static_assert((WORD_SIZE & (WORD_SIZE - 1)) == 0 && WORD_SIZE >= 4 && WORD_SIZE <= 64, "unsupported WORD_SIZE");
static_assert(SECTOR_SIZE % WORD_SIZE == 0 && LINE_SIZE % SECTOR_SIZE == 0, "unsupported line geometry");
static_assert(SECTOR_SIZE <= 64, "memory byte-enables are modeled up to 64 bytes");
static_assert(SETS_PER_BANK >= 1, "cache too small for its banks and ways");
static_assert(TAG_BITS <= 64 && MEM_TAG_BITS <= 64, "tags are modeled up to 64 bits");

// A mismatch between this view of the configuration and the model's port
// widths would silently misplace fields, so it fails the build instead.
constexpr size_t vl_bytes(uint32_t bits) {
  return (bits <= 8) ? 1 : (bits <= 16) ? 2 : (bits <= 32) ? 4 : (bits <= 64) ? 8 : ((bits + 31) / 32) * 4;
}
#define PORT_WIDTH_CHECK(port, bits) \
  static_assert(sizeof(std::declval<VVX_cache_top&>().port[0]) == vl_bytes(bits), \
                #port " width differs from the testbench configuration")
PORT_WIDTH_CHECK(core_req_byteen, WORD_SIZE);
PORT_WIDTH_CHECK(core_req_addr, WORD_ADDR_BITS);
PORT_WIDTH_CHECK(core_req_attr, ATTR_BITS);
PORT_WIDTH_CHECK(core_req_data, WORD_SIZE * 8);
PORT_WIDTH_CHECK(core_req_tag, TAG_BITS);
PORT_WIDTH_CHECK(mem_req_byteen, SECTOR_SIZE);
PORT_WIDTH_CHECK(mem_req_addr, SECTOR_ADDR_BITS);
PORT_WIDTH_CHECK(mem_req_data, SECTOR_SIZE * 8);
PORT_WIDTH_CHECK(mem_req_tag, MEM_TAG_BITS);
static_assert(sizeof(std::declval<VVX_cache_top&>().core_req_valid) == NUM_REQS, "NUM_REQS differs from the model");
static_assert(sizeof(std::declval<VVX_cache_top&>().mem_req_valid) == MEM_PORTS, "MEM_PORTS differs from the model");

// VX_gpu_pkg amo_op_e
enum AmoOp : uint8_t {
  AMO_LR = 0, AMO_SC = 1, AMO_ADD = 2, AMO_SWAP = 3, AMO_XOR = 4,
  AMO_OR = 5, AMO_AND = 6, AMO_MIN = 7, AMO_MAX = 8, AMO_CAS = 9
};

struct AmoKind {
  uint8_t op;
  bool    uns;
};

// Store-bearing atomics other than SC, which is only issued after an LR.
static const AmoKind STORE_AMOS[] = {
  {AMO_ADD, false}, {AMO_SWAP, false}, {AMO_XOR, false}, {AMO_OR, false}, {AMO_AND, false},
  {AMO_MIN, false}, {AMO_MAX, false}, {AMO_MIN, true}, {AMO_MAX, true}, {AMO_CAS, false}
};
constexpr uint32_t NUM_STORE_AMOS = CAS_EN ? 10 : 9;

static const char* amo_name(uint8_t op, bool uns) {
  switch (op) {
  case AMO_LR:   return "LR";
  case AMO_SC:   return "SC";
  case AMO_ADD:  return "AMOADD";
  case AMO_SWAP: return "AMOSWAP";
  case AMO_XOR:  return "AMOXOR";
  case AMO_OR:   return "AMOOR";
  case AMO_AND:  return "AMOAND";
  case AMO_MIN:  return uns ? "AMOMINU" : "AMOMIN";
  case AMO_MAX:  return uns ? "AMOMAXU" : "AMOMAX";
  case AMO_CAS:  return "AMOCAS";
  default:       return "AMO?";
  }
}

///////////////////////////////////////////////////////////////////////////////

// Bit-field access that works on every Verilator port type (CData..QData
// and VlWide), so the testbench is independent of the configured widths.
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

template <typename T>
static void set_bytes(T& sig, const uint8_t* src, uint32_t n) {
  for (uint32_t i = 0; i < n; ++i) {
    set_bits(sig, 8 * i, 8, src[i]);
  }
}

template <typename T>
static void get_bytes(const T& sig, uint8_t* dst, uint32_t n) {
  for (uint32_t i = 0; i < n; ++i) {
    dst[i] = static_cast<uint8_t>(get_bits(sig, 8 * i, 8));
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

using Word   = std::array<uint8_t, WORD_SIZE>;
using Sector = std::array<uint8_t, SECTOR_SIZE>;

// Little-endian value of n bytes at off.
static uint64_t le_get(const uint8_t* p, uint32_t off, uint32_t n) {
  uint64_t v = 0;
  for (uint32_t i = 0; i < n; ++i) {
    v |= static_cast<uint64_t>(p[off + i]) << (8 * i);
  }
  return v;
}

static void le_put(uint8_t* p, uint32_t off, uint32_t n, uint64_t v) {
  for (uint32_t i = 0; i < n; ++i) {
    p[off + i] = static_cast<uint8_t>(v >> (8 * i));
  }
}

static std::string hex_bytes(const uint8_t* p, uint32_t lo, uint32_t hi) {
  std::string s = "0x";
  for (uint32_t i = hi + 1; i-- > lo;) {
    s += strf("%02x", p[i]);
  }
  return s;
}

// {size, offset} of an atomic's naturally-aligned byte-enable.
static std::pair<uint32_t, uint32_t> amo_lane(uint64_t byteen) {
  uint32_t off = 0;
  while (off < 64 && !((byteen >> off) & 1)) {
    ++off;
  }
  uint32_t size = 0;
  while ((off + size) < 64 && ((byteen >> (off + size)) & 1)) {
    ++size;
  }
  return {size, off};
}

// RVA/Zacas read-modify-write on a size-byte operand; *wrote says whether the
// result is a store for reservation purposes (a failed compare is not).
static uint64_t amo_compute(uint8_t op, bool uns, uint32_t size, uint64_t a, uint64_t b, uint64_t c, bool* wrote) {
  uint64_t mask = (size == 8) ? ~0ull : 0xffffffffull;
  a &= mask;
  b &= mask;
  c &= mask;
  int64_t as = (size == 8) ? static_cast<int64_t>(a) : static_cast<int64_t>(static_cast<int32_t>(a));
  int64_t bs = (size == 8) ? static_cast<int64_t>(b) : static_cast<int64_t>(static_cast<int32_t>(b));
  bool lt = uns ? (a < b) : (as < bs);
  uint64_t r = a;
  *wrote = true;
  switch (op) {
  case AMO_SC:
  case AMO_SWAP: r = b; break;
  case AMO_ADD:  r = a + b; break;
  case AMO_XOR:  r = a ^ b; break;
  case AMO_OR:   r = a | b; break;
  case AMO_AND:  r = a & b; break;
  case AMO_MIN:  r = lt ? a : b; break;
  case AMO_MAX:  r = lt ? b : a; break;
  case AMO_CAS:  r = (a == c) ? b : a; *wrote = (a == c); break;
  default:       *wrote = false; break;
  }
  return r & mask;
}

///////////////////////////////////////////////////////////////////////////////
// Sparse byte memory; never-written bytes hold a fixed hash of their address.

class SparseMem {
public:
  uint8_t get(uint64_t addr) const {
    auto it = blocks_.find(addr >> 6);
    return (it == blocks_.end()) ? init_byte(addr) : it->second[addr & 63];
  }

  void set(uint64_t addr, uint8_t value) {
    auto it = blocks_.find(addr >> 6);
    if (it == blocks_.end()) {
      std::array<uint8_t, 64> blk;
      for (uint32_t i = 0; i < 64; ++i) {
        blk[i] = init_byte((addr & ~63ull) + i);
      }
      it = blocks_.emplace(addr >> 6, blk).first;
    }
    it->second[addr & 63] = value;
  }

  static uint8_t init_byte(uint64_t addr) {
    uint64_t x = (addr >> 3) + 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    x ^= x >> 31;
    return static_cast<uint8_t>(x >> (8 * (addr & 7)));
  }

private:
  std::unordered_map<uint64_t, std::array<uint8_t, 64>> blocks_;
};

///////////////////////////////////////////////////////////////////////////////
// DUT driver: core ports, plus a memory model that answers fills.

enum class Role : uint8_t { None, ReadA, ReadB, X };

struct Req {
  bool     rw     = false;
  uint64_t byteen = 0;
  uint64_t addr   = 0; // byte address of the word
  Word     data{};
  bool     amo    = false;
  uint8_t  op     = 0;
  bool     uns    = false;
  uint64_t cmp    = 0;
  uint32_t hart   = 0;
  bool     flush  = false; // is_flush: a fence with `fence`, else a host flush
  bool     fence  = false;
  bool     cta    = false; // a CTA-scope fence
  uint32_t tag    = 0;
  Role     role   = Role::None;
  std::string label;
};

struct CoreRsp {
  uint32_t port;
  uint32_t tag;
  Word     data;
};

struct Events {
  std::array<bool, NUM_REQS> accepted{};
  std::array<Req, NUM_REQS>  req{};
  std::vector<CoreRsp>       rsp;
  bool activity = false;
};

class Dut {
public:
  std::array<std::optional<Req>, NUM_REQS> slot;
  std::array<bool, NUM_REQS> rsp_ready;
  uint32_t lat_min       = 2;
  uint32_t lat_max       = 2;
  uint32_t lat_long_pct  = 0;   // share of fills that take lat_long cycles instead
  uint32_t lat_long      = 0;
  uint32_t mem_ready_pct = 100;
  bool     fill_shuffle  = false;
  uint64_t watch_sector  = ~0ull;
  SparseMem mem;

  explicit Dut(std::mt19937_64& rng) : rng_(rng) {
    rsp_ready.fill(true);
    auto* t = sim_.operator->();
    for (uint32_t i = 0; i < NUM_REQS; ++i) {
      t->core_req_valid[i] = 0;
      t->core_rsp_ready[i] = 1;
    }
    for (uint32_t p = 0; p < MEM_PORTS; ++p) {
      t->mem_req_ready[p] = 0;
      t->mem_rsp_valid[p] = 0;
    }
    t->reset = 1;
    tick_ = sim_.step(tick_, 32);
    t->reset = 0;
  }

  uint64_t cycle() const {
    return cycle_;
  }

  uint64_t mem_reads() const {
    return mem_reads_;
  }

  uint64_t mem_writes() const {
    return mem_writes_;
  }

  // Memory requests that carried a fence or flush below the cache.
  uint64_t fwd_fences() const {
    return fwd_fences_;
  }

  // Sectors read and written by memory requests since the last clear.
  std::set<uint64_t> read_log, write_log;

  // Fills for a held sector wait until release().
  void hold(uint64_t sector) {
    holds_.insert(sector);
  }

  void release(uint64_t sector) {
    holds_.erase(sector);
    for (auto& q : fills_) {
      for (auto& f : q) {
        if (f.sector == sector) {
          f.held = false;
          f.ready = cycle_;
        }
      }
    }
  }

  bool fill_waiting(uint64_t sector) const {
    for (uint32_t p = 0; p < MEM_PORTS; ++p) {
      if (cur_[p] && cur_[p]->sector == sector) {
        return true;
      }
      for (auto& f : fills_[p]) {
        if (f.sector == sector) {
          return true;
        }
      }
    }
    return false;
  }

  size_t fills() const {
    size_t n = 0;
    for (uint32_t p = 0; p < MEM_PORTS; ++p) {
      n += fills_[p].size() + (cur_[p] ? 1 : 0);
    }
    return n;
  }

  // Inputs are applied, outputs sampled, then one rising edge.
  Events clock() {
    auto* t = sim_.operator->();
    for (uint32_t i = 0; i < NUM_REQS; ++i) {
      static const Req idle{};
      const Req& r = slot[i] ? *slot[i] : idle;
      uint64_t attr = 0;
      attr |= static_cast<uint64_t>(r.flush) << ATTR_FLUSH;
      attr |= static_cast<uint64_t>(r.amo) << ATTR_AMO_VALID;
      attr |= static_cast<uint64_t>(r.op & 0xf) << ATTR_AMO_OP;
      attr |= static_cast<uint64_t>(r.uns) << ATTR_AMO_UNS;
      t->core_req_valid[i] = slot[i].has_value();
      t->core_req_rw[i] = r.rw;
      set_bits(t->core_req_byteen[i], 0, WORD_SIZE, r.byteen);
      set_bits(t->core_req_addr[i], 0, WORD_ADDR_BITS, r.addr / WORD_SIZE);
      set_bits(t->core_req_attr[i], 0, ATTR_AMO_CMP, attr);
      if constexpr (CAS_EN) {
        set_bits(t->core_req_attr[i], ATTR_AMO_CMP, XLEN, r.cmp);
      }
      set_bits(t->core_req_attr[i], ATTR_HART, HART_ID_BITS, r.hart);
      set_bits(t->core_req_attr[i], ATTR_FENCE, 1, r.fence);
#ifdef VX_CFG_EXT_DLMEM_ENABLE
      set_bits(t->core_req_attr[i], ATTR_FENCE_CTA, 1, r.cta);
#endif
      set_bytes(t->core_req_data[i], r.data.data(), WORD_SIZE);
      set_bits(t->core_req_tag[i], 0, TAG_BITS, r.tag);
      t->core_rsp_ready[i] = rsp_ready[i];
    }
    for (uint32_t p = 0; p < MEM_PORTS; ++p) {
      if (!cur_[p]) {
        this->pick_fill(p);
      }
      t->mem_rsp_valid[p] = cur_[p].has_value();
      if (cur_[p]) {
        set_bytes(t->mem_rsp_data[p], cur_[p]->data.data(), SECTOR_SIZE);
        set_bits(t->mem_rsp_tag[p], 0, MEM_TAG_BITS, cur_[p]->tag);
      }
      t->mem_req_ready[p] = (rng_() % 100) < mem_ready_pct;
    }

    t->eval();

    Events ev;
    for (uint32_t i = 0; i < NUM_REQS; ++i) {
      if (slot[i] && t->core_req_ready[i]) {
        ev.accepted[i] = true;
        ev.req[i] = *slot[i];
        ev.activity = true;
      }
      if (rsp_ready[i] && t->core_rsp_valid[i]) {
        CoreRsp rsp;
        rsp.port = i;
        rsp.tag = static_cast<uint32_t>(get_bits(t->core_rsp_tag[i], 0, TAG_ID_BITS));
        get_bytes(t->core_rsp_data[i], rsp.data.data(), WORD_SIZE);
        ev.rsp.push_back(rsp);
        ev.activity = true;
      }
    }
    struct MemReq {
      uint32_t port;
      bool     rw;
      uint64_t sector;
      uint64_t byteen;
      uint64_t tag;
      Sector   data;
    };
    std::vector<MemReq> mreqs;
    std::array<bool, MEM_PORTS> consumed{};
    for (uint32_t p = 0; p < MEM_PORTS; ++p) {
      if (t->mem_req_valid[p] && t->mem_req_ready[p]) {
        MemReq m;
        m.port = p;
        m.rw = t->mem_req_rw[p];
        m.sector = get_bits(t->mem_req_addr[p], 0, SECTOR_ADDR_BITS);
        m.byteen = get_bits(t->mem_req_byteen[p], 0, SECTOR_SIZE);
        m.tag = get_bits(t->mem_req_tag[p], 0, MEM_TAG_BITS);
#ifdef VX_CFG_EXT_DLMEM_ENABLE
        if (get_bits(t->mem_req_attr[p], ATTR_FLUSH, 1) || get_bits(t->mem_req_attr[p], ATTR_FENCE, 1)
         || get_bits(t->mem_req_attr[p], ATTR_FENCE_CTA, 1)) {
          ++fwd_fences_;
        }
#endif
        get_bytes(t->mem_req_data[p], m.data.data(), SECTOR_SIZE);
        mreqs.push_back(m);
        ev.activity = true;
      }
      if (cur_[p] && t->mem_rsp_ready[p]) {
        consumed[p] = true;
        ev.activity = true;
        if (cur_[p]->sector == watch_sector) {
          std::printf("  [%" PRIu64 "] watch: fill accepted, sector 0x%" PRIx64 "\n", cycle_, cur_[p]->sector);
        }
      }
    }

    // RTL messages then report the cycle numbers the testbench prints.
    Verilated::threadContextp()->time(cycle_ + 1);
    tick_ = sim_.step(tick_, 2);
    ++cycle_;

    for (uint32_t i = 0; i < NUM_REQS; ++i) {
      if (ev.accepted[i]) {
        slot[i].reset();
      }
    }
    for (uint32_t p = 0; p < MEM_PORTS; ++p) {
      if (consumed[p]) {
        cur_[p].reset();
      }
    }
    for (auto& m : mreqs) {
      uint64_t base = m.sector * SECTOR_SIZE;
      if (m.sector == watch_sector) {
        std::printf("  [%" PRIu64 "] watch: memory %s sector 0x%" PRIx64 " byteen=0x%" PRIx64 "\n",
                    cycle_, m.rw ? "write" : "read", m.sector, m.byteen);
      }
      if (m.rw) {
        for (uint32_t b = 0; b < SECTOR_SIZE; ++b) {
          if ((m.byteen >> b) & 1) {
            mem.set(base + b, m.data[b]);
          }
        }
        ++mem_writes_;
        write_log.insert(m.sector);
      } else {
        Fill f;
        f.tag = m.tag;
        f.sector = m.sector;
        for (uint32_t b = 0; b < SECTOR_SIZE; ++b) {
          f.data[b] = mem.get(base + b);
        }
        f.held = holds_.count(m.sector) != 0;
        f.ready = cycle_ + this->fill_latency();
        fills_[m.port].push_back(f);
        ++mem_reads_;
        read_log.insert(m.sector);
      }
    }
    return ev;
  }

private:
  struct Fill {
    uint64_t tag;
    uint64_t sector;
    Sector   data;
    uint64_t ready;
    bool     held;
  };

  uint32_t fill_latency() {
    if (lat_long_pct != 0 && (rng_() % 100) < lat_long_pct) {
      return lat_long;
    }
    return lat_min + static_cast<uint32_t>(rng_() % (lat_max - lat_min + 1));
  }

  void pick_fill(uint32_t p) {
    std::vector<size_t> ready;
    for (size_t i = 0; i < fills_[p].size(); ++i) {
      const Fill& f = fills_[p][i];
      if (!f.held && f.ready <= cycle_) {
        ready.push_back(i);
        if (!fill_shuffle) {
          break;
        }
      }
    }
    if (ready.empty()) {
      return;
    }
    size_t i = ready[fill_shuffle ? (rng_() % ready.size()) : 0];
    cur_[p] = fills_[p][i];
    fills_[p].erase(fills_[p].begin() + i);
  }

  vl_simulator<VVX_cache_top> sim_;
  std::mt19937_64& rng_;
  std::array<std::deque<Fill>, MEM_PORTS> fills_;
  std::array<std::optional<Fill>, MEM_PORTS> cur_;
  std::set<uint64_t> holds_;
  uint64_t tick_       = 0;
  uint64_t cycle_      = 0;
  uint64_t mem_reads_  = 0;
  uint64_t mem_writes_ = 0;
  uint64_t fwd_fences_ = 0;
};

///////////////////////////////////////////////////////////////////////////////

struct Options {
  uint32_t seed          = 1;
  int      window        = 10;
  uint32_t stress_seeds  = 2;
  uint32_t stress_cycles = 20000;
  uint32_t rsp_ready     = 85;    // stress: % of cycles each response port is ready
  uint32_t mem_ready     = 85;    // stress: % of cycles memory accepts a request
  uint32_t max_report    = 20;
  bool     verbose       = false;
  bool     no_assert     = false;
  uint32_t stress_fence  = 0;     // stress: % of issues that are fences
  uint64_t watch         = ~0ull; // word whose traffic is traced
  std::string tests      = "race,chain,chainbp,wbqburst,bypass,chainload,settle,fence,fencecta,fstorm,stress";
};

struct Hang {};

class Testbench {
public:
  explicit Testbench(const Options& opts)
    : opts_(opts)
    , rng_(opts.seed)
    , dut_(rng_) {
    // Keep running past RTL assertion failures: each one is counted and
    // attributed to the scenario that raised it.
    Verilated::threadContextp()->errorLimit(1 << 30);
    if (opts.no_assert) {
      Verilated::threadContextp()->assertOn(false);
    }
    if (opts.watch != ~0ull) {
      opts_.watch = opts.watch - (opts.watch % WORD_SIZE);
      dut_.watch_sector = opts_.watch / SECTOR_SIZE;
    }
  }

  int run() {
    std::printf("cache: XLEN=%u reqs=%u banks=%u word=%u line=%u sector=%u ways=%u size=%u sets/bank=%u "
                "latency=%u writeback=%d mshr=%u mem_ports=%u amo=%d zacas=%d shared=%d harts=%u seed=%u\n",
                XLEN, NUM_REQS, NUM_BANKS, WORD_SIZE, LINE_SIZE, SECTOR_SIZE, NUM_WAYS, CACHE_SIZE,
                SETS_PER_BANK, LATENCY, WRITEBACK, MSHR_SIZE, MEM_PORTS, AMO_EN, CAS_EN, SHARED, NUM_HARTS, opts_.seed);
    try {
      if (!this->calibrate()) {
        std::printf("FAILED: calibration\n");
        return 1;
      }
      if (AMO_EN) {
        if (this->selected("race")) {
          this->race_sweep();
        }
        if (this->selected("chain")) {
          this->chain_store_sweep();
        }
        if (this->selected("chainbp")) {
          this->chain_bp_sweep();
        }
        if (this->selected("wbqburst")) {
          this->wbq_burst_sweep();
        }
      } else {
        std::printf("atomics disabled (AMO_ENABLE=0): directed atomic sweeps skipped\n");
      }
      if (this->selected("bypass")) {
        this->bypass_sweep();
      }
      if (AMO_EN && this->selected("chainload")) {
        this->chain_load_sweep();
      }
      if (AMO_EN && this->selected("settle")) {
        this->settle_sweep();
      }
      if (this->selected("fence")) {
        this->fence_rounds();
      }
      if (this->selected("fencecta")) {
        this->fence_cta_round();
      }
      if (this->selected("fstorm")) {
        this->fence_storms();
      }
      for (uint32_t s = 0; this->selected("stress") && s < opts_.stress_seeds; ++s) {
        this->stress(opts_.seed * 7919u + s);
      }
    } catch (const Hang&) {
      this->dump_pending();
      ++failures_;
    }
    std::printf("total: cycles=%" PRIu64 " requests=%" PRIu64 " mem_reads=%" PRIu64 " mem_writes=%" PRIu64
                " mismatches=%u (diverged words=%zu, repeat reads of them=%u) rtl_assertions=%" PRIu64 "\n",
                dut_.cycle(), seq_, dut_.mem_reads(), dut_.mem_writes(), mismatches_, diverged_.size(),
                follow_on_, asserts_);
    if (!assert_cycles_.empty()) {
      std::string s;
      for (uint64_t c : assert_cycles_) {
        s += strf(" %" PRIu64, c);
      }
      std::printf("RTL assertion cycles:%s%s\n", s.c_str(), (asserts_ > assert_cycles_.size()) ? " ..." : "");
    }
    if (failures_ != 0 || asserts_ != 0) {
      std::printf("FAILED: %u check(s) failed, %" PRIu64 " RTL assertion(s)\n", failures_, asserts_);
      return 1;
    }
    std::printf("PASSED\n");
    return 0;
  }

private:
  enum class Kind : uint8_t { Load, Amo, Lr, Sc };

  struct Pending {
    Kind     kind;
    uint32_t port;
    uint64_t addr;
    uint64_t byteen;
    Word     expect;
    Word     data;
    uint32_t hart;
    uint64_t seq;
    uint64_t cycle;
    Role     role;
    std::string label;
  };

  struct ResEvent {
    uint64_t seq;
    uint32_t hart;
    uint64_t cycle;
    uint64_t addr;
  };

  struct LrRecord {
    uint64_t seq;
    uint64_t line;
    uint64_t addr;
    uint64_t cycle;
  };

  struct ScRecord {
    uint32_t hart;
    uint64_t line;
    uint64_t addr;
    uint64_t seq;
    uint64_t cycle;
    std::string label;
  };

  static constexpr uint32_t NO_HART = ~0u;

  ///////////////////////////////////////////////////////////////////////////
  // core bookkeeping

  bool selected(const char* name) const {
    std::string list = "," + opts_.tests + ",";
    return list.find(std::string(",") + name + ",") != std::string::npos;
  }

  static uint64_t key(uint32_t port, uint32_t tag) {
    return (static_cast<uint64_t>(port) << 32) | tag;
  }

  uint32_t rand_below(uint32_t n) {
    return static_cast<uint32_t>(rng_() % n);
  }

  void fail(const std::string& msg) {
    ++failures_;
    if (failures_ <= 200) {
      std::printf("%s\n", msg.c_str());
    } else if (failures_ == 201) {
      std::printf("... further failure messages suppressed\n");
    }
  }

  Word ref_word(uint64_t addr) const {
    Word w;
    for (uint32_t b = 0; b < WORD_SIZE; ++b) {
      w[b] = ref_.get(addr + b);
    }
    return w;
  }

  void ref_write(uint64_t addr, uint64_t byteen, const Word& data) {
    for (uint32_t b = 0; b < WORD_SIZE; ++b) {
      if ((byteen >> b) & 1) {
        ref_.set(addr + b, data[b]);
      }
    }
  }

  void log_write(uint64_t addr, uint32_t hart, uint64_t seq, uint64_t cycle) {
    writes_[addr / LINE_SIZE].push_back({seq, hart, cycle, addr});
  }

  bool slot_has_word(uint64_t addr) const {
    for (auto& s : dut_.slot) {
      if (s && s->addr == addr) {
        return true;
      }
    }
    return false;
  }

  bool sc_blocked(uint64_t addr) const {
    return sc_words_.count(addr) != 0;
  }

  uint32_t alloc_tag(uint32_t port) {
    for (;;) {
      tag_ = (tag_ + 1) & ((1u << TAG_ID_BITS) - 1);
      bool busy = pending_.count(key(port, tag_)) != 0;
      for (auto& s : dut_.slot) {
        busy |= (s && s->tag == tag_);
      }
      if (!busy) {
        return tag_;
      }
    }
  }

  void on_accept(uint32_t port, const Req& r) {
    if (r.addr != opts_.watch) {
      this->apply_accept(port, r);
      return;
    }
    Word before = this->ref_word(r.addr);
    this->apply_accept(port, r);
    Word after = this->ref_word(r.addr);
    std::printf("  [%" PRIu64 "] watch: accept port %u tag 0x%x seq %" PRIu64 " %s%s byteen=0x%" PRIx64
                " data=%s model %s -> %s\n", dut_.cycle(), port, r.tag, seq_, r.label.c_str(),
                r.rw ? " (write)" : "", r.byteen, hex_bytes(r.data.data(), 0, WORD_SIZE - 1).c_str(),
                hex_bytes(before.data(), 0, WORD_SIZE - 1).c_str(), hex_bytes(after.data(), 0, WORD_SIZE - 1).c_str());
  }

  void apply_accept(uint32_t port, const Req& r) {
    uint64_t seq = ++seq_;
    uint64_t line = r.addr / LINE_SIZE;
    if (!r.amo && r.rw) {
      this->ref_write(r.addr, r.byteen, r.data);
      this->log_write(r.addr, NO_HART, seq, dut_.cycle());
      return;
    }
    Pending p;
    p.port = port;
    p.addr = r.addr;
    p.byteen = r.byteen;
    p.hart = r.hart;
    p.seq = seq;
    p.cycle = dut_.cycle();
    p.role = r.role;
    p.label = r.label;
    p.data = r.data;
    p.expect = this->ref_word(r.addr);
    if (!r.amo) {
      p.kind = Kind::Load;
    } else if (r.op == AMO_LR) {
      p.kind = Kind::Lr;
      lrs_[r.hart].push_back({seq, line, r.addr, dut_.cycle()});
    } else if (r.op == AMO_SC) {
      p.kind = Kind::Sc;
    } else {
      p.kind = Kind::Amo;
      auto [size, off] = amo_lane(r.byteen);
      Word nw = p.expect;
      bool wrote = false;
      uint64_t v = amo_compute(r.op, r.uns, size, le_get(p.expect.data(), off, size),
                               le_get(r.data.data(), off, size), r.cmp, &wrote);
      le_put(nw.data(), off, size, v);
      this->ref_write(r.addr, r.byteen, nw);
      if (wrote) {
        this->log_write(r.addr, r.hart, seq, dut_.cycle());
      }
    }
    pending_.emplace(key(port, r.tag), p);
    if (opts_.verbose) {
      std::printf("  [%" PRIu64 "] accept port=%u tag=0x%x %s\n", dut_.cycle(), port, r.tag, r.label.c_str());
    }
  }

  void on_response(const CoreRsp& rsp) {
    auto it = pending_.find(key(rsp.port, rsp.tag));
    if (it == pending_.end()) {
      this->fail(strf("FAIL [cycle %" PRIu64 "] unexpected response on port %u, tag 0x%x",
                      dut_.cycle(), rsp.port, rsp.tag));
      return;
    }
    Pending p = it->second;
    pending_.erase(it);
    if (p.addr == opts_.watch) {
      std::printf("  [%" PRIu64 "] watch: response port %u tag 0x%x %s data=%s expected=%s byteen=0x%" PRIx64 "\n",
                  dut_.cycle(), rsp.port, rsp.tag, p.label.c_str(), hex_bytes(rsp.data.data(), 0, WORD_SIZE - 1).c_str(),
                  hex_bytes(p.expect.data(), 0, WORD_SIZE - 1).c_str(), p.byteen);
    }
    if (p.kind == Kind::Sc) {
      auto [size, off] = amo_lane(p.byteen);
      uint64_t v = le_get(rsp.data.data(), off, size);
      uint64_t line = p.addr / LINE_SIZE;
      sc_words_.erase(p.addr);
      if (v > 1) {
        this->fail(strf("FAIL [cycle %" PRIu64 "] %s: SC result 0x%" PRIx64 " is neither 0 nor 1",
                        dut_.cycle(), p.label.c_str(), v));
        return;
      }
      if (v == 0) {
        this->ref_write(p.addr, p.byteen, p.data);
        this->log_write(p.addr, p.hart, p.seq, p.cycle);
        sc_ok_.push_back({p.hart, line, p.addr, p.seq, p.cycle, p.label});
      } else if (p.role == Role::X) {
        // nothing else touches the reserved line in the directed sweep
        this->fail(strf("FAIL [cycle %" PRIu64 "] %s: SC failed although its reservation was undisturbed",
                        dut_.cycle(), p.label.c_str()));
      }
      ++(v == 0 ? sc_pass_ : sc_fail_);
      return;
    }
    uint32_t lo = WORD_SIZE, hi = 0;
    for (uint32_t b = 0; b < WORD_SIZE; ++b) {
      if (((p.byteen >> b) & 1) && rsp.data[b] != p.expect[b]) {
        lo = std::min(lo, b);
        hi = std::max(hi, b);
      }
    }
    if (lo > hi) {
      return;
    }
    ++mismatches_;
    bool first = diverged_.insert(p.addr).second;
    if (!first) {
      ++follow_on_;
      return;
    }
    if (p.role == Role::ReadA) {
      ++lost_x_;
    } else if (p.role == Role::ReadB) {
      ++lost_y_;
    }
    this->fail(strf("FAIL [cycle %" PRIu64 "] %s (port %u, accepted cycle %" PRIu64 "): bytes [%u:%u] expected %s, got %s",
                    dut_.cycle(), p.label.c_str(), p.port, p.cycle, hi, lo,
                    hex_bytes(p.expect.data(), lo, hi).c_str(), hex_bytes(rsp.data.data(), lo, hi).c_str()));
  }

  void tick() {
    Events ev = dut_.clock();
    for (uint32_t i = 0; i < NUM_REQS; ++i) {
      if (ev.accepted[i]) {
        this->on_accept(i, ev.req[i]);
      }
    }
    for (auto& r : ev.rsp) {
      this->on_response(r);
    }
    uint64_t errs = Verilated::threadContextp()->errorCount();
    if (errs != errs_seen_) {
      asserts_ += errs - errs_seen_;
      errs_seen_ = errs;
      if (assert_cycles_.size() < 64) {
        assert_cycles_.push_back(dut_.cycle());
      }
    }
    idle_ = ev.activity ? 0 : (idle_ + 1);
    if (idle_ > 20000) {
      std::printf("FAIL [cycle %" PRIu64 "] no progress for %u cycles\n", dut_.cycle(), idle_);
      throw Hang{};
    }
  }

  void idle(uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
      this->tick();
    }
  }

  // Places a request on a port once the port is free; returns its tag.
  uint32_t present(uint32_t port, Req r) {
    while (dut_.slot[port]) {
      this->tick();
    }
    r.tag = this->alloc_tag(port);
    if (r.amo && r.op == AMO_SC) {
      sc_words_.insert(r.addr);
    }
    dut_.slot[port] = r;
    return r.tag;
  }

  void wait_accept(uint32_t port) {
    while (dut_.slot[port]) {
      this->tick();
    }
  }

  void wait_response(uint32_t port, uint32_t tag) {
    while ((dut_.slot[port] && dut_.slot[port]->tag == tag) || pending_.count(key(port, tag))) {
      this->tick();
    }
  }

  void blocking(uint32_t port, const Req& r) {
    uint32_t tag = this->present(port, r);
    this->wait_response(port, tag);
  }

  void drain() {
    for (;;) {
      bool busy = !pending_.empty() || dut_.fills() != 0;
      for (auto& s : dut_.slot) {
        busy |= s.has_value();
      }
      if (!busy) {
        break;
      }
      this->tick();
    }
    this->idle(4);
  }

  // Pushes the line holding `addr` out of the cache with fills of fresh lines
  // of its set, then reads the word back, so the check covers the memory image.
  void evict_and_reread(uint64_t addr, uint32_t port, Role role, const std::string& label) {
    uint64_t set_bank = (addr / LINE_SIZE) % (uint64_t(SETS_PER_BANK) * NUM_BANKS);
    for (uint32_t i = 0; i <= NUM_WAYS; ++i) {
      uint64_t line = EVICT_LINE_BASE + (evict_tag_++) * SETS_PER_BANK * NUM_BANKS + set_bank;
      this->blocking(port, this->make_load(line * LINE_SIZE, "evict"));
    }
    Req r = this->make_load(addr - (addr % WORD_SIZE), label);
    r.role = role;
    this->blocking(port, r);
  }

  void dump_pending() const {
    std::printf("outstanding requests at cycle %" PRIu64 ": %zu, fills in memory: %zu\n",
                dut_.cycle(), pending_.size(), dut_.fills());
    uint32_t n = 0;
    for (auto& kv : pending_) {
      if (++n > 16) {
        break;
      }
      std::printf("  port %u accepted cycle %" PRIu64 ": %s\n", kv.second.port, kv.second.cycle, kv.second.label.c_str());
    }
  }

  // A successful SC is legal only if its hart's last LR reserved the same line
  // and no other hart wrote the pair's word in between. Acceptance order only
  // binds requests to one word: the bank lets a hit to another word of a line
  // pass that line's chained misses, so a foreign write elsewhere in the
  // reserved line may commit outside the LR/SC window and leave it legal.
  void check_reservations() {
    for (auto& sc : sc_ok_) {
      const LrRecord* lr = nullptr;
      for (auto& e : lrs_[sc.hart]) {
        if (e.seq < sc.seq) {
          lr = &e;
        }
      }
      if (lr == nullptr || lr->line != sc.line) {
        this->fail(strf("FAIL %s (accepted cycle %" PRIu64 ") succeeded without a reservation on its line",
                        sc.label.c_str(), sc.cycle));
        continue;
      }
      for (auto& w : writes_[sc.line]) {
        if (w.hart != sc.hart && w.addr == sc.addr && lr->addr == sc.addr && w.seq > lr->seq && w.seq < sc.seq) {
          this->fail(strf("FAIL %s (accepted cycle %" PRIu64 ") succeeded although %s wrote 0x%" PRIx64 " at cycle %"
                          PRIu64 ", after the LR at cycle %" PRIu64,
                          sc.label.c_str(), sc.cycle,
                          (w.hart == NO_HART) ? "a plain store" : strf("hart %u", w.hart).c_str(), w.addr, w.cycle, lr->cycle));
          break;
        }
      }
    }
    sc_ok_.clear();
    lrs_.clear();
    writes_.clear();
  }

  ///////////////////////////////////////////////////////////////////////////
  // request builders

  Word rand_word() {
    Word w;
    for (auto& b : w) {
      b = static_cast<uint8_t>(rng_());
    }
    return w;
  }

  Req make_load(uint64_t addr, const std::string& label) {
    Req r;
    r.addr = addr;
    r.byteen = (WORD_SIZE == 64) ? ~0ull : ((1ull << WORD_SIZE) - 1);
    r.data = this->rand_word();
    r.label = label;
    return r;
  }

  // An atomic on the size-byte operand at byte address a: the operand rides
  // the request data at its byte offset, the comparand right-aligned.
  Req make_amo(uint8_t op, bool uns, uint64_t a, uint32_t size, uint64_t operand, uint64_t cmp,
               uint32_t hart, const std::string& label) {
    Req r;
    uint32_t off = static_cast<uint32_t>(a % WORD_SIZE);
    r.addr = a - off;
    r.byteen = ((size == 8) ? 0xffull : 0xfull) << off;
    r.data = this->rand_word();
    le_put(r.data.data(), off, size, operand);
    r.amo = true;
    r.op = op;
    r.uns = uns;
    r.cmp = cmp;
    r.hart = hart;
    r.label = strf("%s.%s 0x%" PRIx64 " hart %u %s", amo_name(op, uns), (size == 8) ? "D" : "W", a, hart, label.c_str());
    return r;
  }

  // An operand that makes the atomic change the current value v, so a lost
  // update is visible; ADD stands in where the op cannot change v.
  Req make_visible_amo(AmoKind k, uint64_t a, uint32_t size, uint32_t hart, const std::string& label) {
    uint64_t mask = (size == 8) ? ~0ull : 0xffffffffull;
    Word cur = this->ref_word(a - (a % WORD_SIZE));
    uint64_t v = le_get(cur.data(), static_cast<uint32_t>(a % WORD_SIZE), size);
    uint64_t r = rng_() & mask;
    uint64_t bit = 1ull << rand_below(size * 8);
    uint64_t smin = (size == 8) ? (1ull << 63) : 0x80000000ull;
    uint64_t sv = (v ^ smin);  // signed order as unsigned
    switch (k.op) {
    case AMO_SWAP:
      return this->make_amo(AMO_SWAP, false, a, size, (r == v) ? (r ^ 1) : r, 0, hart, label);
    case AMO_XOR:
      return this->make_amo(AMO_XOR, false, a, size, bit, 0, hart, label);
    case AMO_OR:
      if (v != mask) {
        uint64_t b = bit;
        while (v & b) {
          b = (b << 1) & mask;
          if (b == 0) {
            b = 1;
          }
        }
        return this->make_amo(AMO_OR, false, a, size, b, 0, hart, label);
      }
      break;
    case AMO_AND:
      if (v != 0) {
        uint64_t b = bit;
        while (!(v & b)) {
          b = (b << 1) & mask;
          if (b == 0) {
            b = 1;
          }
        }
        return this->make_amo(AMO_AND, false, a, size, mask & ~b, 0, hart, label);
      }
      break;
    case AMO_MIN:
      if (k.uns ? (v != 0) : (sv != 0)) {
        uint64_t lim = k.uns ? v : sv;
        uint64_t o = r % lim;
        return this->make_amo(AMO_MIN, k.uns, a, size, k.uns ? o : ((o ^ smin) & mask), 0, hart, label);
      }
      break;
    case AMO_MAX:
      if (k.uns ? (v != mask) : (sv != mask)) {
        uint64_t lim = k.uns ? v : sv;
        uint64_t o = lim + 1 + (r % (mask - lim));
        return this->make_amo(AMO_MAX, k.uns, a, size, k.uns ? o : ((o ^ smin) & mask), 0, hart, label);
      }
      break;
    case AMO_CAS:
      return this->make_amo(AMO_CAS, false, a, size, v ^ bit, v, hart, label);
    default:
      break;
    }
    return this->make_amo(AMO_ADD, false, a, size, (r == 0) ? 1 : r, 0, hart, label);
  }

  uint32_t amo_size(uint32_t pick) const {
    return (MAX_AMO_SIZE == 8 && (pick & 1)) ? 8 : 4;
  }

  uint32_t aligned_offset(uint32_t size) {
    return size * rand_below(WORD_SIZE / size);
  }

  ///////////////////////////////////////////////////////////////////////////
  // calibration: the basic store/load/atomic/LR-SC paths respond as modeled

  bool calibrate() {
    uint32_t f0 = failures_;
    uint64_t base = 0x00100000ull;
    for (uint32_t i = 0; i < 4 * NUM_BANKS; ++i) {
      uint64_t a = base + i * LINE_SIZE + (i % WORDS_PER_LINE) * WORD_SIZE;
      Req st;
      st.rw = true;
      st.addr = a;
      st.byteen = (WORD_SIZE == 64) ? ~0ull : ((1ull << WORD_SIZE) - 1);
      st.data = this->rand_word();
      st.label = strf("calibration store 0x%" PRIx64, a);
      this->blocking(i % NUM_REQS, st);
      this->blocking((i + 1) % NUM_REQS, this->make_load(a, strf("calibration load 0x%" PRIx64, a)));
      if (AMO_EN) {
        this->blocking(i % NUM_REQS, this->make_amo(AMO_ADD, false, a, 4, 1, 0, 0, "calibration"));
        this->blocking((i + 2) % NUM_REQS, this->make_load(a, strf("calibration load-after-amo 0x%" PRIx64, a)));
        uint64_t b = a + LINE_SIZE / 2;
        b -= b % 4;
        uint32_t h = i % NUM_HARTS;
        this->blocking(i % NUM_REQS, this->make_amo(AMO_LR, false, b, 4, 0, 0, h, "calibration"));
        Req sc = this->make_amo(AMO_SC, false, b, 4, 0x5a5a0000u + i, 0, h, "calibration");
        sc.role = Role::X;
        this->blocking(i % NUM_REQS, sc);
        this->blocking((i + 3) % NUM_REQS, this->make_load(b - (b % WORD_SIZE), "calibration load-after-sc"));
      }
    }
    this->drain();
    this->check_reservations();
    std::printf("calibration: %s\n", (failures_ == f0) ? "ok" : "FAILED");
    return failures_ == f0;
  }

  ///////////////////////////////////////////////////////////////////////////
  // Directed race. Line B misses; its MSHR chain is {plain read(s), Y} where
  // Y is a store-bearing atomic. The leading reads complete through the fill
  // forward port, which leaves the input free, so a core atomic X on resident
  // line A can enter the pipe right before Y replays. X is presented at every
  // offset around the cycle B's fill is released, in both orders.

  struct RaceCase {
    uint32_t bank;
    int      offset;  // fill release cycle minus X presentation cycle
    uint32_t variant;
    uint32_t nload;   // plain reads heading B's chain
    AmoKind  x, y;
    uint32_t sx, sy;
    bool     x_sc;
    bool     b_first; // B allocated below A
    uint32_t port_x, port_y;
  };

  uint64_t fresh_line(uint32_t bank) {
    uint64_t g = (race_line_++) * NUM_BANKS + bank;
    return RACE_LINE_BASE + g;
  }

  RaceCase make_case(uint32_t bank, int offset, uint32_t variant) {
    RaceCase rc;
    rc.bank = bank;
    rc.offset = offset;
    rc.variant = variant;
    uint32_t k = rot_++;
    switch (variant) {
    case 0:
      rc.nload = 1;
      rc.x = {AMO_ADD, false};
      rc.y = {AMO_ADD, false};
      rc.sx = rc.sy = 4;
      rc.x_sc = false;
      rc.b_first = false;
      rc.port_x = 0;
      rc.port_y = 1 % NUM_REQS;
      break;
    case 1:
      rc.nload = 2;
      rc.x = STORE_AMOS[k % NUM_STORE_AMOS];
      rc.y = STORE_AMOS[(k + 3) % NUM_STORE_AMOS];
      rc.sx = this->amo_size(k);
      rc.sy = this->amo_size(k + 1);
      rc.x_sc = false;
      rc.b_first = false;
      rc.port_x = 2 % NUM_REQS;
      rc.port_y = 3 % NUM_REQS;
      break;
    default:
      rc.nload = 1;
      rc.x = STORE_AMOS[(k + 5) % NUM_STORE_AMOS];
      rc.y = STORE_AMOS[(k + 1) % NUM_STORE_AMOS];
      rc.sx = this->amo_size(k + 1);
      rc.sy = this->amo_size(k);
      rc.x_sc = (k % 2) == 0;
      rc.b_first = true;
      rc.port_x = 1 % NUM_REQS;
      rc.port_y = 0;
      break;
    }
    return rc;
  }

  std::string case_name(const RaceCase& rc) const {
    return strf("bank=%u offset=%+d variant=%u reads=%u X=%s.%s(port %u) Y=%s.%s(port %u)%s",
                rc.bank, rc.offset, rc.variant, rc.nload,
                rc.x_sc ? "SC" : amo_name(rc.x.op, rc.x.uns), (rc.sx == 8) ? "D" : "W", rc.port_x,
                amo_name(rc.y.op, rc.y.uns), (rc.sy == 8) ? "D" : "W", rc.port_y,
                rc.b_first ? " B<A" : "");
  }

  bool run_race(const RaceCase& rc) {
    uint64_t line_a, line_b;
    if (rc.b_first) {
      line_b = this->fresh_line(rc.bank);
      line_a = this->fresh_line(rc.bank);
    } else {
      line_a = this->fresh_line(rc.bank);
      line_b = this->fresh_line(rc.bank);
    }
    uint32_t wa = rand_below(WORDS_PER_LINE);
    uint32_t wb = rand_below(WORDS_PER_LINE);
    uint64_t word_a = line_a * LINE_SIZE + wa * WORD_SIZE;
    uint64_t word_b = line_b * LINE_SIZE + wb * WORD_SIZE;
    uint64_t xa = word_a + this->aligned_offset(rc.sx);
    uint64_t yb = word_b + this->aligned_offset(rc.sy);
    uint64_t sector_b = word_b / SECTOR_SIZE;
    uint64_t sector_word0 = sector_b * SECTOR_SIZE;
    uint32_t hart_x = rand_below(NUM_HARTS);
    uint32_t hart_y = (hart_x + 1) % NUM_HARTS;
    std::string name = this->case_name(rc);

    uint32_t f0 = failures_;
    uint64_t a0 = asserts_;
    uint32_t lx0 = lost_x_, ly0 = lost_y_;
    this->drain();
    if (opts_.verbose) {
      std::printf("race %s A=0x%" PRIx64 " B=0x%" PRIx64 "\n", name.c_str(), xa, yb);
    }

    this->blocking(rc.port_x, this->make_load(word_a, "race: fill A"));
    if (rc.x_sc) {
      this->blocking(rc.port_x, this->make_amo(AMO_LR, false, xa, rc.sx, 0, 0, hart_x, "race: reserve A"));
    }
    dut_.hold(sector_b);
    for (uint32_t i = 0; i < rc.nload; ++i) {
      uint64_t w = sector_word0 + rand_below(WORDS_PER_SECTOR) * WORD_SIZE;
      this->present((rc.port_y + 1 + i) % NUM_REQS, this->make_load(w, "race: read B"));
      this->wait_accept((rc.port_y + 1 + i) % NUM_REQS);
    }
    this->present(rc.port_y, this->make_visible_amo(rc.y, yb, rc.sy, hart_y, "race: Y on missing B"));
    this->wait_accept(rc.port_y);
    while (!dut_.fill_waiting(sector_b)) {
      this->tick();
    }
    this->idle(8);

    Req x;
    if (rc.x_sc) {
      Word cur = this->ref_word(word_a);
      uint64_t v = le_get(cur.data(), static_cast<uint32_t>(xa - word_a), rc.sx);
      x = this->make_amo(AMO_SC, false, xa, rc.sx, ~v, 0, hart_x, "race: X on resident A");
    } else {
      x = this->make_visible_amo(rc.x, xa, rc.sx, hart_x, "race: X on resident A");
    }
    x.role = Role::X;
    int tx = std::max(0, rc.offset);
    int tf = std::max(0, -rc.offset);
    bool x_placed = false, released = false;
    for (int c = 0;; ++c) {
      if (c == tx) {
        this->present(rc.port_x, x);
        x_placed = true;
      }
      if (c == tf) {
        dut_.release(sector_b);
        released = true;
      }
      this->tick();
      if (x_placed && released && !dut_.slot[rc.port_x] && !dut_.fill_waiting(sector_b)) {
        break;
      }
    }
    this->drain();

    Req ra = this->make_load(word_a, "race: read back A");
    ra.role = Role::ReadA;
    this->blocking(rc.port_y, ra);
    Req rb = this->make_load(word_b, "race: read back B");
    rb.role = Role::ReadB;
    this->blocking(rc.port_x, rb);
    this->evict_and_reread(word_a, rc.port_y, Role::ReadA, "race: read back A after eviction");
    this->evict_and_reread(word_b, rc.port_x, Role::ReadB, "race: read back B after eviction");
    this->drain();
    this->check_reservations();

    bool failed = (failures_ != f0) || (asserts_ != a0);
    if (failed) {
      ++race_failing_;
      race_offsets_[rc.bank].insert(rc.offset);
      if (race_failing_ <= opts_.max_report) {
        std::printf("FAIL race %s: A=0x%" PRIx64 " B=0x%" PRIx64 ": rtl_assertions=%" PRIu64 " lost_x=%u lost_y=%u\n",
                    name.c_str(), xa, yb, asserts_ - a0, lost_x_ - lx0, lost_y_ - ly0);
      }
    }
    return !failed;
  }

  void race_sweep() {
    uint32_t runs = 0;
    for (uint32_t bank = 0; bank < NUM_BANKS; ++bank) {
      for (int d = -opts_.window; d <= opts_.window; ++d) {
        for (uint32_t v = 0; v < 3; ++v) {
          this->run_race(this->make_case(bank, d, v));
          ++runs;
        }
      }
    }
    std::printf("race sweep: %u scenarios over offsets [%d, %d], %u failing, lost updates X=%u Y=%u\n",
                runs, -opts_.window, opts_.window, race_failing_, lost_x_, lost_y_);
    for (auto& kv : race_offsets_) {
      std::string s;
      for (int d : kv.second) {
        s += strf(" %+d", d);
      }
      std::printf("  bank %u failing offsets:%s\n", kv.first, s.c_str());
    }
  }

  ///////////////////////////////////////////////////////////////////////////
  // Directed chain: a store-bearing atomic Y and a plain store S to the same
  // word of missing line B, presented `offset` cycles apart (S later when
  // positive) so that S can join B's MSHR chain right behind Y. After the
  // fill, S replays one cycle behind Y and commits while Y's result is still
  // queued for writeback; whichever is younger must own the overlapping bytes.

  bool run_chain_store(uint32_t bank, int offset, uint32_t variant) {
    uint64_t line_b = this->fresh_line(bank);
    uint64_t word_b = line_b * LINE_SIZE + rand_below(WORDS_PER_LINE) * WORD_SIZE;
    uint64_t sector_b = word_b / SECTOR_SIZE;
    AmoKind k = STORE_AMOS[rot_ % NUM_STORE_AMOS];
    uint32_t size = this->amo_size(rot_ / NUM_STORE_AMOS);
    ++rot_;
    uint32_t off = this->aligned_offset(size);
    uint32_t port_y = variant % NUM_REQS;
    uint32_t port_s = (variant + 1) % NUM_REQS;
    uint64_t full = (WORD_SIZE == 64) ? ~0ull : ((1ull << WORD_SIZE) - 1);
    uint64_t ybe = ((size == 8) ? 0xffull : 0xfull) << off;
    uint64_t sbe = full;                                  // the whole word
    if (variant == 1) {
      sbe = ybe;                                          // exactly Y's bytes
    } else if (variant == 2 && WORD_SIZE > size) {
      sbe = ybe | (((off >= size) ? (ybe >> size) : (ybe << size)) & full); // Y's bytes plus a neighbour
    }
    Req s;
    s.rw = true;
    s.addr = word_b;
    s.byteen = sbe;
    s.data = this->rand_word();
    s.label = strf("chain: store S 0x%" PRIx64 " byteen=0x%" PRIx64, word_b, sbe);
    std::string name = strf("bank=%u offset=%+d variant=%u Y=%s.%s@+%u(port %u) S byteen=0x%" PRIx64 "(port %u)",
                            bank, offset, variant, amo_name(k.op, k.uns), (size == 8) ? "D" : "W", off, port_y,
                            sbe, port_s);

    uint32_t f0 = failures_;
    uint64_t a0 = asserts_;
    uint32_t m0 = mismatches_;
    this->drain();
    if (opts_.verbose) {
      std::printf("chain %s B=0x%" PRIx64 "\n", name.c_str(), word_b);
    }
    dut_.hold(sector_b);
    Req y = this->make_visible_amo(k, word_b + off, size, rand_below(NUM_HARTS), "chain: Y on missing B");
    int ty = std::max(0, -offset);
    int ts = std::max(0, offset);
    // S may legitimately wait at the input behind the pending atomic until
    // the fill returns, so only Y's acceptance gates the release.
    uint32_t y_tag = 0;
    bool y_placed = false, s_placed = false;
    for (int c = 0;; ++c) {
      if (c == ty) {
        y_tag = this->present(port_y, y);
        y_placed = true;
      }
      if (c == ts) {
        this->present(port_s, s);
        s_placed = true;
      }
      this->tick();
      if (y_placed && s_placed && !(dut_.slot[port_y] && dut_.slot[port_y]->tag == y_tag)) {
        break;
      }
    }
    while (!dut_.fill_waiting(sector_b)) {
      this->tick();
    }
    this->idle(4);
    dut_.release(sector_b);
    this->drain();
    Req rb = this->make_load(word_b, "chain: read back B");
    rb.role = Role::ReadB;
    this->blocking(port_s, rb);
    this->evict_and_reread(word_b, port_s, Role::ReadB, "chain: read back B after eviction");
    this->drain();
    this->check_reservations();

    bool failed = (failures_ != f0) || (asserts_ != a0);
    if (failed) {
      ++chain_failing_;
      chain_offsets_[bank].insert(offset);
      chain_lost_ += mismatches_ - m0;
      if (chain_failing_ <= opts_.max_report) {
        std::printf("FAIL chain %s: B=0x%" PRIx64 ": rtl_assertions=%" PRIu64 " mismatches=%u\n",
                    name.c_str(), word_b, asserts_ - a0, mismatches_ - m0);
      }
    }
    return !failed;
  }

  void chain_store_sweep() {
    uint32_t runs = 0;
    for (uint32_t bank = 0; bank < NUM_BANKS; ++bank) {
      for (int d = -2; d <= 4; ++d) {
        for (uint32_t v = 0; v < 3; ++v) {
          this->run_chain_store(bank, d, v);
          ++runs;
        }
      }
    }
    std::printf("chain sweep: %u scenarios over offsets [-2, 4], %u failing, lost stores=%u\n",
                runs, chain_failing_, chain_lost_);
    for (auto& kv : chain_offsets_) {
      std::string s;
      for (int d : kv.second) {
        s += strf(" %+d", d);
      }
      std::printf("  bank %u failing offsets:%s\n", kv.first, s.c_str());
    }
  }

  ///////////////////////////////////////////////////////////////////////////
  // Directed chain under memory back-pressure: Y then S to one word of
  // missing line B, back to back on one port. While memory refuses requests,
  // stores to other lines of the bank fill its memory-request queue, so a
  // write-through bank holds S's replay (a write) after Y has replayed and
  // queued its result. When memory resumes, S replays and Y's queued result
  // can be ready to write while S is still ahead of it in the pipe.

  bool run_chain_bp(uint32_t bank, uint32_t nfill, uint32_t hold, uint32_t variant) {
    uint64_t line_b = this->fresh_line(bank);
    uint64_t word_b = line_b * LINE_SIZE + rand_below(WORDS_PER_LINE) * WORD_SIZE;
    uint64_t sector_b = word_b / SECTOR_SIZE;
    AmoKind k = STORE_AMOS[rot_ % NUM_STORE_AMOS];
    uint32_t size = this->amo_size(rot_ / NUM_STORE_AMOS);
    ++rot_;
    uint32_t off = this->aligned_offset(size);
    uint32_t port = 0;
    uint32_t port_f = 1 % NUM_REQS;
    uint64_t full = (WORD_SIZE == 64) ? ~0ull : ((1ull << WORD_SIZE) - 1);
    uint64_t ybe = ((size == 8) ? 0xffull : 0xfull) << off;
    Req s;
    s.rw = true;
    s.addr = word_b;
    s.byteen = (variant & 1) ? ybe : full;
    s.data = this->rand_word();
    s.label = strf("chainbp: store S 0x%" PRIx64, word_b);
    std::string name = strf("bank=%u fillers=%u hold=%u variant=%u Y=%s.%s@+%u S byteen=0x%" PRIx64,
                            bank, nfill, hold, variant, amo_name(k.op, k.uns), (size == 8) ? "D" : "W", off, s.byteen);

    uint32_t f0 = failures_;
    uint64_t a0 = asserts_;
    uint32_t m0 = mismatches_;
    this->drain();
    if (opts_.verbose) {
      std::printf("chainbp %s B=0x%" PRIx64 "\n", name.c_str(), word_b);
    }
    dut_.hold(sector_b);
    this->present(port, this->make_visible_amo(k, word_b + off, size, rand_below(NUM_HARTS), "chainbp: Y on missing B"));
    this->present(port, s);
    // S may legitimately wait at the input behind the pending atomic until
    // the fill returns, so its acceptance does not gate the release.
    for (uint32_t i = 0; i < 16 && dut_.slot[port]; ++i) {
      this->tick();
    }
    while (!dut_.fill_waiting(sector_b)) {
      this->tick();
    }
    dut_.mem_ready_pct = 0;
    for (uint32_t j = 0; j < nfill; ++j) {
      if (!dut_.slot[port_f]) {
        Req f;
        f.rw = true;
        f.addr = this->fresh_line(bank) * LINE_SIZE;
        f.byteen = full;
        f.data = this->rand_word();
        f.label = "chainbp: filler store";
        this->present(port_f, f);
      }
      this->idle(1);
    }
    this->idle(4);
    dut_.release(sector_b);
    this->idle(hold);
    dut_.mem_ready_pct = 100;
    this->drain();
    Req rb = this->make_load(word_b, "chainbp: read back B");
    rb.role = Role::ReadB;
    this->blocking(port, rb);
    this->evict_and_reread(word_b, port, Role::ReadB, "chainbp: read back B after eviction");
    this->drain();
    this->check_reservations();

    bool failed = (failures_ != f0) || (asserts_ != a0);
    if (failed) {
      ++chainbp_failing_;
      if (chainbp_failing_ <= opts_.max_report) {
        std::printf("FAIL chainbp %s: B=0x%" PRIx64 ": rtl_assertions=%" PRIu64 " mismatches=%u\n",
                    name.c_str(), word_b, asserts_ - a0, mismatches_ - m0);
      }
    }
    return !failed;
  }

  void chain_bp_sweep() {
    uint32_t runs = 0;
    for (uint32_t bank = 0; bank < NUM_BANKS; ++bank) {
      for (uint32_t nfill : {0u, 6u, 10u, 16u}) {
        for (uint32_t hold = 0; hold <= 24; hold += 8) {
          for (uint32_t v = 0; v < 2; ++v) {
            this->run_chain_bp(bank, nfill, hold, v);
            ++runs;
          }
        }
      }
    }
    std::printf("chainbp sweep: %u scenarios over fillers {0, 6, 10, 16} and holds [0, 24], %u failing\n",
                runs, chainbp_failing_);
  }

  ///////////////////////////////////////////////////////////////////////////
  // Directed writeback burst: n store-bearing atomics to n distinct missing
  // lines of one bank, their fills held and then released together, or `gap`
  // cycles apart. Fills and the atomics' replays then alternate at the bank
  // input, so every replay queues a result while the queued ones find no slot
  // to drain.

  bool run_wbq_burst(uint32_t bank, uint32_t n, uint32_t gap) {
    std::vector<uint64_t> words, sectors;
    for (uint32_t i = 0; i < n; ++i) {
      uint64_t line = this->fresh_line(bank);
      uint64_t w = line * LINE_SIZE + rand_below(WORDS_PER_LINE) * WORD_SIZE;
      words.push_back(w);
      sectors.push_back(w / SECTOR_SIZE);
    }
    std::string name = strf("bank=%u n=%u gap=%u", bank, n, gap);
    uint32_t f0 = failures_;
    uint64_t a0 = asserts_;
    uint32_t m0 = mismatches_;
    this->drain();
    if (opts_.verbose) {
      std::printf("wbqburst %s start=%" PRIu64 "\n", name.c_str(), dut_.cycle());
    }
    for (auto s : sectors) {
      dut_.hold(s);
    }
    // An input that refuses atomics until a fill returns must not hold the
    // remaining presentations hostage: what does not fit now goes after the
    // release.
    auto make_y = [&](uint32_t i) {
      AmoKind k = STORE_AMOS[rot_ % NUM_STORE_AMOS];
      uint32_t size = this->amo_size(rot_ / NUM_STORE_AMOS);
      ++rot_;
      uint32_t off = this->aligned_offset(size);
      return this->make_visible_amo(k, words[i] + off, size, rand_below(NUM_HARTS), "wbqburst: atomic on missing line");
    };
    uint32_t placed = 0;
    for (; placed < n; ++placed) {
      uint32_t port = placed % NUM_REQS;
      for (uint32_t w = 0; w < 200 && dut_.slot[port]; ++w) {
        this->tick();
      }
      if (dut_.slot[port]) {
        break;
      }
      this->present(port, make_y(placed));
    }
    uint64_t t0 = dut_.cycle();
    for (;;) {
      bool all = true;
      for (auto& sl : dut_.slot) {
        all &= !sl.has_value();
      }
      for (uint32_t i = 0; i < placed; ++i) {
        all &= dut_.fill_waiting(sectors[i]);
      }
      if (all || dut_.cycle() - t0 > 2000) {
        break;
      }
      this->tick();
    }
    this->idle(4);
    if (opts_.verbose) {
      std::printf("wbqburst release=%" PRIu64 "\n", dut_.cycle());
    }
    for (auto s : sectors) {
      dut_.release(s);
      if (gap) {
        this->idle(gap);
      }
    }
    if (opts_.verbose && placed < n) {
      std::printf("wbqburst presented %u of %u before the release\n", placed, n);
    }
    for (; placed < n; ++placed) {
      this->present(placed % NUM_REQS, make_y(placed));
    }
    this->drain();
    for (uint32_t i = 0; i < n; ++i) {
      Req rb = this->make_load(words[i], "wbqburst: read back");
      rb.role = Role::ReadB;
      this->blocking(i % NUM_REQS, rb);
    }
    this->evict_and_reread(words[0], 0, Role::ReadB, "wbqburst: read back after eviction");
    this->drain();
    this->check_reservations();
    bool failed = (failures_ != f0) || (asserts_ != a0);
    if (failed) {
      ++wbq_failing_;
      if (wbq_failing_ <= opts_.max_report) {
        std::printf("FAIL wbqburst %s: rtl_assertions=%" PRIu64 " mismatches=%u\n",
                    name.c_str(), asserts_ - a0, mismatches_ - m0);
      }
    }
    return !failed;
  }

  void wbq_burst_sweep() {
    uint32_t runs = 0;
    for (uint32_t bank = 0; bank < NUM_BANKS; ++bank) {
      for (uint32_t n : {3u, 4u, 5u, 6u, 8u, 12u}) {
        for (uint32_t gap : {0u, 6u}) {
          this->run_wbq_burst(bank, n, gap);
          ++runs;
        }
      }
    }
    std::printf("wbqburst sweep: %u scenarios, %u failing\n", runs, wbq_failing_);
  }

  ///////////////////////////////////////////////////////////////////////////
  // Directed bypass: B's MSHR chain is {plain read(s), S} with S a plain store
  // to word w, and X (an atomic, or a plain load) to w is presented around
  // B's fill release. The leading reads complete through the forward port, so
  // X can reach the freshly filled line while S still waits in the chain; X
  // arrived after S and must observe it (on S's own port this is program order).
  // Variants 4 and 5 turn it around: the chain is {read, S, L} with S a store
  // to another part of the line and L a load of w, and X is a plain store (4)
  // or a store-bearing atomic (5) to w, which L must not observe.

  bool run_bypass(uint32_t bank, int offset, uint32_t variant) {
    uint64_t line_b = this->fresh_line(bank);
    uint32_t wb = rand_below(WORDS_PER_LINE);
    uint64_t word_b = line_b * LINE_SIZE + wb * WORD_SIZE;
    uint64_t sector_b = word_b / SECTOR_SIZE;
    uint64_t sector_word0 = sector_b * SECTOR_SIZE;
    uint32_t nload = (variant == 2) ? 2 : 1;
    uint32_t port_s = 1 % NUM_REQS;
    uint32_t port_x = (variant == 2 || variant >= 4) ? 0 : port_s;
    uint32_t size = this->amo_size(rot_);
    uint32_t off = this->aligned_offset(size);
    AmoKind k = STORE_AMOS[rot_ % NUM_STORE_AMOS];
    ++rot_;
    bool x_lr = (variant == 0);
    bool x_store = (variant == 4);
    bool x_after_load = (variant >= 4);
    bool x_load = (variant == 3) || (!AMO_EN && !x_store);
    std::string x_name = x_store ? std::string("STORE") : x_load ? std::string("LOAD")
                       : strf("%s.%s@+%u", x_lr ? "LR" : amo_name(k.op, k.uns), (size == 8) ? "D" : "W", off);
    std::string name = strf("bank=%u offset=%+d variant=%u reads=%u X=%s(port %u) S(port %u)",
                            bank, offset, variant, nload, x_name.c_str(), port_x, port_s);

    uint32_t f0 = failures_;
    uint64_t a0 = asserts_;
    uint32_t m0 = mismatches_;
    this->drain();
    if (opts_.verbose) {
      std::printf("bypass %s B=0x%" PRIx64 "\n", name.c_str(), word_b);
    }
    dut_.hold(sector_b);
    for (uint32_t i = 0; i < nload; ++i) {
      uint32_t port = (port_s + 1 + i) % NUM_REQS;
      this->present(port, this->make_load(sector_word0 + rand_below(WORDS_PER_SECTOR) * WORD_SIZE, "bypass: read B"));
      this->wait_accept(port);
    }
    uint64_t full = (WORD_SIZE == 64) ? ~0ull : ((1ull << WORD_SIZE) - 1);
    Req s;
    s.rw = true;
    s.addr = word_b;
    s.byteen = full;
    s.data = this->rand_word();
    if (x_after_load) {
      // S closes the forward window without touching w's bytes X will race on
      if (WORDS_PER_SECTOR > 1) {
        s.addr = sector_word0 + ((wb % WORDS_PER_SECTOR + 1) % WORDS_PER_SECTOR) * WORD_SIZE;
      } else {
        s.byteen = full >> (WORD_SIZE / 2);
      }
    }
    s.label = strf("bypass: store S 0x%" PRIx64, s.addr);
    this->present(port_s, s);
    this->wait_accept(port_s);
    if (x_after_load) {
      this->present(port_s, this->make_load(word_b, "bypass: load L before X"));
      this->wait_accept(port_s);
    }
    while (!dut_.fill_waiting(sector_b)) {
      this->tick();
    }
    this->idle(8);

    uint32_t hart = rand_below(NUM_HARTS);
    Req x;
    if (x_store) {
      x.rw = true;
      x.addr = word_b;
      x.byteen = full;
      x.data = this->rand_word();
      x.label = strf("bypass: store X 0x%" PRIx64, word_b);
    } else if (x_load) {
      x = this->make_load(word_b, "bypass: load X after S");
    } else if (x_lr) {
      x = this->make_amo(AMO_LR, false, word_b + off, size, 0, 0, hart, "bypass: X after S");
    } else {
      x = this->make_visible_amo(k, word_b + off, size, hart, x_after_load ? "bypass: X after L" : "bypass: X after S");
    }
    int tx = std::max(0, offset);
    int tf = std::max(0, -offset);
    bool x_placed = false, released = false;
    for (int c = 0;; ++c) {
      if (c == tx) {
        this->present(port_x, x);
        x_placed = true;
      }
      if (c == tf) {
        dut_.release(sector_b);
        released = true;
      }
      this->tick();
      if (x_placed && released && !dut_.slot[port_x] && !dut_.fill_waiting(sector_b)) {
        break;
      }
    }
    this->drain();
    Req rb = this->make_load(word_b, "bypass: read back B");
    rb.role = Role::ReadB;
    this->blocking(port_x, rb);
    this->evict_and_reread(word_b, port_x, Role::ReadB, "bypass: read back B after eviction");
    this->drain();
    this->check_reservations();

    bool failed = (failures_ != f0) || (asserts_ != a0);
    if (failed) {
      ++bypass_failing_;
      bypass_offsets_[bank].insert(offset);
      if (bypass_failing_ <= opts_.max_report) {
        std::printf("FAIL bypass %s: B=0x%" PRIx64 ": rtl_assertions=%" PRIu64 " mismatches=%u\n",
                    name.c_str(), word_b, asserts_ - a0, mismatches_ - m0);
      }
    }
    return !failed;
  }

  void bypass_sweep() {
    uint32_t runs = 0;
    for (uint32_t bank = 0; bank < NUM_BANKS; ++bank) {
      for (int d = -opts_.window; d <= opts_.window; ++d) {
        for (uint32_t v = AMO_EN ? 0 : 3; v < (AMO_EN ? 6u : 5u); ++v) {
          this->run_bypass(bank, d, v);
          ++runs;
        }
      }
    }
    std::printf("bypass sweep: %u scenarios over offsets [%d, %d], %u failing\n",
                runs, -opts_.window, opts_.window, bypass_failing_);
    for (auto& kv : bypass_offsets_) {
      std::string s;
      for (int d : kv.second) {
        s += strf(" %+d", d);
      }
      std::printf("  bank %u failing offsets:%s\n", kv.first, s.c_str());
    }
  }

  ///////////////////////////////////////////////////////////////////////////
  // Directed chained load: B's MSHR chain holds `nread` plain reads of B's
  // sector (the forwardable head), a store-bearing atomic Y on word w joins
  // it, B's fill is released dy cycles after Y is presented, and a plain load
  // X of w is presented dx cycles after Y, on Y's own port (variants 0 and 2:
  // one requester's program order) or on another port (1 and 3). X was
  // accepted after Y, so it must observe Y's result.

  bool run_chain_load(uint32_t bank, int dy, int dx, uint32_t variant) {
    uint64_t line_b = this->fresh_line(bank);
    uint64_t sector_word0 = line_b * LINE_SIZE + rand_below(LINE_SIZE / SECTOR_SIZE) * SECTOR_SIZE;
    uint32_t wsec = rand_below(WORDS_PER_SECTOR);
    uint64_t word_b = sector_word0 + wsec * WORD_SIZE;
    uint64_t sector_b = word_b / SECTOR_SIZE;
    uint32_t nread = (variant >= 2) ? 2 : 1;
    uint32_t port_y = 0;
    uint32_t port_x = (variant & 1) ? (1 % NUM_REQS) : port_y;
    AmoKind k = STORE_AMOS[rot_ % NUM_STORE_AMOS];
    uint32_t size = this->amo_size(rot_ / NUM_STORE_AMOS);
    ++rot_;
    uint32_t off = this->aligned_offset(size);
    std::string name = strf("bank=%u dy=%+d dx=%+d variant=%u reads=%u Y=%s.%s@+%u(port %u) X=LOAD(port %u)",
                            bank, dy, dx, variant, nread, amo_name(k.op, k.uns), (size == 8) ? "D" : "W", off,
                            port_y, port_x);

    uint32_t f0 = failures_;
    uint64_t a0 = asserts_;
    uint32_t m0 = mismatches_;
    this->drain();
    if (opts_.verbose) {
      std::printf("chainload %s B=0x%" PRIx64 "\n", name.c_str(), word_b);
    }
    dut_.hold(sector_b);
    for (uint32_t i = 0; i < nread; ++i) {
      uint32_t port = (port_y + 2 + i) % NUM_REQS;
      uint64_t w = sector_word0 + ((wsec + 1 + i) % WORDS_PER_SECTOR) * WORD_SIZE;
      this->present(port, this->make_load(w, "chainload: read B"));
      this->wait_accept(port);
    }
    while (!dut_.fill_waiting(sector_b)) {
      this->tick();
    }
    this->idle(8);
    Req y = this->make_visible_amo(k, word_b + off, size, rand_below(NUM_HARTS), "chainload: Y on missing B");
    Req x = this->make_load(word_b, "chainload: load X after Y");
    x.role = Role::ReadA;
    int base = std::max(0, -dy);
    int ty = base, tf = base + dy, tx = base + dx;
    bool y_placed = false, y_acc = false, x_placed = false, released = false;
    uint32_t y_tag = 0;
    // Y and X may wait at the input until the fill returns; the bound only
    // keeps a scenario the cache never admits from stalling the sweep.
    for (int c = 0; c <= 1000; ++c) {
      if (!y_placed && c >= ty && !dut_.slot[port_y]) {
        y_tag = this->present(port_y, y);
        y_placed = true;
      }
      if (!released && c >= tf) {
        dut_.release(sector_b);
        released = true;
      }
      if (y_placed && !y_acc && !(dut_.slot[port_y] && dut_.slot[port_y]->tag == y_tag)) {
        y_acc = true;
      }
      if (!x_placed && c >= tx && ((port_x != port_y) ? !dut_.slot[port_x] : y_acc)) {
        this->present(port_x, x);
        x_placed = true;
      }
      this->tick();
      if (x_placed && released && !dut_.slot[port_x] && !dut_.fill_waiting(sector_b)) {
        break;
      }
    }
    this->drain();
    Req rb = this->make_load(word_b, "chainload: read back B");
    rb.role = Role::ReadB;
    this->blocking(port_x, rb);
    this->evict_and_reread(word_b, port_x, Role::ReadB, "chainload: read back B after eviction");
    this->drain();
    this->check_reservations();

    bool failed = (failures_ != f0) || (asserts_ != a0);
    if (failed) {
      ++chainload_failing_;
      chainload_cases_[variant].insert(strf("(%+d,%+d)", dy, dx));
      if (chainload_failing_ <= opts_.max_report) {
        std::printf("FAIL chainload %s: B=0x%" PRIx64 ": rtl_assertions=%" PRIu64 " mismatches=%u\n",
                    name.c_str(), word_b, asserts_ - a0, mismatches_ - m0);
      }
    }
    return !failed;
  }

  void chain_load_sweep() {
    uint32_t runs = 0;
    for (uint32_t bank = 0; bank < NUM_BANKS; ++bank) {
      for (uint32_t v = 0; v < 4; ++v) {
        for (int dy = -2; dy <= 4; ++dy) {
          for (int dx = 1; dx <= 4; ++dx) {
            this->run_chain_load(bank, dy, dx, v);
            ++runs;
          }
        }
      }
    }
    std::printf("chainload sweep: %u scenarios over dy [-2, 4], dx [1, 4], %u failing\n", runs, chainload_failing_);
    for (auto& kv : chainload_cases_) {
      std::string s;
      for (auto& d : kv.second) {
        s += " " + d;
      }
      std::printf("  variant %u failing (dy,dx):%s\n", kv.first, s.c_str());
    }
  }

  ///////////////////////////////////////////////////////////////////////////
  // Directed settle: B's MSHR chain is {Y, n plain loads of another word of
  // B, X}, with Y a store-bearing atomic on word w and X a load of w, or
  // {Y, Y2, n loads, X} with Y2 an atomic on the other word. The chain's
  // response port is held not-ready from before B's fill release until h
  // cycles after, behind k blocker responses, so the pipe stalls at a
  // different point of the replay in each run. X can replay ahead of Y's
  // writeback; it then reads Y's result from the settling writeback, which
  // must survive the stall and a second writeback behind it.

  // Bounded: while a fill is held, the cache may legitimately defer a request.
  bool accepted_within(uint32_t port, uint32_t tag, uint32_t cycles) {
    for (uint32_t i = 0; i < cycles && dut_.slot[port] && dut_.slot[port]->tag == tag; ++i) {
      this->tick();
    }
    return !(dut_.slot[port] && dut_.slot[port]->tag == tag);
  }

  // Puts r into B's chain while B's fill is held. The bank's input guard is
  // not exact (see VX_cache_amo.sv), and two requesters share an unbuffered
  // crossbar: a decoy to resident word_a is refused once on port_d, then r,
  // on the port the arbiter now favours, enters on the decoy's registered
  // same-line probe. The last request to fire must be on port_d.
  bool settle_enter(uint32_t port_d, uint32_t port_r, const Req& r, uint64_t word_a) {
    uint32_t decoy = this->present(port_d, this->make_load(word_a, "settle: decoy on resident A"));
    this->tick();
    uint32_t tag = this->present(port_r, r);
    bool entered = this->accepted_within(port_r, tag, 16);
    return this->accepted_within(port_d, decoy, 16) && entered;
  }

  bool run_settle(uint32_t bank, uint32_t variant, uint32_t nload, uint32_t nblock, uint32_t hold) {
    constexpr uint32_t SECTORS_PER_LINE = LINE_SIZE / SECTOR_SIZE;
    uint64_t line_a = this->fresh_line(bank);
    uint64_t line_b = this->fresh_line(bank);
    uint64_t word_a = line_a * LINE_SIZE;
    uint32_t wy = rand_below(WORDS_PER_LINE);
    uint32_t span = (WORDS_PER_SECTOR > 1) ? WORDS_PER_SECTOR : WORDS_PER_LINE;
    uint32_t wo = (wy - wy % span) + (wy % span + 1 + rand_below(span - 1)) % span;
    uint64_t word_y = line_b * LINE_SIZE + wy * WORD_SIZE;
    uint64_t word_o = line_b * LINE_SIZE + wo * WORD_SIZE;
    uint32_t port_c = 1;  // the chain
    uint32_t port_d = 0;  // decoys
    AmoKind ky = STORE_AMOS[rot_ % NUM_STORE_AMOS];
    uint32_t size = this->amo_size(rot_ / NUM_STORE_AMOS);
    ++rot_;
    uint32_t off = this->aligned_offset(size);
    std::string name = strf("bank=%u variant=%u loads=%u blockers=%u hold=%u Y=%s.%s@+%u",
                            bank, variant, nload, nblock, hold, amo_name(ky.op, ky.uns), (size == 8) ? "D" : "W", off);

    uint32_t f0 = failures_;
    uint64_t a0 = asserts_;
    uint32_t m0 = mismatches_;
    this->drain();
    if (opts_.verbose) {
      std::printf("settle %s B=0x%" PRIx64 " other=0x%" PRIx64 " start=%" PRIu64 "\n",
                  name.c_str(), word_y, word_o, dut_.cycle());
    }
    this->blocking(port_d, this->make_load(word_a, "settle: fill A"));
    for (uint32_t s = 0; s < SECTORS_PER_LINE; ++s) {
      dut_.hold(line_b * SECTORS_PER_LINE + s);
    }
    this->present(port_c, this->make_visible_amo(ky, word_y + off, size, rand_below(NUM_HARTS),
                                                 "settle: Y on missing B"));
    this->wait_accept(port_c);
    while (!dut_.fill_waiting(word_y / SECTOR_SIZE)) {
      this->tick();
    }
    this->idle(4);
    uint32_t decoy = this->present(port_d, this->make_load(word_a, "settle: decoy on resident A"));
    bool formed = this->accepted_within(port_d, decoy, 16);
    if (formed && variant == 1) {
      AmoKind k2 = STORE_AMOS[rot_ % NUM_STORE_AMOS];
      uint32_t size2 = this->amo_size(rot_ / NUM_STORE_AMOS);
      ++rot_;
      Req y2 = this->make_visible_amo(k2, word_o + this->aligned_offset(size2), size2, rand_below(NUM_HARTS),
                                      "settle: Y2 on missing B");
      formed = this->settle_enter(port_d, port_c, y2, word_a);
    }
    for (uint32_t i = 0; formed && i < nload; ++i) {
      formed = this->settle_enter(port_d, port_c, this->make_load(word_o, "settle: load of the other word"), word_a);
    }
    if (formed) {
      formed = this->settle_enter(port_d, port_c, this->make_load(word_y, "settle: load X of Y's word"), word_a);
    }
    if (formed) {
      ++settle_formed_;
      dut_.rsp_ready[port_c] = false;
      for (uint32_t i = 0; i < nblock; ++i) {
        uint32_t tag = this->present(port_c, this->make_load(word_a, "settle: blocker"));
        if (!this->accepted_within(port_c, tag, 16)) {
          break;  // the response path is full: the pipe holds
        }
      }
      this->idle(4);
    }
    if (opts_.verbose) {
      std::printf("settle release=%" PRIu64 " formed=%d\n", dut_.cycle(), formed);
    }
    for (uint32_t s = 0; s < SECTORS_PER_LINE; ++s) {
      dut_.release(line_b * SECTORS_PER_LINE + s);
    }
    this->idle(hold);
    dut_.rsp_ready.fill(true);
    this->drain();
    Req rb = this->make_load(word_y, "settle: read back Y's word");
    rb.role = Role::ReadB;
    this->blocking(port_d, rb);
    Req ro = this->make_load(word_o, "settle: read back the other word");
    ro.role = Role::ReadB;
    this->blocking(port_d, ro);
    this->drain();
    this->check_reservations();

    bool failed = (failures_ != f0) || (asserts_ != a0);
    if (failed) {
      ++settle_failing_;
      if (settle_failing_ <= opts_.max_report) {
        std::printf("FAIL settle %s: B=0x%" PRIx64 ": formed=%d rtl_assertions=%" PRIu64 " mismatches=%u\n",
                    name.c_str(), word_y, formed, asserts_ - a0, mismatches_ - m0);
      }
    }
    return !failed;
  }

  void settle_sweep() {
    if (NUM_REQS != 2 || WORDS_PER_LINE < 2) {
      std::printf("settle sweep: skipped (needs two requesters and two words per line)\n");
      return;
    }
    uint32_t runs = 0;
    for (uint32_t bank = 0; bank < NUM_BANKS; ++bank) {
      for (uint32_t v = 0; v < 2; ++v) {
        for (uint32_t n = 0; n <= 3; ++n) {
          for (uint32_t k = 0; k <= 5; ++k) {
            for (uint32_t h : {4u, 12u, 24u}) {
              this->run_settle(bank, v, n, k, h);
              ++runs;
            }
          }
        }
      }
    }
    std::printf("settle sweep: %u scenarios, %u with the chain formed, %u failing\n",
                runs, settle_formed_, settle_failing_);
  }

  ///////////////////////////////////////////////////////////////////////////
  // Randomized traffic over a footprint that overflows a few sets per bank.

  struct Hart {
    enum State : uint8_t { Idle, LrPending, WaitSc, ScPending } state = Idle;
    uint64_t addr = 0;
    uint32_t size = 4;
    uint32_t port = 0;
    uint32_t tag  = 0;
    uint64_t due  = 0;
  };

  uint64_t rand_byteen() {
    uint64_t full = (WORD_SIZE == 64) ? ~0ull : ((1ull << WORD_SIZE) - 1);
    switch (rand_below(5)) {
    case 0:
      return full;
    case 1: {
      uint64_t m = rng_() & full;
      return (m == 0) ? 1 : m;
    }
    default: {
      uint32_t size = 1u << rand_below(std::min<uint32_t>(clog2(WORD_SIZE), 3) + 1);
      return ((size == 8) ? 0xffull : ((1ull << size) - 1)) << this->aligned_offset(size);
    }
    }
  }

  bool stress_issue(uint32_t port, const std::vector<uint64_t>& words, std::vector<Hart>& harts) {
    for (uint32_t h = 0; h < harts.size(); ++h) {
      Hart& t = harts[h];
      if (t.state == Hart::WaitSc && dut_.cycle() >= t.due) {
        uint64_t w = t.addr - (t.addr % WORD_SIZE);
        if (this->sc_blocked(w) || this->slot_has_word(w)) {
          continue;
        }
        Req sc = this->make_amo(AMO_SC, false, t.addr, t.size, rng_(), 0, h, "stress");
        t.port = port;
        t.tag = this->present(port, sc);
        t.state = Hart::ScPending;
        return true;
      }
    }
    uint64_t w = words[rand_below(static_cast<uint32_t>(words.size()))];
    if (this->sc_blocked(w)) {
      return false;
    }
    if (opts_.stress_fence != 0 && rand_below(100) < opts_.stress_fence) {
      Req f = this->make_flush(w, true, strf("stress fence 0x%" PRIx64, w));
      if (CTA_FENCE) {
        f.cta = (rand_below(2) == 0);
      }
      this->present(port, f);
      return true;
    }
    uint32_t roll = AMO_EN ? rand_below(100) : rand_below(55);
    if (roll < 35) {
      Req r = this->make_load(w, strf("stress load 0x%" PRIx64, w));
      if (rand_below(4) == 0) {
        r.byteen = this->rand_byteen();
      }
      this->present(port, r);
    } else if (roll < 55) {
      Req r;
      r.rw = true;
      r.addr = w;
      r.byteen = this->rand_byteen();
      r.data = this->rand_word();
      r.label = strf("stress store 0x%" PRIx64, w);
      this->present(port, r);
    } else if (roll < 88) {
      AmoKind k = STORE_AMOS[rand_below(NUM_STORE_AMOS)];
      uint32_t size = this->amo_size(rand_below(2));
      uint64_t a = w + this->aligned_offset(size);
      uint64_t cmp = rng_();
      if (k.op == AMO_CAS && rand_below(2) == 0) {
        Word cur = this->ref_word(w);
        cmp = le_get(cur.data(), static_cast<uint32_t>(a - w), size);
      }
      this->present(port, this->make_amo(k.op, k.uns, a, size, rng_(), cmp, rand_below(NUM_HARTS), "stress"));
    } else {
      for (uint32_t h = 0; h < harts.size(); ++h) {
        Hart& t = harts[h];
        if (t.state != Hart::Idle) {
          continue;
        }
        t.size = this->amo_size(rand_below(2));
        t.addr = w + this->aligned_offset(t.size);
        t.port = port;
        t.tag = this->present(port, this->make_amo(AMO_LR, false, t.addr, t.size, 0, 0, h, "stress"));
        t.state = Hart::LrPending;
        return true;
      }
      return false;
    }
    return true;
  }

  void stress(uint32_t seed) {
    rng_.seed(seed);
    std::vector<uint64_t> words;
    uint32_t sets = std::min<uint32_t>(2, SETS_PER_BANK);
    for (uint32_t bank = 0; bank < NUM_BANKS; ++bank) {
      for (uint32_t s = 0; s < sets; ++s) {
        for (uint32_t t = 0; t <= NUM_WAYS; ++t) {
          uint64_t line = STRESS_LINE_BASE + ((uint64_t(t) * SETS_PER_BANK + s) * NUM_BANKS + bank);
          for (uint32_t w = 0; w < WORDS_PER_LINE; ++w) {
            words.push_back(line * LINE_SIZE + w * WORD_SIZE);
          }
        }
      }
    }
    std::vector<Hart> harts(std::min<uint32_t>(NUM_HARTS, 8));
    uint32_t f0 = failures_;
    uint64_t a0 = asserts_;
    uint64_t s0 = seq_;
    uint32_t m0 = mismatches_;
    uint64_t c0 = dut_.cycle();
    dut_.lat_min = 1;
    dut_.lat_max = 24;
    dut_.lat_long_pct = 8;
    dut_.lat_long = 120;
    dut_.fill_shuffle = true;
    dut_.mem_ready_pct = opts_.mem_ready;

    while (dut_.cycle() - c0 < opts_.stress_cycles) {
      for (uint32_t p = 0; p < NUM_REQS; ++p) {
        dut_.rsp_ready[p] = rand_below(100) < opts_.rsp_ready;
      }
      for (uint32_t p = 0; p < NUM_REQS; ++p) {
        if (!dut_.slot[p] && rand_below(100) < 60) {
          this->stress_issue(p, words, harts);
        }
      }
      this->tick();
      for (auto& t : harts) {
        bool done = !pending_.count(key(t.port, t.tag))
                 && !(dut_.slot[t.port] && dut_.slot[t.port]->tag == t.tag);
        if (t.state == Hart::LrPending && done) {
          t.state = Hart::WaitSc;
          t.due = dut_.cycle() + rand_below(32);
        } else if (t.state == Hart::ScPending && done) {
          t.state = Hart::Idle;
        }
      }
    }
    dut_.rsp_ready.fill(true);
    dut_.mem_ready_pct = 100;
    this->drain();
    for (size_t i = 0; i < words.size(); ++i) {
      this->present(static_cast<uint32_t>(i % NUM_REQS), this->make_load(words[i], strf("stress final read 0x%" PRIx64, words[i])));
    }
    this->drain();
    this->check_reservations();
    dut_.lat_min = dut_.lat_max = 2;
    dut_.lat_long_pct = 0;
    dut_.fill_shuffle = false;
    std::printf("stress seed=%u: %" PRIu64 " requests over %" PRIu64 " cycles, %u mismatches, %" PRIu64
                " RTL assertions, SC pass/fail=%u/%u: %s\n",
                seed, seq_ - s0, dut_.cycle() - c0, mismatches_ - m0, asserts_ - a0, sc_pass_, sc_fail_,
                (failures_ == f0 && asserts_ == a0) ? "ok" : "FAILED");
  }

  ///////////////////////////////////////////////////////////////////////////
  // Fence and flush. A fence (is_flush with is_fence) at the shared level
  // drains: it answers once every request the cache accepted before it has
  // left the cache, holds no other port and leaves the lines in place. A
  // fence above the shared level, and a host flush (is_flush alone) at every
  // level, walks: it holds every port, writes dirty lines back and
  // invalidates every line.

  // Cycles one walk spends in a bank: every set, every way of it where dirty
  // lines are written back.
  static constexpr uint32_t WALK_CYCLES = SETS_PER_BANK * (WRITEBACK ? NUM_WAYS : 1);

  Req make_flush(uint64_t addr, bool fence, const std::string& label) {
    Req r = this->make_load(addr, label);
    r.flush = true;
    r.fence = fence;
    return r;
  }

  bool responded(uint32_t port, uint32_t tag) const {
    return !(dut_.slot[port] && dut_.slot[port]->tag == tag) && !pending_.count(key(port, tag));
  }

  // Line i of a round's private group: lines with the same i share a bank and
  // a set across rounds, and no two lines of one round share a set.
  uint64_t fence_line(uint64_t round, uint32_t i) const {
    return FENCE_LINE_BASE + round * (uint64_t(SETS_PER_BANK) * NUM_BANKS) + i;
  }

  void fence_round(bool fence, bool cta = false) {
    const bool walks = !(fence && (SHARED || (cta && CTA_FENCE)));
    const std::string what = cta ? "CTA fence" : (fence ? "fence" : "host flush");
    const uint32_t f0 = failures_;
    const uint64_t a0 = asserts_;
    const uint32_t pf = 0;
    const uint32_t po = (NUM_REQS > 1) ? 1 : 0;
    const uint64_t round = fence_round_++;
    // F is the fence's own word, W are resident lines, H misses and its fill
    // is held while the fence waits behind it.
    const uint32_t nw = 2 * NUM_BANKS;
    const uint64_t f_addr = this->fence_line(round, 0) * LINE_SIZE;
    std::vector<uint64_t> w_addr;
    for (uint32_t i = 1; i <= nw; ++i) {
      w_addr.push_back(this->fence_line(round, i) * LINE_SIZE);
    }
    const uint64_t h_addr = this->fence_line(round, nw + 1) * LINE_SIZE;
    const uint64_t h_sector = h_addr / SECTOR_SIZE;

    std::vector<uint64_t> warm = w_addr;
    // A CTA fence's own word misses, so its read goes below, where it must arrive as a plain read.
    if (!cta) {
      warm.push_back(f_addr);
    }
    for (size_t i = 0; i < warm.size(); ++i) {
      Req st;
      st.rw = true;
      st.addr = warm[i];
      st.byteen = (WORD_SIZE == 64) ? ~0ull : ((1ull << WORD_SIZE) - 1);
      st.data = this->rand_word();
      st.label = strf("%s warm store 0x%" PRIx64, what.c_str(), warm[i]);
      this->blocking(static_cast<uint32_t>(i % NUM_REQS), st);
      this->blocking(static_cast<uint32_t>((i + 1) % NUM_REQS), this->make_load(warm[i], "warm load"));
    }
    this->drain();

    dut_.hold(h_sector);
    uint32_t th = this->present(pf, this->make_load(h_addr, strf("%s held miss 0x%" PRIx64, what.c_str(), h_addr)));
    for (uint32_t c = 0; !dut_.fill_waiting(h_sector); ++c) {
      if (c > 1000) {
        this->fail(strf("FAIL %s: the held load's fill request never reached memory", what.c_str()));
        dut_.release(h_sector);
        this->drain();
        return;
      }
      this->tick();
    }
    dut_.read_log.clear();
    dut_.write_log.clear();
    const uint64_t fwd0 = dut_.fwd_fences();
    // S, a write on another port accepted just before the fence, to a line X
    // whose fill is held: S waits behind that miss, and so must the fence.
    const bool older_write = cta && (NUM_REQS > 1);
    const uint64_t s_addr = this->fence_line(round, nw + 2) * LINE_SIZE;
    const uint64_t s_sector = s_addr / SECTOR_SIZE;
    uint32_t tx = 0;
    if (older_write) {
      dut_.hold(s_sector);
      tx = this->present(po, this->make_load(s_addr, strf("%s held miss 0x%" PRIx64, what.c_str(), s_addr)));
      for (uint32_t c = 0; !dut_.fill_waiting(s_sector); ++c) {
        if (c > 1000) {
          this->fail(strf("FAIL %s: the older write's line never missed to memory", what.c_str()));
          break;
        }
        this->tick();
      }
      Req st;
      st.rw = true;
      st.addr = s_addr;
      st.byteen = (WORD_SIZE == 64) ? ~0ull : ((1ull << WORD_SIZE) - 1);
      st.data = this->rand_word();
      st.label = strf("%s older store 0x%" PRIx64, what.c_str(), s_addr);
      this->present(po, st);
      this->wait_accept(po);
    }
    const uint64_t c_fence = dut_.cycle();
    Req freq = this->make_flush(f_addr, fence, what);
    freq.cta = cta;
    uint32_t tf = this->present(pf, freq);

    // A draining fence holds only its own port: hits on another port go on.
    uint32_t served = 0;
    if (!walks && NUM_REQS > 1) {
      for (uint64_t a : w_addr) {
        uint32_t t = this->present(po, this->make_load(a, strf("load beside a %s", what.c_str())));
        uint32_t c = 0;
        while (!this->responded(po, t) && c < 64) {
          this->tick();
          ++c;
        }
        if (!this->responded(po, t)) {
          this->fail(strf("FAIL %s: a hit on port %u waited %u cycles behind a draining fence", what.c_str(), po, c));
          break;
        }
        ++served;
      }
    }
    this->idle(128);
    if (this->responded(pf, tf)) {
      this->fail(strf("FAIL %s answered while an earlier miss was still waiting for its fill", what.c_str()));
    }
    dut_.release(h_sector);
    if (older_write) {
      this->idle(128);
      if (this->responded(pf, tf)) {
        this->fail(strf("FAIL %s answered while an older write from port %u waited behind a miss", what.c_str(), po));
      }
      dut_.release(s_sector);
    }
    while (!this->responded(pf, tf)) {
      this->tick();
    }
    const uint64_t c_done = dut_.cycle();
    if (older_write && !this->responded(po, tx)) {
      this->fail(strf("FAIL %s answered before the miss the older write waited behind", what.c_str()));
    }
    const uint64_t forwarded = dut_.fwd_fences() - fwd0;
    if (!this->responded(pf, th)) {
      this->fail(strf("FAIL %s answered before the earlier load on its port", what.c_str()));
    }
    this->drain();

    uint32_t written = 0;
    for (uint64_t a : warm) {
      written += dut_.write_log.count(a / SECTOR_SIZE);
    }
    const uint32_t expect_written = (walks && WRITEBACK) ? static_cast<uint32_t>(warm.size()) : 0;
    if (written != expect_written) {
      this->fail(strf("FAIL %s wrote back %u of %zu dirty lines, expected %u", what.c_str(), written, warm.size(), expect_written));
    }
    dut_.read_log.clear();
    for (size_t i = 0; i < w_addr.size(); ++i) {
      this->blocking(static_cast<uint32_t>(i % NUM_REQS), this->make_load(w_addr[i], strf("reload after %s", what.c_str())));
    }
    uint32_t refetched = 0;
    for (uint64_t a : w_addr) {
      refetched += dut_.read_log.count(a / SECTOR_SIZE);
    }
    const uint32_t expect_refetched = walks ? nw : 0;
    if (refetched != expect_refetched) {
      this->fail(strf("FAIL after a %s %u of %u resident lines missed, expected %u (%s)", what.c_str(), refetched, nw,
                      expect_refetched, walks ? "a walk invalidates them" : "a drain keeps them"));
    }
    if (cta && CTA_FENCE && forwarded != 0) {
      this->fail(strf("FAIL %s sent %" PRIu64 " fence or flush request(s) to memory", what.c_str(), forwarded));
    }
    this->drain();
    std::printf("fence round: %s %s, %" PRIu64 " cycles from issue to answer (held fill included), "
                "%u hits served beside it, %u/%u lines written back, %u/%u refetched: %s\n",
                what.c_str(), walks ? "walks" : "drains", c_done - c_fence, served, written,
                static_cast<uint32_t>(warm.size()), refetched, nw,
                (failures_ == f0 && asserts_ == a0) ? "ok" : "FAILED");
  }

  void fence_rounds() {
    this->fence_round(true);
    this->fence_round(false);
  }

  void fence_cta_round() {
    if (!CTA_FENCE) {
      std::printf("fence round: CTA fence skipped, not a CTA_FENCE cache\n");
      return;
    }
    this->fence_round(true, true);
  }

  // A stream of fences or host flushes on port 0 while the other ports load
  // and store: every regular request must be admitted within a bound.
  void fence_storm(bool fence) {
    const std::string what = fence ? "fence" : "host flush";
    if (NUM_REQS < 2) {
      std::printf("fence storm: %s skipped, one port only\n", what.c_str());
      return;
    }
    const bool walks = !fence || !SHARED;
    const uint32_t bound = 4 * (WALK_CYCLES + 128) + 512;
    const uint32_t f0 = failures_;
    const uint64_t a0 = asserts_;
    std::vector<uint64_t> words;
    for (uint32_t i = 0; i < 4 * NUM_BANKS; ++i) {
      uint64_t line = this->fence_line(fence_round_, i);
      for (uint32_t w = 0; w < WORDS_PER_LINE; w += std::max<uint32_t>(1, WORDS_PER_LINE / 4)) {
        words.push_back(line * LINE_SIZE + w * WORD_SIZE);
      }
    }
    ++fence_round_;
    dut_.lat_min = 1;
    dut_.lat_max = 24;
    std::vector<uint64_t> since(NUM_REQS, 0);
    std::vector<bool> waiting(NUM_REQS, false);
    const uint32_t target = 16 * (NUM_REQS - 1);
    uint32_t admitted = 0, flushes = 0, worst = 0;
    bool starved = false;
    const uint64_t c0 = dut_.cycle();
    while ((admitted < target || flushes < 16) && !starved) {
      if (!dut_.slot[0]) {
        uint64_t w = words[rand_below(static_cast<uint32_t>(words.size()))];
        this->present(0, this->make_flush(w, fence, strf("storm %s", what.c_str())));
        ++flushes;
      }
      for (uint32_t p = 1; p < NUM_REQS; ++p) {
        if (dut_.slot[p]) {
          continue;
        }
        uint64_t w = words[rand_below(static_cast<uint32_t>(words.size()))];
        Req r = this->make_load(w, strf("storm load 0x%" PRIx64, w));
        if (rand_below(5) < 2) {
          r.rw = true;
          r.label = strf("storm store 0x%" PRIx64, w);
        }
        this->present(p, r);
        since[p] = dut_.cycle();
        waiting[p] = true;
      }
      this->tick();
      for (uint32_t p = 1; p < NUM_REQS; ++p) {
        if (!waiting[p]) {
          continue;
        }
        uint32_t waited = static_cast<uint32_t>(dut_.cycle() - since[p]);
        if (!dut_.slot[p]) {
          waiting[p] = false;
          ++admitted;
          worst = std::max(worst, waited);
        } else if (waited > bound) {
          this->fail(strf("FAIL %s storm: port %u waited %u cycles for admission (bound %u)", what.c_str(), p, waited, bound));
          starved = true;
          break;
        }
      }
    }
    const uint64_t cycles = dut_.cycle() - c0;
    this->drain();
    dut_.lat_min = dut_.lat_max = 2;
    std::printf("fence storm: %s stream (%s), %u issued over %" PRIu64 " cycles, %u requests admitted beside it, "
                "longest wait %u (bound %u): %s\n",
                what.c_str(), walks ? "walks" : "drains", flushes, cycles, admitted, worst, bound,
                (failures_ == f0 && asserts_ == a0) ? "ok" : "FAILED");
  }

  void fence_storms() {
    this->fence_storm(true);
    this->fence_storm(false);
  }

  ///////////////////////////////////////////////////////////////////////////

  static constexpr uint64_t RACE_LINE_BASE   = 0x10000000ull / LINE_SIZE;
  static constexpr uint64_t STRESS_LINE_BASE = 0x20000000ull / LINE_SIZE;
  static constexpr uint64_t EVICT_LINE_BASE  = 0x30000000ull / LINE_SIZE;
  static constexpr uint64_t FENCE_LINE_BASE  = 0x40000000ull / LINE_SIZE;

  Options opts_;
  std::mt19937_64 rng_;
  Dut dut_;
  SparseMem ref_;
  std::map<uint64_t, Pending> pending_;
  std::unordered_set<uint64_t> sc_words_;
  std::unordered_map<uint32_t, std::vector<LrRecord>> lrs_;
  std::unordered_map<uint64_t, std::vector<ResEvent>> writes_;                  // line -> writers
  std::vector<ScRecord> sc_ok_;
  std::set<uint64_t> diverged_;
  std::map<uint32_t, std::set<int>> race_offsets_;
  std::map<uint32_t, std::set<int>> chain_offsets_;
  std::map<uint32_t, std::set<int>> bypass_offsets_;
  std::vector<uint64_t> assert_cycles_;
  uint64_t seq_       = 0;
  uint64_t asserts_   = 0;
  uint64_t errs_seen_ = 0;
  uint64_t race_line_ = 0;
  uint64_t evict_tag_ = 0;
  uint32_t rot_       = 0;
  uint32_t tag_       = 0;
  uint32_t idle_      = 0;
  uint32_t failures_  = 0;
  uint32_t mismatches_ = 0;
  uint32_t follow_on_ = 0;
  uint32_t lost_x_    = 0;
  uint32_t lost_y_    = 0;
  uint32_t sc_pass_   = 0;
  uint32_t sc_fail_   = 0;
  uint32_t race_failing_ = 0;
  uint32_t chain_failing_ = 0;
  uint32_t chain_lost_ = 0;
  uint32_t chainbp_failing_ = 0;
  uint32_t wbq_failing_ = 0;
  uint32_t bypass_failing_ = 0;
  uint32_t chainload_failing_ = 0;
  uint32_t settle_formed_ = 0;
  uint32_t settle_failing_ = 0;
  uint64_t fence_round_ = 0;
  std::map<uint32_t, std::set<std::string>> chainload_cases_;
};

static void usage(const char* prog) {
  std::printf("usage: %s [--seed=N] [--window=N] [--stress-seeds=N] [--stress-cycles=N] [--rsp-ready=PCT] "
              "[--mem-ready=PCT] [--stress-fence=PCT] [--max-report=N] "
              "[--tests=race,chain,chainbp,wbqburst,bypass,chainload,settle,fence,fstorm,stress] [--no-assert] [--watch=ADDR] [--verbose]\n", prog);
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);

  Options opts;
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    if (a[0] == '+') {
      continue;
    }
    if (std::strncmp(a, "--seed=", 7) == 0) {
      opts.seed = static_cast<uint32_t>(std::strtoul(a + 7, nullptr, 0));
    } else if (std::strncmp(a, "--window=", 9) == 0) {
      opts.window = static_cast<int>(std::strtol(a + 9, nullptr, 0));
    } else if (std::strncmp(a, "--stress-seeds=", 15) == 0) {
      opts.stress_seeds = static_cast<uint32_t>(std::strtoul(a + 15, nullptr, 0));
    } else if (std::strncmp(a, "--stress-cycles=", 16) == 0) {
      opts.stress_cycles = static_cast<uint32_t>(std::strtoul(a + 16, nullptr, 0));
    } else if (std::strncmp(a, "--rsp-ready=", 12) == 0) {
      opts.rsp_ready = static_cast<uint32_t>(std::strtoul(a + 12, nullptr, 0));
    } else if (std::strncmp(a, "--stress-fence=", 15) == 0) {
      opts.stress_fence = static_cast<uint32_t>(std::strtoul(a + 15, nullptr, 0));
    } else if (std::strncmp(a, "--mem-ready=", 12) == 0) {
      opts.mem_ready = static_cast<uint32_t>(std::strtoul(a + 12, nullptr, 0));
    } else if (std::strncmp(a, "--max-report=", 13) == 0) {
      opts.max_report = static_cast<uint32_t>(std::strtoul(a + 13, nullptr, 0));
    } else if (std::strncmp(a, "--tests=", 8) == 0) {
      opts.tests = a + 8;
    } else if (std::strncmp(a, "--watch=", 8) == 0) {
      opts.watch = std::strtoull(a + 8, nullptr, 0);
    } else if (std::strcmp(a, "--no-assert") == 0) {
      opts.no_assert = true;
    } else if (std::strcmp(a, "--verbose") == 0) {
      opts.verbose = true;
    } else {
      usage(argv[0]);
      return 2;
    }
  }

  Testbench tb(opts);
  return tb.run();
}
