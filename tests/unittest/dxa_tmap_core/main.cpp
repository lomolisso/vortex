// Copyright © 2019-2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// The DXA engine's tensor maps in memory, the twin of hw/unittest/dxa_core's
// map vectors. The bench is global memory: it answers the engine's line reads
// after a fixed delay, the map's lines optionally in reverse order. Each
// vector issues map-form (or slot-form) tensor prefetches, whose box reads
// name the descriptor used, and the tensormap ops. Checked: a hit reads no
// map line and starts its box reads on a slot request's cycle; a miss reads
// the map's two lines, back to back, ahead of everything else on the port;
// the victim is the lowest invalid entry, then round robin; INVAL and a
// launch's KMU writes drop entries and no slot, restart the round robin and,
// during a fetch, keep its fill out of the cache (a map use fetches again, a
// PREFETCH ends); PREFETCH fills without box reads; a request behind a
// missing map waits for it; the hit and miss counters.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <vector>

#include "core.h"
#include "dxa_core.h"
#include "dxa_tmap.h"
#include "mem_block_pool.h"
#include "processor.h"
#include "simobject.h"
#include <VX_types.h>

using namespace vortex;

#ifndef DXA_MBAR_ENABLE
#error "tensor maps need a DXA build with mbarriers"
#endif

namespace {

constexpr uint32_t LINE      = VX_CFG_L1_LINE_SIZE;
constexpr uint32_t MAP_LINES = (128 + LINE - 1) / LINE;
constexpr uint32_t BOX_ROWS  = 2;   // a box of 16 x 2 f32: one line a row at 64-byte lines
constexpr uint32_t BOX_LINES = BOX_ROWS * ((64 + LINE - 1) / LINE);
constexpr uint32_t LAT       = 12;  // the bench memory's response delay

[[noreturn]] void fail(const char* what) {
  std::printf("FAILED: %s\n", what);
  std::exit(1);
}

void expect(bool c, const char* what) {
  if (!c)
    fail(what);
}

// The launch vectors report every failure before the run fails.
bool g_failed = false;

void check(bool c, const char* what) {
  if (!c) {
    std::printf("FAILED: %s\n", what);
    g_failed = true;
  }
}

uint64_t map_addr(uint32_t i) { return 0x40000ull + uint64_t(i) * 0x1c0; }  // 64-byte aligned
uint64_t box_base(uint32_t i) { return 0x400000ull + uint64_t(i) * 0x10000; }
constexpr uint32_t SLOT_ID = 100;  // box_base(SLOT_ID): the slot's tensor

// A rank-2 f32 tensor of 64 x 8 with 256-byte rows and a 16 x 2 box.
void make_words(uint32_t* w, uint64_t base) {
  std::memset(w, 0, TMAP_WORDS * 4);
  w[TMAP_W_BASE_LO]    = uint32_t(base);
  w[TMAP_W_BASE_HI]    = uint32_t(base >> 32);
  w[TMAP_W_SIZE0]      = 64;
  w[TMAP_W_SIZE0 + 1]  = 8;
  w[TMAP_W_SIZE0 + 2]  = 1;
  w[TMAP_W_SIZE0 + 3]  = 1;
  w[TMAP_W_SIZE0 + 4]  = 1;
  w[TMAP_W_STRIDE0]    = 256;
  w[TMAP_W_META]       = dxa_tmap::pack_meta(2, 4, 0, 0, 0, 0, false);
  for (uint32_t i = 0; i < 5; ++i)
    w[TMAP_W_ESTRIDE0 + i] = 1;
  w[TMAP_W_TILESIZE01] = dxa_tmap::pack_2x16(16, BOX_ROWS);
  w[TMAP_W_TILESIZE23] = dxa_tmap::pack_2x16(1, 1);
  w[TMAP_W_TILESIZE4]  = 1;
  w[TMAP_W_TYPE]       = dxa_tmap::pack_type(TMAP_ET_F32, TMAP_FILL_ZERO);
}

struct Read {
  uint64_t cycle;
  uint64_t addr;
};

struct Bench {
  DxaCore* dxa;
  Core*    core;
  std::map<uint64_t, uint8_t> mem;       // the maps' bytes; everything else reads 0
  bool     reverse_map = false;          // answer a map's lines last first
  std::vector<Read> reads;               // every line read, in order
  struct Pending { MemRsp rsp; uint64_t due; };
  std::deque<Pending> rsps;
  std::vector<Pending> map_hold;
  uint64_t last_map_rsp = 0;             // the cycle the last map line's response went
  std::vector<uint64_t> map_rsp_at;      // the cycle of every map line's response
  uint64_t uuid = 1000;

  uint64_t now() const { return SimPlatform::instance().cycles(); }

  void put_map(uint32_t i, uint64_t base) {
    uint32_t w[TMAP_WORDS];
    make_words(w, base);
    for (uint32_t b = 0; b < 128; ++b)
      mem[map_addr(i) + b] = uint8_t(w[b / 4] >> (8 * (b % 4)));
  }

  bool is_map_line(uint64_t a) const {
    for (auto& [addr, v] : mem) {
      (void)v;
      if ((addr & ~uint64_t(LINE - 1)) == a)
        return true;
    }
    return false;
  }

  // One cycle: the port's request (at most one a cycle), then a due response.
  void step() {
    SimPlatform::instance().tick();
    auto& rq = dxa->gmem_req_out.at(0);
    uint32_t n = 0;
    while (!rq.empty()) {
      MemReq req = rq.peek();
      rq.pop();
      if (++n > 1)
        fail("two GMEM requests in one cycle");
      if (req.is_write())
        fail("a write from a prefetch");
      reads.push_back({now(), req.addr});
      Pending p{MemRsp(req.tag, req.hart_id, req.uuid, make_mem_block()), now() + LAT};
      for (uint32_t b = 0; b < LINE; ++b) {
        auto it = mem.find(req.addr + b);
        p.rsp.data->data()[b] = (it == mem.end()) ? 0 : it->second;
      }
      if (reverse_map && is_map_line(req.addr)) {
        map_hold.push_back(p);
        if (map_hold.size() == MAP_LINES) {
          for (auto it = map_hold.rbegin(); it != map_hold.rend(); ++it) {
            it->due = now() + LAT;
            rsps.push_back(*it);
          }
          map_hold.clear();
        }
        continue;
      }
      rsps.push_back(p);
    }
    auto& rs = dxa->gmem_rsp_in.at(0);
    if (!rsps.empty() && rsps.front().due <= now() && !rs.full()) {
      if (uint64_t(rsps.front().rsp.tag) >= uint64_t(VX_CFG_DXA_MAX_INFLIGHT) * VX_CFG_NUM_DXA_CORES)
      {
        last_map_rsp = now();
        map_rsp_at.push_back(now());
      }
      rs.send(rsps.front().rsp, 0);
      rsps.pop_front();
    }
  }

  void drain(uint32_t quiet = 60) {
    uint64_t last = now();
    size_t seen = reads.size();
    for (uint32_t k = 0; k < 100000; ++k) {
      step();
      if (reads.size() != seen || !rsps.empty() || !map_hold.empty()) {
        seen = reads.size();
        last = now();
      }
      if (now() > last + quiet)
        return;
    }
    fail("the engine never went quiet");
  }

  // Steps until the DXA has taken the first line response of the fetch whose
  // responses start at map_rsp_at[rsp0]; the second is then in its port.
  void to_first_map_rsp(size_t rsp0) {
    while (map_rsp_at.size() == rsp0)
      step();
    step();
  }

  DxaReq base_req() {
    DxaReq r{};
    r.core     = core;
    r.uuid     = ++uuid;
    r.tload    = true;
    r.prefetch = true;
    return r;
  }

  DxaReq map_load(uint32_t i) {
    DxaReq r = base_req();
    r.map      = true;
    r.map_addr = map_addr(i);
    return r;
  }

  DxaReq slot_load() { return base_req(); }

  DxaReq tmop(uint32_t op, uint64_t addr) {
    DxaReq r{};
    r.core     = core;
    r.uuid     = ++uuid;
    r.tmop     = true;
    r.tmap_op  = op;
    r.map_addr = addr;
    return r;
  }

  void send(DxaReq r) {
    auto& ch = dxa->dxa_req_in.at(0);
    while (ch.full())
      step();
    r.sent = now();
    ch.send(r);
  }

  // Runs requests sent on consecutive cycles to the end; returns the reads.
  std::vector<Read> run(const std::vector<DxaReq>& reqs, uint64_t* t0 = nullptr) {
    const size_t from = reads.size();
    if (t0)
      *t0 = now();
    for (auto& r : reqs) {
      this->send(r);
      step();
    }
    drain();
    return std::vector<Read>(reads.begin() + from, reads.end());
  }
};

uint32_t count_map_reads(const std::vector<Read>& rd, uint32_t i) {
  uint32_t n = 0;
  for (auto& r : rd)
    n += (r.addr >= (map_addr(i) & ~uint64_t(LINE - 1)) && r.addr < map_addr(i) + 128);
  return n;
}

uint32_t count_box_reads(const std::vector<Read>& rd, uint32_t i) {
  uint32_t n = 0;
  for (auto& r : rd)
    n += (r.addr >= box_base(i) && r.addr < box_base(i) + 0x10000);
  return n;
}

uint64_t first_box_read(const std::vector<Read>& rd, uint32_t i) {
  for (auto& r : rd) {
    if (r.addr >= box_base(i) && r.addr < box_base(i) + 0x10000)
      return r.cycle;
  }
  fail("no box read");
}

// A map-form prefetch of map i, alone: its reads and its first box read's
// distance from its send.
struct One {
  std::vector<Read> rd;
  uint64_t t0;
  uint64_t lat;
};

One one(Bench& b, uint32_t i) {
  One o;
  o.rd  = b.run({b.map_load(i)}, &o.t0);
  o.lat = first_box_read(o.rd, i) - o.t0;
  return o;
}

void check_counts(Bench& b, uint64_t hits, uint64_t misses, const char* what) {
  const auto& p = b.dxa->perf_stats();
  if (p.tmap_hits != hits || p.tmap_misses != misses) {
    std::printf("  hits %llu misses %llu, expected %llu %llu\n", (unsigned long long)p.tmap_hits,
                (unsigned long long)p.tmap_misses, (unsigned long long)hits, (unsigned long long)misses);
    fail(what);
  }
}

} // namespace

int main() {
  Processor processor;
  auto dxa = DxaCore::Create("dxa_tmap_core_test", nullptr);
  SimPlatform::instance().reset();
  Bench b;
  b.dxa  = dxa.get();
  b.core = processor.get_first_core();
  std::printf("dxa_tmap_core: line=%u map lines=%u entries=%u latency=%u\n", LINE, MAP_LINES, DXA_TMAP_ENTRIES,
              LAT);
  for (uint32_t i = 0; i < 16; ++i)
    b.put_map(i, box_base(i));

  // The slot: the same box over its own tensor, programmed by DCR.
  uint32_t sw[TMAP_WORDS];
  make_words(sw, box_base(SLOT_ID));
  for (uint32_t k = 0; k < 22; ++k)
    dxa->dcr_write(VX_DCR_DXA_STATE_BEGIN + k, sw[k]);
  uint64_t t0;
  auto rd = b.run({b.slot_load()}, &t0);
  const uint64_t slot_lat = first_box_read(rd, SLOT_ID) - t0;
  expect(rd.size() == BOX_LINES, "slot: not the box's lines");
  std::printf("  slot: first box read %llu cycles after the send\n", (unsigned long long)slot_lat);

  // Miss, its map lines answered in reverse order, then a hit.
  b.reverse_map = true;
  One m = one(b, 0);
  b.reverse_map = false;
  expect(count_map_reads(m.rd, 0) == MAP_LINES, "miss: not the map's lines");
  expect(count_box_reads(m.rd, 0) == BOX_LINES, "miss: not the box of the map's tensor");
  expect(m.rd.at(0).addr == (map_addr(0) & ~uint64_t(LINE - 1)), "miss: the map's lines not first");
  for (uint32_t k = 1; k < MAP_LINES; ++k)
    expect(m.rd.at(k).cycle == m.rd.at(k - 1).cycle + 1, "miss: the map's lines not back to back");
  One h = one(b, 0);
  expect(count_map_reads(h.rd, 0) == 0, "hit: a map line read");
  expect(h.rd.size() == BOX_LINES, "hit: not the box's lines");
  expect(h.lat == slot_lat, "hit: not a slot request's cycle");
  check_counts(b, 1, 1, "hit/miss counters");
  std::printf("  miss: first map read %llu, first box read %llu cycles after the send (slot +%llu)\n",
              (unsigned long long)(m.rd.at(0).cycle - m.t0),
              (unsigned long long)m.lat, (unsigned long long)(m.lat - slot_lat));
  const uint64_t fill_to_box = first_box_read(m.rd, 0) - b.last_map_rsp;
  std::printf("  miss: last map response to first box read %llu cycles\n", (unsigned long long)fill_to_box);
  // In order, the timeline runs from the last response just the same.
  {
    One m2 = one(b, 15);
    expect(first_box_read(m2.rd, 15) - b.last_map_rsp == fill_to_box, "miss: response order changed the timeline");
    std::printf("  miss, in order: first box read %llu cycles after the send\n", (unsigned long long)m2.lat);
  }
  dxa->dcr_write(VX_DCR_KMU_STATE_BEGIN, 0);

  // Eviction: maps 0-8 fill entries 0-7, then 8 replaces entry 0 (map 0);
  // map 0 then replaces entry 1 (map 1); map 2 still hits; map 1 misses.
  for (uint32_t i = 0; i <= 8; ++i) {
    One o = one(b, i);
    expect(count_map_reads(o.rd, i) == MAP_LINES, "evict: a first use without its fetch");
  }
  expect(count_map_reads(one(b, 0).rd, 0) == MAP_LINES, "evict: map 0 not evicted by the ninth map");
  expect(count_map_reads(one(b, 2).rd, 2) == 0, "evict: map 2 evicted out of turn");
  expect(count_map_reads(one(b, 1).rd, 1) == MAP_LINES, "evict: map 1 not the round-robin victim");
  expect(count_map_reads(one(b, 8).rd, 8) == 0, "evict: map 8 evicted out of turn");
  std::printf("  evict: lowest invalid, then round robin\n");

  // INVAL drops its map only; an INVAL of an uncached map does nothing.
  b.run({b.tmop(DXA_TMAP_INVAL, map_addr(2))});
  b.run({b.tmop(DXA_TMAP_INVAL, map_addr(14))});
  expect(count_map_reads(one(b, 8).rd, 8) == 0, "inval: another map dropped");
  expect(count_map_reads(one(b, 2).rd, 2) == MAP_LINES, "inval: the map still cached");
  expect(count_map_reads(one(b, 2).rd, 2) == 0, "inval: the refetched map not cached");
  // INVAL ahead of a use of the same map, in one burst: the use refetches.
  rd = b.run({b.tmop(DXA_TMAP_INVAL, map_addr(2)), b.map_load(2)});
  expect(count_map_reads(rd, 2) == MAP_LINES, "inval: a use behind INVAL hit the stale entry");
  std::printf("  inval: the entry only\n");

  // PREFETCH fills without box reads; a load then hits; a cached PREFETCH reads nothing.
  dxa->dcr_write(VX_DCR_KMU_STATE_BEGIN, 0);
  rd = b.run({b.tmop(DXA_TMAP_PREFETCH, map_addr(3))});
  expect(count_map_reads(rd, 3) == MAP_LINES && rd.size() == MAP_LINES, "prefetch: not just the map's lines");
  One p = one(b, 3);
  expect(count_map_reads(p.rd, 3) == 0 && p.lat == slot_lat, "prefetch: the load did not hit");
  rd = b.run({b.tmop(DXA_TMAP_PREFETCH, map_addr(3))});
  expect(rd.empty(), "prefetch: a cached map read again");
  std::printf("  prefetch: fills, then hits\n");

  // A launch drops every map; the slots stay.
  dxa->dcr_write(VX_DCR_KMU_STATE_BEGIN + 1, 7);
  expect(count_map_reads(one(b, 3).rd, 3) == MAP_LINES, "launch: a map survived the KMU writes");
  rd = b.run({b.slot_load()});
  expect(count_box_reads(rd, SLOT_ID) == BOX_LINES, "launch: the slot changed");
  // A slot write leaves the maps.
  dxa->dcr_write(VX_DCR_DXA_STATE_BEGIN + VX_DCR_DXA_DESC_CFILL_OFF, 0);
  expect(count_map_reads(one(b, 3).rd, 3) == 0, "launch: a slot write dropped a map");
  std::printf("  launch: maps dropped, slots kept\n");

  // A slot request right behind a missing map waits for it, in order.
  rd = b.run({b.map_load(5), b.slot_load()});
  {
    const uint64_t map_box = first_box_read(rd, 5);
    const uint64_t slot_box = first_box_read(rd, SLOT_ID);
    expect(slot_box > map_box, "behind: the slot request passed the missing map");
    uint64_t last_map_line = 0;
    for (auto& r : rd)
      if (count_map_reads({r}, 5))
        last_map_line = r.cycle;
    expect(map_box > last_map_line, "behind: box reads before the map was read");
    std::printf("  behind: slot box %llu cycles after the map's box\n", (unsigned long long)(slot_box - map_box));
  }

  // Every lookup of a map use or a PREFETCH counts once; INVAL does not.
  check_counts(b, 8, 18, "counters");
  std::printf("  counters: %llu hits, %llu misses\n", (unsigned long long)b.dxa->perf_stats().tmap_hits,
              (unsigned long long)b.dxa->perf_stats().tmap_misses);

  // A launch restarts the round robin: after one eviction moved it, the
  // ninth map of the next launch replaces entry 0 again (map 0).
  dxa->dcr_write(VX_DCR_KMU_STATE_BEGIN, 0);
  for (uint32_t i = 0; i <= 8; ++i)
    one(b, i);
  dxa->dcr_write(VX_DCR_KMU_STATE_BEGIN, 0);
  for (uint32_t i = 0; i <= 8; ++i)
    one(b, i);
  {
    bool kept = true;
    for (uint32_t i = 1; i <= 7; ++i)
      kept = kept && (count_map_reads(one(b, i).rd, i) == 0);
    check(kept, "launch rr: the round robin not restarted (an entry other than 0 evicted)");
    check(count_map_reads(one(b, 0).rd, 0) == MAP_LINES, "launch rr: map 0 not the victim");
    std::printf("  launch rr: the ninth map replaces entry 0\n");
  }

  // A launch between a fetch's two line responses: its fill is not kept; the
  // head, looked up again, misses and fetches the map again, as a miss from
  // that lookup (two misses, no hit).
  {
    dxa->dcr_write(VX_DCR_KMU_STATE_BEGIN, 0);
    const size_t from = b.reads.size();
    const size_t rsp0 = b.map_rsp_at.size();
    b.send(b.map_load(6));
    b.to_first_map_rsp(rsp0);
    dxa->dcr_write(VX_DCR_KMU_STATE_BEGIN + 1, 0);
    b.drain();
    rd.assign(b.reads.begin() + from, b.reads.end());
    const uint32_t n = count_map_reads(rd, 6);
    check(n == 2 * MAP_LINES, "kill: the map not fetched again");
    check(count_box_reads(rd, 6) == BOX_LINES, "kill: not the box of the map's tensor");
    if (n == 2 * MAP_LINES && b.map_rsp_at.size() == rsp0 + 2 * MAP_LINES) {
      const uint64_t last1 = b.map_rsp_at.at(rsp0 + MAP_LINES - 1);
      const uint64_t again = rd.at(MAP_LINES).cycle;
      const uint64_t m_first = m.rd.at(0).cycle - m.t0;
      check(again - last1 == fill_to_box - slot_lat + m_first, "kill: the refetch not a miss at the re-lookup");
      for (uint32_t k = 1; k < MAP_LINES; ++k)
        check(rd.at(MAP_LINES + k).cycle == rd.at(MAP_LINES + k - 1).cycle + 1, "kill: the refetch not back to back");
      check(first_box_read(rd, 6) - b.map_rsp_at.back() == fill_to_box, "kill: the refilled map's timeline");
      std::printf("  kill: refetch %llu cycles after the killed fill's last response\n",
                  (unsigned long long)(again - last1));
    }
    check(count_map_reads(one(b, 6).rd, 6) == 0, "kill: the refetched map not cached");
  }

  // A PREFETCH whose fetch a launch kills ends at the re-lookup, on the
  // cycle a filled PREFETCH would; the slot request behind it goes on.
  {
    dxa->dcr_write(VX_DCR_KMU_STATE_BEGIN, 0);
    rd = b.run({b.tmop(DXA_TMAP_PREFETCH, map_addr(9)), b.slot_load()});
    const uint64_t ref = first_box_read(rd, SLOT_ID) - b.last_map_rsp;
    dxa->dcr_write(VX_DCR_KMU_STATE_BEGIN, 0);
    const size_t from = b.reads.size();
    const size_t rsp0 = b.map_rsp_at.size();
    b.send(b.tmop(DXA_TMAP_PREFETCH, map_addr(7)));
    b.step();
    b.send(b.slot_load());
    b.to_first_map_rsp(rsp0);
    dxa->dcr_write(VX_DCR_KMU_STATE_BEGIN + 1, 0);
    b.drain();
    rd.assign(b.reads.begin() + from, b.reads.end());
    check(count_map_reads(rd, 7) == MAP_LINES, "prefetch kill: the map fetched again");
    check(count_box_reads(rd, SLOT_ID) == BOX_LINES && rd.size() == MAP_LINES + BOX_LINES,
          "prefetch kill: not the slot request's box behind it");
    check(first_box_read(rd, SLOT_ID) - b.last_map_rsp == ref, "prefetch kill: not popped on a fill's cycle");
    check(count_map_reads(one(b, 7).rd, 7) == MAP_LINES, "prefetch kill: the killed fill kept");
    std::printf("  prefetch kill: popped, the slot request %llu cycles after the last map response\n",
                (unsigned long long)ref);
  }

  // A slot write between a fetch's line responses leaves the fill alone (SimX
  // DCR writes fall between cycles; the RTL holds that cycle's response).
  {
    dxa->dcr_write(VX_DCR_KMU_STATE_BEGIN, 0);
    const size_t from = b.reads.size();
    const size_t rsp0 = b.map_rsp_at.size();
    b.send(b.map_load(10));
    b.to_first_map_rsp(rsp0);
    dxa->dcr_write(VX_DCR_DXA_STATE_BEGIN + VX_DCR_DXA_DESC_CFILL_OFF, sw[VX_DCR_DXA_DESC_CFILL_OFF]);
    b.drain();
    rd.assign(b.reads.begin() + from, b.reads.end());
    check(count_map_reads(rd, 10) == MAP_LINES && count_box_reads(rd, 10) == BOX_LINES,
          "dcr fill: not one fetch and the map's box");
    check(count_map_reads(one(b, 10).rd, 10) == 0, "dcr fill: the map not cached");
    check(count_box_reads(b.run({b.slot_load()}), SLOT_ID) == BOX_LINES, "dcr fill: the slot changed");
    std::printf("  dcr fill: a slot write during a fetch keeps both\n");
  }

  // A killed transfer counts two misses, a killed PREFETCH one.
  check_counts(b, 17, 43, "launch counters");
  if (g_failed)
    fail("launch vectors");
  std::printf("PASSED\n");
  return 0;
}
