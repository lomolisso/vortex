// l1_slot_ref.h on the host: L1 lines and protected window per carve-out, tag-bank mapping, hit-latency rows,
// store-allocate / partial-word / atomic-clear / borrow / invalidate paths, the age-list draw pipeline, and the
// victim run (allocation rate with fills of latency F, no in-flight victim, protected window, uniform old draws,
// pool invariants every cycle, stragglers) at every configuration and at a 129-line L1.
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <l1_slot_ref.h>

using namespace vortex::l1_slot_ref;

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL %s:%d ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static const uint32_t CONFIGS_KB[9] = {8, 16, 32, 64, 100, 132, 164, 196, 228};

static L1SlotCfg base_cfg(uint32_t s_kb, uint32_t reserve = 0) {
  L1SlotCfg c{};
  c.s_bytes = s_kb * 1024;
  c.sram_bytes = 262144;
  c.reserve_bytes = reserve;
  c.row_bytes = 128;
  c.valid_bytes = 4;
  c.sector_bytes = 32;
  c.tag_ways = 16;
  c.tag_banks = 4;
  c.index_hash = 1;
  c.upd_depth = 6;
  c.protect_pct = 25;
  c.mshr_lines = 128;
  c.max_lines = (c.sram_bytes - c.reserve_bytes - CONFIGS_KB[0] * 1024) / c.row_bytes;
  c.tag_sets = 1u << log2_ceil((c.max_lines + c.tag_ways - 1) / c.tag_ways);
  return c;
}

static void test_tables() {
  const uint32_t lines[9] = {1984, 1920, 1792, 1536, 1248, 992, 736, 480, 224};
  const uint32_t prot[9] = {496, 480, 448, 384, 312, 248, 184, 128, 128};
  for (int i = 0; i < 9; ++i) {
    auto c = base_cfg(CONFIGS_KB[i]);
    CHECK(l1_lines(c) == lines[i], "l1_lines c=%u got %u", CONFIGS_KB[i], l1_lines(c));
    CHECK(l1_protect(c) == prot[i], "protect c=%u got %u", CONFIGS_KB[i], l1_protect(c));
  }
  auto c = base_cfg(8);
  CHECK(c.tag_sets == 128 && l1_protect_max(c) == 496 && l1_old_depth(c) == 2048, "sizes");
  CHECK(l1_valid_bits(c) == 32 && l1_sector_words(0x2, c) == 0xFF00ull, "valid bits");
  CHECK(l1_full_words(2, 8, c) == 0x2 && l1_touched_words(2, 8, c) == 0x7, "word masks");
}

static void test_bank_bijection() {
  for (uint32_t banks : {1u, 2u, 4u, 8u}) {
    for (uint32_t sets : {32u, 64u, 128u, 256u, 512u}) {
      if (sets < banks)
        continue;
      std::vector<std::set<uint32_t>> rows(banks);
      for (uint32_t s = 0; s < sets; ++s) {
        uint32_t b = l1_tag_bank(s, banks);
        CHECK(b < banks, "bank range");
        CHECK(rows[b].insert(l1_tag_row(s, banks)).second, "bank %u sets %u not a bijection at set %u", banks, sets, s);
      }
      for (uint32_t b = 0; b < banks; ++b)
        CHECK(rows[b].size() == sets / banks, "bank %u holds %zu rows", b, rows[b].size());
    }
  }
}

// Wavefronts of one warp load: the most distinct rows any bank sees (4-B lanes, one row = one line).
static uint32_t warp_wavefronts(const uint64_t* addr, uint32_t n) {
  std::map<uint32_t, std::set<uint64_t>> rows;
  uint32_t m = 0;
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t bank = (addr[i] / 4) % 32;
    rows[bank].insert(addr[i] / 128);
    m = std::max<uint32_t>(m, uint32_t(rows[bank].size()));
  }
  return m;
}

static constexpr uint32_t HIT_PIPE = 17;

static uint32_t chase_latency(uint32_t nlanes, uint32_t stride, const L1SlotCfg& c, uint32_t* tc = nullptr) {
  uint64_t addr[32], lines[32];
  uint64_t base = uint64_t(1) << 22;
  for (uint32_t i = 0; i < nlanes; ++i) {
    addr[i] = base + uint64_t(i) * stride;
    lines[i] = addr[i] / 128;
  }
  const uint32_t t = l1_tag_cycles(lines, nlanes, c.tag_banks, c);
  if (tc)
    *tc = t;
  return l1_hit_latency(warp_wavefronts(addr, nlanes), t, HIT_PIPE);
}

// One line: the stages, one wavefront, the response register, whatever the lane count; each further
// row in a bank costs one wavefront-stage cycle and each further line in a tag bank one tag cycle.
static void test_hit_latency() {
  auto c = base_cfg(8);
  CHECK(l1_pipe_stages(1) == 17 && l1_pipe_stages(2) == 18, "pipe stages %u / %u", l1_pipe_stages(1), l1_pipe_stages(2));
  for (uint32_t n = 1; n <= 32; ++n)
    CHECK(chase_latency(n, 4, c) == HIT_PIPE + 1 + L1_ARRAY_RSP_STAGES, "%u lanes one line got %u", n, chase_latency(n, 4, c));
  for (uint32_t n = 1; n <= 32; ++n) {
    uint32_t tc = 0;
    uint32_t got = chase_latency(n, 128, c, &tc);
    CHECK(tc >= 1 && tc <= n, "%u lines: %u tag cycles", n, tc);
    CHECK(got == HIT_PIPE + (tc - 1) + n + L1_ARRAY_RSP_STAGES, "%u lanes at 128 B got %u", n, got);
  }
}

struct VictimRun {
  double rate = 0;
  uint32_t age = 0, inset = 0, cancels = 0, waits[L1SlotDir::W_N] = {};
  double chi2 = 0;
  uint32_t young_line_rank_viol = 0;
};

// One miss offered per cycle to fresh lines, fills after F cycles, at most mshr_lines open.
static VictimRun victim_run(const L1SlotCfg& c, uint32_t F, uint64_t straggle_at, uint32_t straggle_len,
                            uint64_t cycles, uint64_t window, bool verbose_name = false) {
  (void)verbose_name;
  L1SlotDir d;
  d.configure(c);
  VictimRun out;
  std::multimap<uint64_t, uint32_t> due;
  std::map<uint32_t, uint64_t> inserted_at;  // slot -> age-list insertion count
  uint64_t ins = 0, next_line = uint64_t(1) << 24, pend_line = 0, allocs_in_window = 0;
  bool pend = false, stragglers_done = false;
  uint32_t buckets[16] = {};
  uint64_t draws = 0;
  for (uint64_t t = 0; t < cycles; ++t) {
    for (auto it = due.begin(); it != due.end() && it->first <= t;) {
      d.fill(it->second, 0xF);
      d.fill_done(it->second);
      d.replay_done(it->second);
      it = due.erase(it);
    }
    bool straggled = false;
    for (int call = 0; call < 2; ++call) {
      if (!pend && d.open_entries() >= c.mshr_lines)
        break;
      bool was_pend = pend;
      uint64_t line = pend ? pend_line : next_line;
      auto a = d.allocate(line, true, uint32_t(t), 0xF);
      if (!a.ok) {
        ++out.waits[a.wait];
        pend = (a.wait == L1SlotDir::W_DRAW_PENDING || a.wait == L1SlotDir::W_VICTIM_IN_FLIGHT);
        pend_line = line;
        if (a.wait == L1SlotDir::W_VICTIM_IN_FLIGHT)
          ++out.cancels;
        break;
      }
      pend = false;
      ++next_line;
      if (a.victim.slot != L1SlotDir::NONE) {
        if (a.kind == L1SlotDir::K_AGE) {
          ++out.age;
          uint32_t r = d.last_draw_index(), n = d.last_draw_old();
          ++buckets[uint64_t(r) * 16 / n];
          ++draws;
          uint64_t age_rank = ins - inserted_at[a.victim.slot];
          CHECK(age_rank >= d.protect(), "age-list victim among the newest %u insertions (rank %llu)", d.protect(), (unsigned long long)age_rank);
        } else {
          ++out.inset;
        }
        bool vin = false;
        for (auto& e : due)
          vin |= (e.second == a.victim.slot);
        CHECK(!vin, "victim slot %u in flight", a.victim.slot);
      }
      if (a.kind != L1SlotDir::K_INSET && a.kind != L1SlotDir::K_RESIDENT)
        inserted_at[a.slot] = ++ins;
      uint32_t lat = F;
      if (straggle_len && !straggled && !stragglers_done && t >= straggle_at) {
        lat = straggle_len;
        straggled = stragglers_done = true;
      }
      due.emplace(t + lat, a.slot);
      if (t >= cycles - window)
        ++allocs_in_window;
      if (!was_pend)
        break;
    }
    int e = d.check();
    CHECK(e == 0, "invariant %d broken at cycle %llu", e, (unsigned long long)t);
    if (e)
      break;
    d.tick();
  }
  out.rate = double(allocs_in_window) / double(window);
  if (draws >= 160) {
    double exp = double(draws) / 16;
    for (auto b : buckets)
      out.chi2 += (b - exp) * (b - exp) / exp;
  }
  return out;
}

static void test_victims() {
  // chi2(15) at p = 0.01: 30.58
  for (uint32_t F : {64u, 400u}) {
    double want = 0.99 * std::min(1.0, 128.0 / F);
    for (int i = 0; i < 10; ++i) {
      L1SlotCfg c = (i < 9) ? base_cfg(CONFIGS_KB[i]) : base_cfg(8, 262144 - 8192 - 129 * 128);
      if (i == 9)
        c.max_lines = 129, c.tag_sets = 1u << log2_ceil((129 + 15) / 16);
      auto r = victim_run(c, F, 0, 0, 40000, 10000);
      std::printf("F=%u lines=%u rate=%.3f (want >= %.3f) age=%u inset=%u cancels=%u chi2=%.1f waits yf=%u oe=%u sif=%u\n",
                  F, l1_lines(c), r.rate, want, r.age, r.inset, r.cancels, r.chi2,
                  r.waits[L1SlotDir::W_YOUNG_FULL], r.waits[L1SlotDir::W_OLD_EMPTY], r.waits[L1SlotDir::W_SET_IN_FLIGHT]);
      CHECK(r.rate >= want, "rate %.3f < %.3f at F=%u lines=%u", r.rate, want, F, l1_lines(c));
      CHECK(l1_lines(c) - l1_protect(c) < 16 || r.chi2 < 30.58, "chi2 %.1f at F=%u lines=%u", r.chi2, F, l1_lines(c));
    }
  }
  // stragglers: progress and invariants, rate reported
  for (int i : {0, 9}) {
    L1SlotCfg c = (i == 0) ? base_cfg(8) : base_cfg(8, 262144 - 8192 - 129 * 128);
    if (i == 9)
      c.max_lines = 129, c.tag_sets = 1u << log2_ceil((129 + 15) / 16);
    for (uint64_t at : {uint64_t(100), uint64_t(20000)}) {
      auto r = victim_run(c, 64, at, 2000, 40000, 10000);
      std::printf("straggler lines=%u at=%llu rate(last 10000)=%.3f cancels=%u yf=%u\n", l1_lines(c),
                  (unsigned long long)at, r.rate, r.cancels, r.waits[L1SlotDir::W_YOUNG_FULL]);
      auto r2 = victim_run(c, 64, at, 2000, at + 6000, 6000);
      std::printf("  window [at, at+6000): rate=%.3f\n", r2.rate);
      CHECK(r2.rate > 0.5, "straggler stalls the pool too long");
    }
  }
}

// Store-allocate, atomic clear, non-cacheable borrow, fill byte-enables, invalidate.
static void test_paths() {
  auto c = base_cfg(228);
  L1SlotDir d;
  d.configure(c);
  uint64_t L = 0x1000;
  auto a = d.allocate(L, true, 0, 0);
  CHECK(a.ok && a.kind == L1SlotDir::K_BUMP && a.slot == 0 && d.row_of(0) == 2047, "store-allocate");
  d.store(L, l1_full_words(0, 6, c));
  d.tick();
  auto l = d.lookup(L, 0x1);
  CHECK(l.tag && l.hit && !l.pending, "store-allocated word hits");
  l = d.lookup(L, 0x2);
  CHECK(l.tag && !l.hit, "partial word stays invalid");
  auto m = d.allocate(L, true, 1, 0x1);
  CHECK(m.ok && m.kind == L1SlotDir::K_RESIDENT && m.slot == 0, "miss on a resident line keeps its slot");
  CHECK(d.fill_byteen(0) == 0xFEull, "fill byte-enables skip words valid at the miss: %llx", (unsigned long long)d.fill_byteen(0));
  d.fill(0, 0x1);
  d.fill_done(0);
  d.replay_done(0);
  d.tick();
  CHECK(d.lookup(L, 0xFF).hit, "filled sector hits");
  d.clear(L);
  d.tick();
  l = d.lookup(L, 0x1);
  CHECK(l.tag && !l.hit && l.slot == 0, "atomic clear keeps tag and slot");
  auto nc = d.allocate(0x2000, false, 2, 0xF);
  CHECK(nc.ok && nc.kind == L1SlotDir::K_BORROW && d.borrowed() == 1, "non-cacheable borrows");
  CHECK(d.lookup(0x2000, 1).pending && !d.lookup(0x2000, 1).tag, "borrowed line has no tag");
  d.fill(nc.slot, 0xF);
  d.fill_done(nc.slot);
  d.replay_done(nc.slot);
  CHECK(d.borrowed() == 0 && d.free_fifo() == 1, "borrowed slot returns to the free FIFO");
  auto f = d.allocate(0x3000, true, 3, 0xF);
  CHECK(f.ok && f.kind == L1SlotDir::K_FREE && f.slot == nc.slot, "free FIFO before bump");
  CHECK(d.allocate(0x3000, true, 4, 0x2).wait == L1SlotDir::W_ENTRY_OPEN, "one entry per line");
  d.fill(f.slot, 0xF);
  d.fill_done(f.slot);
  d.replay_done(f.slot);
  CHECK(d.check() == 0, "invariants");
  d.invalidate_all();
  CHECK(d.resident() == 0 && d.free_slots() == d.lines() && !d.lookup(L, 0).tag, "invalidate");
}

// The draw pipeline: a lookup of the victim line in the completion cycle still
// hits (old contents), from the next cycle it misses; draws of one index back to back.
static void test_draw_pipeline() {
  auto c = base_cfg(8, 262144 - 8192 - 129 * 128);
  c.max_lines = 129;
  c.tag_sets = 16;
  L1SlotDir d;
  d.configure(c);
  uint64_t line = 0;
  std::vector<uint32_t> slots;
  while (d.bump() < d.lines()) {
    auto a = d.allocate(line++, true, 0, 0xF);
    if (!a.ok) {
      d.tick();
      continue;
    }
    d.fill(a.slot, 0xF);
    d.fill_done(a.slot);
    d.replay_done(a.slot);
    d.tick();
  }
  for (int i = 0; i < 4; ++i)
    d.tick();
  CHECK(d.old() == 1 && d.young() == 128, "129-line pool: young %u old %u", d.young(), d.old());
  for (int k = 0; k < 3; ++k) {
    auto a = d.allocate(line, true, 0, 0xF);
    CHECK(!a.ok && a.wait == L1SlotDir::W_DRAW_PENDING && d.last_draw_index() == 0, "draw starts");
    d.tick();
    a = d.allocate(line, true, 0, 0xF);
    CHECK(a.ok && a.kind == L1SlotDir::K_AGE, "draw completes");
    CHECK(d.lookup(a.victim_line, 0xFFFFFFFF).hit, "victim visible in the completion cycle");
    d.fill(a.slot, 0xF);
    d.fill_done(a.slot);
    d.replay_done(a.slot);
    d.tick();
    CHECK(!d.lookup(a.victim_line, 0x1).tag, "victim cleared from the next cycle");
    CHECK(d.check() == 0, "invariants after draw %d", k);
    ++line;
  }
}

int main() {
  test_tables();
  test_bank_bijection();
  test_hit_latency();
  test_paths();
  test_draw_pipeline();
  test_victims();
  std::printf("%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
