// vx_dram_map.h against the H100's measured DRAM map (dram_hu_vectors.csv, doc 07 model, 5000 lines),
// against the generalised map at the preset and other geometries (golden_presets.csv), then the interim
// channel select and the Ramulator field packing.

#include <vx_dram_map.h>

#include <cinttypes>
#include <cstdio>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#ifndef VECTORS_DIR
#define VECTORS_DIR "."
#endif

namespace {

int g_fail = 0, g_pass = 0;

void expect(const char* what, uint64_t got, uint64_t want) {
  if (got != want) {
    if (g_fail < 20)
      std::printf("FAIL %s: got %#" PRIx64 ", expected %#" PRIx64 "\n", what, got, want);
    ++g_fail;
  } else {
    ++g_pass;
  }
}

const uint64_t HB[] = VX_DRAM_HB_H100;

std::vector<std::map<std::string, std::string>> read_csv(const std::string& path) {
  std::vector<std::map<std::string, std::string>> rows;
  std::ifstream f(path);
  if (!f) {
    std::printf("FAIL cannot open %s\n", path.c_str());
    ++g_fail;
    return rows;
  }
  std::string line;
  std::vector<std::string> cols;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    std::vector<std::string> cells;
    std::stringstream ss(line);
    std::string c;
    while (std::getline(ss, c, ','))
      cells.push_back(c);
    if (cols.empty()) {
      cols = cells;
      continue;
    }
    std::map<std::string, std::string> r;
    for (size_t i = 0; i < cols.size() && i < cells.size(); ++i)
      r[cols[i]] = cells[i];
    rows.push_back(r);
  }
  return rows;
}

uint64_t num(const std::map<std::string, std::string>& r, const char* key) {
  return std::stoull(r.at(key), nullptr, 0);
}

void check_loc(const char* what, uint64_t lc, const vx_dram_geom_t& g, uint64_t row, uint64_t col, uint64_t bank) {
  const vx_dram_loc_t loc = vx_dram_map(lc, &g);
  std::string w(what);
  expect((w + " row").c_str(), loc.row, row);
  expect((w + " col").c_str(), loc.col, col);
  expect((w + " bank").c_str(), loc.bank, bank);
}

void h1_h100_vectors() {
  const auto rows = read_csv(VECTORS_DIR "/dram_hu_vectors.csv");
  const vx_dram_geom_t g = {3, 5, HB};
  const int fail0 = g_fail;
  for (const auto& r : rows)
    check_loc("H1", num(r, "lc"), g, num(r, "row"), num(r, "col"), num(r, "bank"));
  expect("H1 rows", rows.size(), 5000);
  std::printf("H1 H100 vectors: %zu rows, %d failures\n", rows.size(), g_fail - fail0);
}

void h2_golden_presets() {
  const auto rows = read_csv(VECTORS_DIR "/golden_presets.csv");
  std::map<std::string, size_t> per_config;
  const int fail0 = g_fail;
  for (const auto& r : rows) {
    const vx_dram_geom_t g = {(uint32_t)num(r, "lg_lines_per_row"), (uint32_t)num(r, "lg_banks"), HB};
    const std::string cfg = r.at("config");
    check_loc(("H2 " + cfg).c_str(), num(r, "lc"), g, num(r, "dram_row"), num(r, "dram_col"), num(r, "dram_bank"));
    ++per_config[cfg];
  }
  for (const char* p : {"h100", "pocket_h100", "nano_h100", "micro", "mid"})
    expect((std::string("H2 rows ") + p).c_str(), per_config[p], 2000);
  std::printf("H2 golden presets: %zu rows in %zu configurations, %d failures\n", rows.size(), per_config.size(),
              g_fail - fail0);
}

void h3_interim_select() {
  std::mt19937_64 rnd(17);
  const int fail0 = g_fail;
  for (uint32_t lg_line : {6u, 7u}) {
    for (uint32_t lg_c = 0; lg_c <= 7; ++lg_c) {
      const uint64_t C = 1ull << lg_c, line = 1ull << lg_line, addr_bits = 40;
      std::set<std::pair<uint32_t, uint64_t>> seen;
      int bad_inv = 0, bad_line = 0, bad_range = 0, bad_rot = 0, bad_local = 0;
      for (int i = 0; i < 100000; ++i) {
        const uint64_t a = rnd() & ((1ull << addr_bits) - 1);
        const uint32_t ch = vx_dram_interim_channel(a, lg_line, lg_c);
        const uint64_t loc = vx_dram_interim_local(a, lg_line, lg_c);
        bad_inv += vx_dram_interim_addr(ch, loc, lg_line, lg_c) != a;
        bad_range += ch >= C || loc >= (1ull << (addr_bits - lg_c));
        seen.insert({ch, loc});
        const uint64_t base = a & ~(line - 1), last = base + line - 1;
        bad_line += vx_dram_interim_channel(base, lg_line, lg_c) != ch ||
                    vx_dram_interim_channel(last, lg_line, lg_c) != ch ||
                    vx_dram_interim_local(last, lg_line, lg_c) - vx_dram_interim_local(base, lg_line, lg_c) != line - 1;
        bad_rot += vx_dram_interim_channel(a + line, lg_line, lg_c) != ((ch + 1) & (C - 1));
        // C consecutive lines fill one line of every channel: the next line in a channel is C lines away.
        bad_local += vx_dram_interim_local(a + C * line, lg_line, lg_c) != loc + line;
      }
      char w[64];
      std::snprintf(w, sizeof(w), "H3 line %u C %" PRIu64, 1u << lg_line, C);
      std::string s(w);
      expect((s + " inverse").c_str(), bad_inv, 0);
      expect((s + " range").c_str(), bad_range, 0);
      expect((s + " line in one channel").c_str(), bad_line, 0);
      expect((s + " channel rotation").c_str(), bad_rot, 0);
      expect((s + " local stride").c_str(), bad_local, 0);
      std::set<uint64_t> addrs;
      for (const auto& e : seen)
        addrs.insert(vx_dram_interim_addr(e.first, e.second, lg_line, lg_c));
      expect((s + " no collision").c_str(), addrs.size(), seen.size());
    }
  }
  std::printf("H3 interim select: %d failures\n", g_fail - fail0);
}

struct Unpacked { uint32_t pc, bg, ba, column, atom; uint64_t row; };

// ChRaBaRoCo as 2.1's mapper slices it: lowest level (column) first, after the transaction offset.
Unpacked unpack(uint64_t a, uint32_t lg_pcs, uint32_t lg_bgs, uint32_t lg_ba, uint32_t lg_rows,
                uint32_t lg_tx_per_row, uint32_t lg_atom) {
  auto take = [&a](uint32_t bits) { const uint64_t v = a & ((1ull << bits) - 1); a >>= bits; return v; };
  Unpacked u;
  u.atom = (uint32_t)take(lg_atom);
  u.column = (uint32_t)take(lg_tx_per_row);
  u.row = take(lg_rows);
  u.ba = (uint32_t)take(lg_ba);
  u.bg = (uint32_t)take(lg_bgs);
  u.pc = (uint32_t)take(lg_pcs);
  expect("H4 no bits above PC", a, 0);
  return u;
}

void h4_pack() {
  const int fail0 = g_fail;
  const uint32_t lg_atoms_per_line = 2, lg_tx_per_row = 5, lg_atom = 5;
  // width factor 1: 2 PCs x 4 BGs x 4 banks (32); width factor 2: 4 PCs x 4 BGs x 4 banks (64)
  const struct { uint32_t lg_pcs, lg_bgs, lg_ba, lg_rows; } shapes[] = {{1, 2, 2, 15}, {2, 2, 2, 14}};
  std::mt19937_64 rnd(4);
  for (const auto& s : shapes) {
    const vx_dram_geom_t g = {3, s.lg_pcs + s.lg_bgs + s.lg_ba, HB};
    const uint64_t lines = 1ull << (s.lg_rows + g.lg_lines_per_row + g.lg_banks);
    int bad = 0;
    for (int i = 0; i < 100000; ++i) {
      const uint64_t lc = rnd() & (lines - 1);
      const uint32_t sector = (uint32_t)(rnd() & 3);
      const vx_dram_loc_t loc = vx_dram_map(lc, &g);
      const vx_dram_fields_t f = vx_dram_fields(loc, sector, lg_atoms_per_line, s.lg_pcs, s.lg_bgs);
      const Unpacked u = unpack(vx_dram_pack(f, s.lg_pcs, s.lg_bgs, s.lg_ba, s.lg_rows, lg_tx_per_row, lg_atom),
                                s.lg_pcs, s.lg_bgs, s.lg_ba, s.lg_rows, lg_tx_per_row, lg_atom);
      bad += u.pc != f.pc || u.bg != f.bg || u.ba != f.ba || u.row != f.row || u.column != f.column || u.atom != 0;
      bad += (f.pc | f.bg << s.lg_pcs | f.ba << (s.lg_pcs + s.lg_bgs)) != loc.bank;
      bad += f.column != (loc.col << lg_atoms_per_line | sector) || f.row != loc.row;
      // bank bits above the H100's five hash functions are unhashed
      for (uint32_t k = VX_DRAM_HB_MAX; k < g.lg_banks; ++k)
        bad += ((loc.bank >> k) & 1) != ((lc >> (k + 1)) & 1);
    }
    char w[64];
    std::snprintf(w, sizeof(w), "H4 PCs %u round trip", 1u << s.lg_pcs);
    expect(w, bad, 0);
    // map + split + pack is a permutation of the channel's transactions over whole rows
    std::set<uint64_t> packed;
    const uint64_t span = 1ull << 14;
    for (uint64_t lc = 0; lc < span; ++lc)
      for (uint32_t sector = 0; sector < 4; ++sector) {
        const vx_dram_fields_t f = vx_dram_fields(vx_dram_map(lc, &g), sector, lg_atoms_per_line, s.lg_pcs, s.lg_bgs);
        packed.insert(vx_dram_pack(f, s.lg_pcs, s.lg_bgs, s.lg_ba, s.lg_rows, lg_tx_per_row, lg_atom) >> lg_atom);
      }
    std::snprintf(w, sizeof(w), "H4 PCs %u bijective", 1u << s.lg_pcs);
    expect(w, packed.size(), span * 4);
  }
  std::printf("H4 pack: %d failures\n", g_fail - fail0);
}

} // namespace

int main() {
  h1_h100_vectors();
  h2_golden_presets();
  h3_interim_select();
  h4_pack();
  std::printf("dram_map: %d checks passed, %d failed\n", g_pass, g_fail);
  if (g_fail) {
    std::printf("FAILED!\n");
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
