// smem_bank_ref's unified-array part on the host: sram_row_wavefronts against
// smem_wavefronts, the level rule, the port cycle model (SramPortRef), DSMEM
// write merging and the mbarrier op costs.

#include <smem_bank_ref.h>

#include <cinttypes>
#include <cstdio>
#include <random>

using namespace vortex::smem_bank_ref;

namespace {

int g_fail = 0, g_pass = 0;

void expect(const char* what, uint64_t got, uint64_t want) {
  if (got != want) {
    std::printf("FAIL %s: got %" PRIu64 ", expected %" PRIu64 "\n", what, got, want);
    ++g_fail;
  } else {
    ++g_pass;
  }
}

SramPortCfg default_cfg() {
  SramPortCfg c{};
  c.dsmem_wr_bytes = 64;
  for (auto& x : c.rd_pri) x = 1;
  c.rd_pri[RD_LSU] = 0;
  for (auto& x : c.wr_pri) x = 1;
  c.wr_pri[WR_LSU] = 0;
  c.wf_pri_shared = 0;
  c.wf_pri_global = 0;
  for (auto& x : c.amo_pri) x = 0;
  c.bank_pairs = 16;
  c.amo_wb_delay = 2;
  return c;
}

// LDS: row = word / banks, bank = word mod banks; an 8-B lane spans two banks.
void lds_equivalence() {
  std::mt19937 rng(7);
  const uint32_t shapes[][2] = {{32, 4}, {16, 4}, {32, 8}};
  uint32_t checked = 0;
  for (auto& sh : shapes) {
    SmemCfg cfg{sh[0], sh[1], 14};
    for (uint32_t access : {4u, 8u}) {
      if (access < sh[1])
        continue;
      for (int it = 0; it < 4000; ++it) {
        uint64_t addr[32];
        uint32_t row[32], lo[32], nb[32];
        uint32_t span = 1u << (rng() % 16);
        uint32_t mask = (it & 1) ? 0xffffffffu : uint32_t(rng());
        for (uint32_t i = 0; i < 32; ++i) {
          uint32_t mode = rng() % 3;
          uint64_t a = (mode == 0) ? uint64_t(i) * access : (mode == 1) ? uint64_t(i) * span : rng() % (1u << 17);
          addr[i] = a / access * access;
          uint64_t w = addr[i] / sh[1];
          row[i] = uint32_t(w / sh[0]);
          lo[i] = uint32_t(w % sh[0]);
          nb[i] = access / sh[1];
        }
        expect("sram_row_wavefronts == smem_wavefronts",
               sram_row_wavefronts(row, lo, nb, mask, 32), smem_wavefronts(addr, mask, 32, access, cfg));
        ++checked;
      }
    }
  }
  std::printf("lds equivalence: %u accesses\n", checked);
}

// L1 rows: one line per row; lanes of different lines on one bank serialize.
void l1_rows() {
  uint32_t row[32], lo[32], nb[32];
  for (uint32_t i = 0; i < 32; ++i) { row[i] = 100; lo[i] = i; nb[i] = 1; }
  expect("one line, 32 lanes", sram_row_wavefronts(row, lo, nb, ~0u, 32), 1);
  for (uint32_t i = 0; i < 32; ++i) { row[i] = 100 + i; lo[i] = 0; }
  expect("32 lines, same bank", sram_row_wavefronts(row, lo, nb, ~0u, 32), 32);
  for (uint32_t i = 0; i < 32; ++i) { row[i] = 100 + i; lo[i] = i; }
  expect("32 lines, diagonal", sram_row_wavefronts(row, lo, nb, ~0u, 32), 1);
  for (uint32_t i = 0; i < 32; ++i) { row[i] = 100 + i; lo[i] = (i % 4) * 8; }
  expect("32 lines, sector j mod 4", sram_row_wavefronts(row, lo, nb, ~0u, 32), 8);
  for (uint32_t i = 0; i < 16; ++i) { row[i] = 7; lo[i] = 2 * i; nb[i] = 2; }
  expect("16 8-B lanes, one row", sram_row_wavefronts(row, lo, nb, 0xffff, 16), 1);
  expect("no lanes", sram_row_wavefronts(row, lo, nb, 0, 16), 0);
}

void level_rule() {
  uint8_t lvl[4] = {1, 3, 3, 0};
  bool req[4] = {true, true, true, true};
  uint32_t rr = 0;
  expect("highest level", level_pick(lvl, req, 4, rr), 1);
  expect("round robin among equals", level_pick(lvl, req, 4, rr), 2);
  expect("round robin wraps", level_pick(lvl, req, 4, rr), 1);
  bool req2[4] = {true, false, false, true};
  expect("next level", level_pick(lvl, req2, 4, rr), 0);
  bool none[4] = {false, false, false, false};
  expect("no request", level_pick(lvl, none, 4, rr), 4);
}

void dsmem_merge() {
  SramPortCfg c = default_cfg();
  uint32_t same[4] = {5, 5, 5, 5};
  expect("4 slots one row, 64-B merge", dsmem_write_cycles(same, 4, 32, c), 2);
  uint32_t diff[4] = {5, 6, 6, 7};
  expect("rows 5,6,6,7", dsmem_write_cycles(diff, 4, 32, c), 3);
  expect("one slot", dsmem_write_cycles(same, 1, 32, c), 1);
  expect("empty packet", dsmem_write_cycles(same, 0, 32, c), 0);
  c.dsmem_wr_bytes = 128;
  expect("4 slots one row, 128-B merge", dsmem_write_cycles(same, 4, 32, c), 1);
  c.dsmem_wr_bytes = 32;
  expect("4 slots one row, no merge", dsmem_write_cycles(same, 4, 32, c), 4);
}

// Directed port cycles.
void ports() {
  SramPortCfg c = default_cfg();
  {
    SramPortRef p(c);
    p.push_wf(WF_SHARED, WF_READ, 2, 1);
    p.push_read(RD_TCU, 1);
    auto g = p.tick();
    expect("tcu before lsu read", g.rd, RD_TCU);
    expect("lsu item picked", g.wf_pick, WF_SHARED);
    expect("lsu held", g.wf_done, 0);
    g = p.tick();
    expect("lsu read 1", g.rd, RD_LSU);
    g = p.tick();
    expect("lsu read 2", g.rd, RD_LSU);
    expect("lsu done", g.wf_done, 1);
    expect("lsu id", g.wf_id, 1);
    expect("idle", p.idle(), 1);
  }
  {
    // equal levels round robin; the LSU waits for all of them
    SramPortRef p(c);
    p.push_read(RD_TCU, 2);
    p.push_read(RD_DXA, 2);
    p.push_read(RD_DSMEM, 1);
    uint32_t want[5] = {RD_TCU, RD_DXA, RD_DSMEM, RD_TCU, RD_DXA};
    for (uint32_t k = 0; k < 5; ++k)
      expect("read rr", p.tick().rd, want[k]);
  }
  {
    // shared and global alternate at equal levels; a write wavefront takes the write port
    SramPortRef p(c);
    for (uint32_t k = 0; k < 4; ++k) {
      p.push_wf(WF_SHARED, WF_READ, 1, 10 + k);
      p.push_wf(WF_GLOBAL, WF_WRITE, 1, 20 + k);
    }
    for (uint32_t k = 0; k < 8; ++k) {
      auto g = p.tick();
      expect("wf one per cycle", g.wf_done, 1);
      expect("wf alternates", g.wf_pick, k % 2 ? WF_GLOBAL : WF_SHARED);
      expect("wf port", k % 2 ? g.wr : g.rd, k % 2 ? uint32_t(WR_LSU) : uint32_t(RD_LSU));
    }
  }
  {
    // fills before STS; STS gets the cycles fills leave
    SramPortRef p(c);
    p.push_wf(WF_SHARED, WF_WRITE, 2, 1);
    p.push_write(WR_L1FILL, 1);
    p.push_write(WR_CPASYNC, 1);
    expect("fill first", p.tick().wr, WR_L1FILL);
    expect("cp.async next", p.tick().wr, WR_CPASYNC);
    expect("sts 1", p.tick().wr, WR_LSU);
    auto g = p.tick();
    expect("sts 2", g.wr, WR_LSU);
    expect("sts done", g.wf_done, 1);
  }
  {
    // an LSU atomic, a DXA completion and an mbarrier op on different pairs: one RD_AMO grant
    SramPortRef p(c);
    p.push_wf(WF_SHARED, WF_AMO, 0, 5, {{4, 1}, {5, 1}});
    p.push_amo(AMO_DXA, 7);
    p.push_amo_op(AMO_MBAR, {{20, 2}});
    p.push_write(WR_L1FILL, 3);
    auto g = p.tick();
    expect("one amo read", g.rd, RD_AMO);
    expect("three pairs", g.amo_pairs, (1u << 2) | (1u << 7) | (1u << 10));
    expect("lsu atomic done (banks 4 and 5 in one pass)", g.wf_done, 1);
    expect("fill meanwhile", g.wr, WR_L1FILL);
    expect("fill 2", p.tick().wr, WR_L1FILL);
    g = p.tick();
    expect("write-back first", g.amo_wb, 1);
    expect("fill held", g.wr, WR_N);
    expect("fill 3", p.tick().wr, WR_L1FILL);
    expect("idle", p.idle(), 1);
  }
  {
    // two clients on one pair: one per cycle, round robin; 8-B lanes after a 4-B lane on its pair
    SramPortRef p(c);
    p.push_amo(AMO_DXA, 3);
    p.push_amo(AMO_DXA, 3);
    p.push_amo_op(AMO_MBAR, {{6, 2}});
    expect("pair 3: dxa", p.tick().amo_pairs, 1u << 3);
    uint32_t n = 1;
    for (int k = 0; k < 10 && !p.idle(); ++k)
      n += p.tick().amo_pairs ? 1 : 0;
    expect("three updates, one per cycle", n, 3);
    SramPortRef q(c);
    q.push_wf(WF_SHARED, WF_AMO, 0, 9, {{0, 1}, {0, 2}, {1, 1}});
    uint32_t cyc = 0;
    bool done = false;
    while (!done && cyc < 10) {
      done = q.tick().wf_done;
      ++cyc;
    }
    expect("bank-in-order on a pair", cyc, 3);
  }
  {
    // a TCU stream ahead of the AMO winners: the pair winners hold their pick
    SramPortCfg c2 = c;
    c2.rd_pri[RD_TCU] = 3;
    SramPortRef p(c2);
    p.push_read(RD_TCU, 3);
    p.push_amo(AMO_DXA, 1);
    p.push_amo(AMO_DSMEM, 1);
    expect("tcu 1", p.tick().rd, RD_TCU);
    expect("tcu 2", p.tick().rd, RD_TCU);
    expect("tcu 3", p.tick().rd, RD_TCU);
    p.tick();
    p.tick();
    expect("both updated", p.idle(), 0);
  }
}

// U1: LDS and L1 hits share the wavefront stage about evenly, 128 B/cycle in total.
void port_share() {
  SramPortCfg c = default_cfg();
  SramPortRef p(c);
  uint32_t done[WF_N] = {0, 0};
  uint32_t last[WF_N] = {0, 0};
  for (uint32_t k = 0; k < 10000; ++k) {
    for (uint32_t cls = 0; cls < WF_N; ++cls) {
      if (done[cls] + 4 > last[cls]) {
        p.push_wf(cls, WF_READ, 1, cls);
        ++last[cls];
      }
    }
    auto g = p.tick();
    if (g.wf_done)
      ++done[g.wf_id];
  }
  expect("one wavefront per cycle", done[0] + done[1], 10000);
  expect("even split", done[0], 5000);
}

void structural_latency() {
  SmemCfg cfg{32, 4, 14};
  expect("pipe stages", smem_pipe_stages(1, false), 14);
  expect("pipe stages, collector routing and DLMEM", smem_pipe_stages(2, true), 16);
  expect("LDS, one wavefront", smem_load_latency(1, cfg), 16);
  expect("LDS, one cycle per extra wavefront", smem_load_latency(32, cfg) - smem_load_latency(1, cfg), 31);
  uint64_t addr[32];
  for (uint32_t i = 0; i < 32; ++i) addr[i] = 0;
  expect("atomic, one address: a pass per lane", smem_atom_passes(addr, 0xffffffffu, 32, 4, cfg), 32);
  for (uint32_t i = 0; i < 32; ++i) addr[i] = uint64_t(i) * 4;
  expect("atomic, a bank per lane: one pass", smem_atom_passes(addr, 0xffffffffu, 32, 4, cfg), 1);
  expect("atomic chase, one pass", smem_atom_latency(1, cfg), smem_load_latency(1, cfg));
}

} // namespace

int main() {
  lds_equivalence();
  l1_rows();
  level_rule();
  dsmem_merge();
  ports();
  port_share();
  structural_latency();
  std::printf("sram_port_ref: %d passed, %d failed\n", g_pass, g_fail);
  if (g_fail) {
    std::printf("FAILED\n");
    return 1;
  }
  std::printf("PASSED\n");
  return 0;
}
