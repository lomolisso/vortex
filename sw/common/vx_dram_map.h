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

#ifndef VX_DRAM_MAP_H
#define VX_DRAM_MAP_H

#include <stdint.h>

// Row-bit masks of the H100's five DRAM bank functions; a narrower geometry uses the first lg_banks of them,
// a wider one leaves the bank bits above them unhashed (no H100 data).
#define VX_DRAM_HB_H100 {0xe2eull, 0x1e7dull, 0x1c8full, 0x1919ull, 0xa3aull}
#define VX_DRAM_HB_MAX 5

typedef struct { uint32_t lg_lines_per_row, lg_banks; const uint64_t* hb; } vx_dram_geom_t;
typedef struct { uint64_t row; uint32_t col, bank; } vx_dram_loc_t;
typedef struct { uint32_t pc, bg, ba, column; uint64_t row; } vx_dram_fields_t;

static inline uint32_t vx_dram_parity(uint64_t x) {
  x ^= x >> 32; x ^= x >> 16; x ^= x >> 8; x ^= x >> 4; x ^= x >> 2; x ^= x >> 1;
  return (uint32_t)(x & 1);
}

// Lc = line index inside one channel. H100 (8 lines per row, 32 banks): row = Lc >> 8, col = (Lc0, Lc6, Lc7),
// bank bit k = Lc(k+1) ^ par(row & HB[k]), bank bit 0 also ^ Lc6 ^ Lc7 (doc 07 §3, doc 09 h100_dram()). Other
// geometries keep the shape: col = Lc0 and the bits above the bank bits, the bank bits right above Lc0.
static inline vx_dram_loc_t vx_dram_map(uint64_t lc, const vx_dram_geom_t* g) {
  const uint32_t lr = g->lg_lines_per_row, nb = g->lg_banks;
  const uint32_t own_lsb = lr ? 1 : 0;
  const uint64_t upper = lr ? (lc >> (nb + 1)) & ((1ull << (lr - 1)) - 1) : 0;
  const uint32_t s = (uint32_t)((lc >> own_lsb) & ((1ull << nb) - 1)) ^ (nb ? vx_dram_parity(upper) : 0);
  vx_dram_loc_t loc;
  uint32_t k;
  loc.row = lc >> (lr + nb);
  loc.col = lr ? (uint32_t)((lc & 1) | (upper << 1)) : 0;
  loc.bank = 0;
  for (k = 0; k < nb; ++k)
    loc.bank |= (((s >> k) & 1) ^ (k < VX_DRAM_HB_MAX ? vx_dram_parity(loc.row & g->hb[k]) : 0)) << k;
  return loc;
}

// Interim channel select (until the L2 slice hash): channel = the address bits right above one DRAM line.
static inline uint32_t vx_dram_interim_channel(uint64_t addr, uint32_t lg_line, uint32_t lg_channels) {
  return (uint32_t)((addr >> lg_line) & ((1ull << lg_channels) - 1));
}

static inline uint64_t vx_dram_interim_local(uint64_t addr, uint32_t lg_line, uint32_t lg_channels) {
  return ((addr >> (lg_line + lg_channels)) << lg_line) | (addr & ((1ull << lg_line) - 1));
}

static inline uint64_t vx_dram_interim_addr(uint32_t channel, uint64_t local, uint32_t lg_line, uint32_t lg_channels) {
  return ((local >> lg_line) << (lg_line + lg_channels)) | ((uint64_t)channel << lg_line) |
         (local & ((1ull << lg_line) - 1));
}

// Flat bank bits from low to high: pseudo-channel, bank group, bank; column counts atoms (sectors) in the row.
static inline vx_dram_fields_t vx_dram_fields(vx_dram_loc_t loc, uint32_t sector, uint32_t lg_atoms_per_line,
                                              uint32_t lg_pcs, uint32_t lg_bgs) {
  vx_dram_fields_t f;
  f.pc = loc.bank & ((1u << lg_pcs) - 1);
  f.bg = (loc.bank >> lg_pcs) & ((1u << lg_bgs) - 1);
  f.ba = loc.bank >> (lg_pcs + lg_bgs);
  f.column = (loc.col << lg_atoms_per_line) | sector;
  f.row = loc.row;
  return f;
}

// Ramulator's ChRaBaRoCo order inside one channel, MSB to LSB: PC, (SID: one), BG, bank, row, column, atom offset.
static inline uint64_t vx_dram_pack(vx_dram_fields_t f, uint32_t lg_pcs, uint32_t lg_bgs, uint32_t lg_ba,
                                    uint32_t lg_rows, uint32_t lg_tx_per_row, uint32_t lg_atom) {
  uint64_t a = f.pc;
  (void)lg_pcs;
  a = (a << lg_bgs) | f.bg;
  a = (a << lg_ba) | f.ba;
  a = (a << lg_rows) | f.row;
  a = (a << lg_tx_per_row) | f.column;
  return a << lg_atom;
}

#endif
