// Copyright © 2019-2026
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

// Hand-computed DLMEM aperture translations, shared by the SimX unit test and
// the RTL testbench (hw/unittest/dlmem_xlate). Plain data, no dependencies.
//
// Addresses are RV64 byte addresses. The low six bits pass through every kind
// (an out-of-bounds lane lands at its own slot's base plus its line offset), so
// a word-address testbench uses addr >> 3 and out_addr >> 3.
// Every case of a shape uses the shape's my_core, so a testbench built with
// CORE_ID % ncc == my_core runs all of that shape's cases.

#include <cstdint>

namespace dlmem_xlate_vec {

constexpr uint64_t kDlmemBase = 0x0000000200000000ull;  // VX_MEM_DLMEM_BASE_ADDR
constexpr uint64_t kLmemBase  = 0x00000001FFFC0000ull;  // VX_MEM_LMEM_BASE_ADDR

// Same order as dlmem_ref::Kind.
enum : uint32_t { K_NONE = 0, K_LOCAL = 1, K_REMOTE = 2, K_OOB = 3 };

constexpr uint32_t kMaxPeers = 8;

struct Shape {
  const char* name;
  uint32_t ncc;
  uint32_t nw;
  uint32_t max_cluster;
  uint32_t lmem_log;
  uint32_t my_core;
};

struct Peer {
  uint32_t core;
  uint32_t slot;
};

struct Vector {
  const char* name;
  uint32_t shape;              // index into kShapes
  uint32_t k;
  Peer     peers[kMaxPeers];   // peers[r] for r < k; the rest are zero
  uint32_t my_slot;            // the request's cta_id
  uint32_t stride;             // bytes per CTA slot (aligned lmem size)
  uint64_t addr;
  uint32_t kind;
  uint32_t core;               // 0 for K_NONE
  uint64_t out_addr;
};

// The first five are the RTL testbench's build shapes.
constexpr Shape kShapes[] = {
  {"ncc1_max1_lmem14",  1,  4, 1, 14, 0},  // window 16 KB, no rank field
  {"ncc2_max2_lmem14",  2,  4, 2, 14, 1},  // window 32 KB
  {"ncc4_max4_lmem16",  4,  8, 4, 16, 2},  // window 256 KB
  {"ncc8_max8_lmem18",  8,  8, 8, 18, 5},  // window 2 MB
  {"ncc8_max4_lmem17",  8, 16, 4, 17, 6},  // window 512 KB, core field wider than rank field
  {"ncc16_max8_lmem18", 16, 16, 8, 18, 9}, // window 2 MB, core field wider than rank field
};

constexpr uint32_t kNumShapes = sizeof(kShapes) / sizeof(kShapes[0]);

constexpr Vector kVectors[] = {
  // ncc1_max1_lmem14: window [0x2_0000_0000, 0x2_0000_4000), rank always 0.
  {"max1_rank0_self",        0, 1, {{0, 2}}, 2, 0x1000, 0x0000000200000000ull, K_LOCAL,  0, 0x00000001FFFC2000ull},
  {"max1_off_stride_m8",     0, 1, {{0, 2}}, 2, 0x1000, 0x0000000200000FF8ull, K_LOCAL,  0, 0x00000001FFFC2FF8ull},
  {"max1_off_stride",        0, 1, {{0, 2}}, 2, 0x1000, 0x0000000200001000ull, K_OOB,    0, 0x00000001FFFC2000ull},
  {"max1_window_last_byte",  0, 1, {{0, 0}}, 0, 0x4000, 0x0000000200003FFFull, K_LOCAL,  0, 0x00000001FFFC3FFFull},
  {"max1_window_end",        0, 1, {{0, 0}}, 0, 0x4000, 0x0000000200004000ull, K_NONE,   0, 0x0000000200004000ull},
  {"max1_base_m8",           0, 1, {{0, 0}}, 0, 0x4000, 0x00000001FFFFFFF8ull, K_NONE,   0, 0x00000001FFFFFFF8ull},
  {"max1_stride0",           0, 1, {{0, 1}}, 1, 0,      0x0000000200000068ull, K_OOB,    0, 0x00000001FFFC0028ull},

  // ncc2_max2_lmem14: window [0x2_0000_0000, 0x2_0000_8000), rank = bit 14.
  {"max2_rank0_self",        1, 2, {{1, 3}, {0, 1}}, 3, 0x1000, 0x0000000200000040ull, K_LOCAL,  1, 0x00000001FFFC3040ull},
  {"max2_remote_stride_m8",  1, 2, {{1, 3}, {0, 1}}, 3, 0x1000, 0x0000000200004FF8ull, K_REMOTE, 0, 0x0000000200001FF8ull},
  {"max2_remote_off_stride", 1, 2, {{1, 3}, {0, 1}}, 3, 0x1000, 0x0000000200005000ull, K_OOB,    1, 0x00000001FFFC3000ull},
  {"max2_rank_eq_k",         1, 1, {{1, 3}},         3, 0x1000, 0x0000000200004010ull, K_OOB,    1, 0x00000001FFFC3010ull},
  {"max2_own_lmem",          1, 2, {{1, 3}, {0, 1}}, 3, 0x1000, 0x00000001FFFC1238ull, K_NONE,   0, 0x00000001FFFC1238ull},
  {"max2_window_last_byte",  1, 2, {{1, 0}, {0, 0}}, 0, 0x4000, 0x0000000200007FFFull, K_REMOTE, 0, 0x0000000200003FFFull},
  {"max2_window_end",        1, 2, {{1, 3}, {0, 1}}, 3, 0x1000, 0x0000000200008000ull, K_NONE,   0, 0x0000000200008000ull},

  // ncc4_max4_lmem16: window [0x2_0000_0000, 0x2_0004_0000), rank = bits 17:16.
  {"max4_window_first_byte", 2, 3, {{2, 5}, {2, 1}, {3, 7}}, 5, 0x2000, 0x0000000200000000ull, K_LOCAL,  2, 0x00000001FFFCA000ull},
  {"max4_same_core_slot1",   2, 3, {{2, 5}, {2, 1}, {3, 7}}, 5, 0x2000, 0x0000000200011FF8ull, K_LOCAL,  2, 0x00000001FFFC3FF8ull},
  {"max4_remote_core3",      2, 3, {{2, 5}, {2, 1}, {3, 7}}, 5, 0x2000, 0x0000000200020048ull, K_REMOTE, 3, 0x000000020003E048ull},
  {"max4_rank_eq_k",         2, 3, {{2, 5}, {2, 1}, {3, 7}}, 5, 0x2000, 0x0000000200030008ull, K_OOB,    2, 0x00000001FFFCA008ull},
  {"max4_rank_top_gt_k",     2, 2, {{2, 5}, {2, 1}},         5, 0x2000, 0x0000000200030040ull, K_OOB,    2, 0x00000001FFFCA000ull},
  {"max4_window_last_byte",  2, 4, {{2, 0}, {1, 0}, {3, 0}, {0, 0}}, 0, 0x10000, 0x000000020003FFFFull, K_REMOTE, 0, 0x000000020000FFFFull},
  {"max4_io",                2, 3, {{2, 5}, {2, 1}, {3, 7}}, 5, 0x2000, 0x0000000000000040ull, K_NONE,   0, 0x0000000000000040ull},
  {"max4_window_end",        2, 3, {{2, 5}, {2, 1}, {3, 7}}, 5, 0x2000, 0x0000000200040000ull, K_NONE,   0, 0x0000000200040000ull},
  {"max4_base_m8",           2, 3, {{2, 5}, {2, 1}, {3, 7}}, 5, 0x2000, 0x00000001FFFFFFF8ull, K_NONE,   0, 0x00000001FFFFFFF8ull},

  // ncc8_max8_lmem18: window [0x2_0000_0000, 0x2_0020_0000), rank = bits 20:18.
  {"max8_rank0_self",        3, 8, {{5, 4}, {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {6, 6}, {7, 7}}, 4, 0x8000, 0x0000000200001238ull, K_LOCAL,  5, 0x00000001FFFE1238ull},
  {"max8_rank7_stride_m8",   3, 8, {{5, 4}, {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {6, 6}, {7, 7}}, 4, 0x8000, 0x00000002001C7FF8ull, K_REMOTE, 7, 0x00000002001FFFF8ull},
  {"max8_window_last_byte",  3, 8, {{5, 4}, {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {6, 6}, {7, 7}}, 4, 0x8000, 0x00000002001FFFFFull, K_OOB,    5, 0x00000001FFFE003Full},
  {"max8_base_m8_own_lmem",  3, 8, {{5, 4}, {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {6, 6}, {7, 7}}, 4, 0x8000, 0x00000001FFFFFFF8ull, K_NONE,   0, 0x00000001FFFFFFF8ull},
  {"max8_rank2_remote",      3, 8, {{5, 4}, {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {6, 6}, {7, 7}}, 4, 0x8000, 0x0000000200080040ull, K_REMOTE, 1, 0x0000000200048040ull},
  {"max8_off_stride",        3, 8, {{5, 4}, {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {6, 6}, {7, 7}}, 4, 0x8000, 0x00000002000C8000ull, K_OOB,    5, 0x00000001FFFE0000ull},
  {"max8_stride0",           3, 8, {{5, 4}, {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {6, 6}, {7, 7}}, 4, 0,      0x0000000200040000ull, K_OOB,    5, 0x00000001FFFC0000ull},
  {"max8_window_end",        3, 8, {{5, 4}, {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {6, 6}, {7, 7}}, 4, 0x8000, 0x0000000200200000ull, K_NONE,   0, 0x0000000200200000ull},

  // ncc8_max4_lmem17: window [0x2_0000_0000, 0x2_0008_0000), rank = bits 18:17.
  // A remote encoding can lie past the window (core field 3 bits, rank field 2).
  {"ncc8_remote_past_window", 4, 4, {{6, 9}, {7, 15}, {0, 3}, {6, 10}}, 9, 0x2000, 0x0000000200021FF8ull, K_REMOTE, 7, 0x00000002000FFFF8ull},
  {"ncc8_remote_core0",       4, 4, {{6, 9}, {7, 15}, {0, 3}, {6, 10}}, 9, 0x2000, 0x0000000200040000ull, K_REMOTE, 0, 0x0000000200006000ull},
  {"ncc8_same_core_slot10",   4, 4, {{6, 9}, {7, 15}, {0, 3}, {6, 10}}, 9, 0x2000, 0x0000000200060100ull, K_LOCAL,  6, 0x00000001FFFD4100ull},
  {"ncc8_rank0_self",         4, 4, {{6, 9}, {7, 15}, {0, 3}, {6, 10}}, 9, 0x2000, 0x0000000200001FF8ull, K_LOCAL,  6, 0x00000001FFFD3FF8ull},
  {"ncc8_window_end",         4, 4, {{6, 9}, {7, 15}, {0, 3}, {6, 10}}, 9, 0x2000, 0x0000000200080000ull, K_NONE,   0, 0x0000000200080000ull},
  {"ncc8_remote_encoding_in", 4, 4, {{6, 9}, {7, 15}, {0, 3}, {6, 10}}, 9, 0x2000, 0x00000002000FFFF8ull, K_NONE,   0, 0x00000002000FFFF8ull},

  // ncc16_max8_lmem18: window [0x2_0000_0000, 0x2_0020_0000), rank = bits 20:18, core = bits 21:18.
  {"ncc16_remote_core15",     5, 8, {{9, 2}, {15, 14}, {12, 0}, {9, 3}, {1, 1}, {0, 15}, {8, 8}, {10, 4}}, 2, 0x4000, 0x0000000200043FF8ull, K_REMOTE, 15, 0x00000002003FBFF8ull},
  {"ncc16_remote_core12",     5, 8, {{9, 2}, {15, 14}, {12, 0}, {9, 3}, {1, 1}, {0, 15}, {8, 8}, {10, 4}}, 2, 0x4000, 0x0000000200080040ull, K_REMOTE, 12, 0x0000000200300040ull},
  {"ncc16_same_core_slot3",   5, 8, {{9, 2}, {15, 14}, {12, 0}, {9, 3}, {1, 1}, {0, 15}, {8, 8}, {10, 4}}, 2, 0x4000, 0x00000002000C0008ull, K_LOCAL,   9, 0x00000001FFFCC008ull},
  {"ncc16_remote_core0_slot15", 5, 8, {{9, 2}, {15, 14}, {12, 0}, {9, 3}, {1, 1}, {0, 15}, {8, 8}, {10, 4}}, 2, 0x4000, 0x0000000200140010ull, K_REMOTE, 0, 0x000000020003C010ull},
};

constexpr uint32_t kNumVectors = sizeof(kVectors) / sizeof(kVectors[0]);

} // namespace dlmem_xlate_vec
