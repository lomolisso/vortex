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

// DXA descriptor-meta field widths and bit offsets, shared by the host encoder
// (dxa.h) and the SimX decoder (sim/simx/dxa). The offsets are the running sum
// of the field widths, so they live beside the encoder/decoder instead of being
// exported as generated macros. The RTL mirror lives in VX_dxa_pkg.

#pragma once

#define DXA_DESC_META_DIM_BITS        3
#define DXA_DESC_META_ELEMSZ_BITS     2
#define DXA_DESC_META_LAYOUT_BITS     2
#define DXA_DESC_META_SWIZZLE_BITS    2
#define DXA_DESC_META_INTERLEAVE_BITS 2
#define DXA_DESC_META_L2PROMO_BITS    2

#define DXA_DESC_META_DIM_LSB        0
#define DXA_DESC_META_ELEMSZ_LSB     (DXA_DESC_META_DIM_LSB + DXA_DESC_META_DIM_BITS)
#define DXA_DESC_META_LAYOUT_LSB     (DXA_DESC_META_ELEMSZ_LSB + DXA_DESC_META_ELEMSZ_BITS)
#define DXA_DESC_META_SWIZZLE_LSB    (DXA_DESC_META_LAYOUT_LSB + DXA_DESC_META_LAYOUT_BITS)
#define DXA_DESC_META_INTERLEAVE_LSB (DXA_DESC_META_SWIZZLE_LSB + DXA_DESC_META_SWIZZLE_BITS)
#define DXA_DESC_META_L2PROMO_LSB    (DXA_DESC_META_INTERLEAVE_LSB + DXA_DESC_META_INTERLEAVE_BITS)

// META[13]: 0 tile mode, 1 im2col mode (DXA with mbarriers only).
#define DXA_DESC_META_IM2COL_BITS 1
#define DXA_DESC_META_IM2COL_LSB  13

// Stores, reductions and bulk groups (DXA with mbarriers, XLEN 64).
// custom-0 funct7 3: funct3 5 = STORE family, funct3 6 = bulk-group op.
#define DXA_F3_STORE 5
#define DXA_F3_BGOP  6

// STORE ctl word (rs2 lane 0, bits [63:32]): kind, reduction op, type, descriptor slot.
#define DXA_ST_CTL_KIND_LSB  0
#define DXA_ST_CTL_KIND_BITS 3
#define DXA_ST_CTL_OP_LSB    3
#define DXA_ST_CTL_OP_BITS   4
#define DXA_ST_CTL_TYPE_LSB  7
#define DXA_ST_CTL_TYPE_BITS 4
#define DXA_ST_CTL_SLOT_LSB  11
#define DXA_ST_CTL_SLOT_BITS 4

#define DXA_ST_BULK        0  // cp.async.bulk.global.shared::cta.bulk_group
#define DXA_ST_TENSOR      1  // cp.async.bulk.tensor.Nd.global.shared::cta.bulk_group
#define DXA_ST_RED_GLOBAL  2  // cp.reduce.async.bulk.global.shared::cta.bulk_group
#define DXA_ST_RED_CLUSTER 3  // cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier
#define DXA_ST_RED_TENSOR  4  // cp.reduce.async.bulk.tensor.Nd.global.shared::cta.bulk_group

#define DXA_RED_ADD 0
#define DXA_RED_MIN 1
#define DXA_RED_MAX 2
#define DXA_RED_INC 3
#define DXA_RED_DEC 4
#define DXA_RED_AND 5
#define DXA_RED_OR  6
#define DXA_RED_XOR 7

#define DXA_RT_U32  0  // also b32
#define DXA_RT_S32  1
#define DXA_RT_U64  2  // also b64
#define DXA_RT_S64  3
#define DXA_RT_F32  4
#define DXA_RT_F64  5
#define DXA_RT_F16  6
#define DXA_RT_BF16 7

// funct3 6: the rs1 register field is the op, the rs2 register field is N.
#define DXA_BG_COMMIT    0
#define DXA_BG_WAIT      1
#define DXA_BG_WAIT_READ 2

// Per warp: committed groups not yet complete, operations in the open group.
#define DXA_BG_GROUPS  8
#define DXA_BG_MAX_OPS 63

// custom-2 funct3 7 funct7 1 (fence.proxy.async): the rs1 register field.
#define DXA_PFENCE_SHARED 0
#define DXA_PFENCE_GLOBAL 1
#define DXA_PFENCE_ALL    2

// Tensor maps in memory (DXA with mbarriers, XLEN 64; layout in dxa_tmap.h).
// Address-tagged descriptor-cache entries beside the VX_DCR_DXA_DESC_COUNT slots.
#define DXA_TMAP_ENTRIES 8

// TLOAD ctl word (funct3 2, rs2 lane 0 bits [63:32]): bit 5 = the map form,
// with the map address in rs2 lane 2 and the slot bits [3:0] zero.
#define DXA_TL_CTL_MAP_LSB 5

// STORE ctl word (funct3 5): bit 15 = the map form of a tensor store or
// reduction, with the map address in rs2 lane 1 and the slot bits zero.
#define DXA_ST_CTL_MAP_LSB 15

// Reduction type of a map-form tensor reduction: the map's element type.
#define DXA_RT_FROM_MAP 15

// funct3 7: tensormap op; rs1 lane 0 = the map address, the rs2 register field is the op.
#define DXA_F3_TMAP      7
#define DXA_TMAP_PREFETCH 0  // prefetch.tensormap
#define DXA_TMAP_INVAL    1  // fence.proxy.tensormap::generic.acquire
