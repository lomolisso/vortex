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

// Golden model of the uniform datapath: one value per op (the UR result), XLEN bits.
// Op codes are VX_gpu_pkg's INST_UOP_* and SimX's UniformType.

#pragma once

#include <cstdint>

namespace uniform_ref {

enum : uint32_t {
  UOP_ADD, UOP_SUB, UOP_AND, UOP_OR, UOP_XOR, UOP_SLL, UOP_SRL, UOP_SRA,
  UOP_SLT, UOP_SLTU, UOP_SEQ, UOP_R2UR, UOP_UR2R
};

// a, b: the UR sources (R2UR: a = the first active lane's GPR; UR2R: a = the UR).
inline uint64_t execute(uint32_t op, uint32_t xlen, uint64_t a, uint64_t b) {
  const uint64_t xmask = (xlen == 64) ? ~0ull : 0xffffffffull;
  a &= xmask;
  b &= xmask;
  const uint32_t shamt = uint32_t(b & (xlen - 1));
  const int64_t sa = (xlen == 64) ? int64_t(a) : int64_t(int32_t(uint32_t(a)));
  const int64_t sb = (xlen == 64) ? int64_t(b) : int64_t(int32_t(uint32_t(b)));
  uint64_t r;
  switch (op) {
  case UOP_ADD:  r = a + b; break;
  case UOP_SUB:  r = a - b; break;
  case UOP_AND:  r = a & b; break;
  case UOP_OR:   r = a | b; break;
  case UOP_XOR:  r = a ^ b; break;
  case UOP_SLL:  r = a << shamt; break;
  case UOP_SRL:  r = a >> shamt; break;
  case UOP_SRA:  r = uint64_t(sa >> shamt); break;
  case UOP_SLT:  r = sa < sb; break;
  case UOP_SLTU: r = a < b; break;
  case UOP_SEQ:  r = a == b; break;
  default:       r = a; break; // R2UR, UR2R
  }
  return r & xmask;
}

// Operand-collector cycles beyond the first: one UR (or one aligned quad) read per cycle,
// overlapped with the GPR bank reads; a UR slot never takes a GPR bank port.
inline uint32_t collect_stalls(uint32_t gpr_bank_stalls, uint32_t ur_slots) {
  const uint32_t ur = ur_slots ? ur_slots - 1 : 0;
  return gpr_bank_stalls > ur ? gpr_bank_stalls : ur;
}

} // namespace uniform_ref
