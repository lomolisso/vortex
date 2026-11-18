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

#ifndef VX_SIMX_FE_TYPES_H
#define VX_SIMX_FE_TYPES_H

// Front-end op types. Every use in an existing file sits at the end of a line
// that exists without the switch (VX_FE_ONLY), so knob-off __LINE__ values stay.
#ifdef VX_CFG_H100_FE_ENABLE

#define VX_FE_ONLY(...) __VA_ARGS__

namespace vortex {

// Uniform-register ops (uniform PE); codes as VX_gpu_pkg INST_UOP_*.
enum class UniformType {
  ADD, SUB, AND, OR, XOR, SLL, SRL, SRA, SLT, SLTU, SEQ, R2UR, UR2R
};

inline std::ostream &operator<<(std::ostream &os, const UniformType& type) {
  static const char* const names[] = {
    "UADD", "USUB", "UAND", "UOR", "UXOR", "USLL", "USRL", "USRA", "USLT", "USLTU", "USEQ", "R2UR", "UR2R"
  };
  os << names[(int)type];
  return os;
}

// Warp ops (warp-op PE); values are the warp_ops_ref.h codes, as VX_gpu_pkg INST_WOP_*.
enum class WarpOpType {
  REDUX_ADD, REDUX_MIN, REDUX_MAX, REDUX_UMIN, REDUX_UMAX, REDUX_AND, REDUX_OR, REDUX_XOR,
  ELECT_P = 0x10, ELECT_L,
  MATCH_ANY_B32 = 0x18, MATCH_ANY_B64, MATCH_ALL_B32, MATCH_ALL_B64
};

inline std::ostream &operator<<(std::ostream &os, const WarpOpType& type) {
  switch (type) {
  case WarpOpType::REDUX_ADD:     os << "REDUX.ADD"; break;
  case WarpOpType::REDUX_MIN:     os << "REDUX.MIN"; break;
  case WarpOpType::REDUX_MAX:     os << "REDUX.MAX"; break;
  case WarpOpType::REDUX_UMIN:    os << "REDUX.UMIN"; break;
  case WarpOpType::REDUX_UMAX:    os << "REDUX.UMAX"; break;
  case WarpOpType::REDUX_AND:     os << "REDUX.AND"; break;
  case WarpOpType::REDUX_OR:      os << "REDUX.OR"; break;
  case WarpOpType::REDUX_XOR:     os << "REDUX.XOR"; break;
  case WarpOpType::ELECT_P:       os << "ELECT.P"; break;
  case WarpOpType::ELECT_L:       os << "ELECT.L"; break;
  case WarpOpType::MATCH_ANY_B32: os << "MATCH.ANY.B32"; break;
  case WarpOpType::MATCH_ANY_B64: os << "MATCH.ANY.B64"; break;
  case WarpOpType::MATCH_ALL_B32: os << "MATCH.ALL.B32"; break;
  case WarpOpType::MATCH_ALL_B64: os << "MATCH.ALL.B64"; break;
  default: assert(false);
  }
  return os;
}

} // namespace vortex

#define VX_FE_OP_STRING_CASES \
  [&](UniformType type)-> op_string_t { std::ostringstream os; os << type; return {os.str(), ""}; }, \
  [&](WarpOpType type)-> op_string_t { std::ostringstream os; os << type; return {os.str(), ""}; },

#else
#define VX_FE_ONLY(...)
#endif

#endif
