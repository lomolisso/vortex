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

// Built only with H100 WGMMA; the knob may come from CONFIGS or the toml.
#include "VX_config.h"
#ifdef VX_CFG_TCU_H100_ENABLE

#include <cstdint>
#include "svdpi.h"
#include "tcu_fedp_ref.h"

extern "C" {
  unsigned dpi_tcu_fedp(int fmt_s, int fmt_d, int n, const svBitVecVal* a, const svBitVecVal* b, unsigned c);
}

unsigned dpi_tcu_fedp(int fmt_s, int fmt_d, int n, const svBitVecVal* a, const svBitVecVal* b, unsigned c) {
  int eb, sb;
  // integer and idle inputs take the module's other path; the value is unused
  if (n < 1 || n > 8 || !vortex::tcu_fedp_ref::fmt_bits((uint32_t)fmt_s, &eb, &sb))
    return 0;
  uint32_t aw[8], bw[8];
  for (int i = 0; i < n; ++i) {
    aw[i] = a[i];
    bw[i] = b[i];
  }
  return vortex::tcu_fedp_ref::fedp((uint32_t)fmt_s, (uint32_t)fmt_d, aw, bw, c, (uint32_t)n);
}

#endif
