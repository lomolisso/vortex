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

`ifndef DPI_TCU_VH
`define DPI_TCU_VH

// One FEDP call of the TFR datapath model (sw/common/tcu_fedp_ref.h): n words
// of a and b (at most 8), c and the result in the destination format.
import "DPI-C" pure function int unsigned dpi_tcu_fedp(input int fmt_s, input int fmt_d, input int n, input bit [255:0] a, input bit [255:0] b, input int unsigned c);

`endif
