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
//
// Derived from fNToRecFN.v of the Berkeley HardFloat IEEE Floating-Point
// Arithmetic Package, Release 1, by John R. Hauser, under this notice:
//
// Copyright 2019 The Regents of the University of California.  All rights
// reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//  1. Redistributions of source code must retain the above copyright notice,
//     this list of conditions, and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions, and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//  3. Neither the name of the University nor the names of its contributors may
//     be used to endorse or promote products derived from this software without
//     specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
// DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// VX_config.vh preprocesses to nothing, so with the knob off this file is empty.
`include "VX_config.vh"

`ifdef VX_CFG_EXT_FPAMO_ENABLE
`include "VX_define.vh"

// IEEE binary format to HardFloat's recoded format: a port of HardFloat's
// fNToRecFN with VX_lzc as the leading-zero counter. The submodule's
// fNToRecFN instantiates bsg_counting_leading_zeros, which only the BHF tensor
// unit's directory provides, and the synthesis flows cannot add that directory
// without the tensor unit's package.

module VX_amo_fpu_rec #(
    parameter EXPW = 8,
    parameter SIGW = 24     // includes the implicit bit
) (
    input  wire [EXPW+SIGW-1:0] in,
    output wire [EXPW+SIGW:0]   out
);
    localparam NDW = `LOG2UP(SIGW - 1);

    wire            sign;
    wire [EXPW-1:0] exp_in;
    wire [SIGW-2:0] fract_in;
    assign {sign, exp_in, fract_in} = in;
    wire is_zero_exp   = (exp_in == '0);
    wire is_zero_fract = (fract_in == '0);

    wire [NDW-1:0] norm_dist;
    VX_lzc #(
        .N (SIGW - 1)
    ) lzc (
        .data_in   (fract_in),
        .data_out  (norm_dist),
        `UNUSED_PIN (valid_out)
    );

    wire [SIGW-2:0] subnorm_fract = (fract_in << norm_dist) << 1;
    wire [EXPW:0] adj_exp = (is_zero_exp ? ((EXPW+1)'(norm_dist) ^ {(EXPW+1){1'b1}}) : (EXPW+1)'(exp_in))
                          + ((EXPW+1)'(1 << (EXPW - 1)) | (is_zero_exp ? (EXPW+1)'(2) : (EXPW+1)'(1)));
    wire is_zero    = is_zero_exp && is_zero_fract;
    wire is_special = (adj_exp[EXPW:EXPW-1] == 2'b11);

    wire [EXPW:0] exp_out;
    assign exp_out[EXPW:EXPW-2] = is_special ? {2'b11, ~is_zero_fract}
                                : (is_zero ? 3'b000 : adj_exp[EXPW:EXPW-2]);
    assign exp_out[EXPW-3:0]    = adj_exp[EXPW-3:0];
    assign out = {sign, exp_out, is_zero_exp ? subnorm_fract : fract_in};

endmodule
`endif
