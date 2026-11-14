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

// The guard sits above the include so the file preprocesses to nothing
// without H100 WGMMA.
`ifdef VX_CFG_TCU_H100_ENABLE

`include "VX_define.vh"

// Exact f16 -> f32 widening of an f16 accumulator fed back as the FEDP addend.
// f16 subnormals become f32 normals; infinities and NaNs keep their class.
module VX_tcu_tfr_f16_to_f32 (
    input  wire [15:0] h,
    output wire [31:0] f
);
    wire       sign = h[15];
    wire [4:0] exp  = h[14:10];
    wire [9:0] frac = h[9:0];

    // position of the leading one of a subnormal fraction
    logic [3:0] msb;
    always_comb begin
        msb = '0;
        for (integer i = 0; i < 10; ++i) begin
            if (frac[i]) begin
                msb = 4'(i);
            end
        end
    end

    wire [9:0] sub_frac = 10'(frac << (4'd10 - msb));
    wire [7:0] sub_exp  = 8'(msb) + 8'd103;

    logic [31:0] res;
    always_comb begin
        if (exp == 5'd31) begin
            res = {sign, 8'hFF, frac, 13'd0};
        end else if (exp == 5'd0) begin
            res = (frac == 10'd0) ? {sign, 31'd0} : {sign, sub_exp, sub_frac, 13'd0};
        end else begin
            res = {sign, 8'(exp) + 8'd112, frac, 13'd0};
        end
    end

    assign f = res;

endmodule

`endif
