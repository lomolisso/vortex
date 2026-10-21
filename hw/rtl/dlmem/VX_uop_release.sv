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

`include "VX_define.vh"

// VX_uop_release: a release is issued as [CFENCE, op]. The CTA fence orders the
// warp's earlier global accesses before the op performs, as a CTA-scope release
// does: an AMO, SC or LR with .rl (mbarrier arrives included), a blocking
// barrier and the arrive of an arrive/wait barrier.
// The CFENCE uop reads no register and writes none; with TCU metadata it waits
// for the metadata scoreboard bits, so earlier TCU_LD loads are ordered too.

module VX_uop_release import VX_gpu_pkg::*; (
    input  ibuffer_t ibuf_in,
    output wire      is_release,
    output ibuffer_t ibuf_cfence,
    output ibuffer_t ibuf_op
);
`ifdef VX_CFG_EXT_A_ENABLE
    wire is_rl_amo = (ibuf_in.ex_type == EX_LSU)
                  && ibuf_in.op_args.lsu.amo_valid
                  && ibuf_in.op_args.lsu.amo_rl;
`else
    wire is_rl_amo = 1'b0;
`endif
    wire is_bar = (ibuf_in.ex_type == EX_SFU)
               && (ibuf_in.op_type == INST_OP_BITS'(INST_SFU_BAR))
               && (ibuf_in.op_args.wctl.is_sync_bar || ibuf_in.op_args.wctl.is_bar_arrive);

    assign is_release = is_rl_amo || is_bar;

    ibuffer_t cfence;
    always_comb begin
        cfence = ibuf_in;
        cfence.uuid     = get_uop_uuid(ibuf_in.uuid, UOP_CTR_W'(0));
        cfence.ex_type  = EX_LSU;
        cfence.op_type  = INST_OP_BITS'(INST_LSU_CFENCE);
        cfence.op_args  = '0;
        cfence.wb       = 1'b0;
        cfence.rd_xregs = '0;
    `ifdef TCU_META_ENABLE
        cfence.rd_xregs[XREG_0] = 1'b1;
        cfence.rd_xregs[XREG_1] = 1'b1;
    `endif
        cfence.wr_xregs = '0;
        cfence.used_rs  = '0;
        cfence.rd       = '0;
        cfence.bytesel  = BYTESEL_DEFAULT;
    end
    assign ibuf_cfence = cfence;

    ibuffer_t op;
    always_comb begin
        op = ibuf_in;
        op.uuid = get_uop_uuid(ibuf_in.uuid, UOP_CTR_W'(1));
    end
    assign ibuf_op = op;

endmodule
