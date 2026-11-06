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
// The same pair splits a 16-byte st.async into its two 8-byte halves, low then
// high, each completing its own 8 bytes on the mbarrier, and a 16-byte cp.async
// into its two 8-byte copies, low then high, each taking its share of the
// src-size in the LSU.

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

`ifdef VX_CFG_EXT_MBAR_ENABLE
    // A plain store with the reserved funct3=4 also decodes to op_type 4'b1100.
    wire is_q  = (ibuf_in.ex_type == EX_LSU)
              && ibuf_in.op_args.lsu.is_async
              && (ibuf_in.op_type == INST_OP_BITS'(INST_LSU_SQ));
    wire is_sq = is_q && ibuf_in.op_args.lsu.is_store;
    assign is_release = is_rl_amo || is_bar || is_q;
`else
    assign is_release = is_rl_amo || is_bar;
`endif

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
`ifndef VX_CFG_EXT_MBAR_ENABLE
    assign ibuf_cfence = cfence;
`endif

    ibuffer_t op;
    always_comb begin
        op = ibuf_in;
        op.uuid = get_uop_uuid(ibuf_in.uuid, UOP_CTR_W'(1));
    end

`ifdef VX_CFG_EXT_MBAR_ENABLE
    // The high st.async half's data is the odd register of the pair; x0 stands
    // for a zero pair. Both cp.async halves read the same src-size register.
    ibuffer_t sq_lo, sq_hi;
    always_comb begin
        sq_lo = ibuf_in;
        sq_lo.uuid    = get_uop_uuid(ibuf_in.uuid, UOP_CTR_W'(0));
        sq_lo.op_type = is_sq ? INST_OP_BITS'(INST_LSU_SD) : INST_OP_BITS'(INST_LSU_CPA16H);
        sq_hi = ibuf_in;
        sq_hi.uuid    = get_uop_uuid(ibuf_in.uuid, UOP_CTR_W'(1));
        sq_hi.op_type = is_sq ? INST_OP_BITS'(INST_LSU_SD) : INST_OP_BITS'(INST_LSU_CPA16H);
        sq_hi.op_args.lsu.offset = 12'd8;
        if (is_sq) begin
            sq_hi.rs2 = ibuf_in.rs2 | NUM_REGS_BITS'(ibuf_in.rs2 != '0);
        end
    end
    assign ibuf_cfence = is_q ? sq_lo : cfence;
    assign ibuf_op     = is_q ? sq_hi : op;
`else
    assign ibuf_op = op;
`endif

endmodule
