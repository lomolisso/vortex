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

// reset all GPRs in debug mode
`ifdef SIMULATION
`ifndef NDEBUG
`ifndef GPR_RESET
`define GPR_RESET
`endif
`endif
`endif

module VX_opc_unit import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_BANKS = 4,
    parameter OUT_BUF   = 3
) (
    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    output wire [PERF_CTR_BITS-1:0] perf_stalls,
`endif

    VX_writeback_if.slave   writeback_if,
    VX_scoreboard_if.slave  scoreboard_if,
    VX_operands_if.master   operands_if
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam REQ_SEL_WIDTH    = SRC_OPD_WIDTH;
    localparam BANK_SEL_BITS    = `CLOG2(NUM_BANKS);
    localparam BANK_SEL_WIDTH   = `UP(BANK_SEL_BITS);
    localparam BANK_DATA_WIDTH  = `VX_CFG_XLEN * `VX_CFG_SIMD_WIDTH;
    localparam BANK_DATA_SIZE   = BANK_DATA_WIDTH / 8;

    localparam BANK_SIZE        = (NUM_REGS * SIMD_COUNT * PER_OPC_WARPS) / NUM_BANKS;
    localparam BANK_ADDR_WIDTH  = `CLOG2(BANK_SIZE);

    localparam REG_REM_BITS     = NUM_REGS_BITS - BANK_SEL_BITS;

    localparam META_DATAW       = UUID_WIDTH + ISSUE_WIS_W + NCTA_WIDTH + SIMD_IDX_W + `VX_CFG_SIMD_WIDTH + PC_BITS + EX_BITS + INST_OP_BITS + INST_ARGS_BITS + 1 + NUM_XREGS + NUM_REGS_BITS + BYTESEL_BITS + 1 + 1;
    localparam OUT_DATAW        = $bits(operands_t);

    `UNUSED_VAR (writeback_if.data.sop)

    wire [NUM_SRC_OPDS-1:0] src_valid;
    wire [NUM_SRC_OPDS-1:0] req_valid_in, req_ready_in;
    wire [NUM_SRC_OPDS-1:0][REG_REM_BITS-1:0] req_addr_in;
    wire [NUM_SRC_OPDS-1:0][BANK_SEL_WIDTH-1:0] req_bank_idx;

    wire [NUM_BANKS-1:0] gpr_rd_valid, gpr_rd_ready;
    wire [NUM_BANKS-1:0] gpr_rd_valid_st1, gpr_rd_valid_st2;
    wire [NUM_BANKS-1:0][REG_REM_BITS-1:0] gpr_rd_reg, gpr_rd_reg_st1;
    wire [NUM_BANKS-1:0][`VX_CFG_SIMD_WIDTH-1:0][`VX_CFG_XLEN-1:0] gpr_rd_data_st2;
    wire [NUM_BANKS-1:0][REQ_SEL_WIDTH-1:0] gpr_rd_opd, gpr_rd_opd_st1, gpr_rd_opd_st2;

    wire [`VX_CFG_SIMD_WIDTH-1:0] simd_out;
    wire [SIMD_IDX_W-1:0] simd_pid;
    wire simd_sop, simd_eop;

    wire pipe_ready_in;
    wire pipe_valid_st1, pipe_ready_st1;
    wire pipe_valid_st2, pipe_ready_st2;
    wire [META_DATAW-1:0] pipe_mdata, pipe_mdata_st1, pipe_mdata_st2;

    reg [NUM_SRC_OPDS-1:0][(`VX_CFG_SIMD_WIDTH * `VX_CFG_XLEN)-1:0] opd_buffer_st2, opd_buffer_n_st2;

    reg [NUM_SRC_OPDS-1:0] opd_fetched_st1;

    reg has_collision;
    wire has_collision_st1;

    wire [NUM_SRC_OPDS-1:0][NUM_REGS_BITS-1:0] src_regs;
    assign src_regs = {scoreboard_if.data.rs3, scoreboard_if.data.rs2, scoreboard_if.data.rs1};

    for (genvar i = 0; i < NUM_SRC_OPDS; ++i) begin : g_gpr_rd_reg
        assign req_addr_in[i] = src_regs[i][NUM_REGS_BITS-1 -: REG_REM_BITS];
    end

    for (genvar i = 0; i < NUM_SRC_OPDS; ++i) begin : g_req_bank_idx
        if (NUM_BANKS != 1) begin : g_bn
            assign req_bank_idx[i] = src_regs[i][BANK_SEL_BITS-1:0];
        end else begin : g_b1
            assign req_bank_idx[i] = '0;
        end
    end

    for (genvar i = 0; i < NUM_SRC_OPDS; ++i) begin : g_src_valid
        assign src_valid[i] = scoreboard_if.data.used_rs[i] && (src_regs[i] != 0) && ~opd_fetched_st1[i];
    end

    assign req_valid_in = {NUM_SRC_OPDS{scoreboard_if.valid}} & src_valid;

    VX_stream_xbar #(
        .NUM_INPUTS  (NUM_SRC_OPDS),
        .NUM_OUTPUTS (NUM_BANKS),
        .DATAW       (REG_REM_BITS),
        .ARBITER     ("P"), // use priority arbiter
        .OUT_BUF     (0)    // no output buffering
    ) req_xbar (
        .clk       (clk),
        .reset     (reset),
        `UNUSED_PIN(collisions),
        .valid_in  (req_valid_in),
        .data_in   (req_addr_in),
        .sel_in    (req_bank_idx),
        .ready_in  (req_ready_in),
        .valid_out (gpr_rd_valid),
        .data_out  (gpr_rd_reg),
        .sel_out   (gpr_rd_opd),
        .ready_out (gpr_rd_ready)
    );

    assign gpr_rd_ready = {NUM_BANKS{pipe_ready_in}};

    always @(*) begin
        has_collision = 0;
        for (integer i = 0; i < NUM_SRC_OPDS; ++i) begin
            for (integer j = 1; j < (NUM_SRC_OPDS-i); ++j) begin
                has_collision |= src_valid[i]
                              && src_valid[j+i]
                              && (req_bank_idx[i] == req_bank_idx[j+i]);
            end
        end
    end

`ifdef VX_CFG_H100_FE_ENABLE
    // One UR or one UR quad per cycle, overlapped with the GPR reads; counted once per instruction.
    wire [UR_SLOTS-1:0] ur_slots = scoreboard_if.data.ur_rs | UR_SLOTS'(scoreboard_if.data.ur_quad);
    wire [1:0] ur_reads = 2'(ur_slots[0]) + 2'(ur_slots[1]) + 2'(ur_slots[2]);
    reg [1:0] ur_cycles;
    wire ur_done = ~simd_sop || (ur_reads <= 2'd1) || (ur_cycles >= (ur_reads - 2'd1));
    wire opd_more = has_collision || ~ur_done;
    wire opd_last_fetch = pipe_ready_in && ~opd_more;
    always @(posedge clk) begin
        if (reset || opd_last_fetch) begin
            ur_cycles <= '0;
        end else if (scoreboard_if.valid && pipe_ready_in) begin
            ur_cycles <= ur_cycles + 2'd1;
        end
    end
`else
    wire opd_last_fetch = pipe_ready_in && ~has_collision;
`endif

    // A packet narrower than 4 lanes splits a group of 4 threads, which
    // WGATHER and DXA combine: their packets are all issued, active or not.
    localparam SIMD_ALL_PKTS = (SIMD_COUNT != 1) && (`VX_CFG_SIMD_WIDTH < 4);
    localparam SIMD_KEYW     = `VX_CFG_SIMD_WIDTH + SIMD_ALL_PKTS;

    wire [SIMD_COUNT-1:0][SIMD_KEYW-1:0] simd_keys;
    wire [SIMD_KEYW-1:0] simd_key_out;

    if (SIMD_ALL_PKTS) begin : g_simd_all_pkts
        wire is_wgather = (scoreboard_if.data.ex_type == EX_BITS'(EX_ALU))
                       && (scoreboard_if.data.op_args.alu.xtype == ALU_TYPE_OTHER)
                       && scoreboard_if.data.op_type[3];
    `ifdef VX_CFG_EXT_DXA_ENABLE
        wire is_dxa = (scoreboard_if.data.ex_type == EX_BITS'(EX_SFU))
                   && (scoreboard_if.data.op_type == INST_OP_BITS'(INST_SFU_DXA));
    `else
        wire is_dxa = 1'b0;
    `endif
        wire all_pkts = is_wgather || is_dxa;
        for (genvar i = 0; i < SIMD_COUNT; ++i) begin : g_keys
            assign simd_keys[i] = {scoreboard_if.data.tmask[i * `VX_CFG_SIMD_WIDTH +: `VX_CFG_SIMD_WIDTH], all_pkts};
        end
        assign simd_out = simd_key_out[SIMD_KEYW-1:1];
        `UNUSED_VAR (simd_key_out[0])
    end else begin : g_simd_nz_pkts
        assign simd_keys = scoreboard_if.data.tmask;
        assign simd_out  = simd_key_out;
    end

    // simd iterator (skip requests with inactive threads)
    VX_nz_iterator #(
        .DATAW (SIMD_KEYW),
        .N     (SIMD_COUNT)
    ) simd_iter (
        .clk     (clk),
        .reset   (reset),
        .valid_in(scoreboard_if.valid),
        .data_in (simd_keys),
        .next    (opd_last_fetch),
        `UNUSED_PIN (valid_out),
        .data_out(simd_key_out),
        .pid     (simd_pid),
        .sop     (simd_sop),
        .eop     (simd_eop)
    );

    assign pipe_mdata = {
        scoreboard_if.data.uuid,
        scoreboard_if.data.wis,
        scoreboard_if.data.cta_id,
        simd_pid,
        simd_out,
        scoreboard_if.data.PC,
        scoreboard_if.data.wb,
        scoreboard_if.data.wr_xregs,
        scoreboard_if.data.ex_type,
        scoreboard_if.data.op_type,
        scoreboard_if.data.op_args,
        scoreboard_if.data.rd,
        scoreboard_if.data.bytesel,
        simd_sop,
        simd_eop
    };

    assign scoreboard_if.ready = opd_last_fetch && simd_eop;

    wire pipe_fire_st1 = pipe_valid_st1 && pipe_ready_st1;
    wire pipe_fire_st2 = pipe_valid_st2 && pipe_ready_st2;

    VX_pipe_buffer #(
        .DATAW  (NUM_BANKS + META_DATAW + 1 + NUM_BANKS * (REG_REM_BITS + REQ_SEL_WIDTH)),
        .RESETW (1)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .valid_in (scoreboard_if.valid),
        .ready_in (pipe_ready_in),
    `ifdef VX_CFG_H100_FE_ENABLE
        .data_in  ({gpr_rd_valid,     pipe_mdata,     opd_more,          gpr_rd_reg,     gpr_rd_opd}),
    `else
        .data_in  ({gpr_rd_valid,     pipe_mdata,     has_collision,     gpr_rd_reg,     gpr_rd_opd}),
    `endif
        .data_out ({gpr_rd_valid_st1, pipe_mdata_st1, has_collision_st1, gpr_rd_reg_st1, gpr_rd_opd_st1}),
        .valid_out(pipe_valid_st1),
        .ready_out(pipe_ready_st1)
    );

    wire [NUM_SRC_OPDS-1:0] req_fire_in = req_valid_in & req_ready_in;

    always @(posedge clk) begin
        if (reset || opd_last_fetch) begin
            opd_fetched_st1 <= '0;
        end else begin
            opd_fetched_st1 <= opd_fetched_st1 | req_fire_in;
        end
    end

    wire pipe_valid2_st1 = pipe_valid_st1 && ~has_collision_st1;

    VX_pipe_buffer #(
        .DATAW  (NUM_BANKS * (1 + REQ_SEL_WIDTH) + META_DATAW),
        .RESETW (1)
    ) pipe_reg2 (
        .clk      (clk),
        .reset    (reset),
        .valid_in (pipe_valid2_st1),
        .ready_in (pipe_ready_st1),
        .data_in  ({gpr_rd_valid_st1, gpr_rd_opd_st1, pipe_mdata_st1}),
        .data_out ({gpr_rd_valid_st2, gpr_rd_opd_st2, pipe_mdata_st2}),
        .valid_out(pipe_valid_st2),
        .ready_out(pipe_ready_st2)
    );

    always @(*) begin
        opd_buffer_n_st2 = opd_buffer_st2;
        for (integer b = 0; b < NUM_BANKS; ++b) begin
            if (gpr_rd_valid_st2[b]) begin
                opd_buffer_n_st2[gpr_rd_opd_st2[b]] = gpr_rd_data_st2[b];
            end
        end
    end

    always @(posedge clk) begin
        if (reset || pipe_fire_st2) begin
            opd_buffer_st2 <= '0;
        end else begin
            opd_buffer_st2 <= opd_buffer_n_st2;
        end
    end

    wire [BANK_ADDR_WIDTH-1:0] gpr_wr_addr;
    if (SIMD_COUNT != 1) begin : g_gpr_wr_addr_sid
        wire [PER_OPC_NW_BITS + REG_REM_BITS-1:0] tmp;
        `CONCAT(tmp, writeback_if.data.wis[ISSUE_WIS_W-1 -: PER_OPC_NW_W],
              writeback_if.data.rd[NUM_REGS_BITS-1 -: REG_REM_BITS], PER_OPC_NW_BITS, REG_REM_BITS)
        assign gpr_wr_addr = {writeback_if.data.sid, tmp};
    end else begin : g_gpr_wr_addr
        `CONCAT(gpr_wr_addr, writeback_if.data.wis[ISSUE_WIS_W-1 -: PER_OPC_NW_W],
              writeback_if.data.rd[NUM_REGS_BITS-1 -: REG_REM_BITS], PER_OPC_NW_BITS, REG_REM_BITS)
    end

    wire [BANK_SEL_WIDTH-1:0] gpr_wr_bank_idx;
    if (NUM_BANKS != 1) begin : g_gpr_wr_bank_idx_bn
        assign gpr_wr_bank_idx = writeback_if.data.rd[BANK_SEL_BITS-1:0];
    end else begin : g_gpr_wr_bank_idx_b1
        assign gpr_wr_bank_idx = '0;
    end

    wire [BANK_DATA_SIZE-1:0] gpr_wr_byteen;
    for (genvar i = 0; i < `VX_CFG_SIMD_WIDTH; ++i) begin : g_gpr_wr_byteen
        assign gpr_wr_byteen[i*XLENB+:XLENB] = writeback_if.data.byteen[i];
    end

    // GPR banks
    for (genvar b = 0; b < NUM_BANKS; ++b) begin : g_gpr_rams
        wire gpr_wr_enabled;
        if (BANK_SEL_BITS != 0) begin : g_gpr_wr_enabled_bn
        `ifdef VX_CFG_H100_FE_ENABLE
            assign gpr_wr_enabled = writeback_if.valid && writeback_if.data.wb && (gpr_wr_bank_idx == BANK_SEL_BITS'(b));
        `else
            assign gpr_wr_enabled = writeback_if.valid && (gpr_wr_bank_idx == BANK_SEL_BITS'(b));
        `endif
        end else begin : g_gpr_wr_enabled_b1
        `ifdef VX_CFG_H100_FE_ENABLE
            assign gpr_wr_enabled = writeback_if.valid && writeback_if.data.wb;
        `else
            assign gpr_wr_enabled = writeback_if.valid;
        `endif
        end

        wire [BANK_ADDR_WIDTH-1:0] gpr_rd_addr;
        if (SIMD_COUNT != 1) begin : g_gpr_rd_addr_sid
            wire [PER_OPC_NW_BITS + REG_REM_BITS-1:0] tmp;
            `CONCAT(tmp, pipe_mdata_st1[META_DATAW-UUID_WIDTH-1 -: PER_OPC_NW_W],
                gpr_rd_reg_st1[b], PER_OPC_NW_BITS, REG_REM_BITS)
            // pipe_mdata packs {uuid, wis, cta_id, simd_pid, ...}; skip uuid+wis+cta_id
            // to extract simd_pid, which must match writeback_if.data.sid on the write path.
            assign gpr_rd_addr = {pipe_mdata_st1[META_DATAW-UUID_WIDTH-ISSUE_WIS_W-NCTA_WIDTH-1 -: SIMD_IDX_W], tmp};
        end else begin : g_gpr_rd_addr
            `CONCAT(gpr_rd_addr, pipe_mdata_st1[META_DATAW-UUID_WIDTH-1 -: PER_OPC_NW_W],
                gpr_rd_reg_st1[b], PER_OPC_NW_BITS, REG_REM_BITS)
        end

        VX_dp_ram #(
            .DATAW (BANK_DATA_WIDTH),
            .SIZE  (BANK_SIZE),
            .WRENW (BANK_DATA_SIZE),
         `ifdef GPR_RESET
            .RESET_RAM (1),
         `endif
            .OUT_REG (1),
            .RDW_MODE ("R")
        ) gpr_ram (
            .clk   (clk),
            .reset (reset),
            .read  (pipe_fire_st1 && gpr_rd_valid_st1[b]),
            .wren  (gpr_wr_byteen),
            .write (gpr_wr_enabled),
            .waddr (gpr_wr_addr),
            .wdata (writeback_if.data.data),
            .raddr (gpr_rd_addr),
            .rdata (gpr_rd_data_st2[b])
        );
    end

`ifdef VX_CFG_H100_FE_ENABLE
    localparam UR_META_W = UR_SLOTS + UR_QUAD_SLOTS + UR_SLOTS * UREG_BITS;
    wire [UR_META_W-1:0] ur_meta, ur_meta_st1, ur_meta_st2;
    assign ur_meta = {scoreboard_if.data.ur_rs, scoreboard_if.data.ur_quad,
                      scoreboard_if.data.rs3[UREG_BITS-1:0], scoreboard_if.data.rs2[UREG_BITS-1:0], scoreboard_if.data.rs1[UREG_BITS-1:0]};

    // Same control as pipe_reg1/pipe_reg2, so the UR fields stay with their packet.
    VX_pipe_buffer #(
        .DATAW (UR_META_W)
    ) ur_pipe1 (
        .clk      (clk),
        .reset    (reset),
        .valid_in (scoreboard_if.valid),
        `UNUSED_PIN (ready_in),
        .data_in  (ur_meta),
        .data_out (ur_meta_st1),
        `UNUSED_PIN (valid_out),
        .ready_out(pipe_ready_st1)
    );
    VX_pipe_buffer #(
        .DATAW (UR_META_W)
    ) ur_pipe2 (
        .clk      (clk),
        .reset    (reset),
        .valid_in (pipe_valid2_st1),
        `UNUSED_PIN (ready_in),
        .data_in  (ur_meta_st1),
        .data_out (ur_meta_st2),
        `UNUSED_PIN (valid_out),
        .ready_out(pipe_ready_st2)
    );

    wire [UR_SLOTS-1:0] ur_rs_st2;
    wire [UR_QUAD_SLOTS-1:0] ur_quad_st2;
    wire [UR_SLOTS-1:0][UREG_BITS-1:0] ur_idx_st2;
    assign {ur_rs_st2, ur_quad_st2, ur_idx_st2} = ur_meta_st2;

    wire [PER_OPC_NW_W-1:0] ur_rd_slot, ur_wr_slot;
    if (PER_OPC_WARPS != 1) begin : g_ur_slot
        assign ur_rd_slot = pipe_mdata_st2[META_DATAW-UUID_WIDTH-1 -: PER_OPC_NW_W];
        assign ur_wr_slot = writeback_if.data.wis[ISSUE_WIS_W-1 -: PER_OPC_NW_W];
    end else begin : g_ur_slot1
        assign ur_rd_slot = '0;
        assign ur_wr_slot = '0;
    end
    wire [SIMD_IDX_W-1:0] ur_pid_st2 = pipe_mdata_st2[META_DATAW-UUID_WIDTH-ISSUE_WIS_W-NCTA_WIDTH-1 -: SIMD_IDX_W];

    // UR file: the first active lane's value is written; URZ and above read 0.
    reg [PER_OPC_WARPS-1:0][UREG_COUNT-1:0][`VX_CFG_XLEN-1:0] ur_file;
    reg [`VX_CFG_XLEN-1:0] ur_wr_data;
    reg ur_wr_any;
    always @(*) begin
        ur_wr_data = '0;
        ur_wr_any  = 1'b0;
        for (integer l = `VX_CFG_SIMD_WIDTH-1; l >= 0; --l) begin
            if (writeback_if.data.tmask[l]) begin
                ur_wr_data = writeback_if.data.data[l];
                ur_wr_any  = 1'b1;
            end
        end
    end
    wire ur_wr_en = writeback_if.valid && writeback_if.data.wr_xregs[XREG_UR] && ur_wr_any;
    always @(posedge clk) begin
        if (reset) begin
            for (integer w = 0; w < PER_OPC_WARPS; ++w) begin
                for (integer r = 0; r < UREG_COUNT; ++r) begin
                    ur_file[w][r] <= '0;
                end
            end
        end else if (ur_wr_en) begin
            ur_file[ur_wr_slot][writeback_if.data.rd[UREG_BITS-1:0]] <= ur_wr_data;
        end
    end

    function automatic logic [`VX_CFG_XLEN-1:0] ur_read(input logic [UREG_COUNT-1:0][`VX_CFG_XLEN-1:0] urf, input logic [UREG_BITS:0] idx);
        return (idx >= (UREG_BITS+1)'(UREG_ZERO)) ? '0 : urf[idx[UREG_BITS-1:0]];
    endfunction

    // A UR source is broadcast; a quad source fills the warp's lanes 0..3, other lanes 0.
    reg [NUM_SRC_OPDS-1:0][(`VX_CFG_SIMD_WIDTH * `VX_CFG_XLEN)-1:0] opd_data_st2;
    always @(*) begin
        opd_data_st2 = opd_buffer_n_st2;
        for (integer i = 0; i < NUM_SRC_OPDS; ++i) begin
            for (integer l = 0; l < `VX_CFG_SIMD_WIDTH; ++l) begin
                if (ur_rs_st2[i]) begin
                    opd_data_st2[i][l * `VX_CFG_XLEN +: `VX_CFG_XLEN] = ur_read(ur_file[ur_rd_slot], {1'b0, ur_idx_st2[i]});
                end else if (i < UR_QUAD_SLOTS && ur_quad_st2[i % UR_QUAD_SLOTS]) begin
                    opd_data_st2[i][l * `VX_CFG_XLEN +: `VX_CFG_XLEN] = ((32'(ur_pid_st2) * `VX_CFG_SIMD_WIDTH + l) < 4)
                        ? ur_read(ur_file[ur_rd_slot], {1'b0, ur_idx_st2[i]} + (UREG_BITS+1)'(32'(ur_pid_st2) * `VX_CFG_SIMD_WIDTH + l))
                        : '0;
                end
            end
        end
    end
`endif

    // output buffer
    VX_elastic_buffer #(
        .DATAW   (OUT_DATAW),
        .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
        .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF))
    ) out_buf (
        .clk      (clk),
        .reset    (reset),
        .valid_in (pipe_valid_st2),
        .ready_in (pipe_ready_st2),
        .data_in  ({pipe_mdata_st2[META_DATAW-1:2], // remove sop/eop
                `ifdef VX_CFG_H100_FE_ENABLE
                    opd_data_st2,
                `else
                    opd_buffer_n_st2, // operand data
                `endif
                    pipe_mdata_st2[1:0]}), // sop/eop
        .data_out ({
            operands_if.data.uuid,
            operands_if.data.wis,
            operands_if.data.cta_id,
            operands_if.data.sid,
            operands_if.data.tmask,
            operands_if.data.PC,
            operands_if.data.wb,
            operands_if.data.wr_xregs,
            operands_if.data.ex_type,
            operands_if.data.op_type,
            operands_if.data.op_args,
            operands_if.data.rd,
            operands_if.data.bytesel,
            operands_if.data.rs3_data,
            operands_if.data.rs2_data,
            operands_if.data.rs1_data,
            operands_if.data.sop,
            operands_if.data.eop
        }),
        .valid_out(operands_if.valid),
        .ready_out(operands_if.ready)
    );

`ifdef PERF_ENABLE
    reg [PERF_CTR_BITS-1:0] collisions_r;
    always @(posedge clk) begin
        if (reset) begin
            collisions_r <= '0;
        end else begin
            collisions_r <= collisions_r + PERF_CTR_BITS'(scoreboard_if.valid && pipe_ready_in && has_collision);
        end
    end
    assign perf_stalls = collisions_r;
`endif

endmodule
