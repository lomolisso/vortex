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

`ifdef VX_CFG_H100_FE_ENABLE

`include "VX_define.vh"

// ELECT, MATCH and REDUX over the whole warp for NUM_PORTS ALU blocks, picked round robin. The
// operand stage holds one op and owns one row of NUM_LANES comparators: MATCH.ALL compares every
// lane with the first active one in one pass; MATCH.ANY compares with the lowest lane not yet
// matched, one pass per distinct value, and leaves from there. ELECT, REDUX and MATCH.ALL then
// cross a reduction tree with one register stage per level. Cycle model: warp_ops_ref.h Pipe.
module VX_alu_warp import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES = 1,
    parameter NUM_PORTS = 1
) (
    input wire              clk,
    input wire              reset,

    VX_execute_if.slave     execute_if [NUM_PORTS],
    VX_result_if.master     result_if [NUM_PORTS]
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `STATIC_ASSERT (NUM_LANES == `VX_CFG_NUM_THREADS, ("invalid parameter"))
    `STATIC_ASSERT (NUM_LANES <= `VX_CFG_XLEN, ("invalid parameter"))
    `STATIC_ASSERT (NUM_LANES >= 2 && `IS_POW2(NUM_LANES), ("invalid parameter"))

    localparam XLEN      = `VX_CFG_XLEN;
    localparam LANE_BITS = `LOG2UP(NUM_LANES);
    localparam DEPTH     = `CLOG2(NUM_LANES);
    localparam PORT_BITS = `LOG2UP(NUM_PORTS);
    localparam EXE_DATAW = $bits(alu_execute_t);
    // A tree node: {valid, 32-bit value}.
    localparam NODEW     = 33;

    // Input port arbitration.

    wire [NUM_PORTS-1:0] in_valid, in_ready;
    wire [NUM_PORTS-1:0][EXE_DATAW-1:0] in_data;
    for (genvar p = 0; p < NUM_PORTS; ++p) begin : g_in
        assign in_valid[p] = execute_if[p].valid;
        assign in_data[p]  = execute_if[p].data;
        assign execute_if[p].ready = in_ready[p];
    end

    wire arb_valid, arb_ready;
    alu_execute_t arb_data;
    wire [PORT_BITS-1:0] arb_sel;

    VX_stream_arb #(
        .NUM_INPUTS  (NUM_PORTS),
        .NUM_OUTPUTS (1),
        .DATAW       (EXE_DATAW),
        .ARBITER     ("R"),
        .OUT_BUF     (0)
    ) in_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (in_valid),
        .data_in   (in_data),
        .ready_in  (in_ready),
        .valid_out (arb_valid),
        .data_out  (arb_data),
        .ready_out (arb_ready),
        .sel_out   (arb_sel)
    );

    `UNUSED_VAR (arb_data.op_type)
    `UNUSED_VAR (arb_data.rs3_data)

    wire [INST_WOP_BITS-1:0] arb_code = alu_warp_code(arb_data.op_args);

    // Lane words: rs1, and at XLEN 32 rs2 above it for b64 keys.
    logic [NUM_LANES-1:0][63:0] arb_word;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_word
        wire [XLEN-1:0] a = arb_data.rs1_data[i];
        wire [XLEN-1:0] b = arb_data.rs2_data[i];
        if (XLEN == 64) begin : g_x64
            `UNUSED_VAR (b)
            assign arb_word[i] = 64'(a);
        end else begin : g_x32
            assign arb_word[i] = {b[31:0], a[31:0]};
        end
    end

    // Operand stage.

    reg                                 s0_valid;
    alu_header_t                        s0_hdr;
    reg [INST_WOP_BITS-1:0]             s0_code;
    reg [PORT_BITS-1:0]                 s0_port;
    reg [NUM_LANES-1:0][63:0]           s0_word;
    reg [NUM_LANES-1:0]                 s0_left;   // MATCH.ANY: active lanes not matched yet
    reg [NUM_LANES-1:0][NUM_LANES-1:0]  s0_group;  // MATCH.ANY: each matched lane's group

    wire s0_redux = inst_wop_is_redux(s0_code);
    wire s0_elect = inst_wop_is_elect(s0_code);
    wire s0_match = inst_wop_is_match(s0_code);
    wire s0_any   = s0_match & ~s0_code[1];
    wire s0_b64   = s0_match & s0_code[0];
    wire [NUM_LANES-1:0] s0_active = s0_hdr.tmask;

    logic [NUM_LANES-1:0][63:0] key;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_key
        assign key[i] = s0_b64 ? s0_word[i] : {32'b0, s0_word[i][31:0]};
    end

    wire [NUM_LANES-1:0] probe_set = s0_any ? s0_left : s0_active;
    wire [LANE_BITS-1:0] probe;
    VX_priority_encoder #(
        .N (NUM_LANES)
    ) probe_enc (
        .data_in   (probe_set),
        .index_out (probe),
        `UNUSED_PIN (onehot_out),
        `UNUSED_PIN (valid_out)
    );

    wire [63:0] probe_key = key[probe];
    wire [NUM_LANES-1:0] eq;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_cmp
        assign eq[i] = (key[i] == probe_key);
    end
    wire [NUM_LANES-1:0] group = eq & probe_set;
    wire any_last = s0_valid && s0_any && ((s0_left & ~group) == '0);

    // Reduction tree.

    function automatic logic [31:0] redux_step(input logic [2:0] op, input logic [31:0] x, input logic [31:0] y);
        case (op)
        3'd0:    return x + y;
        3'd1:    return ($signed(x) < $signed(y)) ? x : y;
        3'd2:    return ($signed(x) > $signed(y)) ? x : y;
        3'd3:    return (x < y) ? x : y;
        3'd4:    return (x > y) ? x : y;
        3'd5:    return x & y;
        3'd6:    return x | y;
        default: return x ^ y;
        endcase
    endfunction

    // ELECT keeps the lower lane, MATCH.ALL ANDs the equality bits.
    function automatic logic [NODEW-1:0] combine(input logic [INST_WOP_BITS-1:0] op, input logic [NODEW-1:0] lo, input logic [NODEW-1:0] hi);
        if (lo[32] && hi[32]) begin
            if (inst_wop_is_redux(op)) begin
                return {1'b1, redux_step(op[2:0], lo[31:0], hi[31:0])};
            end else if (inst_wop_is_elect(op)) begin
                return lo;
            end else begin
                return {1'b1, lo[31:0] & hi[31:0]};
            end
        end
        return lo[32] ? lo : hi;
    endfunction

    logic [NUM_LANES-1:0][NODEW-1:0] leaf;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_leaf
        always @(*) begin
            if (s0_redux) begin
                leaf[i] = {s0_active[i], s0_word[i][31:0]};
            end else if (s0_elect) begin
                leaf[i] = {s0_active[i] & s0_word[i][i], 32'(i)};
            end else begin
                leaf[i] = {s0_active[i], 31'b0, eq[i]};
            end
        end
    end

    reg   [DEPTH-1:0]                    lvl_valid;
    alu_header_t                         lvl_hdr  [DEPTH];
    reg   [INST_WOP_BITS-1:0]            lvl_code [DEPTH];
    reg   [PORT_BITS-1:0]                lvl_port [DEPTH];
    reg   [NUM_LANES-1:0][NODEW-1:0]     lvl_q    [DEPTH];
    // Each level's register inputs.
    wire  [DEPTH-1:0]                    nxt_valid;
    alu_header_t                         nxt_hdr  [DEPTH];
    wire  [INST_WOP_BITS-1:0]            nxt_code [DEPTH];
    wire  [PORT_BITS-1:0]                nxt_port [DEPTH];
    wire  [NUM_LANES-1:0][NODEW-1:0]     nxt_node [DEPTH];

    // Output: the tree's result first, else MATCH.ANY's on its last pass.

    wire tree_out = lvl_valid[DEPTH-1];
    wire [NUM_PORTS-1:0] port_ready;
    wire [PORT_BITS-1:0] out_port = tree_out ? lvl_port[DEPTH-1] : s0_port;
    wire out_ready = | (port_ready & (NUM_PORTS'(1) << out_port));
    wire tree_en   = ~tree_out || out_ready;
    wire any_fire  = any_last && ~tree_out && out_ready;
    wire s0_leave  = s0_valid && (s0_any ? any_fire : tree_en);
    assign arb_ready = ~s0_valid || s0_leave;
    wire accept = arb_valid && arb_ready;

    for (genvar l = 0; l < DEPTH; ++l) begin : g_tree
        localparam NN = NUM_LANES >> (l + 1);
        wire [NUM_LANES-1:0][NODEW-1:0] src;
        if (l == 0) begin : g_first
            assign src         = leaf;
            assign nxt_code[l] = s0_code;
            assign nxt_hdr[l]  = s0_hdr;
            assign nxt_port[l] = s0_port;
            assign nxt_valid[l] = s0_valid && ~s0_any;
        end else begin : g_next
            assign src         = lvl_q[l-1];
            assign nxt_code[l] = lvl_code[l-1];
            assign nxt_hdr[l]  = lvl_hdr[l-1];
            assign nxt_port[l] = lvl_port[l-1];
            assign nxt_valid[l] = lvl_valid[l-1];
        end
        `UNUSED_VAR (src)
        for (genvar j = 0; j < NUM_LANES; ++j) begin : g_node
            if (j < NN) begin : g_used
                assign nxt_node[l][j] = combine(nxt_code[l], src[2*j], src[2*j+1]);
            end else begin : g_unused
                assign nxt_node[l][j] = '0;
            end
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            lvl_valid <= '0;
        end else if (tree_en) begin
            lvl_valid <= nxt_valid;
        end
        if (tree_en) begin
            for (integer l = 0; l < DEPTH; ++l) begin
                lvl_q[l]    <= nxt_node[l];
                lvl_code[l] <= nxt_code[l];
                lvl_hdr[l]  <= nxt_hdr[l];
                lvl_port[l] <= nxt_port[l];
            end
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            s0_valid <= 1'b0;
        end else if (accept) begin
            s0_valid <= 1'b1;
        end else if (s0_leave) begin
            s0_valid <= 1'b0;
        end
        if (accept) begin
            s0_hdr   <= arb_data.header;
            s0_code  <= arb_code;
            s0_port  <= arb_sel;
            s0_word  <= arb_word;
            s0_left  <= arb_data.header.tmask;
            s0_group <= '0;
        end else if (s0_valid && s0_any && ~any_last) begin
            s0_left <= s0_left & ~group;
            for (integer i = 0; i < NUM_LANES; ++i) begin
                if (group[i]) begin
                    s0_group[i] <= group;
                end
            end
        end
    end

    wire [NODEW-1:0] root = lvl_q[DEPTH-1][0];
    alu_header_t root_hdr;
    assign root_hdr = lvl_hdr[DEPTH-1];
    wire [INST_WOP_BITS-1:0] root_code = lvl_code[DEPTH-1];
    wire root_redux = inst_wop_is_redux(root_code);
    wire root_elect = inst_wop_is_elect(root_code);

    logic [NUM_LANES-1:0][XLEN-1:0] out_data;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_out
        always @(*) begin
            if (~tree_out) begin
                out_data[i] = XLEN'(group[i] ? group : s0_group[i]);
            end else if (root_redux) begin
                out_data[i] = XLEN'($signed(root[31:0]));
            end else if (root_elect) begin
                if (root_code[0]) begin
                    out_data[i] = root[32] ? XLEN'(root[31:0]) : '1;
                end else begin
                    out_data[i] = XLEN'(root[32] && (root[LANE_BITS-1:0] == LANE_BITS'(i)));
                end
            end else begin
                out_data[i] = root[0] ? XLEN'(root_hdr.tmask) : '0;
            end
        end
    end

    alu_header_t out_hdr;
    assign out_hdr = tree_out ? root_hdr : s0_hdr;
    wire out_valid = tree_out || any_last;

    for (genvar p = 0; p < NUM_PORTS; ++p) begin : g_out_port
        assign result_if[p].valid       = out_valid && (out_port == PORT_BITS'(p));
        assign result_if[p].data.header = out_hdr;
        assign result_if[p].data.data   = out_data;
        assign port_ready[p]            = result_if[p].ready;
    end

endmodule

`endif
