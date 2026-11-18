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

`ifdef DXA_TMAP_L2_ENABLE

// Level 2 of the tensor-map cache, one per cluster (the H100's GPC-level
// constant cache), shared by the cluster's sockets: 128-B lines in L1-line
// sectors. The sockets' DXA cores read level 3 on its misses and fill it; the
// one on port 0 loads the launch's parameter window into it. One operation at
// a time; a port's lookup waits parked, so fills never queue behind it, and
// no lookup runs while the window loads.
module VX_tmap_l2 import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_PORTS = 1
) (
    input wire clk,
    input wire reset,
    VX_tmap_l2_bus_if.slave bus_if [NUM_PORTS]
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam AW      = `VX_CFG_MEM_ADDR_WIDTH;
    localparam LINE    = 128;
    localparam LG_LINE = 7;
    localparam SECTOR  = `VX_CFG_L1_LINE_SIZE;
    localparam LG_SEC  = `CLOG2(SECTOR);
    localparam SECTORS = LINE / SECTOR;
    localparam SEC_B   = `CLOG2(SECTORS);
    localparam WAYS    = `VX_CFG_TMAP_L2_WAYS;
    localparam LG_WAYS = `CLOG2(WAYS);
    localparam SETS    = `VX_CFG_TMAP_L2_ENTRIES / WAYS;
    localparam SET_B   = `CLOG2(SETS);
    localparam SET_W   = `UP(SET_B);
    localparam TAG_W   = AW - LG_LINE - SET_B;
    localparam NLINES  = SETS * WAYS;
    localparam DATA_B  = SET_B + LG_WAYS + SEC_B;
    localparam PEND_W  = `CLOG2(`VX_CFG_TMAP_PARAM_BYTES / SECTOR + 2);
    localparam PORT_W  = `UP(`CLOG2(NUM_PORTS));
    localparam BEAT_W  = 2;

    `STATIC_ASSERT ((SECTOR <= LINE) && (SECTOR >= 64), ("the tensor-map level 2 needs L1 lines of 64 or 128 bytes"))
    `STATIC_ASSERT (WAYS >= 2 && (WAYS & (WAYS - 1)) == 0, ("invalid parameter"))

    localparam S_IDLE = 3'd0, S_TAG = 3'd1, S_CMP = 3'd2, S_DATA = 3'd3, S_RSP = 3'd4;
    reg [2:0] state_r;

    reg [1:0]                 cur_op_r;
    reg [PORT_W-1:0]          cur_port_r;
    reg [AW-1:0]              cur_addr_r;   // the sector (LOOKUP, FILL) or the line (INVAL) at hand
    reg [AW-1:0]              cur_end_r;    // INVAL: the map's last line
    reg                       cur_param_r;
    reg [SECTOR*8-1:0]        cur_data_r;
    reg [BEAT_W-1:0]          beat_r, nbeats_r;
    reg [PEND_W-1:0]          pending_r;

    reg                       rsp_valid_r;
    tmap_l2_rsp_t             rsp_r;

    // ---- Ports: lookups park, other operations are arbitrated ----
    reg  [NUM_PORTS-1:0]          park_valid_r;
    reg  [NUM_PORTS-1:0][AW-1:0]  park_addr_r;
    wire [NUM_PORTS-1:0]          op_req, req_valid_p, req_ready_p;
    tmap_l2_req_t [NUM_PORTS-1:0] req_data;

    for (genvar p = 0; p < NUM_PORTS; ++p) begin : g_ports
        assign req_data[p]    = bus_if[p].req_data;
        assign req_valid_p[p] = bus_if[p].req_valid;
        assign req_ready_p[p] = bus_if[p].req_ready;
        assign op_req[p]   = bus_if[p].req_valid && (bus_if[p].req_data.op != TMAP_L2_LOOKUP);
        assign bus_if[p].rsp_valid = rsp_valid_r && (cur_port_r == PORT_W'(p));
        assign bus_if[p].rsp_data  = rsp_r;
    end

    wire [PORT_W-1:0] op_idx, lk_idx;
    wire              op_valid, lk_valid;
    wire              idle   = (state_r == S_IDLE);
    wire              op_go  = idle && op_valid;
    wire              lk_go  = idle && ~op_valid && lk_valid && (pending_r == '0);

    VX_rr_arbiter #(
        .NUM_REQS (NUM_PORTS)
    ) op_arb (
        .clk          (clk),
        .reset        (reset),
        .requests     (op_req),
        .grant_index  (op_idx),
        `UNUSED_PIN   (grant_onehot),
        .grant_valid  (op_valid),
        .grant_ready  (op_go)
    );

    VX_rr_arbiter #(
        .NUM_REQS (NUM_PORTS)
    ) lk_arb (
        .clk          (clk),
        .reset        (reset),
        .requests     (park_valid_r),
        .grant_index  (lk_idx),
        `UNUSED_PIN   (grant_onehot),
        .grant_valid  (lk_valid),
        .grant_ready  (lk_go)
    );

    for (genvar p = 0; p < NUM_PORTS; ++p) begin : g_ready
        assign bus_if[p].req_ready = (bus_if[p].req_data.op == TMAP_L2_LOOKUP) ? ~park_valid_r[p]
                                                                               : (op_go && (op_idx == PORT_W'(p)));
    end

    tmap_l2_req_t op_data;
    assign op_data = req_data[op_idx];

    // ---- Address fields of the operation at hand ----
    wire [AW-LG_LINE-1:0] cur_line = cur_addr_r[AW-1:LG_LINE];
    wire [SET_W-1:0]      cur_set;
    wire [TAG_W-1:0]      cur_tag = cur_line[AW-LG_LINE-1:SET_B];
    wire [SEC_B > 0 ? SEC_B-1 : 0:0] cur_sec;
    if (SET_B != 0) begin : g_set
        assign cur_set = cur_line[SET_B-1:0];
    end else begin : g_set0
        assign cur_set = '0;
    end
    if (SEC_B != 0) begin : g_sec
        assign cur_sec = cur_addr_r[LG_SEC +: SEC_B];
    end else begin : g_sec0
        assign cur_sec = '0;
    end

    // ---- State per line (flops: a flush empties them in one cycle) ----
    reg [NLINES-1:0]               valid_r;
    reg [NLINES-1:0][SECTORS-1:0]  sectors_r;
    reg [SETS-1:0][WAYS-1:0]       plru_r;   // tree nodes 1..WAYS-1: the half that holds the victim
    reg [SETS-1:0][LG_WAYS-1:0]    fifo_r;

    wire [WAYS*TAG_W-1:0] tag_rdata;
    wire                  tag_write;
    wire [WAYS-1:0]       tag_wren;

    VX_dp_ram #(
        .DATAW   (WAYS * TAG_W),
        .SIZE    (SETS),
        .WRENW   (WAYS),
        .OUT_REG (1),
        .RDW_MODE("W")
    ) tag_ram (
        .clk   (clk),
        .reset (reset),
        .read  (state_r == S_TAG),
        .write (tag_write),
        .wren  (tag_wren),
        .waddr (cur_set),
        .wdata ({WAYS{cur_tag}}),
        .raddr (cur_set),
        .rdata (tag_rdata)
    );

    wire [WAYS-1:0] way_match;
    for (genvar w = 0; w < WAYS; ++w) begin : g_match
        wire [`LOG2UP(NLINES)-1:0] li = `LOG2UP(NLINES)'(cur_set) * `LOG2UP(NLINES)'(WAYS) + `LOG2UP(NLINES)'(w);
        assign way_match[w] = valid_r[li] && (tag_rdata[w * TAG_W +: TAG_W] == cur_tag);
    end
    wire                 hit_any = (| way_match);
    wire [LG_WAYS-1:0]   hit_way, free_way;
    wire                 has_free;
    wire [WAYS-1:0]      set_valid;
    for (genvar w = 0; w < WAYS; ++w) begin : g_sv
        assign set_valid[w] = valid_r[`LOG2UP(NLINES)'(cur_set) * `LOG2UP(NLINES)'(WAYS) + `LOG2UP(NLINES)'(w)];
    end

    VX_priority_encoder #(
        .N (WAYS)
    ) hit_enc (
        .data_in   (way_match),
        .index_out (hit_way),
        `UNUSED_PIN (onehot_out),
        `UNUSED_PIN (valid_out)
    );

    VX_priority_encoder #(
        .N (WAYS)
    ) free_enc (
        .data_in   (~set_valid),
        .index_out (free_way),
        `UNUSED_PIN (onehot_out),
        .valid_out (has_free)
    );

    reg [LG_WAYS-1:0] plru_way;
    always @(*) begin
        integer node;
        node = 1;
        plru_way = '0;
        for (integer l = 0; l < LG_WAYS; ++l) begin
            if (plru_r[cur_set][LG_WAYS'(node)]) begin
                plru_way = plru_way | LG_WAYS'(WAYS >> (l + 1));
                node = 2 * node + 1;
            end else begin
                node = 2 * node;
            end
        end
    end
    wire [LG_WAYS-1:0] victim = has_free ? free_way
                              : ((`VX_CFG_TMAP_L2_REPL == 0) ? fifo_r[cur_set] : plru_way);

    wire [`LOG2UP(NLINES)-1:0] hit_li = `LOG2UP(NLINES)'(cur_set) * `LOG2UP(NLINES)'(WAYS) + `LOG2UP(NLINES)'(hit_way);
    wire [`LOG2UP(NLINES)-1:0] vic_li = `LOG2UP(NLINES)'(cur_set) * `LOG2UP(NLINES)'(WAYS) + `LOG2UP(NLINES)'(victim);
    wire sec_hit = hit_any && sectors_r[hit_li][cur_sec];

    wire in_cmp   = (state_r == S_CMP);
    wire fill_now = in_cmp && (cur_op_r == TMAP_L2_FILL);
    wire [LG_WAYS-1:0] fill_way = hit_any ? hit_way : victim;
    assign tag_write = fill_now && ~hit_any;
    assign tag_wren  = WAYS'(1) << victim;

    // ---- Data: one L1 line per (set, way, sector) ----
    wire [SECTOR*8-1:0] data_rdata;
    wire [`UP(DATA_B)-1:0] data_waddr, data_raddr;
    if (SEC_B != 0) begin : g_daddr
        assign data_waddr = {cur_set, fill_way, cur_sec};
        assign data_raddr = {cur_set, hit_way, cur_sec};
    end else begin : g_daddr0
        assign data_waddr = `UP(DATA_B)'({cur_set, fill_way});
        assign data_raddr = `UP(DATA_B)'({cur_set, hit_way});
    end

    VX_dp_ram #(
        .DATAW   (SECTOR * 8),
        .SIZE    (NLINES * SECTORS),
        .OUT_REG (1),
        .RDW_MODE("W")
    ) data_ram (
        .clk   (clk),
        .reset (reset),
        .read  (in_cmp && (cur_op_r == TMAP_L2_LOOKUP)),
        .write (fill_now),
        .wren  (1'b1),
        .waddr (data_waddr),
        .wdata (cur_data_r),
        .raddr (data_raddr),
        .rdata (data_rdata)
    );

    // ---- A lookup's beats: the map's L1 lines from the one holding its first byte ----
    wire [AW-1:0] lk_map = park_addr_r[lk_idx];
    wire [AW-1:0] lk_first = {lk_map[AW-1:LG_SEC], LG_SEC'(0)};
    wire [BEAT_W-1:0] lk_nbeats = BEAT_W'((32'(lk_map[LG_SEC-1:0]) + DXA_TMAP_BYTES - 1) / SECTOR + 1);

    wire [PEND_W-1:0] win_lines = PEND_W'((32'(op_data.addr[LG_SEC-1:0]) + `VX_CFG_TMAP_PARAM_BYTES + SECTOR - 1) / SECTOR);

    wire [NUM_PORTS-1:0] rsp_ready_p;
    for (genvar p = 0; p < NUM_PORTS; ++p) begin : g_rsp_ready
        assign rsp_ready_p[p] = bus_if[p].rsp_ready;
    end
    wire bus_if_rsp_ready = rsp_ready_p[cur_port_r];

    always @(posedge clk) begin
        if (reset) begin
            state_r      <= S_IDLE;
            park_valid_r <= '0;
            pending_r    <= '0;
            rsp_valid_r  <= 1'b0;
            valid_r      <= '0;
            sectors_r    <= '0;
            plru_r       <= '0;
            fifo_r       <= '0;
        end else begin
            for (integer p = 0; p < NUM_PORTS; ++p) begin
                if (req_valid_p[p] && req_ready_p[p] && (req_data[p].op == TMAP_L2_LOOKUP)) begin
                    park_valid_r[p] <= 1'b1;
                    park_addr_r[p]  <= req_data[p].addr;
                end
            end
            case (state_r)
            S_IDLE: begin
                if (op_go) begin
                    cur_op_r    <= op_data.op;
                    cur_port_r  <= op_idx;
                    cur_param_r <= op_data.param;
                    cur_data_r  <= op_data.data;
                    beat_r      <= '0;
                    if (op_data.op == TMAP_L2_FLUSH) begin
                        // Every socket sees the launch; port 0 loads the window.
                        if (op_idx == PORT_W'(0)) begin
                            valid_r     <= '0;
                            sectors_r   <= '0;
                            plru_r      <= '0;
                            fifo_r      <= '0;
                            pending_r   <= win_lines;
                            rsp_r.kind  <= TMAP_L2_PRELOAD_K;
                            rsp_r.last  <= 1'b1;
                            rsp_r.addr  <= op_data.addr;
                            rsp_r.data  <= '0;
                            rsp_valid_r <= 1'b1;
                            state_r     <= S_RSP;
                        end
                    end else if (op_data.op == TMAP_L2_INVAL) begin
                        cur_addr_r <= op_data.addr & ~AW'(LINE - 1);
                        cur_end_r  <= (op_data.addr + AW'(DXA_TMAP_BYTES - 1)) & ~AW'(LINE - 1);
                        state_r    <= S_TAG;
                    end else begin
                        cur_addr_r <= op_data.addr;
                        state_r    <= S_TAG;
                    end
                end else if (lk_go) begin
                    park_valid_r[lk_idx] <= 1'b0;
                    cur_op_r   <= TMAP_L2_LOOKUP;
                    cur_port_r <= lk_idx;
                    cur_addr_r <= lk_first;
                    beat_r     <= '0;
                    nbeats_r   <= lk_nbeats;
                    state_r    <= S_TAG;
                end
            end
            S_TAG: state_r <= S_CMP;
            S_CMP: begin
                case (cur_op_r)
                TMAP_L2_LOOKUP: begin
                    if (sec_hit) begin
                        state_r <= S_DATA;
                    end else begin
                        rsp_r.kind  <= TMAP_L2_MISS_K;
                        rsp_r.last  <= 1'b1;
                        rsp_r.addr  <= '0;
                        rsp_r.data  <= '0;
                        rsp_valid_r <= 1'b1;
                        state_r     <= S_RSP;
                    end
                end
                TMAP_L2_FILL: begin
                    if (hit_any) begin
                        sectors_r[hit_li][cur_sec] <= 1'b1;
                    end else begin
                        valid_r[vic_li]   <= 1'b1;
                        sectors_r[vic_li] <= SECTORS'(1) << cur_sec;
                        if (~has_free)
                            fifo_r[cur_set] <= fifo_r[cur_set] + LG_WAYS'(1);
                    end
                    if (cur_param_r && (pending_r != '0))
                        pending_r <= pending_r - PEND_W'(1);
                    state_r <= S_IDLE;
                end
                default: begin // INVAL, line by line
                    if (hit_any)
                        valid_r[hit_li] <= 1'b0;
                    if (cur_addr_r == cur_end_r) begin
                        state_r <= S_IDLE;
                    end else begin
                        cur_addr_r <= cur_addr_r + AW'(LINE);
                        state_r    <= S_TAG;
                    end
                end
                endcase
            end
            S_DATA: begin
                rsp_r.kind  <= TMAP_L2_HIT_K;
                rsp_r.last  <= (beat_r + BEAT_W'(1)) == nbeats_r;
                rsp_r.addr  <= '0;
                rsp_r.data  <= data_rdata;
                rsp_valid_r <= 1'b1;
                state_r     <= S_RSP;
            end
            default: begin // S_RSP
                if (bus_if_rsp_ready) begin
                    rsp_valid_r <= 1'b0;
                    if (rsp_r.last) begin
                        state_r <= S_IDLE;
                    end else begin
                        beat_r     <= beat_r + BEAT_W'(1);
                        cur_addr_r <= cur_addr_r + AW'(SECTOR);
                        state_r    <= S_TAG;
                    end
                end
            end
            endcase

            // Recency: a lookup's hit and a fill touch their way.
            if (in_cmp && ((cur_op_r == TMAP_L2_LOOKUP && sec_hit) || (cur_op_r == TMAP_L2_FILL))) begin
                integer node;
                logic [LG_WAYS-1:0] tw;
                tw = (cur_op_r == TMAP_L2_FILL) ? fill_way : hit_way;
                node = 1;
                for (integer l = 0; l < LG_WAYS; ++l) begin
                    if ((tw & LG_WAYS'(WAYS >> (l + 1))) != '0) begin
                        plru_r[cur_set][LG_WAYS'(node)] <= 1'b0;
                        node = 2 * node + 1;
                    end else begin
                        plru_r[cur_set][LG_WAYS'(node)] <= 1'b1;
                        node = 2 * node;
                    end
                end
            end
        end
    end

    `RUNTIME_ASSERT(~(fill_now && cur_param_r && (pending_r == '0)),
        ("%t: *** %s parameter-window fill without a load", $time, INSTANCE_ID))

endmodule

`endif

`endif
