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

// AXI4 INCR-burst width converter from a wide master to a narrower slave port.
// Each wide beat becomes S_DATA_WIDTH/M_DATA_WIDTH narrow beats; a burst longer
// than 256 narrow beats leaves as several sub-bursts. One burst per direction is
// in flight and its sub-bursts go one after another, so narrow read beats return
// in order and pack without reordering;
// the wide write response is sent only after every sub-burst's B, keeping the
// master's write-then-signal ordering. Equal widths are a pass-through.
module VX_mem_axi_downsize #(
    parameter ADDR_WIDTH   = 64,
    parameter S_DATA_WIDTH = 512,
    parameter M_DATA_WIDTH = 512,
    parameter ID_WIDTH     = 32
) (
    input  wire          clk,
    input  wire          reset,
    VX_mem_axi_if.slave  s,
    VX_mem_axi_if.master m
);
    localparam R = S_DATA_WIDTH / M_DATA_WIDTH;
    `STATIC_ASSERT (R >= 1 && `IS_POW2(R) && (R * M_DATA_WIDTH == S_DATA_WIDTH), ("invalid width ratio"))

    if (R == 1) begin : g_pass
        `UNUSED_VAR ({clk, reset})
        assign m.awvalid = s.awvalid;  assign s.awready = m.awready;
        assign m.awaddr  = s.awaddr;   assign m.awid    = s.awid;
        assign m.awlen   = s.awlen;    assign m.awsize  = s.awsize;
        assign m.awburst = s.awburst;
        assign m.wvalid  = s.wvalid;   assign s.wready  = m.wready;
        assign m.wdata   = s.wdata;    assign m.wstrb   = s.wstrb;
        assign m.wlast   = s.wlast;
        assign s.bvalid  = m.bvalid;   assign m.bready  = s.bready;
        assign s.bid     = m.bid;      assign s.bresp   = m.bresp;
        assign m.arvalid = s.arvalid;  assign s.arready = m.arready;
        assign m.araddr  = s.araddr;   assign m.arid    = s.arid;
        assign m.arlen   = s.arlen;    assign m.arsize  = s.arsize;
        assign m.arburst = s.arburst;
        assign s.rvalid  = m.rvalid;   assign m.rready  = s.rready;
        assign s.rdata   = m.rdata;    assign s.rid     = m.rid;
        assign s.rlast   = m.rlast;    assign s.rresp   = m.rresp;
    end else begin : g_down
        localparam M_BYTES = M_DATA_WIDTH / 8;
        localparam LR      = `CLOG2(R);
        localparam CNT_W   = 8 + LR + 1;   // narrow beats of one burst: up to 256 * R
        localparam SUBN_W  = LR + 1;       // sub-bursts of one burst: up to R
        localparam RBUF_W  = (R - 1) * M_DATA_WIDTH;

        `UNUSED_VAR ({s.awsize, s.awburst, s.arsize, s.arburst, m.bid, m.rid})

        function automatic [8:0] sub_beats(input [CNT_W-1:0] left);
            sub_beats = (left > CNT_W'(256)) ? 9'd256 : 9'(left);
        endfunction

        // ---- write: AW -> sub-burst AWs, each wide W beat -> R narrow beats ----
        logic                  w_busy, w_aw_pend, w_data;
        logic [ADDR_WIDTH-1:0] w_addr;
        logic [ID_WIDTH-1:0]   w_id;
        logic [CNT_W-1:0]      w_left;      // narrow beats not yet sent
        logic [8:0]            w_sub_left;  // narrow beats left in the current sub-burst
        logic [LR-1:0]         w_slice;
        logic [SUBN_W-1:0]     w_subs, w_bs;
        logic [1:0]            w_resp;

        wire aw_fire = m.awvalid && m.awready;
        wire w_fire  = m.wvalid && m.wready;
        wire b_fire  = m.bvalid && m.bready;
        wire w_done  = w_busy && ~w_aw_pend && ~w_data && (w_bs == w_subs) && (w_left == '0);

        always @(posedge clk) begin
            if (reset) begin
                w_busy    <= 1'b0;
                w_aw_pend <= 1'b0;
                w_data    <= 1'b0;
            end else begin
                if (s.awvalid && s.awready) begin
                    w_busy    <= 1'b1;
                    w_aw_pend <= 1'b1;
                end
                if (aw_fire) begin
                    w_aw_pend <= 1'b0;
                    w_data    <= 1'b1;
                end
                if (w_fire && m.wlast) begin
                    w_data    <= 1'b0;
                    w_aw_pend <= (w_left != CNT_W'(1));
                end
                if (s.bvalid && s.bready) begin
                    w_busy <= 1'b0;
                end
            end
            if (s.awvalid && s.awready) begin
                w_addr  <= s.awaddr;
                w_id    <= s.awid;
                w_left  <= CNT_W'({1'b0, s.awlen} + 9'd1) << LR;
                w_slice <= '0;
                w_subs  <= '0;
                w_bs    <= '0;
                w_resp  <= 2'b00;
            end
            if (aw_fire) begin
                w_sub_left <= sub_beats(w_left);
                w_subs     <= w_subs + SUBN_W'(1);
            end
            if (w_fire) begin
                w_addr     <= w_addr + ADDR_WIDTH'(M_BYTES);
                w_left     <= w_left - CNT_W'(1);
                w_sub_left <= w_sub_left - 9'd1;
                w_slice    <= w_slice + LR'(1);
            end
            if (b_fire) begin
                w_bs <= w_bs + SUBN_W'(1);
                if (m.bresp > w_resp)
                    w_resp <= m.bresp;
            end
        end

        assign s.awready = ~w_busy;
        assign m.awvalid = w_aw_pend;
        assign m.awaddr  = w_addr;
        assign m.awid    = w_id;
        assign m.awlen   = 8'(sub_beats(w_left) - 9'd1);
        assign m.awsize  = 3'(`CLOG2(M_BYTES));
        assign m.awburst = 2'b01;

        assign m.wvalid  = w_data && s.wvalid;
        assign m.wdata   = s.wdata[w_slice * M_DATA_WIDTH +: M_DATA_WIDTH];
        assign m.wstrb   = s.wstrb[w_slice * M_BYTES +: M_BYTES];
        assign m.wlast   = (w_sub_left == 9'd1);
        assign s.wready  = w_data && m.wready && (w_slice == LR'(R - 1));
        `UNUSED_VAR (s.wlast)

        assign m.bready  = 1'b1;
        assign s.bvalid  = w_done;
        assign s.bid     = w_id;
        assign s.bresp   = w_resp;

        // ---- read: AR -> sub-burst ARs; R narrow beats pack into one wide beat ----
        logic                  r_busy;
        logic [ADDR_WIDTH-1:0] r_addr;
        logic [ID_WIDTH-1:0]   r_id;
        logic [CNT_W-1:0]      r_ar_left;   // narrow beats not yet requested
        logic [CNT_W-1:0]      r_left;      // narrow beats not yet received
        logic [LR-1:0]         r_slice;
        logic [RBUF_W-1:0]     r_buf;       // earlier slices of the wide beat, oldest lowest
        logic [1:0]            r_resp;

        wire ar_fire  = m.arvalid && m.arready;
        wire r_fire   = m.rvalid && m.rready;
        wire r_packed = (r_slice == LR'(R - 1));

        always @(posedge clk) begin
            if (reset) begin
                r_busy <= 1'b0;
            end else begin
                if (s.arvalid && s.arready)
                    r_busy <= 1'b1;
                if (r_fire && (r_left == CNT_W'(1)))
                    r_busy <= 1'b0;
            end
            if (s.arvalid && s.arready) begin
                r_addr    <= s.araddr;
                r_id      <= s.arid;
                r_ar_left <= CNT_W'({1'b0, s.arlen} + 9'd1) << LR;
                r_left    <= CNT_W'({1'b0, s.arlen} + 9'd1) << LR;
                r_slice   <= '0;
                r_resp    <= 2'b00;
            end
            if (ar_fire) begin
                r_addr    <= r_addr + ADDR_WIDTH'(M_BYTES) * ADDR_WIDTH'(sub_beats(r_ar_left));
                r_ar_left <= r_ar_left - CNT_W'(sub_beats(r_ar_left));
            end
            if (r_fire) begin
                r_left  <= r_left - CNT_W'(1);
                r_slice <= r_slice + LR'(1);
                if (~r_packed) begin
                    r_buf <= RBUF_W'({m.rdata, r_buf} >> M_DATA_WIDTH);
                    if (m.rresp > r_resp)
                        r_resp <= m.rresp;
                end else begin
                    r_resp <= 2'b00;
                end
            end
        end

        assign s.arready = ~r_busy;
        assign m.arvalid = r_busy && (r_ar_left != '0) && (r_ar_left == r_left);
        assign m.araddr  = r_addr;
        assign m.arid    = r_id;
        assign m.arlen   = 8'(sub_beats(r_ar_left) - 9'd1);
        assign m.arsize  = 3'(`CLOG2(M_BYTES));
        assign m.arburst = 2'b01;

        assign m.rready  = r_busy && (~r_packed || s.rready);
        assign s.rvalid  = r_busy && m.rvalid && r_packed;
        assign s.rdata   = {m.rdata, r_buf};
        assign s.rid     = r_id;
        assign s.rlast   = (r_left == CNT_W'(1));
        assign s.rresp   = (m.rresp > r_resp) ? m.rresp : r_resp;
        `UNUSED_VAR (m.rlast)
    end

endmodule
