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

// DXA Descriptor Table — BRAM-based, single read port.
// DCR writes update individual 32-bit words via byte-enable.

`include "VX_define.vh"

module VX_dxa_desc_table import VX_gpu_pkg::*, VX_dxa_pkg::*; (
    input wire clk,
    input wire reset,

    VX_dcr_bus_if.slave dcr_bus_if,

`ifdef DXA_MBAR_ENABLE
    // Map entries are RAM indexes COUNT + e, filled one map line at a time.
    input wire [DXA_DESC_ADDR_W-1:0] read_addr,
    input wire                       fill_valid,
    input wire [DXA_TMAP_IDX_W-1:0]  fill_entry,
    input wire [`VX_DCR_DXA_DESC_STRIDE-1:0] fill_wren,
    input wire [`VX_DCR_DXA_DESC_STRIDE*32-1:0] fill_wdata,
    output wire                      fill_ready,
    output wire [31:0]               read_w22,
    output wire [31:0]               read_w23,
`else
    input wire [DXA_DESC_SLOT_W-1:0] read_addr,
`endif
    output dxa_desc_t read_desc
);
    localparam NUM_SLOTS = `VX_DCR_DXA_DESC_COUNT;
    localparam STRIDE    = `VX_DCR_DXA_DESC_STRIDE;
    localparam ENTRY_W   = STRIDE * 32;
    // Offsets within the descriptor DCR window are bounded by NUM_SLOTS*STRIDE;
    // decode the slot/word at that width so the (non-power-of-two) stride
    // divide/modulo stays a few LUTs instead of a 32-bit carry-chain divider
    // on the BRAM write-enable path.
    localparam OFF_W     = `CLOG2(NUM_SLOTS * STRIDE);

    // ---- DCR write logic ----
    wire dcr_write = dcr_bus_if.req_valid
                  && dcr_bus_if.req_data.rw
                  && (dcr_bus_if.req_data.addr >= `VX_DCR_DXA_DESC_BASE)
                  && (dcr_bus_if.req_data.addr < (`VX_DCR_DXA_DESC_BASE + (NUM_SLOTS * STRIDE)));

    wire [OFF_W-1:0] dcr_off = OFF_W'(dcr_bus_if.req_data.addr - `VX_DCR_DXA_DESC_BASE);
    wire [DXA_DESC_SLOT_W-1:0]  dcr_slot = DXA_DESC_SLOT_W'(dcr_off / STRIDE);
    wire [`CLOG2(STRIDE)-1:0]   dcr_word = `CLOG2(STRIDE)'(dcr_off % STRIDE);

    wire [STRIDE-1:0] dcr_wren;
    for (genvar j = 0; j < STRIDE; ++j) begin : g_wren
        assign dcr_wren[j] = dcr_write && (dcr_word == `CLOG2(STRIDE)'(j));
    end

    wire [ENTRY_W-1:0] dcr_wdata;
    for (genvar j = 0; j < STRIDE; ++j) begin : g_wdata
        assign dcr_wdata[j*32 +: 32] = dcr_bus_if.req_data.data;
    end

    // ---- BRAM storage ----
    wire [ENTRY_W-1:0] entry_rdata;
    `UNUSED_VAR (entry_rdata)

    // OUT_REG=1 gives the RAM a registered read port, which lets the wide
    // (ENTRY_W × NUM_SLOTS) descriptor store infer true BRAM instead of
    // distributed LUTRAM. The 1-cycle read latency is absorbed by the fetch
    // register in VX_dxa_core.
`ifdef DXA_MBAR_ENABLE
    // DCR writes keep the write port; a fill is held while one is presented.
    assign fill_ready = ~dcr_write;
    wire                       ram_write = dcr_write || fill_valid;
    wire [STRIDE-1:0]          ram_wren  = dcr_write ? dcr_wren : fill_wren;
    wire [DXA_DESC_ADDR_W-1:0] ram_waddr = dcr_write ? DXA_DESC_ADDR_W'(dcr_slot)
                                         : DXA_DESC_ADDR_W'(NUM_SLOTS + 32'(fill_entry));
    wire [ENTRY_W-1:0]         ram_wdata = dcr_write ? dcr_wdata : fill_wdata;

    VX_dp_ram #(
        .DATAW    (ENTRY_W),
        .SIZE     (DXA_DESC_ENTRIES),
        .WRENW    (STRIDE),
        .OUT_REG  (1),
        .RDW_MODE ("W")
    ) desc_store (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (ram_write),
        .wren  (ram_wren),
        .raddr (read_addr),
        .waddr (ram_waddr),
        .wdata (ram_wdata),
        .rdata (entry_rdata)
    );

    assign read_w22 = entry_rdata[DXA_TMAP_W_STRIDE_HI*32 +: 32];
    assign read_w23 = entry_rdata[DXA_TMAP_W_ETYPE*32 +: 32];
`else
    VX_dp_ram #(
        .DATAW    (ENTRY_W),
        .SIZE     (NUM_SLOTS),
        .WRENW    (STRIDE),
        .OUT_REG  (1),
        .RDW_MODE ("W")
    ) desc_store (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (dcr_write),
        .wren  (dcr_wren),
        .raddr (read_addr),
        .waddr (dcr_slot),
        .wdata (dcr_wdata),
        .rdata (entry_rdata)
    );
`endif

    // Unpack fields
    assign read_desc.base_addr = `VX_CFG_MEM_ADDR_WIDTH'({
        entry_rdata[`VX_DCR_DXA_DESC_BASE_HI_OFF*32 +: 32],
        entry_rdata[`VX_DCR_DXA_DESC_BASE_LO_OFF*32 +: 32]
    });
    assign read_desc.meta    = entry_rdata[`VX_DCR_DXA_DESC_META_OFF*32 +: 32];
    assign read_desc.tile01  = entry_rdata[`VX_DCR_DXA_DESC_TILESIZE01_OFF*32 +: 32];
    assign read_desc.tile23  = entry_rdata[`VX_DCR_DXA_DESC_TILESIZE23_OFF*32 +: 32];
    assign read_desc.tile4   = entry_rdata[`VX_DCR_DXA_DESC_TILESIZE4_OFF*32 +: 32];
    assign read_desc.cfill   = entry_rdata[`VX_DCR_DXA_DESC_CFILL_OFF*32 +: 32];
    assign read_desc.size0   = entry_rdata[`VX_DCR_DXA_DESC_SIZE0_OFF*32 +: 32];
    assign read_desc.size1   = entry_rdata[`VX_DCR_DXA_DESC_SIZE1_OFF*32 +: 32];
    assign read_desc.size2   = entry_rdata[`VX_DCR_DXA_DESC_SIZE2_OFF*32 +: 32];
    assign read_desc.size3   = entry_rdata[`VX_DCR_DXA_DESC_SIZE3_OFF*32 +: 32];
    assign read_desc.size4   = entry_rdata[`VX_DCR_DXA_DESC_SIZE4_OFF*32 +: 32];
    assign read_desc.stride0 = entry_rdata[`VX_DCR_DXA_DESC_STRIDE0_OFF*32 +: 32];
    assign read_desc.stride1 = entry_rdata[`VX_DCR_DXA_DESC_STRIDE1_OFF*32 +: 32];
    assign read_desc.stride2 = entry_rdata[`VX_DCR_DXA_DESC_STRIDE2_OFF*32 +: 32];
    assign read_desc.stride3 = entry_rdata[`VX_DCR_DXA_DESC_STRIDE3_OFF*32 +: 32];
    assign read_desc.smem_stride = entry_rdata[`VX_DCR_DXA_DESC_SMEM_STRIDE_OFF*32 +: 32];
    assign read_desc.estride2    = entry_rdata[`VX_DCR_DXA_DESC_ESTRIDE2_OFF*32 +: 32];
`ifdef DXA_MBAR_ENABLE
    assign read_desc.estride0    = entry_rdata[`VX_DCR_DXA_DESC_ESTRIDE0_OFF*32 +: 32];
    assign read_desc.estride1    = entry_rdata[`VX_DCR_DXA_DESC_ESTRIDE1_OFF*32 +: 32];
    assign read_desc.estride3    = entry_rdata[`VX_DCR_DXA_DESC_ESTRIDE3_OFF*32 +: 32];
    assign read_desc.estride4    = entry_rdata[`VX_DCR_DXA_DESC_ESTRIDE4_OFF*32 +: 32];
`endif

    assign dcr_bus_if.rsp_valid = '0;
    assign dcr_bus_if.rsp_data  = '0;

    `UNUSED_VAR (reset)

endmodule
