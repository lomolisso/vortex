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

module VX_cluster import VX_gpu_pkg::*; #(
    parameter CLUSTER_ID = 0,
    parameter `STRING INSTANCE_ID = ""
) (
    `SCOPE_IO_DECL

    // Clock
    input  wire                 clk,
    input  wire                 reset,

`ifdef PERF_ENABLE
    input sysmem_perf_t         sysmem_perf,
`endif

    // DCRs
    VX_dcr_bus_if.slave         dcr_bus_if,

    // Memory
    VX_mem_bus_if.master        mem_bus_if [`L2_MEM_PORTS],

    // Status
    output wire                 busy
);

`ifdef SCOPE
    localparam scope_socket = 0;
    `SCOPE_IO_SWITCH (NUM_SOCKETS);
`endif

`ifdef PERF_ENABLE
    cache_perf_t l2_perf;
    sysmem_perf_t sysmem_perf_tmp;
    always @(*) begin
        sysmem_perf_tmp = sysmem_perf;
        sysmem_perf_tmp.l2cache = l2_perf;
    end
`endif

`ifdef GBAR_ENABLE

    VX_gbar_bus_if per_socket_gbar_bus_if[NUM_SOCKETS]();
    VX_gbar_bus_if gbar_bus_if();

    VX_gbar_arb #(
        .NUM_REQS (NUM_SOCKETS),
        .OUT_BUF  ((NUM_SOCKETS > 2) ? 1 : 0) // bgar_unit has no backpressure
    ) gbar_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (per_socket_gbar_bus_if),
        .bus_out_if (gbar_bus_if)
    );

    VX_gbar_unit #(
        .INSTANCE_ID (`SFORMATF(("gbar%0d", CLUSTER_ID)))
    ) gbar_unit (
        .clk         (clk),
        .reset       (reset),
        .gbar_bus_if (gbar_bus_if)
    );

`endif

    VX_mem_bus_if #(
        .DATA_SIZE (`L1_LINE_SIZE),
        .TAG_WIDTH (L1_MEM_ARB_TAG_WIDTH)
    ) per_socket_mem_bus_if[NUM_SOCKETS * `L1_MEM_PORTS]();

    `RESET_RELAY (l2_reset, reset);

    VX_cache_wrap #(
        .INSTANCE_ID    (`SFORMATF(("%s-l2cache", INSTANCE_ID))),
        .CACHE_SIZE     (`L2_CACHE_SIZE),
        .LINE_SIZE      (`L2_LINE_SIZE),
        .NUM_BANKS      (`L2_NUM_BANKS),
        .NUM_WAYS       (`L2_NUM_WAYS),
        .WORD_SIZE      (L2_WORD_SIZE),
        .NUM_REQS       (L2_NUM_REQS),
        .MEM_PORTS      (`L2_MEM_PORTS),
        .CRSQ_SIZE      (`L2_CRSQ_SIZE),
        .MSHR_SIZE      (`L2_MSHR_SIZE),
        .MRSQ_SIZE      (`L2_MRSQ_SIZE),
        .MREQ_SIZE      (`L2_WRITEBACK ? `L2_MSHR_SIZE : `L2_MREQ_SIZE),
        .TAG_WIDTH      (L2_TAG_WIDTH),
        .WRITE_ENABLE   (1),
        .WRITEBACK      (`L2_WRITEBACK),
        .DIRTY_BYTES    (`L2_DIRTYBYTES),
        .REPL_POLICY    (`L2_REPL_POLICY),
        .CORE_OUT_BUF   (3),
        .MEM_OUT_BUF    (3),
        .NC_ENABLE      (1),
        .PASSTHRU       (!`L2_ENABLED)
    ) l2cache (
        .clk            (clk),
        .reset          (l2_reset),
    `ifdef PERF_ENABLE
        .cache_perf     (l2_perf),
    `endif
        .core_bus_if    (per_socket_mem_bus_if),
        .mem_bus_if     (mem_bus_if)
    );

    ///////////////////////////////////////////////////////////////////////////

    wire [NUM_SOCKETS-1:0] per_socket_busy;

    // Generate all sockets
    //
    // In ASIC synthesis flows we instantiate VX_socket_top (a flat-port
    // wrapper around VX_socket). This makes the socket module the clean
    // blackbox boundary for the two-step flow: synthesize + PnR one
    // VX_socket_top in isolation, then later swap its RTL for a .lef/.lib
    // macro in the full-GPGPU synthesis run. The flat ports match 1:1 with
    // the pins produced by PnR.
    //
    // In all other flows (simulation, FPGA synth) we keep the original
    // interface-port VX_socket so that performance counters and SCOPE
    // debug hooks continue to work — the wrapper would otherwise stub
    // sysmem_perf and drop SCOPE_IO_BIND.
    for (genvar socket_id = 0; socket_id < NUM_SOCKETS; ++socket_id) begin : g_sockets

        `RESET_RELAY (socket_reset, reset);

        VX_dcr_bus_if socket_dcr_bus_if();
        wire is_base_dcr_addr = (dcr_bus_if.write_addr >= `VX_DCR_BASE_STATE_BEGIN && dcr_bus_if.write_addr < `VX_DCR_BASE_STATE_END);
        `BUFFER_DCR_BUS_IF (socket_dcr_bus_if, dcr_bus_if, is_base_dcr_addr, (NUM_SOCKETS > 1))

    `ifdef ASIC_SYNTHESIS
        // Flatten this socket's slice of the per_socket_mem_bus_if array
        // into plain wires so we can drive VX_socket_top's flat ports.
        wire [`L1_MEM_PORTS-1:0]                                             sock_mem_req_valid;
        wire [`L1_MEM_PORTS-1:0]                                             sock_mem_req_rw;
        wire [`L1_MEM_PORTS-1:0][`L1_LINE_SIZE-1:0]                          sock_mem_req_byteen;
        wire [`L1_MEM_PORTS-1:0][`MEM_ADDR_WIDTH-`CLOG2(`L1_LINE_SIZE)-1:0]  sock_mem_req_addr;
        wire [`L1_MEM_PORTS-1:0][`L1_LINE_SIZE*8-1:0]                        sock_mem_req_data;
        wire [`L1_MEM_PORTS-1:0][MEM_FLAGS_WIDTH-1:0]                        sock_mem_req_flags;
        wire [`L1_MEM_PORTS-1:0][L1_MEM_ARB_TAG_WIDTH-1:0]                   sock_mem_req_tag;
        wire [`L1_MEM_PORTS-1:0]                                             sock_mem_req_ready;
        wire [`L1_MEM_PORTS-1:0]                                             sock_mem_rsp_valid;
        wire [`L1_MEM_PORTS-1:0][`L1_LINE_SIZE*8-1:0]                        sock_mem_rsp_data;
        wire [`L1_MEM_PORTS-1:0][L1_MEM_ARB_TAG_WIDTH-1:0]                   sock_mem_rsp_tag;
        wire [`L1_MEM_PORTS-1:0]                                             sock_mem_rsp_ready;

        for (genvar p = 0; p < `L1_MEM_PORTS; ++p) begin : g_mem_flat
            assign per_socket_mem_bus_if[socket_id * `L1_MEM_PORTS + p].req_valid       = sock_mem_req_valid[p];
            assign per_socket_mem_bus_if[socket_id * `L1_MEM_PORTS + p].req_data.rw     = sock_mem_req_rw[p];
            assign per_socket_mem_bus_if[socket_id * `L1_MEM_PORTS + p].req_data.byteen = sock_mem_req_byteen[p];
            assign per_socket_mem_bus_if[socket_id * `L1_MEM_PORTS + p].req_data.addr   = sock_mem_req_addr[p];
            assign per_socket_mem_bus_if[socket_id * `L1_MEM_PORTS + p].req_data.data   = sock_mem_req_data[p];
            assign per_socket_mem_bus_if[socket_id * `L1_MEM_PORTS + p].req_data.flags  = sock_mem_req_flags[p];
            assign per_socket_mem_bus_if[socket_id * `L1_MEM_PORTS + p].req_data.tag    = sock_mem_req_tag[p];
            assign sock_mem_req_ready[p]                                                = per_socket_mem_bus_if[socket_id * `L1_MEM_PORTS + p].req_ready;

            assign sock_mem_rsp_valid[p]                                                = per_socket_mem_bus_if[socket_id * `L1_MEM_PORTS + p].rsp_valid;
            assign sock_mem_rsp_data[p]                                                 = per_socket_mem_bus_if[socket_id * `L1_MEM_PORTS + p].rsp_data.data;
            assign sock_mem_rsp_tag[p]                                                  = per_socket_mem_bus_if[socket_id * `L1_MEM_PORTS + p].rsp_data.tag;
            assign per_socket_mem_bus_if[socket_id * `L1_MEM_PORTS + p].rsp_ready       = sock_mem_rsp_ready[p];
        end

    `ifdef GBAR_ENABLE
        // Flatten the socket's gbar interface instance into wires so
        // VX_socket_top's flat ports can drive / be driven by them.
        wire                  sock_gbar_req_valid;
        wire [NB_WIDTH-1:0]   sock_gbar_req_id;
        wire [NC_WIDTH-1:0]   sock_gbar_req_size_m1;
        wire [NC_WIDTH-1:0]   sock_gbar_req_core_id;
        wire                  sock_gbar_req_ready;
        wire                  sock_gbar_rsp_valid;
        wire [NB_WIDTH-1:0]   sock_gbar_rsp_id;

        assign per_socket_gbar_bus_if[socket_id].req_valid       = sock_gbar_req_valid;
        assign per_socket_gbar_bus_if[socket_id].req_data.id     = sock_gbar_req_id;
        assign per_socket_gbar_bus_if[socket_id].req_data.size_m1= sock_gbar_req_size_m1;
        assign per_socket_gbar_bus_if[socket_id].req_data.core_id= sock_gbar_req_core_id;
        assign sock_gbar_req_ready                               = per_socket_gbar_bus_if[socket_id].req_ready;
        assign sock_gbar_rsp_valid                               = per_socket_gbar_bus_if[socket_id].rsp_valid;
        assign sock_gbar_rsp_id                                  = per_socket_gbar_bus_if[socket_id].rsp_data.id;
    `endif

        VX_socket_top #(
            .SOCKET_ID ((CLUSTER_ID * NUM_SOCKETS) + socket_id)
        ) socket (
            .clk             (clk),
            .reset           (socket_reset),

            .dcr_write_valid (socket_dcr_bus_if.write_valid),
            .dcr_write_addr  (socket_dcr_bus_if.write_addr),
            .dcr_write_data  (socket_dcr_bus_if.write_data),

            .mem_req_valid   (sock_mem_req_valid),
            .mem_req_rw      (sock_mem_req_rw),
            .mem_req_byteen  (sock_mem_req_byteen),
            .mem_req_addr    (sock_mem_req_addr),
            .mem_req_data    (sock_mem_req_data),
            .mem_req_flags   (sock_mem_req_flags),
            .mem_req_tag     (sock_mem_req_tag),
            .mem_req_ready   (sock_mem_req_ready),

            .mem_rsp_valid   (sock_mem_rsp_valid),
            .mem_rsp_data    (sock_mem_rsp_data),
            .mem_rsp_tag     (sock_mem_rsp_tag),
            .mem_rsp_ready   (sock_mem_rsp_ready),

        `ifdef GBAR_ENABLE
            .gbar_req_valid    (sock_gbar_req_valid),
            .gbar_req_id       (sock_gbar_req_id),
            .gbar_req_size_m1  (sock_gbar_req_size_m1),
            .gbar_req_core_id  (sock_gbar_req_core_id),
            .gbar_req_ready    (sock_gbar_req_ready),
            .gbar_rsp_valid    (sock_gbar_rsp_valid),
            .gbar_rsp_id       (sock_gbar_rsp_id),
        `endif

            .busy            (per_socket_busy[socket_id])
        );
    `else
        VX_socket #(
            .SOCKET_ID ((CLUSTER_ID * NUM_SOCKETS) + socket_id),
            .INSTANCE_ID (`SFORMATF(("%s-socket%0d", INSTANCE_ID, socket_id)))
        ) socket (
            `SCOPE_IO_BIND  (scope_socket+socket_id)

            .clk            (clk),
            .reset          (socket_reset),

        `ifdef PERF_ENABLE
            .sysmem_perf    (sysmem_perf_tmp),
        `endif

            .dcr_bus_if     (socket_dcr_bus_if),

            .mem_bus_if     (per_socket_mem_bus_if[socket_id * `L1_MEM_PORTS +: `L1_MEM_PORTS]),

        `ifdef GBAR_ENABLE
            .gbar_bus_if    (per_socket_gbar_bus_if[socket_id]),
        `endif

            .busy           (per_socket_busy[socket_id])
        );
    `endif
    end

    `BUFFER_EX(busy, (| per_socket_busy), 1'b1, 1, (NUM_SOCKETS > 1));

endmodule
