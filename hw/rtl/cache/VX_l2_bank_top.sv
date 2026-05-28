// Copyright © 2019-2023. Apache 2.0.
//
// Fixed-parameter hard-macro wrapper around VX_cache_bank for the L2 bank.
// Same wrapper for 1c4s (4 banks) and 1c8s (8 banks) — per-bank size
// (L2_CACHE_SIZE / L2_NUM_BANKS) is 131072 B in both cluster configs.
// Dispatched from VX_cache.sv under ASIC_SYNTHESIS when CACHE_KIND="l2".

`include "VX_define.vh"
`include "VX_cache_define.vh"

`ifdef ASIC_SYNTHESIS
module VX_l2_bank_top import VX_gpu_pkg::*; #(
    parameter NUM_REQS       = L2_NUM_REQS,
    parameter NUM_BANKS      = `L2_NUM_BANKS,
    parameter WORD_SIZE      = L2_WORD_SIZE,
    parameter LINE_SIZE      = `L2_LINE_SIZE,
    parameter TAG_WIDTH      = L2_TAG_WIDTH,
    parameter MSHR_SIZE      = `L2_MSHR_SIZE,
    parameter MSHR_ADDR_WIDTH= `LOG2UP(MSHR_SIZE),
    parameter MEM_TAG_WIDTH  = UUID_WIDTH + MSHR_ADDR_WIDTH,
    parameter REQ_SEL_WIDTH  = `UP(`CS_REQ_SEL_BITS),
    parameter WORD_SEL_WIDTH = `UP(`CS_WORD_SEL_BITS)
) (
    input  wire                          clk,
    input  wire                          reset,
    input  wire                          core_req_valid,
    input  wire [`CS_LINE_ADDR_WIDTH-1:0] core_req_addr,
    input  wire                          core_req_rw,
    input  wire [WORD_SEL_WIDTH-1:0]     core_req_wsel,
    input  wire [WORD_SIZE-1:0]          core_req_byteen,
    input  wire [`CS_WORD_WIDTH-1:0]     core_req_data,
    input  wire [TAG_WIDTH-1:0]          core_req_tag,
    input  wire [REQ_SEL_WIDTH-1:0]      core_req_idx,
    input  wire [`UP(MEM_FLAGS_WIDTH)-1:0] core_req_flags,
    output wire                          core_req_ready,
    output wire                          core_rsp_valid,
    output wire [`CS_WORD_WIDTH-1:0]     core_rsp_data,
    output wire [TAG_WIDTH-1:0]          core_rsp_tag,
    output wire [REQ_SEL_WIDTH-1:0]      core_rsp_idx,
    input  wire                          core_rsp_ready,
    output wire                          mem_req_valid,
    output wire [`CS_LINE_ADDR_WIDTH-1:0] mem_req_addr,
    output wire                          mem_req_rw,
    output wire [LINE_SIZE-1:0]          mem_req_byteen,
    output wire [`CS_LINE_WIDTH-1:0]     mem_req_data,
    output wire [MEM_TAG_WIDTH-1:0]      mem_req_tag,
    output wire [`UP(MEM_FLAGS_WIDTH)-1:0] mem_req_flags,
    input  wire                          mem_req_ready,
    input  wire                          mem_rsp_valid,
    input  wire [`CS_LINE_WIDTH-1:0]     mem_rsp_data,
    input  wire [MEM_TAG_WIDTH-1:0]      mem_rsp_tag,
    output wire                          mem_rsp_ready,
    input  wire                          flush_begin,
    input  wire [`UP(UUID_WIDTH)-1:0]    flush_uuid,
    output wire                          flush_end
);

    VX_cache_bank #(
        .BANK_ID      (0),
        .INSTANCE_ID  (`SFORMATF(("l2_bank"))),
        .CACHE_SIZE   (`L2_CACHE_SIZE),
        .LINE_SIZE    (`L2_LINE_SIZE),
        .NUM_BANKS    (`L2_NUM_BANKS),
        .NUM_WAYS     (`L2_NUM_WAYS),
        .WORD_SIZE    (L2_WORD_SIZE),
        .NUM_REQS     (L2_NUM_REQS),
        .WRITE_ENABLE (1),
        .WRITEBACK    (`L2_WRITEBACK),
        .DIRTY_BYTES  (`L2_DIRTYBYTES),
        .REPL_POLICY  (`L2_REPL_POLICY),
        .CRSQ_SIZE    (`L2_CRSQ_SIZE),
        .MSHR_SIZE    (`L2_MSHR_SIZE),
        .MREQ_SIZE    (`L2_WRITEBACK ? `L2_MSHR_SIZE : `L2_MREQ_SIZE),
        .TAG_WIDTH    (L2_TAG_WIDTH),
        .CORE_OUT_REG (0),
        .MEM_OUT_REG  (0)
    ) bank (.*);

endmodule
`endif
