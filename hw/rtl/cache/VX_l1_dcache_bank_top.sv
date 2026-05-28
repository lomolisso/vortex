// Copyright © 2019-2023. Apache 2.0.
//
// Fixed-parameter hard-macro wrapper around VX_cache_bank for the L1 D-cache
// bank. Dispatched from VX_cache.sv under ASIC_SYNTHESIS when CACHE_KIND="l1_dcache".

`include "VX_define.vh"
`include "VX_cache_define.vh"

`ifdef ASIC_SYNTHESIS
module VX_l1_dcache_bank_top import VX_gpu_pkg::*; #(
    parameter NUM_REQS       = DCACHE_NUM_REQS,
    parameter NUM_BANKS      = `DCACHE_NUM_BANKS,
    parameter WORD_SIZE      = DCACHE_WORD_SIZE,
    parameter LINE_SIZE      = DCACHE_LINE_SIZE,
    parameter TAG_WIDTH      = DCACHE_TAG_WIDTH,
    parameter MSHR_SIZE      = `DCACHE_MSHR_SIZE,
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
        .INSTANCE_ID  (`SFORMATF(("l1_dcache_bank"))),
        .CACHE_SIZE   (`DCACHE_SIZE),
        .LINE_SIZE    (DCACHE_LINE_SIZE),
        .NUM_BANKS    (`DCACHE_NUM_BANKS),
        .NUM_WAYS     (`DCACHE_NUM_WAYS),
        .WORD_SIZE    (DCACHE_WORD_SIZE),
        .NUM_REQS     (DCACHE_NUM_REQS),
        .WRITE_ENABLE (1),
        .WRITEBACK    (`DCACHE_WRITEBACK),
        .DIRTY_BYTES  (`DCACHE_DIRTYBYTES),
        .REPL_POLICY  (`DCACHE_REPL_POLICY),
        .CRSQ_SIZE    (`DCACHE_CRSQ_SIZE),
        .MSHR_SIZE    (`DCACHE_MSHR_SIZE),
        .MREQ_SIZE    (`DCACHE_WRITEBACK ? `DCACHE_MSHR_SIZE : `DCACHE_MREQ_SIZE),
        .TAG_WIDTH    (DCACHE_TAG_WIDTH),
        .CORE_OUT_REG (0),
        .MEM_OUT_REG  (0)
    ) bank (.*);

endmodule
`endif
