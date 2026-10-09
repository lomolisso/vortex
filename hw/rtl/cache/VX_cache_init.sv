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

`include "VX_cache_define.vh"

// Flush admission. A flush request (is_flush) walks the cache: every port is
// held, the banks drain and walk their lines, then the waiting flush requests
// go. At the device's shared level (SHARED_LEVEL) a fence (is_flush with
// is_fence) drains instead: only the fence waits, until the banks have retired
// every request accepted before it, and no line is walked.
module VX_cache_init import VX_gpu_pkg::*; #(
    parameter NUM_REQS  = 4,                  // Number of Word requests per cycle
    parameter NUM_BANKS = 1,                  // Number of banks
    parameter TAG_WIDTH = UUID_WIDTH + 1,     // core request tag size
    parameter BANK_SEL_LATENCY = 1,           // Bank select latency
    parameter SHARED_LEVEL = 0                // every memory agent reaches this cache
) (
    input wire              clk,
    input wire              reset,
    VX_mem_bus_if.slave     core_bus_in_if [NUM_REQS],
    VX_mem_bus_if.master    core_bus_out_if [NUM_REQS],
    input wire [NUM_BANKS-1:0] bank_req_fire,
    // acceptance epoch the bank select stage carries with each request
    output wire             req_epoch,
    input wire [NUM_BANKS-1:0] bank_req_epoch,
    output wire [NUM_BANKS-1:0] flush_begin,
    output wire             flush_drain,
    output wire [`UP(UUID_WIDTH)-1:0] flush_uuid,
    input wire [NUM_BANKS-1:0] flush_end
);
    `UNUSED_PARAM (TAG_WIDTH)

    localparam STATE_IDLE  = 0;
    localparam STATE_WAIT1 = 1;
    localparam STATE_FLUSH = 2;
    localparam STATE_WAIT2 = 3;
    localparam STATE_DONE  = 4;
    localparam STATE_GRACE = 5;
    localparam STATE_DRAIN = 6;
    localparam STATE_DREL  = 7;

    localparam NUM_REQS_W  = `CLOG2(NUM_REQS+1);
    localparam NUM_BANKS_W = `CLOG2(NUM_BANKS+1);

    reg [2:0] state, state_n;

    wire [NUM_REQS-1:0] core_bus_out_fire;
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_bus_out_fire
        assign core_bus_out_fire[i] = core_bus_out_if[i].req_valid && core_bus_out_if[i].req_ready;
    end

    wire [NUM_REQS_W-1:0] core_bus_out_cnt;
    `POP_COUNT(core_bus_out_cnt, core_bus_out_fire);

    // track in-flight core requests

    wire no_inflight_reqs;

    if (BANK_SEL_LATENCY != 0) begin : g_bank_sel_latency

        wire [NUM_BANKS_W-1:0] bank_req_cnt;
        `POP_COUNT(bank_req_cnt, bank_req_fire);

        VX_pending_size #(
            .SIZE  (BANK_SEL_LATENCY * NUM_BANKS),
            .INCRW (NUM_BANKS_W),
            .DECRW (NUM_BANKS_W)
        ) pending_size (
            .clk   (clk),
            .reset (reset),
            .incr  (NUM_BANKS_W'(core_bus_out_cnt)),
            .decr  (bank_req_cnt),
            .empty (no_inflight_reqs),
            `UNUSED_PIN (alm_empty),
            `UNUSED_PIN (full),
            `UNUSED_PIN (alm_full),
            `UNUSED_PIN (size)
        );

    end else begin : g_no_bank_sel_latency
        assign no_inflight_reqs = 0;
    end

    reg [NUM_BANKS-1:0] flush_done, flush_done_n;

    wire [NUM_REQS-1:0] flush_req_mask;
    wire [NUM_REQS-1:0] fence_req_mask;
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_flush_req_mask
        assign flush_req_mask[i] = core_bus_in_if[i].req_valid && core_bus_in_if[i].req_data.attr[MEM_ATTR_FLUSH_OFFS];
        assign fence_req_mask[i] = flush_req_mask[i] && (SHARED_LEVEL != 0)
                                && core_bus_in_if[i].req_data.attr[MEM_ATTR_FENCE_OFFS];
    end
    wire flush_req_enable = (| flush_req_mask);
    wire walk_req_enable  = (| (flush_req_mask & ~fence_req_mask));
    wire fence_req_enable = (| fence_req_mask);

    reg [NUM_REQS-1:0] lock_released, lock_released_n;
    reg [NUM_REQS-1:0] grace_mask, grace_mask_n;
    reg [`UP(UUID_WIDTH)-1:0] flush_uuid_r, flush_uuid_n;

    // A walk holds every port from the cycle its request shows up until the
    // walk ends, then lets only the flush requests it covers go. A port whose
    // regular request waited through the walk is admitted once before another
    // walk may start (STATE_GRACE), so a flush stream cannot starve it. A drain
    // holds only the fence requests. Holding whole ports keeps each port's
    // requests in order.
    reg [NUM_REQS-1:0] input_enable;
    always @(*) begin
        case (state)
        STATE_IDLE:  input_enable = walk_req_enable ? '0 : ~flush_req_mask;
        STATE_DONE:  input_enable = {NUM_REQS{~flush_req_enable}} | lock_released;
        STATE_GRACE,
        STATE_DRAIN: input_enable = ~flush_req_mask;
        STATE_DREL:  input_enable = ~flush_req_mask | lock_released;
        default:     input_enable = '0;
        endcase
    end

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_bus_out_req
        assign core_bus_out_if[i].req_valid = core_bus_in_if[i].req_valid && input_enable[i];
        assign core_bus_out_if[i].req_data  = core_bus_in_if[i].req_data;
        assign core_bus_in_if[i].req_ready  = core_bus_out_if[i].req_ready && input_enable[i];
    end

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_bus_in_rsp
        assign core_bus_in_if[i].rsp_valid  = core_bus_out_if[i].rsp_valid;
        assign core_bus_in_if[i].rsp_data   = core_bus_out_if[i].rsp_data;
        assign core_bus_out_if[i].rsp_ready = core_bus_in_if[i].rsp_ready;
    end

    reg [NUM_REQS-1:0][`UP(UUID_WIDTH)-1:0] core_bus_out_uuid;
    wire [NUM_REQS-1:0] core_bus_out_ready;
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_bus_out_uuid
        if (UUID_WIDTH != 0) begin : g_uuid
            assign core_bus_out_uuid[i] = core_bus_in_if[i].req_data.tag.uuid;
        end else begin : g_no_uuid
            assign core_bus_out_uuid[i] = 0;
        end
    end

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_bus_out_ready
        assign core_bus_out_ready[i] = core_bus_out_if[i].req_ready;
    end

    // regular requests still waiting at their port after this cycle
    wire [NUM_REQS-1:0] regular_waiting;
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_regular_waiting
        assign regular_waiting[i] = core_bus_in_if[i].req_valid && ~flush_req_mask[i] && ~core_bus_out_fire[i];
    end

    wire drain_start = (state == STATE_IDLE) && ~walk_req_enable && fence_req_enable;
    wire drain_banks;

    if (SHARED_LEVEL != 0) begin : g_drain

        // The banks may start a drain only once every request accepted before
        // it has entered its bank. The bank select stage reorders requests, so
        // each one carries its acceptance epoch; a drain flips the epoch and
        // waits until none of the old epoch is left in transit.
        localparam TRANSIT_W = `CLOG2(4 * (NUM_REQS + NUM_BANKS) + 1);

        reg epoch_r;
        reg [1:0][TRANSIT_W-1:0] transit_r;
        reg drain_begun_r;

        wire [NUM_BANKS_W-1:0] bank_fire_cnt0, bank_fire_cnt1;
        `POP_COUNT(bank_fire_cnt0, bank_req_fire & ~bank_req_epoch);
        `POP_COUNT(bank_fire_cnt1, bank_req_fire & bank_req_epoch);

        always @(posedge clk) begin
            if (reset) begin
                epoch_r       <= 1'b0;
                transit_r     <= '0;
                drain_begun_r <= 1'b0;
            end else begin
                transit_r[0] <= transit_r[0] + TRANSIT_W'(epoch_r ? NUM_REQS_W'(0) : core_bus_out_cnt) - TRANSIT_W'(bank_fire_cnt0);
                transit_r[1] <= transit_r[1] + TRANSIT_W'(epoch_r ? core_bus_out_cnt : NUM_REQS_W'(0)) - TRANSIT_W'(bank_fire_cnt1);
                if (drain_start) begin
                    epoch_r <= ~epoch_r;
                end
                drain_begun_r <= (state == STATE_DRAIN) && (drain_begun_r || drain_banks);
            end
        end

        assign req_epoch = epoch_r;
        assign drain_banks = (state == STATE_DRAIN) && ~drain_begun_r && (transit_r[~epoch_r] == '0);

        `RUNTIME_ASSERT (~drain_start || (transit_r[~epoch_r] == '0), ("%t: cache drain started with the next epoch in transit", $time))

    end else begin : g_no_drain
        assign req_epoch   = 1'b0;
        assign drain_banks = 1'b0;
        `UNUSED_VAR (bank_req_epoch)
        `UNUSED_VAR (fence_req_enable)
        if (BANK_SEL_LATENCY == 0) begin : g_unused
            `UNUSED_VAR (bank_req_fire)
            `UNUSED_VAR (core_bus_out_cnt)
        end
    end

    always @(*) begin
        state_n = state;
        flush_done_n = flush_done;
        lock_released_n = lock_released;
        grace_mask_n = grace_mask;
        flush_uuid_n = flush_uuid_r;
        case (state)
            // STATE_IDLE:
            default: begin
                if (walk_req_enable) begin
                    state_n = (BANK_SEL_LATENCY != 0) ? STATE_WAIT1 : STATE_FLUSH;
                    for (integer i = NUM_REQS-1; i >= 0; --i) begin
                        if (flush_req_mask[i]) begin
                            flush_uuid_n = core_bus_out_uuid[i];
                        end
                    end
                end else if (drain_start) begin
                    state_n = STATE_DRAIN;
                    // only the fences waiting now: a later fence's earlier
                    // requests may be accepted after the drain started
                    lock_released_n = fence_req_mask;
                    for (integer i = NUM_REQS-1; i >= 0; --i) begin
                        if (fence_req_mask[i]) begin
                            flush_uuid_n = core_bus_out_uuid[i];
                        end
                    end
                end
            end
            STATE_WAIT1: begin
                if (no_inflight_reqs) begin
                    state_n = STATE_FLUSH;
                end
            end
            STATE_FLUSH: begin
                // generate a flush request pulse
                state_n = STATE_WAIT2;
            end
            STATE_WAIT2: begin
                // wait for all banks to finish flushing
                flush_done_n = flush_done | flush_end;
                if (flush_done_n == {NUM_BANKS{1'b1}}) begin
                    state_n = STATE_DONE;
                    flush_done_n = '0;
                    // only release current flush requests
                    // and keep normal requests locked
                    lock_released_n = flush_req_mask;
                end
            end
            STATE_DONE: begin
                // wait until released flush requests are issued
                lock_released_n = lock_released & ~core_bus_out_ready;
                if (lock_released_n == 0) begin
                    grace_mask_n = regular_waiting;
                    state_n = (regular_waiting != 0) ? STATE_GRACE : STATE_IDLE;
                end
            end
            STATE_GRACE: begin
                grace_mask_n = grace_mask & regular_waiting;
                if (grace_mask_n == 0) begin
                    state_n = STATE_IDLE;
                end
            end
            STATE_DRAIN: begin
                flush_done_n = flush_done | flush_end;
                if (flush_done_n == {NUM_BANKS{1'b1}}) begin
                    state_n = STATE_DREL;
                    flush_done_n = '0;
                end
            end
            STATE_DREL: begin
                // the drained fences go on as reads of their word, which answer them
                lock_released_n = lock_released & ~core_bus_out_ready;
                if (lock_released_n == 0) begin
                    state_n = STATE_IDLE;
                end
            end
        endcase
    end

    always @(posedge clk) begin
        if (reset) begin
            state <= STATE_IDLE;
            flush_done <= '0;
            lock_released <= '0;
            grace_mask <= '0;
        end else begin
            state <= state_n;
            flush_done <= flush_done_n;
            lock_released <= lock_released_n;
            grace_mask <= grace_mask_n;
        end
        flush_uuid_r <= flush_uuid_n;
    end

    assign flush_begin = {NUM_BANKS{(state == STATE_FLUSH) || drain_banks}};
    assign flush_drain = (state == STATE_DRAIN);
    assign flush_uuid  = flush_uuid_r;

endmodule
