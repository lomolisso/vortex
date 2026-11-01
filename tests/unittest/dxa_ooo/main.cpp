// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include "core.h"
#include "dxa_core.h"
#include "mem_block_pool.h"
#include "processor.h"
#include "simobject.h"
#include <VX_types.h>

using namespace vortex;

namespace {

[[noreturn]] void fail(const char *message) {
  std::cerr << "FAILED: " << message << std::endl;
  std::exit(1);
}

void tick(uint32_t cycles = 1) {
  for (uint32_t i = 0; i < cycles; ++i)
    SimPlatform::instance().tick();
}

MemReq wait_for_lmem_write(DxaCore *dxa, uint32_t limit = 100) {
  for (uint32_t i = 0; i < limit; ++i) {
    tick();
    auto &output = dxa->lmem_req_out.at(0);
    if (!output.empty()) {
      MemReq req = output.peek();
      output.pop();
      return req;
    }
  }
  fail("timed out waiting for LMEM write");
}

std::shared_ptr<mem_block_t> make_response_data(uint8_t value) {
  auto data = make_mem_block();
  std::fill(data->begin(), data->end(), value);
  return data;
}

void send_response(DxaCore *dxa, const MemReq &request, uint8_t value) {
  dxa->gmem_rsp_in.at(0).send(
      MemRsp(request.tag, 0, 1, make_response_data(value)));
}

void check_write(const MemReq &write, uint64_t expected_addr,
                 uint8_t expected_data, bool expected_notify) {
  if (write.addr != expected_addr)
    fail("response wrote an unexpected LMEM address");
  if (!write.data || write.data->at(0) != expected_data)
    fail("response wrote unexpected data");
  if (write.flags.dxa_notify_done != expected_notify)
    fail("response carried an unexpected completion flag");
}

// A barrier multicast replays each beat to the receivers of its warp mask in
// increasing bit order: bit j writes at the issuer's address + j * smem_stride
// and releases barrier + j, whatever bits are clear below it.
void check_sparse_mask(DxaCore *dxa, Core *core, uint32_t mask) {
  constexpr uint32_t slot = 1;
  constexpr uint32_t line_size = VX_CFG_L1_LINE_SIZE;
  constexpr uint32_t stride = 4 * line_size;
  constexpr uint32_t gmem_base = 64 * line_size;
  constexpr uint32_t smem_base = 2 * line_size;
  constexpr uint32_t raw_bar = (3u << 8) | 0x10u;
  constexpr uint32_t dcr = VX_DCR_DXA_DESC_BASE + slot * VX_DCR_DXA_DESC_STRIDE;
  dxa->dcr_write(dcr + VX_DCR_DXA_DESC_BASE_LO_OFF, gmem_base);
  dxa->dcr_write(dcr + VX_DCR_DXA_DESC_BASE_HI_OFF, 0);
  dxa->dcr_write(dcr + VX_DCR_DXA_DESC_SIZE0_OFF, line_size);
  dxa->dcr_write(dcr + VX_DCR_DXA_DESC_META_OFF, 1);
  dxa->dcr_write(dcr + VX_DCR_DXA_DESC_ESTRIDE0_OFF, 1);
  dxa->dcr_write(dcr + VX_DCR_DXA_DESC_TILESIZE01_OFF, line_size);
  dxa->dcr_write(dcr + VX_DCR_DXA_DESC_SMEM_STRIDE_OFF, stride);

  DxaReq req{};
  req.core = core;
  req.uuid = 2;
  req.desc_slot = slot;
  req.bar_id = raw_bar;
  req.cta_mask = mask;
  req.smem_addr = smem_base;
  dxa->dxa_req_in.at(0).send(req);

  MemReq read;
  bool have_read = false;
  for (uint32_t i = 0; i < 100 && !have_read; ++i) {
    tick();
    auto &output = dxa->gmem_req_out.at(0);
    if (!output.empty()) {
      read = output.peek();
      output.pop();
      have_read = true;
    }
  }
  if (!have_read || read.addr != gmem_base)
    fail("sparse mask: expected one GMEM read of the line");
  send_response(dxa, read, 0x5a);

  std::vector<uint32_t> bits;
  for (uint32_t j = 0; j < 32; ++j) {
    if ((mask >> j) & 1)
      bits.push_back(j);
  }
  uint32_t rcv = 0;
  for (uint32_t n = 0; n < 64 && rcv < bits.size(); ++n) {
    MemReq write = wait_for_lmem_write(dxa);
    const uint64_t base = smem_base + uint64_t(bits.at(rcv)) * stride;
    if (write.addr < base || write.addr >= base + line_size)
      fail("sparse mask: data of a receiver not at the issuer's address + bit * smem_stride");
    if (!write.data || write.byteen == 0 || write.data->at(__builtin_ctzll(write.byteen)) != 0x5a)
      fail("sparse mask: unexpected data");
    if (write.flags.dxa_notify_done) {
      if (write.flags.dxa_notify_bar_id != raw_bar + bits.at(rcv))
        fail("sparse mask: a receiver's release is not barrier + bit");
      ++rcv;
    }
  }
  if (rcv != bits.size())
    fail("sparse mask: a receiver got no completion");
  tick(8);
  if (!dxa->lmem_req_out.at(0).empty())
    fail("sparse mask: a write past the last receiver");
}

} // namespace

int main() {
  Processor processor;
  auto *core = processor.get_first_core();
  auto dxa = DxaCore::Create("dxa_ooo_test", nullptr);

  SimPlatform::instance().reset();

  constexpr uint32_t slot = 0;
  constexpr uint32_t line_size = VX_CFG_L1_LINE_SIZE;
  constexpr uint32_t transfer_size = 3 * line_size;
  constexpr uint32_t gmem_base = 16 * line_size;
  constexpr uint32_t smem_base = 8 * line_size;
  constexpr uint32_t dcr = VX_DCR_DXA_DESC_BASE + slot * VX_DCR_DXA_DESC_STRIDE;
  dxa->dcr_write(dcr + VX_DCR_DXA_DESC_BASE_LO_OFF, gmem_base);
  dxa->dcr_write(dcr + VX_DCR_DXA_DESC_BASE_HI_OFF, 0);
  dxa->dcr_write(dcr + VX_DCR_DXA_DESC_SIZE0_OFF, transfer_size);
  dxa->dcr_write(dcr + VX_DCR_DXA_DESC_META_OFF, 1);
  dxa->dcr_write(dcr + VX_DCR_DXA_DESC_ESTRIDE0_OFF, 1);
  dxa->dcr_write(dcr + VX_DCR_DXA_DESC_TILESIZE01_OFF, transfer_size);

  DxaReq req{};
  req.core = core;
  req.uuid = 1;
  req.desc_slot = slot;
  req.cta_mask = 1;
  req.smem_addr = smem_base;
  dxa->dxa_req_in.at(0).send(req);

  std::vector<MemReq> reads;
  for (uint32_t i = 0; i < 100 && reads.size() < 3; ++i) {
    tick();
    auto &output = dxa->gmem_req_out.at(0);
    if (!output.empty()) {
      reads.push_back(output.peek());
      output.pop();
    }
  }

  if (reads.size() != 3)
    fail("expected exactly three GMEM reads");
  if (reads.at(0).addr != gmem_base ||
      reads.at(1).addr != gmem_base + line_size ||
      reads.at(2).addr != gmem_base + 2 * line_size)
    fail("unexpected GMEM request addresses");

  // C is the last work item. Even if it returns first, its completion write
  // must remain deferred while A and B are outstanding.
  send_response(dxa.get(), reads.at(2), 0xcc);
  tick(8);
  if (!dxa->lmem_req_out.at(0).empty())
    fail("last response drained before older outstanding requests");

  // B is ready and is not the last work item, so it must bypass A.
  send_response(dxa.get(), reads.at(1), 0xbb);
  check_write(wait_for_lmem_write(dxa.get()), smem_base + line_size, 0xbb,
              false);

  // Once A completes, the previously deferred C response must drain last.
  send_response(dxa.get(), reads.at(0), 0xaa);
  check_write(wait_for_lmem_write(dxa.get()), smem_base, 0xaa, false);
  check_write(wait_for_lmem_write(dxa.get()), smem_base + 2 * line_size, 0xcc,
              true);

  tick(8);
  check_sparse_mask(dxa.get(), core, 0x5);
  check_sparse_mask(dxa.get(), core, 0xa);
  check_sparse_mask(dxa.get(), core, 0xf);

  std::cout << "PASSED" << std::endl;
  return 0;
}
