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

#include "kmu.h"
#include <cassert>
#include "constants.h"
#include "debug.h"
#include "launch_tree.h"

using namespace vortex;

namespace {
using Level = LaunchTree::Level;
// The RTL's launch fan-out levels below the KMU, device first.
std::vector<Level> launch_levels() {
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  // Each cluster's gang unit is an endpoint; it owns the levels below.
  return {Level(VX_CFG_NUM_CLUSTERS)};
#elif defined(VX_CFG_EXT_RASTER_ENABLE)
  // The cluster registers the stream it merges with its raster engines.
  return {Level(VX_CFG_NUM_CLUSTERS), Level(1, true), Level(NUM_SOCKETS), Level(VX_CFG_SOCKET_SIZE)};
#else
  return {Level(VX_CFG_NUM_CLUSTERS), Level(NUM_SOCKETS), Level(VX_CFG_SOCKET_SIZE)};
#endif
}

} // namespace

Kmu::Kmu(const SimContext& ctx, const char* name, uint32_t num_lanes)
  : SimObject<Kmu>(ctx, name)
  , bus_out(num_lanes, this)
  , credit_in(num_lanes, this)
  , PC_(0)
  , entry_(0)
  , param_(0)
  , lmem_size_(0)
  , block_size_(0)
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  , policy_(KMU_POLICY_DEFAULT)
#endif
  , running_(false)
  , cta_id_(0)
  , ctx_id_(0)
  , tree_(new LaunchTree(launch_levels()))
{
  assert(tree_->num_leaves() == num_lanes);
  block_dim_[0]    = block_dim_[1]    = block_dim_[2]    = 1;
  grid_dim_[0]     = grid_dim_[1]     = grid_dim_[2]     = 1;
  cluster_dim_[0]  = cluster_dim_[1]  = cluster_dim_[2]  = 1;
  warp_step_[0]    = warp_step_[1]    = warp_step_[2]    = 1;
  group_origin_[0] = group_origin_[1] = group_origin_[2] = 0;
  intra_offset_[0] = intra_offset_[1] = intra_offset_[2] = 0;
}

Kmu::~Kmu() {}

bool Kmu::running() const {
  return running_ || !tree_->empty();
}

void Kmu::on_reset() {
  // Reset only the per-run progression state. The kernel descriptor (PC,
  // param, dims, block_size, lmem_size, warp_step, cluster_dim) is set
  // by dcr_write() before run() and must persist across
  // SimPlatform::on_reset().
  // ctx_id_ is excluded: it is what distinguishes one launch from the next, so
  // re-zeroing it here would pin it to the same value on every launch and
  // destroy the distinction. It is free-running from device reset, and
  // zero-initialized at construction.
  running_         = false;
  cta_id_          = 0;
  group_origin_[0] = group_origin_[1] = group_origin_[2] = 0;
  intra_offset_[0] = intra_offset_[1] = intra_offset_[2] = 0;
  tree_->reset();
}

void Kmu::dcr_write(uint32_t addr, uint32_t value) {
  switch (addr) {
  case VX_DCR_KMU_STARTUP_ADDR0: PC_    = (PC_    & ~uint64_t(0xFFFFFFFF)) | value; break;
  case VX_DCR_KMU_STARTUP_ADDR1: PC_    = (PC_    &  uint64_t(0xFFFFFFFF)) | (uint64_t(value) << 32); break;
  case VX_DCR_KMU_KERNEL_ENTRY0: entry_ = (entry_ & ~uint64_t(0xFFFFFFFF)) | value; break;
  case VX_DCR_KMU_KERNEL_ENTRY1: entry_ = (entry_ &  uint64_t(0xFFFFFFFF)) | (uint64_t(value) << 32); break;
  case VX_DCR_KMU_STARTUP_ARG0:  param_ = (param_ & ~uint64_t(0xFFFFFFFF)) | value; break;
  case VX_DCR_KMU_STARTUP_ARG1:  param_ = (param_ &  uint64_t(0xFFFFFFFF)) | (uint64_t(value) << 32); break;
  case VX_DCR_KMU_BLOCK_DIM_X:   block_dim_[0] = value; break;
  case VX_DCR_KMU_BLOCK_DIM_Y:   block_dim_[1] = value; break;
  case VX_DCR_KMU_BLOCK_DIM_Z:   block_dim_[2] = value; break;
  case VX_DCR_KMU_GRID_DIM_X:    grid_dim_[0]  = value; break;
  case VX_DCR_KMU_GRID_DIM_Y:    grid_dim_[1]  = value; break;
  case VX_DCR_KMU_GRID_DIM_Z:    grid_dim_[2]  = value; break;
  case VX_DCR_KMU_LMEM_SIZE:     lmem_size_    = value; break;
  case VX_DCR_KMU_BLOCK_SIZE:    block_size_   = value; break;
  case VX_DCR_KMU_WARP_STEP_X:   warp_step_[0] = value; break;
  case VX_DCR_KMU_WARP_STEP_Y:   warp_step_[1] = value; break;
  case VX_DCR_KMU_WARP_STEP_Z:   warp_step_[2] = value; break;
  case VX_DCR_KMU_CLUSTER_DIM_X: cluster_dim_[0] = value; break;
  case VX_DCR_KMU_CLUSTER_DIM_Y: cluster_dim_[1] = value; break;
  case VX_DCR_KMU_CLUSTER_DIM_Z: cluster_dim_[2] = value; break;
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  case VX_DCR_KMU_CLUSTER_POLICY: policy_ = value & 1; break;
#endif
  default: break;
  }
}

void Kmu::start() {
  running_ = (block_size_ > 0)
           && (grid_dim_[0] > 0)
           && (grid_dim_[1] > 0)
           && (grid_dim_[2] > 0)
           && (cluster_dim_[0] > 0)
           && (cluster_dim_[1] > 0)
           && (cluster_dim_[2] > 0);
  ctx_id_ += 1;
  if (running_) {
    cta_id_          = 0;
    group_origin_[0] = group_origin_[1] = group_origin_[2] = 0;
    intra_offset_[0] = intra_offset_[1] = intra_offset_[2] = 0;
    this->tick_wake();
  }
}

void Kmu::on_tick() {
  uint64_t now = SimPlatform::instance().cycles();
  for (uint32_t c = 0, n = bus_out.size(); c < n; ++c) {
    auto& credits = credit_in.at(c);
    while (!credits.empty()) {
      credits.pop();
      tree_->credit(c);
    }
  }
  auto sink = [this](uint32_t lane, const kmu_req_t& req) {
    DT(2, this->name() << " launch: lane=" << lane << ", cta_id=" << req.cta_id);
    bus_out.at(lane).send(req, 0);
  };
  tree_->begin_cycle();
  tree_->advance(now, sink);
  if (running_) {
    // One CTA per cycle; a cluster is one message, its last member the eop.
    LaunchTree::Beat beat{};
    beat.eop = ((intra_offset_[0] + 1) == cluster_dim_[0])
            && ((intra_offset_[1] + 1) == cluster_dim_[1])
            && ((intra_offset_[2] + 1) == cluster_dim_[2]);
    if (tree_->ready(beat)) {
      this->next_cta(&beat.req);
      DT(1, this->name() << " cta-fire: cta_id=" << beat.req.cta_id << ", block_idx=["
            << beat.req.block_idx[0] << "," << beat.req.block_idx[1] << "," << beat.req.block_idx[2] << "]");
      tree_->push(beat, now, sink);
    }
  }
  if (!running_ && tree_->empty() && this->credits_idle()) {
    this->tick_sleep();
  }
}

void Kmu::next_cta(kmu_req_t* req) {
  assert(running_);

  // Effective block_idx = group_origin + intra_offset.
  uint32_t block_idx[3] = {
    group_origin_[0] + intra_offset_[0],
    group_origin_[1] + intra_offset_[1],
    group_origin_[2] + intra_offset_[2],
  };

  req->PC           = PC_;
  req->entry        = entry_;
  req->param        = param_;
  req->cta_id       = cta_id_;
  req->ctx_id       = ctx_id_;
  req->block_idx[0] = block_idx[0];
  req->block_idx[1] = block_idx[1];
  req->block_idx[2] = block_idx[2];
  req->block_dim[0] = block_dim_[0];
  req->block_dim[1] = block_dim_[1];
  req->block_dim[2] = block_dim_[2];
  req->grid_dim[0]  = grid_dim_[0];
  req->grid_dim[1]  = grid_dim_[1];
  req->grid_dim[2]  = grid_dim_[2];
  req->lmem_size    = lmem_size_;
  req->block_size   = block_size_;
  req->warp_step[0] = warp_step_[0];
  req->warp_step[1] = warp_step_[1];
  req->warp_step[2] = warp_step_[2];
  req->cluster_dim[0] = cluster_dim_[0];
  req->cluster_dim[1] = cluster_dim_[1];
  req->cluster_dim[2] = cluster_dim_[2];
  req->is_first_of_cluster = (intra_offset_[0] == 0)
                          && (intra_offset_[1] == 0)
                          && (intra_offset_[2] == 0);
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  // Walk position inside the cluster (x fastest), the order the RTL counts ranks in.
  req->cluster_rank = intra_offset_[0]
                    + cluster_dim_[0] * (intra_offset_[1] + cluster_dim_[1] * intra_offset_[2]);
  req->policy       = policy_;
#endif

  // Advance the intra-cluster offset first (fills the cluster), then the
  // group_origin in (X, Y, Z) order when the inner loop wraps.
  ++cta_id_;
  bool intra_x_wrap = (intra_offset_[0] + 1 == cluster_dim_[0]);
  if (!intra_x_wrap) {
    intra_offset_[0]++;
  } else {
    intra_offset_[0] = 0;
    bool intra_y_wrap = (intra_offset_[1] + 1 == cluster_dim_[1]);
    if (!intra_y_wrap) {
      intra_offset_[1]++;
    } else {
      intra_offset_[1] = 0;
      bool intra_z_wrap = (intra_offset_[2] + 1 == cluster_dim_[2]);
      if (!intra_z_wrap) {
        intra_offset_[2]++;
      } else {
        intra_offset_[2] = 0;
        // Cluster complete — advance group_origin in (X, Y, Z) order.
        uint32_t ox = group_origin_[0] + cluster_dim_[0];
        if (ox == grid_dim_[0]) {
          group_origin_[0] = 0;
          uint32_t oy = group_origin_[1] + cluster_dim_[1];
          if (oy == grid_dim_[1]) {
            group_origin_[1] = 0;
            uint32_t oz = group_origin_[2] + cluster_dim_[2];
            if (oz == grid_dim_[2]) {
              group_origin_[2] = 0;
              running_ = false;
            } else {
              group_origin_[2] = oz;
            }
          } else {
            group_origin_[1] = oy;
          }
        } else {
          group_origin_[0] = ox;
        }
      }
    }
  }
}

// A credit still in flight is delivered without waking this unit.
bool Kmu::credits_idle() const {
  for (const auto& credits : credit_in) {
    if (credits.size() != 0) {
      return false;
    }
  }
  return true;
}
