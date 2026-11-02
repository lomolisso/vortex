// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// ============================================================================
// runtime_helpers.cpp — vortex2.h utility entry points.
//
// These wrap common multi-call patterns (occupancy computation) so user
// code calling vortex2.h doesn't reimplement them. All implementations
// call only public vortex2.h primitives.
// ============================================================================

#include <vortex2.h>
#include "vortex2_internal.h"
#include <dlmem_ref.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

extern "C" vx_result_t vx_device_max_occupancy_grid(vx_device_h dev,
                                                    uint32_t ndim,
                                                    const uint32_t* global_dim,
                                                    uint32_t* grid_out,
                                                    uint32_t* block_out) {
    if (!dev || ndim == 0 || ndim > 3 || !global_dim ||
        !grid_out || !block_out) return VX_ERR_INVALID_VALUE;

    uint64_t num_threads = 0, num_warps = 0;
    auto r = vx_device_query(dev, VX_CAPS_NUM_THREADS, &num_threads);
    if (r != VX_SUCCESS) return r;
    r = vx_device_query(dev, VX_CAPS_NUM_WARPS, &num_warps);
    if (r != VX_SUCCESS) return r;

    // One block fills a core: NUM_THREADS lanes on axis 0, NUM_WARPS warps on
    // axis 1. A 1D launch has no axis 1, so fold the warp count onto axis 0 —
    // otherwise the block is a single warp (1/NUM_WARPS of the core), defeating
    // the maximum-occupancy contract. The launch path maps a >warp-size axis 0
    // onto multiple warps (see prepare_kernel_launch_params warp stepping).
    uint64_t auto_block[3];
    if (ndim == 1) {
        auto_block[0] = num_threads * num_warps;
        auto_block[1] = 1;
        auto_block[2] = 1;
    } else {
        auto_block[0] = num_threads;
        auto_block[1] = num_warps;
        auto_block[2] = 1;
    }
    for (uint32_t i = 0; i < ndim; ++i) {
        block_out[i] = (uint32_t)auto_block[i];
        grid_out[i]  = (uint32_t)((global_dim[i] + block_out[i] - 1) / block_out[i]);
    }
    return VX_SUCCESS;
}

namespace vx {

vx_result_t cluster_info(vx_device_h dev, const vx_launch_info_t* info, ClusterInfo* out) {
    ClusterInfo ci = {};
    ci.policy = VX_CLUSTER_POLICY_DEFAULT;
    if (info->struct_size >= offsetof(vx_launch_info_t, cluster_policy) + sizeof(info->cluster_policy))
        ci.policy = static_cast<uint32_t>(info->cluster_policy);
    if (info->struct_size >= offsetof(vx_launch_info_t, cluster_flags) + sizeof(info->cluster_flags))
        ci.flags = info->cluster_flags;
    if (ci.policy > VX_CLUSTER_POLICY_SINGLE_CORE) return VX_ERR_INVALID_VALUE;
    if (ci.flags & ~VX_CLUSTER_FLAG_NON_PORTABLE) return VX_ERR_INVALID_VALUE;

    uint64_t isa = 0, line = 0;
    auto r = vx_device_query(dev, VX_CAPS_ISA_FLAGS, &isa);
    if (r != VX_SUCCESS) return r;
    ci.dlmem = (isa & VX_ISA_EXT_DLMEM) != 0;
    if (ci.policy == VX_CLUSTER_POLICY_SPREAD && !ci.dlmem) return VX_ERR_NOT_SUPPORTED;
    r = vx_device_query(dev, VX_CAPS_NUM_WARPS, &ci.nw);
    if (r == VX_SUCCESS) r = vx_device_query(dev, VX_CAPS_NUM_THREADS, &ci.nt);
    if (r == VX_SUCCESS) r = vx_device_query(dev, VX_CAPS_NUM_CORES, &ci.cores);
    if (r == VX_SUCCESS) r = vx_device_query(dev, VX_CAPS_NUM_CLUSTERS, &ci.num_clusters);
    if (r == VX_SUCCESS) r = vx_device_query(dev, VX_CAPS_DLMEM_MAX_CLUSTER, &ci.max_cluster);
    if (r == VX_SUCCESS) r = vx_device_query(dev, VX_CAPS_LOCAL_MEM_SIZE, &ci.lmem_cap);
    if (r == VX_SUCCESS) r = vx_device_query(dev, VX_CAPS_CACHE_LINE_SIZE, &line);
    if (r != VX_SUCCESS) return r;
    ci.ncc = ci.cores / std::max<uint64_t>(ci.num_clusters, 1);

    uint64_t block_size = 1;
    ci.k = 1;
    for (uint32_t i = 0; i < info->ndim && i < 3; ++i) {
        ci.explicit_cluster |= (info->cluster_dim[i] != 0);
        ci.k *= std::max<uint32_t>(info->cluster_dim[i], 1);
        block_size *= info->block_dim[i];
    }
    ci.spread = ci.dlmem && (ci.policy == VX_CLUSTER_POLICY_SPREAD
                          || (ci.policy == VX_CLUSTER_POLICY_DEFAULT && ci.k > 1));
    ci.warps = std::max<uint64_t>((block_size + ci.nt - 1) / std::max<uint64_t>(ci.nt, 1), 1);
    // The dispatcher's slot pitch: the request rounded up to a cache line, or
    // to a local-memory row in a DXA build.
    uint64_t align = line;
#if VX_CFG_EXT_DXA_ENABLED
    align = std::max<uint64_t>(align, (uint64_t)VX_CFG_LMEM_NUM_BANKS * (VX_CFG_XLEN / 8));
#endif
    const uint64_t stride = (info->lmem_size + align - 1) / align * align;
    ci.slots = (stride == 0) ? ci.nw : std::min(ci.nw, ci.lmem_cap / stride);
    ci.per_core = (ci.warps > ci.nw) ? 0 : std::min(ci.slots, ci.nw / ci.warps);
    *out = ci;
    return VX_SUCCESS;
}

} // namespace vx

namespace {

// SPREAD clusters of k blocks one hardware cluster holds at once, on an idle
// one: the gang placement, run on an empty mirror until it waits.
uint64_t spread_fit(const vx::ClusterInfo& ci, uint64_t k) {
    if (k == 0 || k > ci.max_cluster || k > ci.ncc || ci.per_core == 0)
        return 0;
    vortex::dlmem_ref::Cfg cfg = {};
    cfg.ncc         = (uint32_t)ci.ncc;
    cfg.nw          = (uint32_t)ci.nw;
    cfg.nt          = (uint32_t)ci.nt;
    cfg.max_cluster = (uint32_t)ci.max_cluster;
    cfg.lmem_log    = vortex::dlmem_ref::clog2(ci.lmem_cap);
    cfg.num_gangs   = 1;
    cfg.socket_size = 1;
    cfg.num_sockets = 1;
    vortex::dlmem_ref::Mirror m;
    vortex::dlmem_ref::reset(cfg, m);
    std::vector<vortex::dlmem_ref::Peer> peers;
    uint64_t n = 0;
    while (vortex::dlmem_ref::place(cfg, m, (uint32_t)k, (uint32_t)ci.warps, (uint32_t)ci.slots, peers))
        ++n;
    return n;
}

// Single-core clusters of k blocks one core holds at once.
uint64_t single_core_fit(const vx::ClusterInfo& ci, uint64_t k) {
    if (k == 0 || k > std::max(ci.nw, ci.max_cluster))
        return 0;
    return std::min(ci.nw / (k * ci.warps), ci.slots / k);
}

} // namespace

extern "C" vx_result_t vx_occupancy_max_active_clusters(vx_device_h dev,
                                                        const vx_launch_info_t* info,
                                                        uint32_t* out) {
    if (!dev || !info || !out) return VX_ERR_INVALID_VALUE;
    if (info->struct_size < offsetof(vx_launch_info_t, reserved0) || info->ndim > 3)
        return VX_ERR_INVALID_INFO;
    vx::ClusterInfo ci;
    auto r = vx::cluster_info(dev, info, &ci);
    if (r != VX_SUCCESS) return r;
    if (ci.dlmem && ci.k > VX_CLUSTER_MAX_PORTABLE && !(ci.flags & VX_CLUSTER_FLAG_NON_PORTABLE))
        return VX_ERR_INVALID_VALUE;
    const uint64_t n = ci.spread ? spread_fit(ci, ci.k) * ci.num_clusters
                                 : single_core_fit(ci, ci.k) * ci.cores;
    *out = (uint32_t)n;
    return VX_SUCCESS;
}

extern "C" vx_result_t vx_occupancy_max_potential_cluster_size(vx_device_h dev,
                                                               const vx_launch_info_t* info,
                                                               uint32_t* out) {
    if (!dev || !info || !out) return VX_ERR_INVALID_VALUE;
    if (info->struct_size < offsetof(vx_launch_info_t, reserved0) || info->ndim > 3)
        return VX_ERR_INVALID_INFO;
    vx::ClusterInfo ci;
    auto r = vx::cluster_info(dev, info, &ci);
    if (r != VX_SUCCESS) return r;
    uint64_t best = 0;
    const bool portable = ci.dlmem && !(ci.flags & VX_CLUSTER_FLAG_NON_PORTABLE);
    if (ci.dlmem && ci.policy != VX_CLUSTER_POLICY_SINGLE_CORE) {
        const uint64_t limit = portable ? std::min<uint64_t>(ci.max_cluster, VX_CLUSTER_MAX_PORTABLE)
                                        : ci.max_cluster;
        for (uint64_t k = limit; k >= 2 && best == 0; --k) {
            if (spread_fit(ci, k) != 0)
                best = k;
        }
        // A cluster of one block takes the normal dispatch path.
        if (best == 0 && ci.per_core != 0)
            best = 1;
    } else {
        uint64_t limit = std::max(ci.nw, ci.max_cluster);
        if (portable)
            limit = std::min<uint64_t>(limit, VX_CLUSTER_MAX_PORTABLE);
        for (uint64_t k = limit; k >= 1 && best == 0; --k) {
            if (single_core_fit(ci, k) != 0)
                best = k;
        }
    }
    *out = (uint32_t)best;
    return VX_SUCCESS;
}
