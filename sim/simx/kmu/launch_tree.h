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

#pragma once

#include <cassert>
#include <cstdint>
#include <deque>
#include <utility>
#include <vector>
#include "kmu.h"

namespace vortex {

// The launch bus from a launch source down to its endpoints, built like the
// RTL's chain of VX_kmu_bus_arb fan-outs. A level with several outputs gives
// each one a two-entry skid (OUT_BUF=3) whose ready is registered, and sends a
// compute message to the nearest ready output from its round-robin pointer
// (a pinned one by its destination); a message keeps its output until eop. A
// level with one output is a wire unless it is registered too.
//
// The skids of the last level sit in the endpoints: the tree counts their
// entries and learns of a pop from the endpoint's credit one cycle later,
// which is when the RTL's registered ready sees it. A beat entering one is
// handed to the sink, which must make it visible to the endpoint the next
// cycle, as the skid's output register does.
//
// With no registered level at all the RTL path is combinational. The model
// approximates it with one two-entry registered stage, since the endpoint is
// in another domain: the first beat arrives a cycle late, and later beats keep
// the RTL's pace.
class LaunchTree {
public:
  struct Beat {
    kmu_req_t req;
    bool      eop;
    bool      pinned;
    uint32_t  dest;   // the endpoint of a pinned beat
  };

  struct Level {
    uint32_t outputs;
    bool     skid;
    Level(uint32_t outputs) : outputs(outputs), skid(outputs > 1) {}
    Level(uint32_t outputs, bool skid) : outputs(outputs), skid(skid) {}
  };

  // The levels from the source down.
  explicit LaunchTree(const std::vector<Level>& levels)
    : num_leaves_(1) {
    for (const auto& level : levels) {
      assert(level.outputs != 0);
      num_leaves_ *= level.outputs;
      if (level.skid) {
        radix_.push_back(level.outputs);
      }
    }
    stride_.resize(radix_.size());
    uint32_t below = num_leaves_;
    uint32_t nodes = 1;
    for (size_t l = 0; l < radix_.size(); ++l) {
      below /= radix_[l];
      stride_[l] = below;
      nodes_.emplace_back(nodes);
      nodes *= radix_[l];
      if (l + 1 < radix_.size()) {
        skids_.emplace_back(nodes);
      }
    }
    leaves_.resize(num_leaves_);
  }

  uint32_t num_leaves() const { return num_leaves_; }

  void reset() {
    for (auto& level : skids_) {
      for (auto& s : level) {
        s = Skid{};
      }
    }
    for (auto& level : nodes_) {
      for (auto& n : level) {
        n = Node{};
      }
    }
    for (auto& s : leaves_) {
      s = Skid{};
    }
  }

  // The endpoint popped a beat of leaf `leaf`.
  void credit(uint32_t leaf) {
    auto& s = leaves_.at(leaf);
    assert(s.count != 0);
    --s.count;
  }

  // Start of a cycle: every skid's ready is its occupancy now.
  void begin_cycle() {
    for (auto& level : skids_) {
      for (auto& s : level) {
        s.ready = (s.q.size() < SKID_SIZE);
      }
    }
    for (auto& s : leaves_) {
      s.ready = (s.count < SKID_SIZE);
    }
  }

  // Every internal skid offers its head one level down.
  template <typename Sink>
  void advance(uint64_t now, Sink&& sink) {
    for (size_t l = 0; l < skids_.size(); ++l) {
      for (uint32_t i = 0; i < skids_[l].size(); ++i) {
        this->drain(skids_[l][i], l + 1, i, now, sink);
      }
    }
  }

  // Whether the source can fire `beat` this cycle.
  bool ready(const Beat& beat) const {
    if (radix_.empty()) {
      return leaves_[0].ready;
    }
    return this->out_ready(0, this->select(0, 0, beat));
  }

  template <typename Sink>
  void push(const Beat& beat, uint64_t now, Sink&& sink) {
    assert(this->ready(beat));
    if (radix_.empty()) {
      ++leaves_[0].count;
      sink(0u, beat.req);
    } else {
      this->fire(0, 0, this->select(0, 0, beat), beat, now, sink);
    }
  }

  // No beat waits in an internal skid.
  bool empty() const {
    for (auto& level : skids_) {
      for (auto& s : level) {
        if (!s.q.empty()) {
          return false;
        }
      }
    }
    return true;
  }

private:
  static constexpr uint32_t SKID_SIZE = 2;

  struct Skid {
    std::deque<std::pair<Beat, uint64_t>> q;  // internal: beats and the cycle each shows at the output
    uint32_t count = 0;                       // leaf: beats sent, not yet popped
    bool     ready = true;
  };

  struct Node {
    uint32_t rr       = 0;
    bool     lock     = false;
    uint32_t lock_sel = 0;
  };

  uint32_t select(size_t l, uint32_t n, const Beat& beat) const {
    const auto& node = nodes_[l][n];
    uint32_t num = radix_[l];
    if (node.lock) {
      return node.lock_sel;
    }
    if (beat.pinned) {
      return (beat.dest / stride_[l]) % num;
    }
    for (uint32_t k = 0; k < num; ++k) {
      uint32_t o = (node.rr + k) % num;
      if (this->out_ready(l, n * num + o)) {
        return o;
      }
    }
    return node.rr;
  }

  bool out_ready(size_t l, uint32_t o) const {
    return (l + 1 == radix_.size()) ? leaves_[o].ready : skids_[l][o].ready;
  }

  template <typename Sink>
  void fire(size_t l, uint32_t n, uint32_t sel, const Beat& beat, uint64_t now, Sink&& sink) {
    uint32_t num = radix_[l];
    uint32_t o = n * num + sel;
    if (l + 1 == radix_.size()) {
      ++leaves_[o].count;
      sink(o, beat.req);
    } else {
      skids_[l][o].q.emplace_back(beat, now + 1);
    }
    auto& node = nodes_[l][n];
    node.lock     = !beat.eop;
    node.lock_sel = sel;
    if (beat.eop) {
      node.rr = (sel + 1) % num;
    }
  }

  template <typename Sink>
  void drain(Skid& s, size_t l, uint32_t n, uint64_t now, Sink&& sink) {
    if (s.q.empty() || s.q.front().second > now) {
      return;
    }
    const Beat& beat = s.q.front().first;
    uint32_t sel = this->select(l, n, beat);
    if (!this->out_ready(l, n * radix_[l] + sel)) {
      return;
    }
    this->fire(l, n, sel, beat, now, sink);
    s.q.pop_front();
  }

  std::vector<uint32_t> radix_;               // outputs of each registered level
  std::vector<uint32_t> stride_;              // endpoints under one output of the level
  uint32_t num_leaves_;
  std::vector<std::vector<Node>> nodes_;      // per multi-output level
  std::vector<std::vector<Skid>> skids_;      // the internal outputs of every level but the last
  std::vector<Skid> leaves_;
};

#ifdef VX_CFG_EXT_DLMEM_ENABLE
// A cluster's launch fan-out to its cores, below the gang unit: the unit's
// output skid drains into it, and its lanes are the cores' last skids.
class LaunchFanout : public SimObject<LaunchFanout> {
public:
  SimChannel<kmu_req_t> bus_in;
  SimChannel<uint8_t>   credit_out;
  std::vector<SimChannel<kmu_req_t>> bus_out;
  std::vector<SimChannel<uint8_t>>   credit_in;

  LaunchFanout(const SimContext& ctx, const char* name, uint32_t num_sockets, uint32_t socket_size)
    : SimObject<LaunchFanout>(ctx, name)
    , bus_in(this)
    , credit_out(this)
    , bus_out(num_sockets * socket_size, this)
    , credit_in(num_sockets * socket_size, this)
    , tree_({LaunchTree::Level(num_sockets), LaunchTree::Level(socket_size)})
  {}

  bool busy() const {
    return (bus_in.size() != 0) || !tree_.empty();
  }

protected:
  void on_reset() {
    tree_.reset();
  }

  void on_tick() {
    uint64_t now = SimPlatform::instance().cycles();
    for (uint32_t c = 0, n = bus_out.size(); c < n; ++c) {
      while (!credit_in.at(c).empty()) {
        credit_in.at(c).pop();
        tree_.credit(c);
      }
    }
    auto sink = [this](uint32_t lane, const kmu_req_t& req) {
      bus_out.at(lane).send(req, 0);
    };
    tree_.begin_cycle();
    tree_.advance(now, sink);
    if (!bus_in.empty()) {
      // A placed block is a whole message; a same-core cluster ends at its last rank.
      const auto& req = bus_in.peek();
      uint32_t k = req.cluster_dim[0] * req.cluster_dim[1] * req.cluster_dim[2];
      LaunchTree::Beat beat{req, req.pinned || (req.cluster_rank + 1 >= k), req.pinned, req.dest_core};
      if (tree_.ready(beat)) {
        tree_.push(beat, now, sink);
        bus_in.pop();
        credit_out.send(1, 1);
      }
    }
    // size() counts a packet still in flight: its delivery does not wake us.
    bool credits = false;
    for (const auto& c : credit_in) {
      credits = credits || (c.size() != 0);
    }
    if (bus_in.size() == 0 && !credits && tree_.empty()) {
      this->tick_sleep();
    }
  }

private:
  LaunchTree tree_;

  friend class SimObject<LaunchFanout>;
};
#endif

} // namespace vortex
