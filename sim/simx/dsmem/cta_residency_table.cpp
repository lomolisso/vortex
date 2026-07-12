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

#include "cta_residency_table.h"
#include <cassert>

using namespace vortex;

CtaResidencyTable::CtaResidencyTable(uint32_t num_entries)
  : entries_(num_entries)
{}

void CtaResidencyTable::reset() {
  for (auto& e : entries_) {
    e = Entry{};
  }
}

const CtaResidencyTable::Entry* CtaResidencyTable::find(uint32_t cluster_id) const {
  for (auto& e : entries_) {
    if (e.valid && e.cluster_id == cluster_id)
      return &e;
  }
  return nullptr;
}

CtaResidencyTable::Entry* CtaResidencyTable::find(uint32_t cluster_id) {
  return const_cast<Entry*>(const_cast<const CtaResidencyTable*>(this)->find(cluster_id));
}

bool CtaResidencyTable::can_admit(uint32_t cluster_id) const {
  if (this->find(cluster_id) != nullptr)
    return true;
  for (auto& e : entries_) {
    if (!e.valid)
      return true;
  }
  return false;
}

void CtaResidencyTable::admit(uint32_t cluster_id, uint32_t rank, uint32_t num_ranks,
                              uint32_t core_id, uint32_t cta_slot, uint64_t lmem_base) {
  auto* entry = this->find(cluster_id);
  if (entry == nullptr) {
    for (auto& e : entries_) {
      if (!e.valid) {
        entry = &e;
        break;
      }
    }
    assert(entry != nullptr && "residency table full; gate must check can_admit()");
    entry->valid      = true;
    entry->cluster_id = cluster_id;
    entry->num_ranks  = num_ranks;
    entry->admitted   = 0;
    entry->retired    = 0;
    entry->ranks.assign(num_ranks, RankSlot{});
  }
  assert(rank < entry->num_ranks);
  assert(!entry->ranks[rank].valid);
  entry->ranks[rank] = RankSlot{true, core_id, cta_slot, lmem_base};
  ++entry->admitted;
}

bool CtaResidencyTable::fully_admitted(uint32_t cluster_id) const {
  auto* entry = this->find(cluster_id);
  return entry != nullptr && entry->admitted == entry->num_ranks;
}

bool CtaResidencyTable::retire(uint32_t cluster_id, uint32_t rank) {
  auto* entry = this->find(cluster_id);
  assert(entry != nullptr && rank < entry->num_ranks && entry->ranks[rank].valid);
  entry->ranks[rank].valid = false;
  ++entry->retired;
  if (entry->retired == entry->num_ranks) {
    assert(entry->admitted == entry->num_ranks);
    *entry = Entry{};
    return true;
  }
  return false;
}

const CtaResidencyTable::RankSlot* CtaResidencyTable::resolve(uint32_t cluster_id, uint32_t rank) const {
  auto* entry = this->find(cluster_id);
  if (entry == nullptr || rank >= entry->num_ranks || !entry->ranks[rank].valid)
    return nullptr;
  return &entry->ranks[rank];
}

uint64_t CtaResidencyTable::participant_mask(uint32_t cluster_id) const {
  auto* entry = this->find(cluster_id);
  if (entry == nullptr)
    return 0;
  uint64_t mask = 0;
  for (auto& r : entry->ranks) {
    if (r.valid)
      mask |= (1ull << r.core_id);
  }
  return mask;
}

uint32_t CtaResidencyTable::num_ranks(uint32_t cluster_id) const {
  auto* entry = this->find(cluster_id);
  return entry ? entry->num_ranks : 0;
}

///////////////////////////////////////////////////////////////////////////////
// CtaReplica — per-SM read cache filled by the master's broadcast.

CtaReplica::CtaReplica(uint32_t num_entries)
  : entries_(num_entries)
{}

void CtaReplica::reset() {
  for (auto& e : entries_) {
    e = Entry{};
  }
}

const CtaReplica::Entry* CtaReplica::find(uint32_t cluster_id) const {
  for (auto& e : entries_) {
    if (e.valid && e.cluster_id == cluster_id)
      return &e;
  }
  return nullptr;
}

CtaReplica::Entry* CtaReplica::find(uint32_t cluster_id) {
  return const_cast<Entry*>(const_cast<const CtaReplica*>(this)->find(cluster_id));
}

void CtaReplica::install(uint32_t cluster_id, uint32_t rank, uint32_t num_ranks,
                         const RankSlot& slot) {
  auto* entry = this->find(cluster_id);
  if (entry == nullptr) {
    for (auto& e : entries_) {
      if (!e.valid) {
        entry = &e;
        break;
      }
    }
    assert(entry != nullptr && "cta replica full");
    entry->valid      = true;
    entry->cluster_id = cluster_id;
    entry->num_ranks  = num_ranks;
    entry->ranks.assign(num_ranks, RankSlot{});
  }
  assert(rank < entry->num_ranks);
  entry->ranks[rank] = slot;
}

void CtaReplica::evict(uint32_t cluster_id) {
  auto* entry = this->find(cluster_id);
  if (entry != nullptr)
    *entry = Entry{};
}

const CtaReplica::RankSlot* CtaReplica::resolve(uint32_t cluster_id, uint32_t rank) const {
  auto* entry = this->find(cluster_id);
  if (entry == nullptr || rank >= entry->num_ranks || !entry->ranks[rank].valid)
    return nullptr;
  return &entry->ranks[rank];
}
