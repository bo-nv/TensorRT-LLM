# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Peer-to-peer view matching, compatibility validation and mapper selection.

Everything here is a pure function of two page tables. A rank's page table
carries, per view, the cache manager's shard declaration; comparing two
declarations answers whether the views pair up, whether they are compatible,
who sends, and which mapper moves the bytes. No parallel-configuration
lookups happen at this level.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.base.region import RegionMapperBase
from tensorrt_llm._torch.disaggregation.native.mixers.mappers import (
    HNDShardMapper,
    IntactMapper,
    NHDShardMapper,
    ReplicatedMapper,
    SectionedMapper,
    shard_ratio,
    shards_paired,
)
from tensorrt_llm._torch.disaggregation.resource.page import (
    KVCachePageTable,
    LayerGroup,
    MapperKind,
    PoolView,
)
from tensorrt_llm._torch.disaggregation.resource.utils import (
    get_layer_byte_ranges,
    get_layer_to_layer_group,
    get_pool_view_global_layer_ids,
)

# (layer_group_idx, pool_view_idx)
LGPoolKey = Tuple[int, int]


# ---------------------------------------------------------------------------
# View matching
# ---------------------------------------------------------------------------


def match_pool_views(
    self_pt: Optional[KVCachePageTable], peer_pt: Optional[KVCachePageTable]
) -> Dict[LGPoolKey, LGPoolKey]:
    """Map every local view to the peer view holding the same role class.

    Two-step matching:
    1. Find the peer layer group via global-layer-id overlap, scoped to the
       same token-axis class (hybrid models may list one global layer in
       both a paged and a state group).
    2. Within that peer layer group, pick the unique peer view whose
       ``pool_role`` equals ours and whose global layer ids overlap.

    Layer overlap is required: a peer view with the same role but zero layer
    overlap covers disjoint layers and has nothing to transfer. A local view
    never spans multiple peer layer groups (both sides derive layer grouping
    from the same model config; PP only changes which layers overlap), so a
    multi-group hit is an unsupported topology and raises.
    """
    mapping: Dict[LGPoolKey, LGPoolKey] = {}
    if self_pt is None or peer_pt is None:
        return mapping
    if not self_pt.layer_groups or not peer_pt.layer_groups:
        return mapping

    peer_l2g_cache: Dict[bool, Dict[int, int]] = {}

    def _peer_l2g(has_token_axis: bool) -> Dict[int, int]:
        if has_token_axis not in peer_l2g_cache:
            peer_l2g_cache[has_token_axis] = get_layer_to_layer_group(peer_pt, has_token_axis)
        return peer_l2g_cache[has_token_axis]

    for self_lg_idx, self_lg in enumerate(self_pt.layer_groups):
        for self_pi, self_pv in enumerate(self_lg.pool_views):
            pv_global_ids = get_pool_view_global_layer_ids(self_pv, self_lg)
            if not pv_global_ids:
                continue
            peer_layer_to_group = _peer_l2g(self_lg.has_token_axis)
            peer_lg_indices = {
                peer_layer_to_group[g] for g in pv_global_ids if g in peer_layer_to_group
            }
            if not peer_lg_indices:
                continue
            if len(peer_lg_indices) > 1:
                raise ValueError(
                    "match_pool_views: pool view "
                    f"(lg={self_lg_idx}, pool={self_pi}) spans multiple peer "
                    f"layer groups {sorted(peer_lg_indices)}; mismatched layer "
                    "grouping between peers is not supported"
                )
            peer_lg_idx = next(iter(peer_lg_indices))
            peer_lg = peer_pt.layer_groups[peer_lg_idx]

            self_layer_set = set(pv_global_ids)
            for peer_pi, peer_pv in enumerate(peer_lg.pool_views):
                if peer_pv.pool_role != self_pv.pool_role:
                    continue
                peer_global_ids = get_pool_view_global_layer_ids(peer_pv, peer_lg)
                if not set(peer_global_ids) & self_layer_set:
                    continue
                if peer_pv.mapper_kind != self_pv.mapper_kind:
                    raise ValueError(
                        "match_pool_views: incompatible mapper kinds for pool role "
                        f"{sorted(self_pv.pool_role)} (local={self_pv.mapper_kind.name}, "
                        f"peer={peer_pv.mapper_kind.name}, peer_pool={peer_pi})"
                    )
                mapping[(self_lg_idx, self_pi)] = (peer_lg_idx, peer_pi)
                break
    return mapping


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------


def _view_name(lg_idx: int, pi: int, pv: PoolView) -> str:
    return f"view(lg={lg_idx}, pool={pi}, role={sorted(pv.pool_role)}, kind={pv.mapper_kind.name})"


def _needs_equal_tokens_per_slot(self_pv: PoolView, peer_pv: PoolView) -> bool:
    """NHD and replicated views address bytes inside a slot, so their geometry
    only lines up when both sides use the same tokens_per_slot."""
    exact = (MapperKind.NHD, MapperKind.REPLICATED)
    return self_pv.mapper_kind in exact or peer_pv.mapper_kind in exact


def validate_page_tables(
    self_pt: Optional[KVCachePageTable], peer_pt: Optional[KVCachePageTable]
) -> Dict[LGPoolKey, LGPoolKey]:
    """Reject peers whose declared layouts cannot be lined up with ours.

    Runs once per peer at registration. Raises ``ValueError`` naming the first
    offending view pair; returns the view mapping on success. Checks, per
    matched view pair (plus the two layer groups that own them):

    * same ``mapper_kind`` and element dtype / shape;
    * ``bytes_per_layer * num_shards`` equal on both sides (the TP-aggregated
      size of the role is topology-invariant);
    * divisible shard counts, equal ``shard_unit_bytes``, and per-layer sizes
      that are whole multiples of it;
    * for ``SECTIONED``: equal section count and equal aggregated size per
      section;
    * ``tokens_per_slot``: both ``None`` (per-request state), or equal when a
      view addresses bytes inside a slot (NHD / replicated), or divisible
      otherwise (block boundaries then align every ``lcm`` tokens).

    Views without a peer counterpart (layers the peer does not hold, one-sided
    side caches) are legal and skipped.
    """
    mapping = match_pool_views(self_pt, peer_pt)
    if not mapping:
        return mapping
    assert self_pt is not None and peer_pt is not None

    for (self_lg_idx, self_pi), (peer_lg_idx, peer_pi) in mapping.items():
        self_lg = self_pt.layer_groups[self_lg_idx]
        peer_lg = peer_pt.layer_groups[peer_lg_idx]
        self_pv = self_lg.pool_views[self_pi]
        peer_pv = peer_lg.pool_views[peer_pi]
        where = (
            f"local {_view_name(self_lg_idx, self_pi, self_pv)} vs "
            f"peer {_view_name(peer_lg_idx, peer_pi, peer_pv)}"
        )

        if self_pv.mapper_kind != peer_pv.mapper_kind:
            raise ValueError(f"validate_page_tables: mapper kind differs: {where}")
        self_elem = (self_pv.layout.elem_dtype, self_pv.layout.elem_shape)
        peer_elem = (peer_pv.layout.elem_dtype, peer_pv.layout.elem_shape)
        if self_elem != peer_elem:
            raise ValueError(
                "validate_page_tables: element dtype/shape differs "
                f"(local={self_elem}, peer={peer_elem}): {where}"
            )
        if self_lg.has_token_axis != peer_lg.has_token_axis:
            raise ValueError(f"validate_page_tables: token-axis class differs: {where}")

        _, self_bpl = get_layer_byte_ranges(self_pv)
        _, peer_bpl = get_layer_byte_ranges(peer_pv)
        self_shards = (self_pv.num_shards, self_pv.shard_index)
        peer_shards = (peer_pv.num_shards, peer_pv.shard_index)
        try:
            shard_ratio(self_shards, peer_shards)
        except ValueError as e:
            raise ValueError(f"validate_page_tables: {e}: {where}") from e
        if self_bpl * self_pv.num_shards != peer_bpl * peer_pv.num_shards:
            raise ValueError(
                "validate_page_tables: aggregated region size differs: "
                f"local {self_bpl} bytes x {self_pv.num_shards} shards vs "
                f"peer {peer_bpl} bytes x {peer_pv.num_shards} shards; the per-rank sizes are "
                "inconsistent with a sharded layout across the two sides (state shape/dtype "
                f"differs, or a replicated role is declared as sharded): {where}"
            )
        if (
            self_pv.num_shards != peer_pv.num_shards
            and self_pv.mapper_kind in (MapperKind.HND, MapperKind.NHD)
            and (self_pv.shard_unit_bytes is None or peer_pv.shard_unit_bytes is None)
        ):
            raise ValueError(
                "validate_page_tables: re-splitting across different shard counts requires a "
                "byte-aligned shard_unit_bytes on both sides (sub-byte head slicing is not "
                f"byte-aligned): {where}"
            )
        if self_pv.shard_unit_bytes is not None and peer_pv.shard_unit_bytes is not None:
            if self_pv.shard_unit_bytes != peer_pv.shard_unit_bytes:
                raise ValueError(
                    "validate_page_tables: shard_unit_bytes differs "
                    f"(local={self_pv.shard_unit_bytes}, peer={peer_pv.shard_unit_bytes}): {where}"
                )
        for side, pv, bpl in (("local", self_pv, self_bpl), ("peer", peer_pv, peer_bpl)):
            if pv.shard_unit_bytes is not None and bpl % pv.shard_unit_bytes:
                raise ValueError(
                    f"validate_page_tables: {side} per-layer size {bpl} is not a multiple of "
                    f"shard_unit_bytes {pv.shard_unit_bytes}: {where}"
                )
        if self_pv.mapper_kind == MapperKind.SECTIONED:
            if self_pv.section_bytes is None or peer_pv.section_bytes is None:
                raise ValueError(
                    f"validate_page_tables: SECTIONED view lacks section_bytes: {where}"
                )
            if len(self_pv.section_bytes) != len(peer_pv.section_bytes):
                raise ValueError(
                    "validate_page_tables: section count differs "
                    f"(local={len(self_pv.section_bytes)}, "
                    f"peer={len(peer_pv.section_bytes)}): {where}"
                )
            for i, (s, p) in enumerate(zip(self_pv.section_bytes, peer_pv.section_bytes)):
                if s * self_pv.num_shards != p * peer_pv.num_shards:
                    raise ValueError(
                        f"validate_page_tables: aggregated section_bytes[{i}] differs "
                        f"(local {s} x {self_pv.num_shards} vs "
                        f"peer {p} x {peer_pv.num_shards}): {where}"
                    )

        self_tps, peer_tps = self_lg.tokens_per_slot, peer_lg.tokens_per_slot
        if self_tps is not None and peer_tps is not None and self_tps != peer_tps:
            if _needs_equal_tokens_per_slot(self_pv, peer_pv):
                raise ValueError(
                    "validate_page_tables: tokens_per_slot must match for NHD/replicated views "
                    f"(local={self_tps}, peer={peer_tps}): {where}"
                )
            larger, smaller = max(self_tps, peer_tps), min(self_tps, peer_tps)
            if larger % smaller != 0:
                raise ValueError(
                    "validate_page_tables: tokens_per_slot not divisible "
                    f"(local={self_tps}, peer={peer_tps}): {where}"
                )
            logger.warning(
                "validate_page_tables: tokens_per_slot mismatch (local=%d, peer=%d); "
                "KV transfer proceeds — ensure block boundaries align during transfer.",
                self_tps,
                peer_tps,
            )
    return mapping


# ---------------------------------------------------------------------------
# Ownership and mapper selection
# ---------------------------------------------------------------------------


class TransferPolicy:
    """Decide, per matched view pair, whether this rank sends and how bytes map.

    Stateless: every method is a function of the two views' declarations (and
    the destination's DP rank for replica election).
    """

    @staticmethod
    def is_paired(self_pv: PoolView, peer_pv: PoolView) -> bool:
        """Whether our shard and the peer's shard overlap."""
        return shards_paired(
            (self_pv.num_shards, self_pv.shard_index), (peer_pv.num_shards, peer_pv.shard_index)
        )

    @staticmethod
    def send_ratio(self_pv: PoolView, peer_pv: PoolView) -> int:
        """How many local replicas of a shard target one peer rank (1 = no election)."""
        return max(1, self_pv.num_replicas // max(1, peer_pv.num_replicas))

    @classmethod
    def should_send(cls, self_pv: PoolView, peer_pv: PoolView, peer_dp_rank: int) -> bool:
        """Whether this rank owns the transfer of one view pair to one peer rank.

        Unpaired shards never send. When several local ranks hold the same
        bytes the peer needs (``num_replicas`` larger than the peer's), exactly
        one is elected; the election is rotated by the destination's DP rank
        so replicated traffic spreads across local ranks instead of always
        landing on the first replica.
        """
        if not cls.is_paired(self_pv, peer_pv):
            return False
        ratio = cls.send_ratio(self_pv, peer_pv)
        if ratio <= 1:
            return True
        return self_pv.replica_index % ratio == peer_dp_rank % ratio

    @staticmethod
    def buffers_per_layer(pool_view: PoolView, *, context: str = "PoolView") -> int:
        """Per-layer buffer count of a view (e.g. K+V -> 2, key-only -> 1).

        Views are bucketed per (layer group, pool, layout) at page-table build
        time, so every layer in a view carries the same role set and hence the
        same entry count. Verified per layer rather than via total-count
        divisibility: 1 + 3 entries over two layers would pass the latter yet
        make shard mappers split every layer at wrong offsets.
        """
        entries = pool_view.buffer_entries
        if len(entries) == 0:
            return 1
        counts = Counter(int(e["local_layer_id"]) for e in entries)
        distinct = set(counts.values())
        if len(distinct) != 1:
            raise ValueError(
                f"{context}: buffer entries are not evenly distributed across layers: "
                f"per-layer entry counts={sorted(counts.items())}"
            )
        return distinct.pop()

    @classmethod
    def build_mapper(
        cls,
        *,
        self_pv: PoolView,
        peer_pv: PoolView,
        self_lg: LayerGroup,
        peer_lg: LayerGroup,
        self_layer_offsets: np.ndarray,
        peer_layer_offsets: np.ndarray,
        self_bytes_per_layer: int,
        peer_bytes_per_layer: int,
    ) -> RegionMapperBase:
        """Pick the mapper for one view pair.

        Equal shard counts collapse into ``IntactMapper`` for every kind
        (the run merging degrades to a single whole-region copy per slot when
        the selected layers are contiguous on both sides). Otherwise the kind
        decides how a layer's region is re-split.
        """
        self_shards = (self_pv.num_shards, self_pv.shard_index)
        peer_shards = (peer_pv.num_shards, peer_pv.shard_index)
        kind = self_pv.mapper_kind
        if kind == MapperKind.REPLICATED:
            return ReplicatedMapper(
                self_layer_offsets, peer_layer_offsets, self_bytes_per_layer, peer_bytes_per_layer
            )
        if self_pv.num_shards == peer_pv.num_shards:
            return IntactMapper(
                self_layer_offsets,
                peer_layer_offsets,
                self_bytes_per_layer,
                peer_bytes_per_layer,
                mapper_name=kind.name,
            )
        if kind == MapperKind.SECTIONED:
            if self_pv.section_bytes is None or peer_pv.section_bytes is None:
                raise ValueError("SECTIONED view is missing section_bytes")
            return SectionedMapper(
                src_layer_offsets=self_layer_offsets,
                dst_layer_offsets=peer_layer_offsets,
                self_section_bytes=self_pv.section_bytes,
                peer_section_bytes=peer_pv.section_bytes,
                self_shards=self_shards,
                peer_shards=peer_shards,
            )
        buffers = cls.buffers_per_layer(self_pv, context="local view")
        if buffers != cls.buffers_per_layer(peer_pv, context="peer view"):
            raise ValueError(
                f"{kind.name} buffer count per layer mismatch: local={buffers}, "
                f"peer={cls.buffers_per_layer(peer_pv)}"
            )
        if kind == MapperKind.NHD:
            tps = self_lg.tokens_per_slot
            if tps is None or tps != peer_lg.tokens_per_slot:
                raise ValueError(
                    "NHD mapper requires equal tokens_per_slot; "
                    f"local={self_lg.tokens_per_slot}, peer={peer_lg.tokens_per_slot}"
                )
            return NHDShardMapper(
                src_layer_offsets=self_layer_offsets,
                dst_layer_offsets=peer_layer_offsets,
                self_bytes_per_layer=self_bytes_per_layer,
                peer_bytes_per_layer=peer_bytes_per_layer,
                buffers_per_layer=buffers,
                tokens_per_slot=self_lg.tokens_per_slot,
                self_shards=self_shards,
                peer_shards=peer_shards,
            )
        return HNDShardMapper(
            src_layer_offsets=self_layer_offsets,
            dst_layer_offsets=peer_layer_offsets,
            self_bytes_per_layer=self_bytes_per_layer,
            peer_bytes_per_layer=peer_bytes_per_layer,
            buffers_per_layer=buffers,
            self_shards=self_shards,
            peer_shards=peer_shards,
        )


# ---------------------------------------------------------------------------
# Per-request state payload accounting
# ---------------------------------------------------------------------------


def state_payload_bytes(
    sender_page_table: KVCachePageTable,
    receiver_page_table: KVCachePageTable,
) -> int:
    """Per-request state bytes that will land in the receiver's state slots.

    Receiver-local invariant: regardless of the sender-side shard pairing, each
    state group's slot receives exactly the receiver's own per-layer region
    bytes over the layers both sides hold. Summed over every state group. Used
    to size the bounce region, which the sender's state fragments share with
    the paged KV.
    """
    receiver_states: List[LayerGroup] = [
        lg for lg in receiver_page_table.layer_groups if not lg.has_token_axis
    ]
    sender_globals = {
        ll.global_layer_id
        for lg in sender_page_table.layer_groups
        if not lg.has_token_axis
        for ll in lg.local_layers
    }
    if not receiver_states or not sender_globals:
        return 0
    total = 0
    for lg in receiver_states:
        overlap = {ll.global_layer_id for ll in lg.local_layers} & sender_globals
        if not overlap:
            continue
        total += len(overlap) * sum(int(pv.bytes_per_layer or 0) for pv in lg.pool_views)
    return total
