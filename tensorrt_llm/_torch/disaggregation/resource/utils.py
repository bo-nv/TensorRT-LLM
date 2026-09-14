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

from __future__ import annotations

from typing import Dict, List, Set

import numpy as np

from .page import KVCachePageTable, LayerGroup, PhysicalPool, PoolView

# -------------------------------------------------------------------------
# PhysicalPool helpers
# -------------------------------------------------------------------------


def get_pool_bytes(pool: PhysicalPool) -> int:
    """Total transferable payload bytes across all slots in this pool."""
    return pool.slot_bytes * pool.num_slots


# -------------------------------------------------------------------------
# PoolView helpers
# -------------------------------------------------------------------------


def compute_layer_byte_ranges(
    buffer_entries,
    *,
    declared_bytes_per_layer: "int | None" = None,
    context: str = "PoolView",
) -> tuple[Dict[int, int], int]:
    """Per-layer slot-relative byte offsets from raw buffer entries.

    Returns ``({local_layer_id: start_offset}, bytes_per_layer)``. A layer's
    region is the concatenation of its buffer entries, which must be
    contiguous within the slot; the region size must be uniform across
    layers (the slot may interleave other role classes between layers, so
    only the *size* is uniform — offsets are per layer). ``context`` labels
    error messages; ``declared_bytes_per_layer`` cross-checks a size that
    was recorded elsewhere.
    """
    starts: Dict[int, int] = {}
    totals: Dict[int, int] = {}
    entries_by_layer: Dict[int, list] = {}
    for entry in buffer_entries:
        entries_by_layer.setdefault(int(entry["local_layer_id"]), []).append(
            (int(entry["offset"]), int(entry["size"]))
        )
    if not entries_by_layer:
        raise ValueError(f"{context} has no buffer entries; per-layer byte ranges are undefined")
    for layer_id, spans in entries_by_layer.items():
        spans.sort()
        for (off, size), (next_off, _) in zip(spans, spans[1:]):
            if off + size != next_off:
                raise ValueError(
                    f"{context} layer {layer_id} buffers are "
                    f"not contiguous: [{off}, {off + size}) is followed by offset {next_off}"
                )
        starts[layer_id] = spans[0][0]
        totals[layer_id] = sum(size for _, size in spans)
    distinct_totals = set(totals.values())
    if len(distinct_totals) != 1:
        raise ValueError(
            f"{context} per-layer region sizes are not uniform: {sorted(totals.items())}"
        )
    bytes_per_layer = distinct_totals.pop()
    if declared_bytes_per_layer is not None and declared_bytes_per_layer != bytes_per_layer:
        raise ValueError(
            f"{context} declares bytes_per_layer={declared_bytes_per_layer} but buffer "
            f"entries sum to {bytes_per_layer} per layer"
        )
    return starts, bytes_per_layer


def get_layer_byte_ranges(pool_view: PoolView) -> tuple[Dict[int, int], int]:
    """Per-layer byte ranges of a view; see :func:`compute_layer_byte_ranges`."""
    return compute_layer_byte_ranges(
        pool_view.buffer_entries,
        declared_bytes_per_layer=pool_view.bytes_per_layer,
        context=f"PoolView(pool_idx={pool_view.pool_idx}, role={sorted(pool_view.pool_role)})",
    )


def get_unique_layers(pool_view: PoolView) -> Set[int]:
    """Unique local layer IDs in *pool_view*."""
    return {int(e["local_layer_id"]) for e in pool_view.buffer_entries}


def get_num_buffer_entries(pool_view: PoolView) -> int:
    """Number of buffer entries."""
    return len(pool_view.buffer_entries)


def get_pool_view_num_layers(pool_view: PoolView) -> int:
    """
    Number of unique layers represented in *pool_view*
    """
    return len(get_unique_layers(pool_view))


def get_pool_view_global_layer_ids(pool_view: PoolView, layer_group: "LayerGroup") -> List[int]:
    """
    Global layer IDs for the layers that appear in *pool_view*, ordered by their
    physical offset within the coalesced buffer (ascending).

    The order is derived from the buffer entries' physical offsets rather than
    from ``layer_group.local_layers`` order on purpose: the KV transfer maps
    layers positionally (a layer's position in this list times the per-layer
    slot size gives its byte offset), so the position must reflect the physical
    slot layout. Deriving it from offsets keeps the transceiver decoupled from
    the KV-cache manager's layer-grouping order (which is an implementation
    detail, not an API contract). This mirrors ``get_aggregated_pages``, which
    likewise sorts buffers by their offset inside the coalesced buffer.
    """
    local_to_global = {ll.local_layer_id: ll.global_layer_id for ll in layer_group.local_layers}
    # A layer may contribute several buffer entries (e.g. KEY and VALUE); use the
    # smallest offset as that layer's position within the slot.
    min_offset: dict[int, int] = {}
    for entry in pool_view.buffer_entries:
        local_layer_id = int(entry["local_layer_id"])
        offset = int(entry["offset"])
        if local_layer_id not in min_offset or offset < min_offset[local_layer_id]:
            min_offset[local_layer_id] = offset
    ordered_local_ids = sorted(min_offset, key=lambda lid: min_offset[lid])
    return [local_to_global[lid] for lid in ordered_local_ids]


# -------------------------------------------------------------------------
# LayerGroup helpers
# -------------------------------------------------------------------------


def get_global_layer_ids(layer_group: LayerGroup) -> List[int]:
    """
    Ordered global layer IDs for *layer_group*
    """
    return [ll.global_layer_id for ll in layer_group.local_layers]


def get_layer_group_num_layers(layer_group: LayerGroup) -> int:
    """
    Number of layers in *layer_group*
    """
    return len(layer_group.local_layers)


# -------------------------------------------------------------------------
# Physical pool lookup helpers
# -------------------------------------------------------------------------


def get_physical_pool(page_table: KVCachePageTable, lg_idx: int, pool_idx: int) -> PhysicalPool:
    """
    Return the :class:`PhysicalPool` backing *pool_idx* within layer group *lg_idx*
    """
    lg = page_table.layer_groups[int(lg_idx)]
    return page_table.pool_groups[int(lg.pool_group_idx)].pools[int(pool_idx)]


# -------------------------------------------------------------------------
# NIXL memory registration helpers
# -------------------------------------------------------------------------


def get_unique_pool_memory_descs(
    page_table: KVCachePageTable, device_id: int
) -> list[tuple[int, int, int, str]]:
    """Return deduplicated (ptr, size, device_id, name) tuples for all physical pools.

    A pool's footprint follows the same addressing contract the extractor and
    mappers use: ``slot_stride * (num_slots - 1) + extent``, where ``extent`` is
    the furthest byte any referencing view touches inside a slot (at least
    ``slot_bytes``). Views whose offsets encode a layer pitch (legacy V1
    layer-major recurrent state) therefore register the whole layer-major
    allocation; slot-spanning views register ``num_slots * slot_stride``.
    Pools referenced from several layer groups are registered once.
    """
    footprints: dict[int, int] = {}  # base_address -> footprint bytes
    order: list[int] = []
    for lg_idx, lg in enumerate(page_table.layer_groups):
        for pv in lg.pool_views:
            pool = get_physical_pool(page_table, lg_idx, pv.pool_idx)
            extent = int(pool.slot_bytes)
            if len(pv.buffer_entries):
                entries_end = pv.buffer_entries["offset"].astype(np.int64) + pv.buffer_entries[
                    "size"
                ].astype(np.int64)
                extent = max(extent, int(entries_end.max()))
            size = int(pool.slot_stride_bytes) * (int(pool.num_slots) - 1) + extent
            base = int(pool.base_address)
            if base not in footprints:
                order.append(base)
                footprints[base] = size
            else:
                footprints[base] = max(footprints[base], size)
    return [
        (base, footprints[base], device_id, f"kv_cache_memory_pool{idx}")
        for idx, base in enumerate(order)
    ]


# -------------------------------------------------------------------------
# KVCachePageTable aggregate helpers
# -------------------------------------------------------------------------


def get_layer_to_layer_group(
    page_table: KVCachePageTable,
    has_token_axis: bool | None = True,
) -> Dict[int, int]:
    """
    Build ``{global_layer_id: lg_idx}`` mapping.

    Layer groups are filtered by whether they have a token axis (paged KV vs
    per-request state): in hybrid models one global layer id may legitimately
    appear in both a paged group and a state group, so peer matching scopes
    the lookup to one class. ``True`` (default) indexes paged groups, ``False``
    state groups, ``None`` all groups. Within the selected class every
    global_layer_id must belong to exactly one group; a duplicate raises
    instead of silently keeping the last group.
    """
    out: Dict[int, int] = {}
    for lg_idx, lg in enumerate(page_table.layer_groups):
        if has_token_axis is not None and lg.has_token_axis != has_token_axis:
            continue
        for ll in lg.local_layers:
            gid = int(ll.global_layer_id)
            if gid in out:
                raise ValueError(
                    f"global_layer_id {gid} appears in layer groups "
                    f"{out[gid]} and {lg_idx}; layer groups must partition "
                    "a rank's layers (within one token-axis class)"
                )
            out[gid] = int(lg_idx)
    return out


def _paged_groups(page_table: KVCachePageTable):
    return ((i, lg) for i, lg in enumerate(page_table.layer_groups) if lg.has_token_axis)


def get_num_layers(page_table: KVCachePageTable) -> int:
    """Total number of layers across paged (token-axis) layer groups."""
    return sum(len(lg.local_layers) for _, lg in _paged_groups(page_table))


def get_num_layer_groups(page_table: KVCachePageTable) -> int:
    """Layer group count."""
    return len(page_table.layer_groups)


def get_pool_views(page_table: KVCachePageTable) -> List[List[PoolView]]:
    """Pool views per paged layer group."""
    return [lg.pool_views for _, lg in _paged_groups(page_table)]


def get_total_pools(page_table: KVCachePageTable) -> int:
    """Total pool-view count over paged layer groups."""
    return sum(len(lg.pool_views) for _, lg in _paged_groups(page_table))


def get_total_buffer_entries(page_table: KVCachePageTable) -> int:
    """Total buffer entries across paged layer groups."""
    return sum(
        get_num_buffer_entries(pv) for _, lg in _paged_groups(page_table) for pv in lg.pool_views
    )


def get_total_pool_bytes(page_table: KVCachePageTable) -> int:
    """Total allocated bytes across the physical pools of paged layer groups."""
    return sum(
        get_pool_bytes(get_physical_pool(page_table, lg_idx, pv.pool_idx))
        for lg_idx, lg in _paged_groups(page_table)
        for pv in lg.pool_views
    )


def get_total_slots(page_table: KVCachePageTable) -> int:
    """Total slot count across the physical pools of paged layer groups."""
    return sum(
        get_physical_pool(page_table, lg_idx, pv.pool_idx).num_slots
        for lg_idx, lg in _paged_groups(page_table)
        for pv in lg.pool_views
    )
