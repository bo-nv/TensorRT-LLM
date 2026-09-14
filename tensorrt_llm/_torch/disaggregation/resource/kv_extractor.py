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

"""Build the disaggregation page table from a V2 cache manager, and extract slot pointers.

``_build_page_table_v2`` reads the V2 storage descriptors
(``impl.pool_group_descs``): every buffer is ``(layer_id, role, offset,
size)``. It is one loop over life cycles; every role goes through the same
code. The manager's only disaggregation-facing statement is
``get_disagg_role_mapper_kinds()`` (how each role's bytes are arranged);
how those bytes are sharded across ranks is derived from the role table
(:mod:`.role_rules`), the manager's ``mapping`` and its geometry attributes.

The legacy C++-backed V1 manager is served by :mod:`.page_table_v1`.
"""

from collections import defaultdict
from typing import Dict, List

import numpy as np

from tensorrt_llm._torch.disaggregation.base.region import (
    DataLayout,
    MemRegionGroup,
    RegionExtractorBase,
    SpecRegion,
)
from tensorrt_llm._torch.disaggregation.resource.page import (
    BUFFER_ENTRY_DTYPE,
    KVCachePageTable,
    LayerGroup,
    LocalLayer,
    MapperKind,
    PhysicalPool,
    PhysicalPoolGroup,
    PoolView,
    RoleLayout,
)
from tensorrt_llm._torch.disaggregation.resource.page_table_v1 import build_page_table
from tensorrt_llm._torch.disaggregation.resource.role_rules import (
    build_role_layout,
    resolve_role_rule,
)
from tensorrt_llm._torch.disaggregation.resource.utils import (
    compute_layer_byte_ranges,
    get_physical_pool,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.runtime.kv_cache_manager_v2 import SsmLayerConfig


class KVRegionExtractorV1(RegionExtractorBase):
    """
    Descriptor and region extractor for KV cache pool managed by
    KVCacheManager, KVCacheManagerV2, or described by a KVCachePageTable.

    Provides region descriptors for adapting block-wise view.
    """

    def __init__(self, kv_arg):
        if isinstance(kv_arg, KVCachePageTable):
            self._page_table = kv_arg
        else:
            # Assume it is a manager (KVCacheManager / KVCacheManagerV2)
            self._page_table = build_page_table_from_manager(kv_arg)
        self._data_layout = DataLayout.HND

    @property
    def page_table(self) -> KVCachePageTable:
        return self._page_table

    @nvtx_range("KVRegionExtractorV1.extract")
    def extract(
        self,
        region_ids: np.ndarray,
        layer_group_id: int = 0,
        pool_idx: int = 0,
    ) -> SpecRegion:
        """
        Given a list of slot ids, returns a single SpecRegion whose memory is a
        MemRegionGroup with one pointer per slot:
        ``ptr = base_address + slot_id * slot_stride_bytes``.

        Sub-slot selection (layers, role classes, shards) is the mappers'
        responsibility; logical views carry that geometry in their buffer
        entries. The same call serves groups with a token axis (many slots
        per request) and per-request state (one slot per request).

        Args:
            layer_group_id: The layer group index (= life cycle index).
            pool_idx: The pool-view index within the layer group.
        """
        lg = self._page_table.layer_groups[layer_group_id]
        pv = lg.pool_views[pool_idx]
        pool = get_physical_pool(self._page_table, layer_group_id, pv.pool_idx)

        base_ptr = pool.base_address
        block_size = pool.slot_bytes
        block_stride = pool.slot_stride_bytes
        assert block_stride is not None

        # Filter out invalid slot ids (BAD_PAGE_INDEX = -1)
        valid = region_ids >= 0
        ptrs = base_ptr + block_stride * region_ids[valid]
        memory = MemRegionGroup(ptrs=ptrs, bytes_per_region=block_size)
        return SpecRegion(memory=memory)


# ---------------------------------------------------------------------------
# V2 (KVCacheManagerV2) page table
# ---------------------------------------------------------------------------


def _v2_role_mapper_kinds(manager) -> Dict[str, MapperKind]:
    """Read the manager's ``{role: MapperKind}`` declaration, keyed by role name.

    ``Role.ALL`` is the required fallback for paged roles without an explicit
    entry. This is the manager's only disaggregation-facing statement: how
    each role's bytes are arranged. Sharding is derived in ``role_rules``.
    """
    get_kinds = getattr(manager, "get_disagg_role_mapper_kinds", None)
    if get_kinds is None:
        raise AttributeError("V2 cache manager must implement get_disagg_role_mapper_kinds()")
    kinds = dict(get_kinds())
    if Role.ALL not in kinds:
        raise ValueError("Disaggregation role mapper kinds must define Role.ALL")
    for role, kind in kinds.items():
        if not isinstance(kind, MapperKind):
            raise ValueError(
                f"Invalid disaggregation mapper kind {kind!r} for role {role!s}; "
                "expected MapperKind"
            )
    return {str(role): kind for role, kind in kinds.items()}


def _v2_layer_key_fn(manager):
    """Return ``local_layer_id -> peer-matching key``.

    Peers match layers by this key, so it must be collision-free on a rank
    and identical for the same layer on every PP rank. Managers whose
    internal layers are not model layers (DeepSeek-V4 registers one internal
    layer per attention type) publish ``layer_keys``; otherwise the model
    layer index from ``pp_layers`` is used.
    """
    layer_keys = getattr(manager, "layer_keys", None)
    if layer_keys is not None:
        return lambda local_layer_id: int(layer_keys[local_layer_id])
    pp_layers = manager.pp_layers
    return lambda local_layer_id: int(pp_layers[local_layer_id])


def _build_page_table_v2(manager) -> KVCachePageTable:
    """Build a KVCachePageTable from a KVCacheManagerV2.

    One uniform pass over the storage layer's ``pool_group_descs``: every
    ``SlotDescVariant`` is one layer group (life cycle) drawing slots from
    one physical pool group, and every buffer of a variant is described by
    ``(layer_id, role, offset, size)``. Every role goes through the same
    code: the manager's ``get_disagg_role_mapper_kinds()`` gives the byte arrangement
    of each role, and the role name selects a shard rule from
    :data:`ROLE_RULES`, instantiated with ``mapping`` and geometry attributes.

    A physical pool group may be shared by several layer groups (life cycles
    whose coalesced-buffer sizes are identical); ``layer_groups`` stays
    indexed by layer_group_id while ``pool_group_idx`` points at the shared
    physical pool group entry, so request-side slot lists and this table use
    the same indices as the V2 core.

    Layer groups the V2 core marks ``SsmLayerConfig`` have no token axis (one
    slot per request); every role in such a group must have a declared kind
    and a shard rule, because the ``Role.ALL`` fallback describes paged roles.
    """
    config = manager.impl.init_config
    pool_group_descs = manager.impl.pool_group_descs
    layer_key = _v2_layer_key_fn(manager)
    layer_configs = list(config.layers)
    tokens_per_block = int(config.tokens_per_block)

    role_kinds = _v2_role_mapper_kinds(manager)
    default_kind = role_kinds[str(Role.ALL)]
    layout_cache: Dict[tuple, RoleLayout] = {}

    def _role_layout(role: str, has_token_axis: bool, context: str) -> RoleLayout:
        layout = layout_cache.get((role, has_token_axis))
        if layout is None:
            kind = role_kinds.get(role)
            if kind is None:
                if not has_token_axis:
                    raise ValueError(
                        f"{context}: per-request state role {role!r} has no mapper kind; the "
                        "cache manager must declare it in get_disagg_role_mapper_kinds()"
                    )
                kind = default_kind
            rule = resolve_role_rule(role, kind, has_token_axis=has_token_axis, context=context)
            layout = build_role_layout(rule, manager, kind, context=f"{context} role {role!r}")
            layout_cache[(role, has_token_axis)] = layout
        return layout

    def _layer_config(internal_layer_id: int):
        if internal_layer_id >= len(layer_configs):
            raise ValueError(
                f"Cannot resolve layer config for internal layer {internal_layer_id} "
                f"({len(layer_configs)} layers configured)"
            )
        return layer_configs[internal_layer_id]

    pool_groups: List[PhysicalPoolGroup] = []
    storage_pg_to_list_idx: Dict[int, int] = {}
    layer_groups_by_id: List[LayerGroup | None] = [None] * len(manager.impl.layer_grouping)

    for pg_desc in pool_group_descs:
        storage_pg_idx = int(pg_desc.pool_group_index)
        storage_pg_to_list_idx[storage_pg_idx] = len(pool_groups)
        pool_groups.append(
            PhysicalPoolGroup(
                pools=[
                    PhysicalPool(
                        base_address=int(pool.base_address),
                        slot_bytes=int(pool.slot_bytes),
                        num_slots=int(pg_desc.num_slots),
                    )
                    for pool in pg_desc.pools
                ]
            )
        )

        # Each variant is one layer group (life cycle) drawing slots from
        # this pool group. Multiple layer groups share a pool group when
        # their coalesced-buffer sizes are identical; within a slot, each
        # layer group's buffer offsets start from 0 independently — the
        # memory is reused, not concatenated.
        for variant in pg_desc.slot_desc.variants:
            layer_group_id = int(variant.layer_group_id)
            if layer_group_id >= len(layer_groups_by_id) or layer_groups_by_id[layer_group_id]:
                raise ValueError(
                    f"V2 storage describes layer group {layer_group_id} twice or out of range "
                    f"({len(layer_groups_by_id)} layer groups); page-table indices must equal "
                    "V2 layer group ids"
                )
            internal_layer_ids = [int(lid) for lid in manager.impl.layer_grouping[layer_group_id]]
            is_state = [
                isinstance(_layer_config(lid), SsmLayerConfig) for lid in internal_layer_ids
            ]
            if any(is_state) and not all(is_state):
                raise ValueError(
                    f"V2 layer group {layer_group_id} mixes per-request-state and paged layers: "
                    f"{internal_layer_ids}"
                )
            has_token_axis = not (is_state and all(is_state))

            local_layers = [
                LocalLayer(local_layer_id=lid, global_layer_id=layer_key(lid))
                for lid in internal_layer_ids
            ]

            # Bucket buffer entries by (pool, role layout): one PoolView per
            # bucket, spanning every layer of that role class. V2 storage
            # coalesces buffers purely by size within a layer group, so one
            # physical pool may hold several role classes; each still gets its
            # own view, which keeps peer matching independent of that
            # coalescing decision. Buffer offsets within a slot follow
            # ``buffer_ids`` order: the i-th buffer lives at
            # ``i * single_buffer_size``.
            bucket_entries: Dict[tuple, list] = defaultdict(list)
            bucket_roles: Dict[tuple, set] = defaultdict(set)
            for pool_idx, coalesced_buffer in enumerate(variant.coalesced_buffers):
                single_buffer_size = int(coalesced_buffer.single_buffer_size)
                offset = 0
                for buffer_id in coalesced_buffer.buffer_ids:
                    layout = _role_layout(
                        str(buffer_id.role),
                        has_token_axis,
                        f"V2 layer group {layer_group_id}",
                    )
                    bucket_key = (pool_idx, layout)
                    bucket_entries[bucket_key].append(
                        (int(buffer_id.layer_id), offset, single_buffer_size)
                    )
                    bucket_roles[bucket_key].add(str(buffer_id.role))
                    offset += single_buffer_size

            # Emit this layer group's views: one per (pool, role layout).
            # Roles sharing a layout share a view (KEY+VALUE); roles with
            # different layouts in the same physical pool get separate views.
            # All ordering below is canonicalization — the page table is
            # serialized and matched against peers, so view order (pool, then
            # lowest slot offset), entry order (slot offset), and role text
            # must not depend on dict/set iteration order.
            pool_views = []
            lg_bucket_keys = sorted(
                bucket_entries,
                key=lambda key: (key[0], min(entry[1] for entry in bucket_entries[key])),
            )
            for bucket_key in lg_bucket_keys:
                pool_idx, layout = bucket_key
                roles = frozenset(bucket_roles[bucket_key])
                entries = np.array(
                    sorted(bucket_entries[bucket_key], key=lambda entry: entry[1]),
                    dtype=BUFFER_ENTRY_DTYPE,
                )
                view_name = (
                    f"View(layer_group={layer_group_id}, pool={pool_idx}, "
                    f"kind={layout.mapper_kind.name}, role={sorted(roles)})"
                )
                # Fail fast on invalid geometry and record the uniform
                # per-layer region size on the wire.
                _, bytes_per_layer = compute_layer_byte_ranges(entries, context=view_name)
                sections = layout.section_bytes
                if sections is not None and sum(sections) != bytes_per_layer:
                    raise ValueError(
                        f"{view_name} section_bytes {list(layout.section_bytes)} do not sum "
                        f"to the per-layer size {bytes_per_layer}"
                    )
                unit = layout.shard_unit_bytes
                if unit is not None and bytes_per_layer % unit:
                    raise ValueError(
                        f"{view_name} per-layer size {bytes_per_layer} is not a multiple of "
                        f"shard_unit_bytes {layout.shard_unit_bytes}"
                    )
                pool_views.append(
                    PoolView(
                        pool_idx=pool_idx,
                        buffer_entries=entries,
                        pool_role=roles,
                        layout=layout,
                        bytes_per_layer=bytes_per_layer,
                    )
                )

            first_cfg = _layer_config(internal_layer_ids[0]) if internal_layer_ids else None
            layer_groups_by_id[layer_group_id] = LayerGroup(
                pool_group_idx=storage_pg_to_list_idx[storage_pg_idx],
                local_layers=local_layers,
                pool_views=pool_views,
                tokens_per_slot=tokens_per_block if has_token_axis else None,
                live_token_window=(
                    getattr(first_cfg, "window_size", None) if has_token_axis else None
                ),
            )

    layer_groups: List[LayerGroup] = []
    for layer_group_id, layer_group in enumerate(layer_groups_by_id):
        if layer_group is None:
            raise ValueError(f"Missing V2 layer group descriptor for layer group {layer_group_id}")
        layer_groups.append(layer_group)

    return KVCachePageTable(
        tokens_per_block=tokens_per_block,
        layer_groups=layer_groups,
        pool_groups=pool_groups,
    )


def _is_kv_cache_manager_v2(obj) -> bool:
    return hasattr(obj, "impl") and hasattr(obj.impl, "layer_grouping")


def build_page_table_from_manager(manager) -> KVCachePageTable:
    """Unified entry point: build a KVCachePageTable from any manager type.

    Supports KVCacheManager (V1) and KVCacheManagerV2.
    """
    if _is_kv_cache_manager_v2(manager):
        return _build_page_table_v2(manager)
    else:
        return build_page_table(manager)
