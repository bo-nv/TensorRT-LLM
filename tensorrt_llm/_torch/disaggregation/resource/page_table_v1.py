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

"""Page table for the C++-backed V1 ``KVCacheManager`` (legacy path).

The V1 manager has no storage-descriptor API, so the pool geometry is
computed here from its pool pointers, and the hybrid manager's recurrent
state is read from its Python-side tensors. The output obeys the same
addressing contract as the V2 builder (``base + slot * slot_stride +
entry.offset``) and uses the same role rules, so the transfer side does not
know which builder produced a table.
"""

from types import SimpleNamespace
from typing import List, Sequence

import numpy as np

from tensorrt_llm._torch.disaggregation.resource.page import (
    BUFFER_ENTRY_DTYPE,
    KVCachePageTable,
    LayerGroup,
    LocalLayer,
    PhysicalPool,
    PhysicalPoolGroup,
    PoolView,
    RoleLayout,
)
from tensorrt_llm._torch.disaggregation.resource.role_rules import (
    KV_RULE,
    REPLICATED_RULE,
    ROLE_RULES,
    build_role_layout,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManager
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import get_size_in_bytes
from tensorrt_llm.bindings import DataType

# Native V2 role names of the recurrent-state buffers, so V1 and V2 page
# tables label the same bytes the same way.
SSM_STATE_ROLE = frozenset({"ssm_state"})
CONV_STATE_ROLE = frozenset({"conv_state"})


def _layer_major_view(
    pool_idx: int,
    pool: PhysicalPool,
    layer_stride_bytes: int,
    local_layers: Sequence[LocalLayer],
    *,
    pool_role: frozenset,
    layout: RoleLayout,
) -> PoolView:
    """View over a legacy ``[layer][slot]`` allocation (V1 recurrent state).

    Each layer's buffer entry carries ``offset = local_layer_id *
    layer_stride_bytes``, so the universal ``base + slot * slot_stride +
    offset`` addressing reaches layer-major storage without any pool-kind
    branching downstream.
    """
    sorted_lids = sorted(ll.local_layer_id for ll in local_layers)
    return PoolView(
        pool_idx=pool_idx,
        buffer_entries=np.array(
            [(lid, lid * layer_stride_bytes, pool.slot_bytes) for lid in sorted_lids],
            dtype=BUFFER_ENTRY_DTYPE,
        ),
        pool_role=pool_role,
        layout=layout,
        bytes_per_layer=pool.slot_bytes,
    )


def _build_layer_group_for_mamba(
    manager: MambaHybridCacheManager, pool_group_idx: int
) -> "tuple[LayerGroup, PhysicalPoolGroup]":
    local_layers = [
        LocalLayer(local_layer_id=int(lid), global_layer_id=int(gid))
        for gid, lid in sorted(manager._impl.mamba_layer_offsets.items(), key=lambda x: x[1])
    ]

    conv_state = manager._impl.mamba_cache.conv
    ssm_state = manager._impl.mamba_cache.temporal

    conv_pool = PhysicalPool(
        base_address=conv_state.data_ptr(),
        slot_bytes=conv_state.stride(1) * conv_state.element_size(),
        num_slots=conv_state.shape[1],
    )
    conv_layer_stride = conv_state.stride(0) * conv_state.element_size()

    ssm_pool = PhysicalPool(
        base_address=ssm_state.data_ptr(),
        slot_bytes=ssm_state.stride(1) * ssm_state.element_size(),
        num_slots=ssm_state.shape[1],
    )
    ssm_layer_stride = ssm_state.stride(0) * ssm_state.element_size()

    # The V1 hybrid manager keeps the state geometry on its inner
    # PythonMambaCacheManager and in the tensors themselves; present it under
    # the same attribute names the V2 manager exposes so the shared role
    # table applies unchanged. Tensors are (layers, slots, ...) here.
    geom = SimpleNamespace(
        mapping=manager.mapping,
        conv_state_shape=tuple(conv_state.shape[2:]),
        conv_state_dtype=conv_state.dtype,
        conv_section_dims=tuple(manager._impl.conv_section_dims),
        ssm_state_shape=tuple(ssm_state.shape[2:]),
        ssm_state_dtype=ssm_state.dtype,
    )
    conv_layout = build_role_layout(ROLE_RULES["conv_state"], geom)
    ssm_layout = build_role_layout(ROLE_RULES["ssm_state"], geom)

    pool_group = PhysicalPoolGroup(pools=[conv_pool, ssm_pool])
    layer_group = LayerGroup(
        pool_group_idx=pool_group_idx,
        local_layers=local_layers,
        pool_views=[
            _layer_major_view(
                0,
                conv_pool,
                conv_layer_stride,
                local_layers,
                pool_role=CONV_STATE_ROLE,
                layout=conv_layout,
            ),
            _layer_major_view(
                1,
                ssm_pool,
                ssm_layer_stride,
                local_layers,
                pool_role=SSM_STATE_ROLE,
                layout=ssm_layout,
            ),
        ],
        tokens_per_slot=None,
        live_token_window=None,
    )
    return layer_group, pool_group


def _build_non_kv_layers(
    manager,
    layer_groups: List[LayerGroup],
    pool_groups: List[PhysicalPoolGroup],
) -> None:
    """Append the V1 hybrid manager's layer-major recurrent-state group.

    V1 only: the C++-backed ``KVCacheManager`` does not describe the Mamba
    state pools, so they are read from ``MambaHybridCacheManager``. V2
    recurrent state is ordinary V2 storage and is described by
    :func:`_build_page_table_v2` like any other layer group.
    """
    if isinstance(manager, MambaHybridCacheManager):
        pool_group_idx = len(pool_groups)
        layer_group, pool_group = _build_layer_group_for_mamba(manager, pool_group_idx)
        layer_groups.append(layer_group)
        pool_groups.append(pool_group)


def build_page_table(kv_cache_manager: KVCacheManager) -> KVCachePageTable:
    """Build a KVCachePageTable from a KVCacheManager (V1)."""
    if kv_cache_manager.dtype == DataType.NVFP4:
        raise NotImplementedError("NVFP4 quantization not supported")

    tokens_per_block = kv_cache_manager.tokens_per_block

    # Group local layers by their window size (layer group)
    window_size_to_local_layer_ids = kv_cache_manager._get_window_size_to_layers()
    layer_offsets = kv_cache_manager.layer_offsets
    local_to_global = {local_id: global_id for global_id, local_id in layer_offsets.items()}

    if len(window_size_to_local_layer_ids) < 1:
        raise ValueError("KVRegionExtractorV1: window_size_to_local_layer_ids is empty")

    sorted_window_sizes = sorted(
        window_size_to_local_layer_ids.keys(), key=lambda x: (x is None, x)
    )

    pool_groups: List[PhysicalPoolGroup] = []
    layer_groups: List[LayerGroup] = []

    for group_id, window_size in enumerate(sorted_window_sizes):
        local_layer_ids = window_size_to_local_layer_ids[window_size]
        first_local_layer = local_layer_ids[0]

        # Get pool base address via pool_mapping -> pool_pointers
        pool_id = int(kv_cache_manager.kv_cache_pool_mapping[first_local_layer][0].item())
        base_addr = int(kv_cache_manager.kv_cache_pool_pointers[pool_id][0].item())

        # Get num_blocks from per-layer pool view: shape = (numBlocks, kvFactor, blockSize)
        pool_layer_view = kv_cache_manager.impl.get_primary_pool_data(first_local_layer)
        num_blocks = pool_layer_view.shape[0]

        num_kv_heads = kv_cache_manager.num_kv_heads_per_layer[first_local_layer]
        kv_factor = kv_cache_manager.kv_factor
        is_key_only = kv_factor == 1

        elements_per_buffer = tokens_per_block * num_kv_heads * kv_cache_manager.head_dim
        buffer_size = get_size_in_bytes(elements_per_buffer, kv_cache_manager.dtype)
        stride = buffer_size * kv_factor
        slot_bytes = stride * len(local_layer_ids)

        entries = []
        kv_role_names: set[str] = {"key"}
        if not is_key_only:
            kv_role_names.add("value")
        for i, lid in enumerate(local_layer_ids):
            base_offset = i * stride
            entries.append((lid, base_offset, buffer_size))
            if not is_key_only:
                entries.append((lid, base_offset + buffer_size, buffer_size))

        kv_layout = build_role_layout(
            KV_RULE, kv_cache_manager, model_layer=int(local_to_global[first_local_layer])
        )
        kv_physical = PhysicalPool(
            base_address=base_addr, slot_bytes=slot_bytes, num_slots=num_blocks
        )
        kv_view = PoolView(
            pool_idx=0,
            buffer_entries=np.array(entries, dtype=BUFFER_ENTRY_DTYPE),
            pool_role=frozenset(kv_role_names),
            layout=kv_layout,
            bytes_per_layer=stride,
        )
        physical_pools = [kv_physical]
        pool_views = [kv_view]

        # Indexer K cache support. The DSA indexer K cache is identical on
        # every TP rank (single index head), so its view is REPLICATED. With a
        # per-layer indexer mask (cross-layer indexer sharing, e.g. GLM 5.2)
        # only the "full" indexer-owning layers get a pool row, so the view
        # covers that subset: one buffer entry per owning layer, each mapped to
        # its packed row in the (possibly masked) pool. When the mask is absent
        # every layer owns a row (dense/legacy layout) and this reduces to the
        # equal-sized packing in local-layer order.
        if kv_cache_manager.enable_indexer_k_cache:
            local_indexer_mask = kv_cache_manager.indexer_k_cache_local_layer_mask
            owning_layer_ids = [
                lid
                for lid in local_layer_ids
                if local_indexer_mask is None or local_indexer_mask[lid]
            ]
            # A layer group whose layers are all masked out owns no indexer pool
            # row on this rank (the pool getter would raise); skip it so the peer
            # simply transfers nothing for this rank's indexer.
            if owning_layer_ids:
                indexer_pool = kv_cache_manager.impl.get_indexer_k_cache_pool()
                if indexer_pool.shape[1] != len(owning_layer_ids):
                    raise RuntimeError(
                        "The DSA indexer K-cache pool row count does not match "
                        "the number of indexer-owning layers in its layer group: "
                        f"{indexer_pool.shape[1]} rows for {len(owning_layer_ids)} layers"
                    )
                # indexer_pool shape: (numBlocks, numIndexerLayers, kvFactor,
                # blockSize), dtype=UINT8. numIndexerLayers is the number of
                # owning layers on this rank (== the attention layer count when
                # unmasked). slot_bytes packs every owning-layer row.
                per_block_elems = 1
                for d in indexer_pool.shape[1:]:  # skip numBlocks dim
                    per_block_elems *= d
                indexer_slot_bytes = per_block_elems * indexer_pool.element_size()
                indexer_bytes_per_layer = indexer_slot_bytes // indexer_pool.shape[1]
                indexer_physical = PhysicalPool(
                    base_address=int(indexer_pool.data_ptr()),
                    slot_bytes=indexer_slot_bytes,
                    num_slots=num_blocks,
                )
                indexer_view = PoolView(
                    pool_idx=len(physical_pools),
                    buffer_entries=np.array(
                        [
                            (
                                lid,
                                kv_cache_manager.impl.get_indexer_k_cache_pool_layer_idx(lid)
                                * indexer_bytes_per_layer,
                                indexer_bytes_per_layer,
                            )
                            for lid in owning_layer_ids
                        ],
                        dtype=BUFFER_ENTRY_DTYPE,
                    ),
                    pool_role=frozenset({"indexer_k"}),
                    layout=build_role_layout(REPLICATED_RULE, kv_cache_manager),
                    bytes_per_layer=indexer_bytes_per_layer,
                )
                physical_pools.append(indexer_physical)
                pool_views.append(indexer_view)

        pool_groups.append(PhysicalPoolGroup(pools=physical_pools))
        local_layers = [
            LocalLayer(local_layer_id=int(lid), global_layer_id=int(local_to_global[lid]))
            for lid in local_layer_ids
        ]
        layer_groups.append(
            LayerGroup(
                pool_group_idx=group_id,
                local_layers=local_layers,
                pool_views=pool_views,
                tokens_per_slot=tokens_per_block,
                live_token_window=window_size,
            )
        )
    _build_non_kv_layers(kv_cache_manager, layer_groups, pool_groups)

    return KVCachePageTable(
        tokens_per_block=tokens_per_block,
        layer_groups=layer_groups,
        pool_groups=pool_groups,
    )
