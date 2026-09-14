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

from abc import ABC, abstractmethod
from typing import List, Sequence, Union

import numpy as np

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.runtime.kv_cache_manager_v2._common import BAD_PAGE_INDEX

from .page import LayerGroup
from .utils import get_global_layer_ids


class CacheReuseAdapter(ABC):
    """Uniform per-request slot and prefix-reuse API over KVCacheManager V1/V2."""

    @property
    @abstractmethod
    def enable_block_reuse(self) -> bool: ...

    @property
    @abstractmethod
    def tokens_per_block(self) -> int: ...

    @property
    @abstractmethod
    def num_extra_kv_tokens(self) -> int:
        """Extra KV slots the manager allocates past the prompt (speculative decoding)."""

    @abstractmethod
    def _global_cached_token_count(self, req: LlmRequest) -> int:
        """Block-aligned cached prefix length reported by the cache manager."""

    def get_cached_token_count_per_layer_group(
        self,
        req: LlmRequest,
        layer_groups: Sequence[LayerGroup],
    ) -> List[int]:
        """Per-layer-group cached prefix in tokens (block-aligned).

        Returns the reuse-hit prefix only; SWA stale-region handling lives at
        the transfer call site (it is a transport concern, not a cache one).
        """
        if not self.enable_block_reuse:
            return [0] * len(layer_groups)
        scalar = max(0, self._global_cached_token_count(req))
        return [scalar] * len(layer_groups)

    @abstractmethod
    def get_slot_ids(
        self,
        req: LlmRequest,
        group_idx: int,
        lg: LayerGroup,
    ) -> np.ndarray:
        """All slot ids *req* owns in layer group *group_idx* (dtype ``int64``).

        For a group with a token axis this is the request's block list in
        token order; for a per-request state group it is the single state slot
        (or empty when the rank holds no state for the request).

        Returned values are **primary memory-pool slot indices**, not raw block
        IDs: ``KVRegionExtractorV1.extract`` and downstream transfer code do
        ``base_ptr + slot_idx * slot_stride`` and require the value to be a
        current primary-pool offset. With host offload enabled, a block's
        logical ID can diverge from its primary slot index after
        offload/onboard, so each backend must translate before returning.
        """

    @abstractmethod
    def commit_blocks_for_reuse(self, req: LlmRequest) -> None:
        """Commit KV blocks to radix tree for future prefix reuse.

        Must be called after ``req.context_current_position = req.prompt_len``.
        """


class _CacheReuseAdapterV1(CacheReuseAdapter):
    """C++-backed KVCacheManager."""

    def __init__(self, mgr: KVCacheManager) -> None:
        self._mgr = mgr

    @property
    def enable_block_reuse(self) -> bool:
        return self._mgr.enable_block_reuse

    @property
    def tokens_per_block(self) -> int:
        return self._mgr.tokens_per_block

    @property
    def num_extra_kv_tokens(self) -> int:
        return int(getattr(self._mgr, "num_extra_kv_tokens", 0))

    def _global_cached_token_count(self, req: LlmRequest) -> int:
        if not self.enable_block_reuse:
            return 0
        tpb = self.tokens_per_block
        return (req.prepopulated_prompt_len // tpb) * tpb

    def get_slot_ids(self, req, group_idx, lg):  # noqa: ARG002
        if not lg.has_token_axis:
            # V1 hybrid manager keeps the recurrent state outside the C++ KV
            # cache; its per-request slot is the mamba cache index.
            index_map = getattr(self._mgr, "mamba_cache_index", None)
            if index_map is None or req.py_request_id not in index_map:
                return np.array([], dtype=np.int64)
            return np.array([int(index_map[req.py_request_id])], dtype=np.int64)
        first_layer = get_global_layer_ids(lg)[0]
        beam_width = req.py_beam_width
        raw_ids = self._mgr.get_batch_cache_indices(
            [req.py_request_id], layer_idx=first_layer, beam_width=beam_width
        )[0]
        if not raw_ids:
            return np.array([], dtype=np.int64)
        # block_id != primary-pool slot index once host offload kicks in; translate
        # so the cache transceiver's pointer arithmetic is correct. The manager aborts
        # if any referenced block is currently offloaded — disagg transfer cannot read
        # from the secondary pool, and a held block can never be offloaded.
        window_size = lg.live_token_window
        # V1 layer groups carry the manager's window key (full-attention layers get the
        # max window), so this is always set; see page_table_v1.build_page_table.
        assert window_size is not None
        pool_indices = self._mgr.get_memory_pool_block_indices(
            list(raw_ids), window_size=window_size
        )
        return np.asarray(pool_indices, dtype=np.int64)

    def commit_blocks_for_reuse(self, req: LlmRequest) -> None:
        if not self.enable_block_reuse:
            return
        self._mgr.store_blocks_for_reuse(req, pin_blocks=False)


class _CacheReuseAdapterV2(CacheReuseAdapter):
    """Python-based KVCacheManagerV2."""

    def __init__(self, mgr: KVCacheManagerV2) -> None:
        self._mgr = mgr

    @property
    def enable_block_reuse(self) -> bool:
        return self._mgr.enable_block_reuse

    @property
    def tokens_per_block(self) -> int:
        return self._mgr.tokens_per_block

    @property
    def num_extra_kv_tokens(self) -> int:
        return int(getattr(self._mgr, "num_extra_kv_tokens", 0))

    def _global_cached_token_count(self, req: LlmRequest) -> int:
        if not self.enable_block_reuse:
            return 0
        kv_cache = self._mgr.kv_cache_map.get(req.py_request_id)
        if kv_cache is None:
            return 0
        tpb = self.tokens_per_block
        return (kv_cache.num_committed_tokens // tpb) * tpb

    def get_slot_ids(self, req, group_idx, lg):
        # V2 already returns per-cache-level pool slot indices (not logical block
        # IDs), and active sequences GPU-lock their pages (_UniqPageLock enforces
        # cache_level==GPU), so the slot_ids yielded here are already the right
        # offsets for primary-pool pointer arithmetic. No translation is needed,
        # unlike V1 (see _CacheReuseAdapterV1.get_slot_ids).
        kv_cache = self._mgr.kv_cache_map.get(req.py_request_id)
        if kv_cache is None:
            return np.array([], dtype=np.int64)
        if not lg.has_token_axis:
            # Per-request state lives in the life cycle's single SSM slot,
            # which the V2 core tracks separately from the paged blocks.
            slot = int(kv_cache.get_ssm_block_base_index(group_idx))
            if slot == BAD_PAGE_INDEX or slot < 0:
                return np.array([], dtype=np.int64)
            return np.array([slot], dtype=np.int64)
        return np.fromiter(
            kv_cache.get_aggregated_page_indices(group_idx, valid_only=True),
            dtype=np.int64,
        )

    def commit_blocks_for_reuse(self, req: LlmRequest) -> None:
        self._mgr.try_commit_blocks(req)


def create_cache_reuse_adapter(
    mgr: Union[KVCacheManager, KVCacheManagerV2],
) -> CacheReuseAdapter:
    """Factory — pick the right adapter for the concrete manager type."""
    if isinstance(mgr, KVCacheManagerV2):
        return _CacheReuseAdapterV2(mgr)
    return _CacheReuseAdapterV1(mgr)
