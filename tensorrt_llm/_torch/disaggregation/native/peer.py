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

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.base.region import RegionMapperBase
from tensorrt_llm._torch.disaggregation.native.auxiliary import AuxTransferLayout
from tensorrt_llm._torch.disaggregation.native.mixers.policy import (
    LGPoolKey,
    TransferPolicy,
    match_pool_views,
    validate_page_tables,
)
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.disaggregation.resource.page import MapperKind
from tensorrt_llm._torch.disaggregation.resource.utils import (
    get_layer_byte_ranges,
    get_pool_view_global_layer_ids,
)


@dataclass
class PeerOverlap:
    overlap_pp_size: int = 0
    overlap_tp_size: int = 0
    overlap_cp_size: int = 0
    ranks: List[int] = field(default_factory=list)


class PeerRegistrar:
    """Per-peer bookkeeping: rank info, page-table view matching, mappers, ownership.

    Shard geometry never comes from ``RankInfo`` here; it is read from the two
    page tables' view declarations and compared by :class:`TransferPolicy`.
    ``RankInfo`` is used only for instance-level topology (which peer ranks
    overlap ours in PP/TP/CP, aux ownership, helix constraints).
    """

    def __init__(self, self_rank_info: RankInfo, self_extractor: KVRegionExtractorV1):
        self._ri = self_rank_info
        self._peer_ri_cache: Dict[str, RankInfo] = {}
        self._kv_map_cache: Dict[
            tuple, RegionMapperBase
        ] = {}  # key: (peer_key, self_lg_pool_key, peer_lg_pool_key)
        self._self_ext_cache = self_extractor
        self._peer_ext_cache: Dict[str, KVRegionExtractorV1] = {}
        self._overlap_cache: Dict[str, PeerOverlap] = {}
        self._aux_transfer_layout_cache: Dict[str, AuxTransferLayout] = {}
        self._lg_pool_mapping_cache: Dict[
            str, Dict[LGPoolKey, LGPoolKey]
        ] = {}  # peer_key -> {(self_lg, self_pi) -> (peer_lg, peer_pi)}

    def register(self, peer_name: str, peer_rank: int, peer_ri: RankInfo):
        assert self._self_ext_cache is not None
        mapping = self._check_peer_compatible(peer_ri)
        if mapping is None:
            raise ValueError(
                f"PeerRegistrar.register: peer {peer_name} (rank={peer_rank}) is incompatible with local rank."
            )
        key = self._unique_key(peer_name, peer_rank)
        self._peer_ri_cache[key] = peer_ri
        # Validation already matched every view pair; keep it for get_pool_mapping.
        mapping_key = self._unique_key(peer_ri.instance_name, peer_ri.instance_rank)
        self._lg_pool_mapping_cache[mapping_key] = mapping
        self._aux_transfer_layout_cache.pop(key, None)
        peer_ri = self.get_peer_rank_info(peer_name, peer_rank)
        extractor = KVRegionExtractorV1(peer_ri.page_table)
        self._peer_ext_cache[key] = extractor
        self._warn_nhd_resplit(peer_ri)

    def _warn_nhd_resplit(self, peer_ri: RankInfo) -> None:
        """NHD re-splitting has no contiguous staging path: one descriptor per token."""
        self_pt = self._self_ext_cache.page_table
        peer_pt = peer_ri.page_table
        if self_pt is None or peer_pt is None:
            return
        nhd_fragments_per_token = 0
        for (self_lg, self_pi), (peer_lg, peer_pi) in self.get_pool_mapping(peer_ri).items():
            self_pv = self_pt.layer_groups[self_lg].pool_views[self_pi]
            peer_pv = peer_pt.layer_groups[peer_lg].pool_views[peer_pi]
            if self_pv.mapper_kind == MapperKind.NHD and self_pv.num_shards != peer_pv.num_shards:
                nhd_fragments_per_token += len(self_pv.buffer_entries)
        if nhd_fragments_per_token:
            logger.warning_once(
                "NHD shard-mismatched disaggregated KV transfer has no contiguous "
                "staging path and will emit approximately "
                f"{nhd_fragments_per_token} NIXL descriptors per transferred token "
                "per peer, excluding block-level replicated pools. Long-context "
                "TEP/DEP transfers may have high latency.",
                key=f"native-nhd-shard-mismatch-{nhd_fragments_per_token}",
            )

    def peer_extractor(self, peer_name: str, peer_rank: int) -> KVRegionExtractorV1:
        return self._peer_ext_cache[self._unique_key(peer_name, peer_rank)]

    @property
    def self_extractor(self) -> KVRegionExtractorV1:
        assert self._self_ext_cache is not None
        return self._self_ext_cache

    def unregister(self, peer_name: str, peer_rank: int):
        key = self._unique_key(peer_name, peer_rank)
        if key in self._peer_ri_cache:
            del self._peer_ri_cache[key]
        if key in self._peer_ext_cache:
            del self._peer_ext_cache[key]
        self._aux_transfer_layout_cache.pop(key, None)
        # Clean up kv_map_cache entries for this peer
        keys_to_remove = [k for k in self._kv_map_cache if k[0] == key]
        for k in keys_to_remove:
            del self._kv_map_cache[k]
        if key in self._lg_pool_mapping_cache:
            del self._lg_pool_mapping_cache[key]

    def get_peer_rank_info(self, peer_name: str, peer_rank: int):
        return self._peer_ri_cache[self._unique_key(peer_name, peer_rank)]

    def get_aux_transfer_layout(
        self, peer_name: str, peer_rank: int
    ) -> Optional[AuxTransferLayout]:
        return self._aux_transfer_layout_cache.get(self._unique_key(peer_name, peer_rank))

    def cache_aux_transfer_layout(
        self, peer_name: str, peer_rank: int, layout: AuxTransferLayout
    ) -> None:
        self._aux_transfer_layout_cache[self._unique_key(peer_name, peer_rank)] = layout

    @property
    def self_rank_info(self) -> RankInfo:
        return self._ri

    def _unique_key(self, name: str, rank: int) -> str:
        return name + str(rank)

    def _check_peer_compatible(self, peer_ri: RankInfo) -> Optional[Dict[LGPoolKey, LGPoolKey]]:
        """Instance-level topology rules, then the page-table layout gate.

        Returns the matched view mapping on success and ``None`` when an
        instance-level rule fails. The layout gate raises ``ValueError`` with a
        field-level diagnostic instead of returning None, so the precise
        mismatch reaches the caller of register().
        """
        if self._ri.cp_size != 1 and peer_ri.cp_size != 1:
            logger.warning(
                "PeerRegistrar: incompatible: cp_size must be 1 on at least one side "
                "(helix pairs a cp=1 context instance with a cp=N generation instance); "
                "local=%d peer=%d",
                self._ri.cp_size,
                peer_ri.cp_size,
            )
            return None
        self_pt = self._self_ext_cache.page_table if self._self_ext_cache is not None else None
        peer_pt = peer_ri.page_table
        if (self._ri.cp_size != 1 or peer_ri.cp_size != 1) and (
            self_pt is not None
            and peer_pt is not None
            and self_pt.tokens_per_block != peer_pt.tokens_per_block
        ):
            logger.warning(
                "PeerRegistrar: incompatible: helix block-interleaved transfer requires equal "
                "tokens_per_block on both sides (block boundaries must coincide for "
                "[cp_rank::cp_size] ownership); local=%d peer=%d",
                self_pt.tokens_per_block,
                peer_pt.tokens_per_block,
            )
            return None

        mapping = validate_page_tables(self_pt, peer_pt)

        self_layers = sum(self._ri.layer_num_per_pp)
        peer_layers = sum(peer_ri.layer_num_per_pp)
        if self_layers != peer_layers:
            # Allow mismatch when one side has speculative (e.g. MTP) layers
            # that the other side doesn't. The pool_mapping logic will only
            # transfer layers that exist on both sides.
            logger.warning(
                "PeerRegistrar: layer count differs "
                f"(local={self_layers}, peer={peer_layers}), "
                "allowing partial layer transfer."
            )
        return mapping

    def get_pool_mapping(self, peer_ri: RankInfo) -> Dict[LGPoolKey, LGPoolKey]:
        """Cached ``(self_lg_idx, self_pool_idx) -> (peer_lg_idx, peer_pool_idx)``.

        See :func:`match_pool_views` for the matching rules.
        """
        key = self._unique_key(peer_ri.instance_name, peer_ri.instance_rank)
        if key in self._lg_pool_mapping_cache:
            return self._lg_pool_mapping_cache[key]
        mapping = match_pool_views(self._self_ext_cache.page_table, peer_ri.page_table)
        self._lg_pool_mapping_cache[key] = mapping
        return mapping

    def get_kv_map(
        self,
        peer_ri: RankInfo,
        self_pool_key: LGPoolKey,
        peer_pool_key: LGPoolKey,
    ) -> RegionMapperBase:
        """Get mapper for a specific pool pair.

        Args:
            peer_ri: Peer rank info.
            self_pool_key: (self_lg_idx, self_pool_idx).
            peer_pool_key: (peer_lg_idx, peer_pool_idx).
        """
        peer_key = self._unique_key(peer_ri.instance_name, peer_ri.instance_rank)
        cache_key = (peer_key, self_pool_key, peer_pool_key)
        if cache_key in self._kv_map_cache:
            return self._kv_map_cache[cache_key]

        self_pt = self._self_ext_cache.page_table
        peer_pt = peer_ri.page_table
        assert self_pt is not None
        assert peer_pt is not None
        self_lg_idx, self_pi = self_pool_key
        peer_lg_idx, peer_pi = peer_pool_key
        self_lg = self_pt.layer_groups[self_lg_idx]
        peer_lg = peer_pt.layer_groups[peer_lg_idx]
        self_pv = self_lg.pool_views[self_pi]
        peer_pv = peer_lg.pool_views[peer_pi]

        if self_pv.mapper_kind != peer_pv.mapper_kind:
            raise ValueError(
                "PeerRegistrar.get_kv_map: incompatible mapper kinds "
                f"(local={self_pv.mapper_kind.name}, peer={peer_pv.mapper_kind.name})"
            )

        # Every view is entries-driven: resolve the overlap layers to
        # slot-relative byte offsets on each side from the views' buffer
        # entries. Layer selection is explicit, so mappers never assume a
        # uniform layer stride (other role classes may interleave), and no
        # convention about global-id/byte-offset ordering is needed.
        self_global_ids = get_pool_view_global_layer_ids(self_pv, self_lg)
        peer_global_ids = get_pool_view_global_layer_ids(peer_pv, peer_lg)
        # Iterate the overlap in self's physical slot order (not sorted by
        # global id) so that layers whose regions are contiguous on both
        # sides stay adjacent in the offset arrays and the mappers can merge
        # them into one fragment even when global-id order diverges from the
        # physical layout. Order only affects run merging (a perf property):
        # each layer's byte offset is looked up explicitly below, so any
        # iteration order transfers correct bytes.
        overlap = set(self_global_ids) & set(peer_global_ids)
        overlapping_layers = [gid for gid in self_global_ids if gid in overlap]

        self_starts, self_bytes_per_layer = get_layer_byte_ranges(self_pv)
        peer_starts, peer_bytes_per_layer = get_layer_byte_ranges(peer_pv)
        self_g2l = {ll.global_layer_id: ll.local_layer_id for ll in self_lg.local_layers}
        peer_g2l = {ll.global_layer_id: ll.local_layer_id for ll in peer_lg.local_layers}
        self_layer_offsets = np.array(
            [self_starts[self_g2l[gid]] for gid in overlapping_layers], dtype=np.int64
        )
        peer_layer_offsets = np.array(
            [peer_starts[peer_g2l[gid]] for gid in overlapping_layers], dtype=np.int64
        )

        mapper = TransferPolicy.build_mapper(
            self_pv=self_pv,
            peer_pv=peer_pv,
            self_lg=self_lg,
            peer_lg=peer_lg,
            self_layer_offsets=self_layer_offsets,
            peer_layer_offsets=peer_layer_offsets,
            self_bytes_per_layer=self_bytes_per_layer,
            peer_bytes_per_layer=peer_bytes_per_layer,
        )

        self._kv_map_cache[cache_key] = mapper
        return mapper

    @staticmethod
    def _find_overlap(self_val, peer_val, self_rank, peer_rank=None):
        if self_val <= peer_val:
            overlap = peer_val // self_val
            start = self_rank * overlap + (peer_rank * peer_val if peer_rank is not None else 0)
            end = start + overlap
        else:
            ratio = self_val // peer_val
            start = (self_rank // ratio) + (peer_rank * peer_val if peer_rank is not None else 0)
            overlap = 1
            end = start + overlap

        return overlap, start, end

    def get_peer_overlap(self, peer_rank_info: RankInfo, peer_dp_rank: int) -> PeerOverlap:
        """Which peer ranks this rank exchanges data with (PP x TP x CP overlap).

        Instance-level topology only; whether a given view pair is actually
        sent between two of these ranks is decided per view by
        :meth:`should_send_pool` from the page-table declarations.
        """
        peer_ri = peer_rank_info
        key = self._unique_key(peer_ri.instance_name, peer_dp_rank)
        if key in self._overlap_cache:
            return self._overlap_cache[key]

        # compute pp overlap and target layers
        self_start_layer = sum(self._ri.layer_num_per_pp[: self._ri.pp_rank])
        self_end_layer = self_start_layer + self._ri.layer_num_per_pp[self._ri.pp_rank]

        pre = 0
        tgt_pp_ranks: List[int] = []
        for p in range(peer_ri.pp_size):
            peer_start_layer = pre
            peer_end_layer = peer_start_layer + peer_ri.layer_num_per_pp[p]
            if self_start_layer < peer_end_layer and self_end_layer > peer_start_layer:
                tgt_pp_ranks.append(p)
            pre += peer_ri.layer_num_per_pp[p]

        if tgt_pp_ranks == []:
            targets = PeerOverlap()
            self._overlap_cache[key] = targets
            return targets

        peer_start_pp = tgt_pp_ranks[0]
        overlap_pp_size = len(tgt_pp_ranks)
        peer_end_pp = peer_start_pp + overlap_pp_size

        self_tp_per_dp = self._ri.tp_size_per_dp_group
        peer_tp_per_dp = peer_ri.tp_size_per_dp_group
        self_tp_rank_in_dp = self._ri.tp_rank % self_tp_per_dp

        overlap_tp_size, peer_start_tp, peer_end_tp = self._find_overlap(
            self_tp_per_dp, peer_tp_per_dp, self_tp_rank_in_dp, peer_dp_rank
        )
        overlap_cp_size, peer_start_cp, peer_end_cp = self._find_overlap(
            self._ri.cp_size, peer_ri.cp_size, self._ri.cp_rank
        )

        ranks: List[int] = []
        for pp in range(peer_start_pp, peer_end_pp):
            for tp in range(peer_start_tp, peer_end_tp):
                for cp in range(peer_start_cp, peer_end_cp):
                    # CP-minor flat rank (ppRank*(TP*CP) + tpRank*CP +
                    # cpRank), matching Mapping and the C++ transceiver; a
                    # TP-minor formula mis-routes when tp > 1 and cp > 1.
                    # Loop nesting mirrors the layout so ranks come out in
                    # ascending order.
                    ranks.append(pp * peer_ri.tp_size * peer_ri.cp_size + tp * peer_ri.cp_size + cp)

        targets = PeerOverlap(
            overlap_pp_size=overlap_pp_size,
            overlap_tp_size=overlap_tp_size,
            overlap_cp_size=overlap_cp_size,
            ranks=ranks,
        )
        self._overlap_cache[key] = targets
        return targets

    def should_send_pool(
        self,
        peer_rank_info: RankInfo,
        layer_group_id: int,
        pool_idx: int,
    ) -> bool:
        """Return whether this rank owns the transfer of one view pair.

        ``pool_idx`` indexes the layer group's ``pool_views`` list (one view
        per role class; several views may share a physical pool). The
        decision is :meth:`TransferPolicy.should_send` on the two views'
        shard declarations: unpaired shards never send, and when several
        local replicas hold the bytes one is elected per destination.
        """
        peer_pt = peer_rank_info.page_table
        if peer_pt is None:
            return False
        peer_key = self.get_pool_mapping(peer_rank_info).get((layer_group_id, pool_idx))
        if peer_key is None:
            return False
        self_pv = self._self_ext_cache.page_table.layer_groups[layer_group_id].pool_views[pool_idx]
        peer_pv = peer_pt.layer_groups[peer_key[0]].pool_views[peer_key[1]]
        return TransferPolicy.should_send(self_pv, peer_pv, peer_rank_info.dp_rank)

    def should_send_aux(self, peer_rank_info: RankInfo) -> bool:
        # to ensure the transfer aux is not duplicated

        # TP: only the first rank in each peer-TP-sized group sends aux
        ratio = max(1, self._ri.tp_size_per_dp_group // peer_rank_info.tp_size_per_dp_group)
        self_tp_rank_in_dp_group = self._ri.tp_rank % self._ri.tp_size_per_dp_group
        should_send_in_tp = self_tp_rank_in_dp_group % ratio == 0

        # PP: only the first self-PP rank whose layers overlap with the peer's PP rank sends aux.
        # All tp/pp ranks have the same aux data, so pick the first overlapping one to avoid duplication.
        peer_start_layer = sum(peer_rank_info.layer_num_per_pp[: peer_rank_info.pp_rank])
        peer_end_layer = peer_start_layer + peer_rank_info.layer_num_per_pp[peer_rank_info.pp_rank]
        offset = 0
        for p, n in enumerate(self._ri.layer_num_per_pp):
            if offset < peer_end_layer and offset + n > peer_start_layer:
                return should_send_in_tp and p == self._ri.pp_rank
            offset += n
        return False
