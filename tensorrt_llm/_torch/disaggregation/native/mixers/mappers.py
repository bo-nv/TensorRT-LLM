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

"""Region mappers: turn two peers' slot pointers into (src, dst, bytes) fragments.

Every mapper consumes slot-base pointers from ``KVRegionExtractorV1.extract``
and the slot-relative per-layer byte offsets of the two views, and knows the
two sides' shard positions ``(num_shards, shard_index)``. Nothing here reads
the parallel configuration or knows what model the bytes belong to; the same
five mappers serve attention K/V, replicated side caches and recurrent state.

Shard arithmetic (``shard_split``): when one side holds ``ratio`` times more
shards than the other, the coarser side's region contains the finer side's
region as a contiguous sub-range at ``(fine.shard_index % ratio) * fine_bytes``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import List, Tuple

import numpy as np

from tensorrt_llm._torch.disaggregation.base.region import (
    MemRegionGroup,
    RegionMapperBase,
    SpecRegion,
    SpecRegionPair,
)
from tensorrt_llm._utils import nvtx_range

ShardPos = Tuple[int, int]  # (num_shards, shard_index)


def shard_ratio(self_shards: ShardPos, peer_shards: ShardPos) -> int:
    """``max(num_shards) // min(num_shards)``; raises when not divisible."""
    self_n, _ = self_shards
    peer_n, _ = peer_shards
    larger, smaller = max(self_n, peer_n), min(self_n, peer_n)
    if smaller <= 0 or larger % smaller != 0:
        raise ValueError(f"shard counts must be divisible: local={self_n}, peer={peer_n}")
    return larger // smaller


def shards_paired(self_shards: ShardPos, peer_shards: ShardPos) -> bool:
    """Whether the two shards overlap (one contains the other)."""
    self_n, self_i = self_shards
    peer_n, peer_i = peer_shards
    ratio = shard_ratio(self_shards, peer_shards)
    if self_n <= peer_n:
        return peer_i // ratio == self_i
    return self_i // ratio == peer_i


def shard_split(
    self_shards: ShardPos,
    peer_shards: ShardPos,
    self_bytes: int,
    peer_bytes: int,
) -> Tuple[int, int, int]:
    """Return ``(src_offset, dst_offset, copy_bytes)`` for one shardable region.

    ``self_bytes`` / ``peer_bytes`` are the two sides' sizes of the region
    (one buffer, one section, one token row, ...). The finer side's whole
    region is copied; the coarser side addresses it at an inner offset.
    """
    self_n, self_i = self_shards
    peer_n, peer_i = peer_shards
    if not shards_paired(self_shards, peer_shards):
        raise ValueError(
            f"shards are not paired: local=({self_n}, {self_i}), peer=({peer_n}, {peer_i})"
        )
    ratio = shard_ratio(self_shards, peer_shards)
    if self_n == peer_n:
        if self_bytes != peer_bytes:
            raise ValueError(
                f"equal shard counts but region sizes differ: local={self_bytes}, peer={peer_bytes}"
            )
        return 0, 0, self_bytes
    if self_n < peer_n:
        # Local is coarser: the peer's region is a sub-range of ours.
        if self_bytes != peer_bytes * ratio:
            raise ValueError(
                f"region sizes inconsistent with shard ratio {ratio}: "
                f"local={self_bytes}, peer={peer_bytes}"
            )
        return (peer_i % ratio) * peer_bytes, 0, peer_bytes
    if peer_bytes != self_bytes * ratio:
        raise ValueError(
            f"region sizes inconsistent with shard ratio {ratio}: "
            f"local={self_bytes}, peer={peer_bytes}"
        )
    return 0, (self_i % ratio) * self_bytes, self_bytes


def _as_offsets(name: str, src, dst) -> Tuple[np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    if src.size == 0 or src.size != dst.size:
        raise ValueError(
            f"{name} layer offsets must be non-empty and equal-length: "
            f"src={src.size}, dst={dst.size}"
        )
    return src, dst


def _buffer_bytes(name: str, bytes_per_layer: int, buffers_per_layer: int, side: str) -> int:
    if buffers_per_layer <= 0 or bytes_per_layer % buffers_per_layer != 0:
        raise ValueError(
            f"{name} layer geometry is not evenly divisible ({side}): "
            f"bytes_per_layer={bytes_per_layer}, buffers_per_layer={buffers_per_layer}"
        )
    return bytes_per_layer // buffers_per_layer


class IntactMapper(RegionMapperBase):
    """Copy the selected layers' regions between slots unchanged.

    Used whenever both sides hold the same shard of a role (equal shard
    counts) — attention K/V with equal TP, recurrent state with equal TP,
    replicated side caches. Consumes slot-base pointers and expands them with
    the slot-relative per-layer byte offsets from the views' buffer entries,
    so non-uniform layer strides (interleaved role classes) are handled.

    src slot: [ base ]--+off(L2)--> [L2 region] --+off(L3)--> [L3 region] ...
    dst slot: [ base ]--+off'(L2)-> [L2 region] --+off'(L3)-> [L3 region] ...

    Layers whose regions are contiguous on BOTH sides are merged into one
    fragment, so a fully contiguous class degrades to a single whole-region
    copy per slot.
    """

    def __init__(
        self,
        src_layer_offsets: Sequence[int] | np.ndarray,
        dst_layer_offsets: Sequence[int] | np.ndarray,
        self_bytes_per_layer: int,
        peer_bytes_per_layer: int,
        *,
        mapper_name: str = "Intact",
    ) -> None:
        if self_bytes_per_layer != peer_bytes_per_layer:
            raise ValueError(
                f"{mapper_name} cache region size mismatch: "
                f"local={self_bytes_per_layer}, peer={peer_bytes_per_layer}"
            )
        src, dst = _as_offsets(mapper_name, src_layer_offsets, dst_layer_offsets)
        self._runs = self._merge_contiguous(src, dst, self_bytes_per_layer)

    @staticmethod
    def _merge_contiguous(
        src: np.ndarray, dst: np.ndarray, bytes_per_layer: int
    ) -> list[tuple[int, int, int]]:
        runs: list[tuple[int, int, int]] = []
        run_start = 0
        for i in range(1, src.size + 1):
            if (
                i == src.size
                or src[i] != src[i - 1] + bytes_per_layer
                or dst[i] != dst[i - 1] + bytes_per_layer
            ):
                run_layers = i - run_start
                runs.append(
                    (int(src[run_start]), int(dst[run_start]), run_layers * bytes_per_layer)
                )
                run_start = i
        return runs

    @nvtx_range("IntactMapper.map")
    def map(self, src_regions: SpecRegion, dst_regions: SpecRegion):
        src_group = src_regions.memory
        dst_group = dst_regions.memory
        if src_group.ptrs.size != dst_group.ptrs.size:
            raise ValueError(
                f"Number of regions of src({src_group.ptrs.size}) and "
                f"dst({dst_group.ptrs.size}) must match"
            )
        pairs = [
            SpecRegionPair(
                src=SpecRegion(
                    memory=MemRegionGroup(
                        ptrs=src_group.ptrs + src_off, bytes_per_region=run_bytes
                    ),
                    spec=src_regions.spec,
                ),
                dst=SpecRegion(
                    memory=MemRegionGroup(
                        ptrs=dst_group.ptrs + dst_off, bytes_per_region=run_bytes
                    ),
                    spec=dst_regions.spec,
                ),
            )
            for src_off, dst_off, run_bytes in self._runs
        ]
        return pairs[0] if len(pairs) == 1 else pairs


class ReplicatedMapper(IntactMapper):
    """Copy per-layer regions that are identical on every rank.

    No shard slicing applies; layer selection under partial PP overlap happens
    through the per-layer offsets, and ownership among replicas (one sender
    per destination) is decided upstream by ``TransferPolicy.should_send``.
    """

    def __init__(
        self,
        src_layer_offsets: Sequence[int] | np.ndarray,
        dst_layer_offsets: Sequence[int] | np.ndarray,
        self_bytes_per_layer: int,
        peer_bytes_per_layer: int,
    ) -> None:
        super().__init__(
            src_layer_offsets,
            dst_layer_offsets,
            self_bytes_per_layer,
            peer_bytes_per_layer,
            mapper_name="Replicated",
        )


class _FlatOffsetMapper(RegionMapperBase):
    """Shared ``map`` for mappers that precompute one flat offset array per side.

    ``np.add.outer(slot_ptrs, flat_offsets).ravel()`` emits, for every slot,
    every fragment; all fragments share ``_copy_bytes``.
    """

    _src_flat_offsets: np.ndarray
    _dst_flat_offsets: np.ndarray
    _copy_bytes: int

    @nvtx_range("ShardMapper.map")
    def map(self, src_regions: SpecRegion, dst_regions: SpecRegion) -> SpecRegionPair:
        src_group = src_regions.memory
        dst_group = dst_regions.memory
        if src_group.ptrs.size != dst_group.ptrs.size:
            raise ValueError(
                f"Number of regions of src({src_group.ptrs.size}) and "
                f"dst({dst_group.ptrs.size}) must match"
            )
        all_src_ptrs = np.add.outer(src_group.ptrs, self._src_flat_offsets).ravel()
        all_dst_ptrs = np.add.outer(dst_group.ptrs, self._dst_flat_offsets).ravel()
        return SpecRegionPair(
            src=SpecRegion(
                memory=MemRegionGroup(ptrs=all_src_ptrs, bytes_per_region=self._copy_bytes),
                spec=src_regions.spec,
            ),
            dst=SpecRegion(
                memory=MemRegionGroup(ptrs=all_dst_ptrs, bytes_per_region=self._copy_bytes),
                spec=dst_regions.spec,
            ),
        )


class HNDShardMapper(_FlatOffsetMapper):
    """Re-split a region whose shard units are contiguous (head-major K/V, SSM state).

    Each layer's region consists of ``buffers_per_layer`` equal buffers (K and
    V, or a single state buffer). Within every buffer the finer side's data is
    one contiguous sub-range, so one fragment per ``(layer, buffer)`` suffices.

    Source (layers x heads, coarse):     L0: [S00 S01] [S02 S03]
    Destination (layers x heads, fine):  L0': [D00]  L0'': [D01] ...
    Each arrow copies ``copy_bytes = min(local, peer)`` buffer bytes.
    """

    def __init__(
        self,
        *,
        src_layer_offsets: Sequence[int] | np.ndarray,
        dst_layer_offsets: Sequence[int] | np.ndarray,
        self_bytes_per_layer: int,
        peer_bytes_per_layer: int,
        buffers_per_layer: int,
        self_shards: ShardPos,
        peer_shards: ShardPos,
    ) -> None:
        src, dst = _as_offsets("HND", src_layer_offsets, dst_layer_offsets)
        src_buf = _buffer_bytes("HND", self_bytes_per_layer, buffers_per_layer, "local")
        dst_buf = _buffer_bytes("HND", peer_bytes_per_layer, buffers_per_layer, "peer")
        src_off, dst_off, self._copy_bytes = shard_split(self_shards, peer_shards, src_buf, dst_buf)
        buffers = np.arange(buffers_per_layer, dtype=np.int64)
        self._src_flat_offsets = (src[:, None] + src_buf * buffers[None, :] + src_off).ravel()
        self._dst_flat_offsets = (dst[:, None] + dst_buf * buffers[None, :] + dst_off).ravel()


class NHDShardMapper(_FlatOffsetMapper):
    """Re-split token-major ``[token, head, dim]`` buffers.

    The finer side's head range is contiguous only inside one token, so this
    emits one fragment per ``(layer, buffer, token)``. Requires both sides to
    use the same ``tokens_per_slot``.
    """

    def __init__(
        self,
        *,
        src_layer_offsets: Sequence[int] | np.ndarray,
        dst_layer_offsets: Sequence[int] | np.ndarray,
        self_bytes_per_layer: int,
        peer_bytes_per_layer: int,
        buffers_per_layer: int,
        tokens_per_slot: int,
        self_shards: ShardPos,
        peer_shards: ShardPos,
    ) -> None:
        src, dst = _as_offsets("NHD", src_layer_offsets, dst_layer_offsets)
        if tokens_per_slot <= 0:
            raise ValueError("NHD mapper requires a positive tokens_per_slot")
        src_buf = _buffer_bytes("NHD", self_bytes_per_layer, buffers_per_layer, "local")
        dst_buf = _buffer_bytes("NHD", peer_bytes_per_layer, buffers_per_layer, "peer")
        if src_buf % tokens_per_slot or dst_buf % tokens_per_slot:
            raise ValueError(
                "NHD buffer bytes are not divisible by tokens_per_slot: "
                f"local={src_buf}, peer={dst_buf}, tokens_per_slot={tokens_per_slot}"
            )
        src_tok = src_buf // tokens_per_slot
        dst_tok = dst_buf // tokens_per_slot
        src_off, dst_off, self._copy_bytes = shard_split(self_shards, peer_shards, src_tok, dst_tok)
        buffers = np.arange(buffers_per_layer, dtype=np.int64)
        tokens = np.arange(tokens_per_slot, dtype=np.int64)
        self._src_flat_offsets = (
            src[:, None, None]
            + src_buf * buffers[None, :, None]
            + src_tok * tokens[None, None, :]
            + src_off
        ).ravel()
        self._dst_flat_offsets = (
            dst[:, None, None]
            + dst_buf * buffers[None, :, None]
            + dst_tok * tokens[None, None, :]
            + dst_off
        ).ravel()


class SectionedMapper(RegionMapperBase):
    """Re-split ``[Sec0|Sec1|...]`` regions where each section is sharded independently.

    Convolution state of Mamba2 (``[x|B|C]``) and GDN (``[q|k|v]``): a TP
    change re-splits every section on its own, so one fragment per
    ``(layer, section)`` with a per-section size. Returns one
    ``SpecRegionPair`` per section (sections may differ in size).
    """

    def __init__(
        self,
        *,
        src_layer_offsets: Sequence[int] | np.ndarray,
        dst_layer_offsets: Sequence[int] | np.ndarray,
        self_section_bytes: Sequence[int],
        peer_section_bytes: Sequence[int],
        self_shards: ShardPos,
        peer_shards: ShardPos,
    ) -> None:
        if len(self_section_bytes) != len(peer_section_bytes) or not self_section_bytes:
            raise ValueError(
                f"Section count mismatch: local={len(self_section_bytes)}, "
                f"peer={len(peer_section_bytes)}"
            )
        src, dst = _as_offsets("Sectioned", src_layer_offsets, dst_layer_offsets)
        self._plans: List[Tuple[np.ndarray, np.ndarray, int]] = []
        src_start = 0
        dst_start = 0
        for self_sec, peer_sec in zip(self_section_bytes, peer_section_bytes):
            src_off, dst_off, copy_bytes = shard_split(
                self_shards, peer_shards, int(self_sec), int(peer_sec)
            )
            self._plans.append((src + src_start + src_off, dst + dst_start + dst_off, copy_bytes))
            src_start += int(self_sec)
            dst_start += int(peer_sec)

    @nvtx_range("SectionedMapper.map")
    def map(self, src_regions: SpecRegion, dst_regions: SpecRegion) -> List[SpecRegionPair]:
        src_group = src_regions.memory
        dst_group = dst_regions.memory
        if src_group.ptrs.size != dst_group.ptrs.size:
            raise ValueError(
                f"Number of regions of src({src_group.ptrs.size}) and "
                f"dst({dst_group.ptrs.size}) must match"
            )
        return [
            SpecRegionPair(
                src=SpecRegion(
                    memory=MemRegionGroup(
                        ptrs=np.add.outer(src_group.ptrs, src_offs).ravel(),
                        bytes_per_region=copy_bytes,
                    ),
                    spec=src_regions.spec,
                ),
                dst=SpecRegion(
                    memory=MemRegionGroup(
                        ptrs=np.add.outer(dst_group.ptrs, dst_offs).ravel(),
                        bytes_per_region=copy_bytes,
                    ),
                    spec=dst_regions.spec,
                ),
            )
            for src_offs, dst_offs, copy_bytes in self._plans
        ]
