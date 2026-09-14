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

"""Wire-level description of one rank's transferable cache memory.

A rank publishes its page table to its peers. It answers two questions and
names no model family or cache type:

* **Where is the memory?** :class:`PhysicalPool` (base, slot pitch, slot
  count) and :attr:`PoolView.buffer_entries` (per-layer byte ranges inside a
  slot). Every consumer addresses bytes as
  ``pool.base_address + slot * pool.slot_stride_bytes + entry.offset``.
* **How is it sharded?** :attr:`PoolView.layout` (:class:`RoleLayout`): how
  many shards the role is split into across the parallel group, which shard
  and replica this rank holds, and the granularity the bytes may be re-split
  at. Peers compare layouts and derive byte ranges with integer arithmetic;
  no side re-derives geometry from tensors or from the other side's parallel
  configuration.

:class:`LayerGroup` is the request-level unit: one slot list per request
covers every view in the group. :attr:`LayerGroup.tokens_per_slot` says
whether that list grows with the prompt or is a single per-request slot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import FrozenSet, List, Optional, Tuple

import numpy as np

BUFFER_ENTRY_DTYPE = np.dtype(
    [
        ("local_layer_id", np.uint32),
        ("offset", np.uint32),
        ("size", np.uint32),
    ]
)


class MapperKind(IntEnum):
    """How one layer's region of a role may be cut when peers hold different shard counts.

    HND: The region (or each of its ``buffers_per_layer`` equal buffers) is a
        contiguous run of shard units, so a finer-sharded peer's data is one
        contiguous sub-range.
    REPLICATED: Every rank holds identical bytes (``num_shards == 1``). Copied
        whole; one sender is elected among the replicas.
    NHD: Unit-minor storage ``[token][unit][...]``: a finer peer's sub-range is
        contiguous only inside one token, so re-splitting emits one fragment
        per ``(layer, buffer, token)``. Requires equal ``tokens_per_slot``.
    SECTIONED: ``[Sec0|Sec1|...]``, each section sharded independently; sizes
        come from :attr:`RoleLayout.section_bytes`.

    A physical pool may hold roles of different layouts, so the page-table
    builder emits one PoolView per ``(physical pool, RoleLayout)``.
    """

    HND = 0
    REPLICATED = 1
    NHD = 2
    SECTIONED = 3


@dataclass(frozen=True)
class RoleLayout:
    """How one role's bytes are sharded across the parallel group.

    Built once per role by the page-table builder (``role_rules.py``) and
    attached to every :class:`PoolView` of that role.

    Fields:
        mapper_kind: See :class:`MapperKind`.
        num_shards: Number of *distinct* shards across the parallel group.
            Ranks holding identical bytes are replicas, not extra shards.
        shard_index: Which shard this rank holds.
        num_replicas: How many ranks hold each shard.
        replica_index: Which replica this rank is; used only to elect one
            sender among ranks holding the same bytes.
        shard_unit_bytes: Smallest unit a layer's region may be re-split at
            for ``HND``/``NHD``; ``None`` when re-splitting is impossible.
        section_bytes: Per-section byte sizes for ``SECTIONED``.
        elem_dtype, elem_shape: Element type and topology-invariant shape of
            one unit; peers only compare them for equality.
    """

    mapper_kind: MapperKind
    num_shards: int = 1
    shard_index: int = 0
    num_replicas: int = 1
    replica_index: int = 0
    shard_unit_bytes: Optional[int] = None
    section_bytes: Optional[Tuple[int, ...]] = None
    elem_dtype: str = ""
    elem_shape: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.mapper_kind, MapperKind):
            raise ValueError(f"Invalid disaggregation mapper kind {self.mapper_kind!r}")
        if self.num_shards <= 0 or not (0 <= self.shard_index < self.num_shards):
            raise ValueError(
                f"RoleLayout shard_index {self.shard_index} out of range for "
                f"num_shards {self.num_shards}"
            )
        if self.num_replicas <= 0 or not (0 <= self.replica_index < self.num_replicas):
            raise ValueError(
                f"RoleLayout replica_index {self.replica_index} out of range for "
                f"num_replicas {self.num_replicas}"
            )
        if self.section_bytes is not None:
            object.__setattr__(self, "section_bytes", tuple(int(b) for b in self.section_bytes))
        object.__setattr__(self, "elem_shape", tuple(int(d) for d in self.elem_shape))
        object.__setattr__(self, "elem_dtype", str(self.elem_dtype))
        if self.mapper_kind == MapperKind.SECTIONED:
            if not self.section_bytes:
                raise ValueError("SECTIONED RoleLayout requires non-empty section_bytes")
            if any(b <= 0 for b in self.section_bytes):
                raise ValueError("RoleLayout section_bytes must all be positive")
        elif self.section_bytes is not None:
            raise ValueError(
                f"section_bytes is only valid for SECTIONED, got {self.mapper_kind.name}"
            )
        if self.mapper_kind == MapperKind.REPLICATED:
            if self.num_shards != 1:
                raise ValueError("REPLICATED RoleLayout must have num_shards == 1")
            if self.shard_unit_bytes is not None:
                raise ValueError("REPLICATED RoleLayout must not declare shard_unit_bytes")
        if self.shard_unit_bytes is not None and self.shard_unit_bytes <= 0:
            raise ValueError("RoleLayout shard_unit_bytes must be positive")

    # ---- wire ---------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "mapper_kind": int(self.mapper_kind),
            "num_shards": int(self.num_shards),
            "shard_index": int(self.shard_index),
            "num_replicas": int(self.num_replicas),
            "replica_index": int(self.replica_index),
            "shard_unit_bytes": (
                int(self.shard_unit_bytes) if self.shard_unit_bytes is not None else None
            ),
            "section_bytes": (
                [int(b) for b in self.section_bytes] if self.section_bytes is not None else None
            ),
            "elem_dtype": self.elem_dtype,
            "elem_shape": [int(d) for d in self.elem_shape],
        }

    @staticmethod
    def from_dict(data: dict) -> "RoleLayout":
        return RoleLayout(
            MapperKind(int(data["mapper_kind"])),
            num_shards=int(data.get("num_shards", 1)),
            shard_index=int(data.get("shard_index", 0)),
            num_replicas=int(data.get("num_replicas", 1)),
            replica_index=int(data.get("replica_index", 0)),
            shard_unit_bytes=(
                int(data["shard_unit_bytes"]) if data.get("shard_unit_bytes") is not None else None
            ),
            section_bytes=(
                tuple(int(b) for b in data["section_bytes"])
                if data.get("section_bytes") is not None
                else None
            ),
            elem_dtype=str(data.get("elem_dtype", "")),
            elem_shape=tuple(int(d) for d in data.get("elem_shape", ())),
        )


@dataclass
class PhysicalPool:
    """One physical memory pool addressed by slot.

    ``num_slots`` slots start at ``base_address`` with pitch
    ``slot_stride_bytes`` (defaults to ``slot_bytes``). Where a layer's bytes
    sit relative to the slot start comes from the ``buffer_entries`` offsets
    of the :class:`PoolView` referencing this pool; those offsets may exceed
    ``slot_bytes`` when the allocation is layer-major.
    """

    base_address: int  # uint64
    slot_bytes: int
    num_slots: int
    slot_stride_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        if self.slot_stride_bytes is None:
            self.slot_stride_bytes = self.slot_bytes
        if self.slot_stride_bytes < self.slot_bytes:
            raise ValueError("slot_stride_bytes must be greater than or equal to slot_bytes")

    def to_dict(self) -> dict:
        return {
            "base_address": int(self.base_address),
            "slot_bytes": int(self.slot_bytes),
            "num_slots": int(self.num_slots),
            "slot_stride_bytes": int(self.slot_stride_bytes),
        }

    @staticmethod
    def from_dict(data: dict) -> "PhysicalPool":
        return PhysicalPool(
            base_address=int(data["base_address"]),
            slot_bytes=int(data["slot_bytes"]),
            num_slots=int(data["num_slots"]),
            slot_stride_bytes=(
                int(data["slot_stride_bytes"])
                if data.get("slot_stride_bytes") is not None
                else None
            ),
        )


@dataclass
class PhysicalPoolGroup:
    pools: List[PhysicalPool]

    def to_dict(self) -> dict:
        return {"pools": [p.to_dict() for p in self.pools]}

    @classmethod
    def from_dict(cls, data: dict) -> "PhysicalPoolGroup":
        return cls(pools=[PhysicalPool.from_dict(p) for p in data.get("pools", [])])


@dataclass(frozen=True)
class LocalLayer:
    """Mapping between a local/internal layer id and a collision-free global layer id."""

    local_layer_id: int
    global_layer_id: int

    def to_dict(self) -> dict:
        return {
            "local_layer_id": int(self.local_layer_id),
            "global_layer_id": int(self.global_layer_id),
        }

    @staticmethod
    def from_dict(data: dict) -> "LocalLayer":
        return LocalLayer(
            local_layer_id=int(data["local_layer_id"]),
            global_layer_id=int(data["global_layer_id"]),
        )


@dataclass
class PoolView:
    """One role class's bytes inside a physical pool, for one layer group.

    Fields:
        pool_idx: Index of the physical pool within its pool group.
        buffer_entries: Structured array using ``BUFFER_ENTRY_DTYPE``. Each
            entry records a buffer's ``local_layer_id`` and its byte ``offset``
            and ``size`` within the pool slot.
        pool_role: Set of the manager's role-name strings living in this
            view. Two peer views match iff their ``pool_role`` sets are equal;
            the transfer code never enumerates the vocabulary.
        layout: This rank's :class:`RoleLayout` for the role class.
        bytes_per_layer: Uniform byte size of one layer's region within the
            slot; the per-layer offsets live in ``buffer_entries``.

    A layer's bytes for slot ``s`` live at ``pool.base_address + s *
    pool.slot_stride_bytes + entry.offset``.
    """

    pool_idx: int
    buffer_entries: np.ndarray  # dtype=BUFFER_ENTRY_DTYPE
    pool_role: FrozenSet[str] = field(default_factory=frozenset)
    layout: RoleLayout = field(default_factory=lambda: RoleLayout(MapperKind.HND))
    bytes_per_layer: Optional[int] = None

    # Read-through accessors for the layout fields consumers use most.
    @property
    def mapper_kind(self) -> MapperKind:
        return self.layout.mapper_kind

    @property
    def num_shards(self) -> int:
        return self.layout.num_shards

    @property
    def shard_index(self) -> int:
        return self.layout.shard_index

    @property
    def num_replicas(self) -> int:
        return self.layout.num_replicas

    @property
    def replica_index(self) -> int:
        return self.layout.replica_index

    @property
    def shard_unit_bytes(self) -> Optional[int]:
        return self.layout.shard_unit_bytes

    @property
    def section_bytes(self) -> Optional[Tuple[int, ...]]:
        return self.layout.section_bytes

    @property
    def shards(self) -> Tuple[int, int]:
        return self.layout.num_shards, self.layout.shard_index

    def to_dict(self) -> dict:
        return {
            "pool_idx": int(self.pool_idx),
            "buffer_entries": self.buffer_entries.tolist(),
            "pool_role": sorted(self.pool_role),
            "layout": self.layout.to_dict(),
            "bytes_per_layer": (
                int(self.bytes_per_layer) if self.bytes_per_layer is not None else None
            ),
        }

    @staticmethod
    def from_dict(data: dict) -> "PoolView":
        # msgpack deserializes tuples as lists; np.array requires tuples for
        # structured dtypes (enforced in numpy >=2.0), so convert explicitly.
        return PoolView(
            pool_idx=int(data["pool_idx"]),
            buffer_entries=np.array(
                [tuple(row) for row in data.get("buffer_entries", [])],
                dtype=BUFFER_ENTRY_DTYPE,
            ),
            pool_role=frozenset(data["pool_role"]),
            layout=RoleLayout.from_dict(data["layout"]),
            bytes_per_layer=(
                int(data["bytes_per_layer"]) if data.get("bytes_per_layer") is not None else None
            ),
        )


@dataclass
class LayerGroup:
    """One life cycle: the set of views a single per-request slot list covers.

    Fields:
        pool_group_idx: Index into ``KVCachePageTable.pool_groups``.
        local_layers: Local ↔ global layer id mapping; peers match layer
            groups by global-id overlap.
        pool_views: Logical views into ``pool_groups[pool_group_idx].pools``.
        tokens_per_slot: Tokens covered by one slot when the request's slot
            list grows with the prompt. ``None`` means the request owns exactly
            one slot regardless of length, so no token-range arithmetic
            applies to this group.
        live_token_window: When set, only the slots covering the most recent
            ``live_token_window`` tokens hold valid data; older slots are
            skipped. Only meaningful with a token axis.
    """

    pool_group_idx: int
    local_layers: List[LocalLayer] = field(default_factory=list)
    pool_views: List[PoolView] = field(default_factory=list)
    tokens_per_slot: Optional[int] = None
    live_token_window: Optional[int] = None

    @property
    def has_token_axis(self) -> bool:
        return self.tokens_per_slot is not None

    def to_dict(self) -> dict:
        return {
            "pool_group_idx": int(self.pool_group_idx),
            "local_layers": [ll.to_dict() for ll in self.local_layers],
            "pool_views": [pv.to_dict() for pv in self.pool_views],
            "tokens_per_slot": (
                int(self.tokens_per_slot) if self.tokens_per_slot is not None else None
            ),
            "live_token_window": (
                int(self.live_token_window) if self.live_token_window is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LayerGroup":
        return cls(
            pool_group_idx=int(data["pool_group_idx"]),
            local_layers=[LocalLayer.from_dict(x) for x in data.get("local_layers", [])],
            pool_views=[PoolView.from_dict(pv) for pv in data.get("pool_views", [])],
            tokens_per_slot=(
                int(data["tokens_per_slot"]) if data.get("tokens_per_slot") is not None else None
            ),
            live_token_window=(
                int(data["live_token_window"])
                if data.get("live_token_window") is not None
                else None
            ),
        )


@dataclass
class KVCachePageTable:
    tokens_per_block: int
    layer_groups: List[LayerGroup]
    pool_groups: List[PhysicalPoolGroup]  # indexed by LayerGroup.pool_group_idx

    def to_dict(self) -> dict:
        return {
            "tokens_per_block": int(self.tokens_per_block),
            "layer_groups": [lg.to_dict() for lg in self.layer_groups],
            "pool_groups": [pg.to_dict() for pg in self.pool_groups],
        }

    @staticmethod
    def from_dict(data: dict) -> "KVCachePageTable":
        return KVCachePageTable(
            tokens_per_block=int(data["tokens_per_block"]),
            layer_groups=[LayerGroup.from_dict(lg) for lg in data.get("layer_groups", [])],
            pool_groups=[PhysicalPoolGroup.from_dict(pg) for pg in data.get("pool_groups", [])],
        )
