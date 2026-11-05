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
"""A byte oracle for page layouts.

Each rank's page holds, at every byte of every buffer, the id of the byte of its role's space lying
there (``RoleLayout.pieces``). After every source rank sends what ``layout_mapping.map_buffers``
gives it, every byte a destination rank holds of every buffer the sources transfer must have
arrived exactly once, holding its own byte of the space, and only from source ranks the destination
pairs with (``layout_mapping.fan``); no run reaches past a pool slot, and no other byte of the
destination's slots is written. A buffer the destination transfers of a model layer the sources
hold, of a role they transfer none of for it, is one the destination rebuilds per request
(``PageLayout.request_rebuilt_roles``).

A buffer of token positions (``RoleLayout.positional``) holds, at the index its layout gives each
position it holds of the page (``RoleLayout.positions``, ``RoleLayout.index_of``), that position's
bytes; a destination must receive exactly the positions both sides hold, each at its own index,
applying runs as the native transfer does (``layout_mapping.page_runs``). Which positions a layout
holds, and where, the oracle takes from the layout: that is the declaration, which tests running
the kernels show to be right.

Use it on hand-written layouts, or on the page layouts real managers build
(``manager.get_page_table().page_layout``) to check a new cache's declarations across the
topologies it is served with. It checks that the ranks' declarations agree with one another; that
they describe what the kernels write is for tests running the kernels to show.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from tensorrt_llm._torch.pyexecutor.kv_cache.layout_mapping import (
    TransferWindow,
    fan,
    layer_group_of,
    map_buffers,
    page_runs,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.page_layout import (
    GRID_DIMENSIONS,
    BufferDesc,
    PageLayout,
    RankTopology,
    grid_place,
)

# (pool group, pool index, layer group): layer groups sharing a pool group draw slots of their
# own, so one page of each holds that group's buffers alone.
Slot = Tuple[int, int, int]


def _slot(groups: Dict[Tuple[int, int], int], buffer: BufferDesc) -> Slot:
    return buffer.pool_group, buffer.pool_index, groups[buffer.pool_group, buffer.layer]


def _slots(page: PageLayout, groups: Dict[Tuple[int, int], int]) -> Dict[Slot, np.ndarray]:
    """An empty slot of each pool for each layer group drawing from it, -1 at every byte."""
    return {
        _slot(groups, b): np.full(page.pool_slot_bytes[b.pool_group][b.pool_index], -1, np.int64)
        for b in page.buffers
    }


class Elements:
    """Ids of the bytes of every (layer, role)'s space, each space numbered from a base of its own.

    A buffer of token positions numbers a position's bytes from that position times the bytes
    between its indices, for page ``page`` of a request reaching token position ``end``
    (exclusive).
    """

    def __init__(self, page: int, end: int) -> None:
        self._bases: Dict[tuple, int] = {}
        self._page = page
        self._end = end

    def _base(self, buffer: BufferDesc) -> int:
        key = (buffer.layer, buffer.role)
        return self._bases.setdefault(key, len(self._bases) << 40)

    def _positions(self, page: PageLayout, buffer: BufferDesc) -> np.ndarray:
        """The positions the buffer holds of the page once the request is written up to ``end``."""
        held = buffer.layout.positions(self._page, self._end, page.tokens_per_page)
        return np.asarray(held, dtype=np.int64)

    def of(
        self, page: PageLayout, buffer: BufferDesc, peer: Optional[BufferDesc] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """The bytes of the buffer on the rank of ``page``.

        Of a buffer of token positions, those of the positions it holds; with ``peer``, the other
        side's buffer of the same layer and role, only those both hold.

        Returns:
            Each byte's offset within the pool slot, and the id of its byte of the space.
        """
        layout = buffer.layout
        pieces = layout.pieces(buffer.size, *grid_place(page.topology, layout.split))
        offsets = buffer.slot_offset + np.concatenate(
            [np.arange(p.offset, p.offset + p.size, dtype=np.int64) for p in pieces]
        )
        ids = np.concatenate([np.arange(p.at, p.at + p.size, dtype=np.int64) for p in pieces])
        if layout.positional:
            positions = self._positions(page, buffer)
            if peer is not None:
                positions = np.intersect1d(positions, self._positions(page, peer))
            stride = layout.stride(buffer.size)
            index = np.asarray(layout.index_of(positions, page.tokens_per_page), dtype=np.int64)
            offsets = (offsets[None, :] + index[:, None] * stride).ravel()
            ids = (ids[None, :] + positions[:, None] * stride).ravel()
        return offsets, ids + self._base(buffer)

    def memory(self, page: PageLayout) -> Dict[Slot, np.ndarray]:
        """One slot of each pool per layer group drawing from it, holding byte ids.

        Each byte holds the id of the byte of the space there, -1 where none lies.
        """
        groups = layer_group_of(page)
        mem = _slots(page, groups)
        for buffer in page.buffers:
            offsets, ids = self.of(page, buffer)
            mem[_slot(groups, buffer)][offsets] = ids
        return mem


def _model_layer(page: PageLayout, groups: Dict[Tuple[int, int], int], buffer: BufferDesc) -> int:
    group = page.layer_groups[groups[buffer.pool_group, buffer.layer]]
    return group.model_layers[group.layers.index(buffer.layer)]


def paired(src: RankTopology, dst: RankTopology) -> bool:
    """Whether the source rank's share overlaps the destination's along every dimension."""
    for dim in GRID_DIMENSIONS:
        src_size, src_rank = grid_place(src, (dim,))
        dst_size, dst_rank = grid_place(dst, (dim,))
        if src_rank not in fan(src_size, dst_size, dst_rank):
            return False
    return True


def assert_layouts_reshard(
    sources: Sequence[PageLayout],
    destinations: Sequence[PageLayout],
    *,
    page: int = 0,
    end: Optional[int] = None,
) -> None:
    """Check the byte oracle for one page of one request.

    Args:
        sources: The page layouts of every source rank serving the request: one attention
            data-parallel group, all pipeline stages and context ranks.
        destinations: Page layouts of destination ranks, each checked against all sources.
        page: The page's position among the request's pages, from 0.
        end: Token position (exclusive) the request's transfer reaches; the page's end by
            default, the page written whole.
    """
    if end is None:
        end = (page + 1) * sources[0].tokens_per_page
    elements = Elements(page, end)
    source_memory = [elements.memory(layout) for layout in sources]
    source_groups = [layer_group_of(layout) for layout in sources]
    sent = {(b.layer, b.role) for layout in sources for b in layout.buffers if b.transferred}
    # The sources' buffers of token positions, so that a destination expects the positions both
    # hold.
    source_positions = {
        (b.layer, b.role): b
        for layout in sources
        for b in layout.buffers
        if b.transferred and b.positional
    }
    sent_models = {
        (_model_layer(layout, groups, b), b.role)
        for layout, groups in zip(sources, source_groups)
        for b in layout.buffers
        if b.transferred
    }
    held_models = {
        _model_layer(layout, groups, b)
        for layout, groups in zip(sources, source_groups)
        for b in layout.buffers
    }
    for dst in destinations:
        dst_groups = layer_group_of(dst)
        for buffer in dst.buffers:
            model = _model_layer(dst, dst_groups, buffer)
            if (
                buffer.transferred
                and model in held_models
                and (model, buffer.role) not in sent_models
            ):
                assert buffer.role in dst.request_rebuilt_roles, (
                    f"{dst.topology} buffer ({buffer.layer}, {buffer.role}) of model layer {model}: "
                    "no source sends it, and the destination does not rebuild it"
                )
        got = _slots(dst, dst_groups)
        writes = {pool: np.zeros(arr.size, dtype=np.int64) for pool, arr in got.items()}
        for src, memory, groups in zip(sources, source_memory, source_groups):
            maps = map_buffers(src, dst)
            if maps:
                assert paired(src.topology, dst.topology), (src.topology, dst.topology)
            for runs in maps:
                src_pool = _slot(groups, runs.src)
                dst_pool = _slot(dst_groups, runs.dst)
                for s, d, n in zip(*page_runs(runs, page, TransferWindow(0, end))):
                    assert n > 0 and 0 <= s and s + n <= memory[src_pool].size, (src_pool, s, n)
                    assert 0 <= d and d + n <= got[dst_pool].size, (dst.topology, dst_pool, d, n)
                    got[dst_pool][d : d + n] = memory[src_pool][s : s + n]
                    writes[dst_pool][d : d + n] += 1
        expected = {pool: np.zeros(arr.size, dtype=bool) for pool, arr in got.items()}
        for buffer in dst.buffers:
            if not buffer.transferred or (buffer.layer, buffer.role) not in sent:
                continue
            pool = _slot(dst_groups, buffer)
            peer = source_positions.get((buffer.layer, buffer.role))
            offsets, ids = elements.of(dst, buffer, peer)
            expected[pool][offsets] = True
            where = f"{dst.topology} buffer ({buffer.layer}, {buffer.role})"
            miscount = np.flatnonzero(writes[pool][offsets] != 1)
            assert not miscount.size, (
                f"{where} byte {offsets[miscount[0]]}: {writes[pool][offsets[miscount[0]]]} writes"
            )
            wrong = np.flatnonzero(got[pool][offsets] != ids)
            assert not wrong.size, (
                f"{where} byte {offsets[wrong[0]]}: byte {got[pool][offsets[wrong[0]]]} of the "
                f"space, not {ids[wrong[0]]}"
            )
        for pool, arr in writes.items():
            stray = np.flatnonzero((arr != 0) & ~expected[pool])
            assert not stray.size, (
                f"{dst.topology} pool {pool} byte {stray[0]}: written, but no transferred buffer "
                "holds it"
            )
