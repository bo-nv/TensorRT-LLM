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
"""How two ranks' pages correspond, read from their page layouts alone.

A buffer's layout (``page_layout.RoleLayout``) says which pieces of its role's space a rank holds.
From a source's and a destination's layouts this module derives what the source sends; nothing
here names a cache, a model or a layout:

- ``fan``: which source ranks a destination hears from, by each dimension's shares of ranks.
- ``map_buffers``: the byte runs of every buffer both ranks transfer, once per pair of ranks: each
  byte of the space the destination holds comes from the elected one of the paired source ranks
  holding it.
- ``page_runs``: what one page moves; for a layout of token positions, every position both sides
  hold (``RoleLayout.positions``) moved to its own index on each side (``RoleLayout.index_of``).
- ``pair_pages``: which slots of a request pair in one transfer, as the layer group's kind says.
- ``check_compatible``: whether two ranks' layouts may pair at all.

Example: K of one layer, 8 KV heads of 256 bytes, head-major, from TP2 to TP4. Destination rank 1
holds heads 2..3, bytes 512..1023 of the space. It hears from source rank 0 alone
(``fan(2, 4, 1) == range(0, 1)``), which holds heads 0..3 and sends heads 2..3 as one run of 512
bytes.

Election: where several paired source ranks hold a byte (heads every rank keeps, or a dimension the
layout does not split over), the one at position ``key`` modulo their count sends it, ``key`` being
the destination's attention data-parallel rank, so that the destinations of several data-parallel
groups spread over the sources. A destination holding a byte no paired source rank holds, as an
uneven head split can make, is refused rather than left without it.

Adding a layout: a ``RoleLayout`` subclass says which pieces of the space each rank holds; this
module derives the rest. A layer-group kind registers which slots pair (``pair_slots``).
"""

from __future__ import annotations

import functools
import itertools
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .page_layout import (
    GRID_DIMENSIONS,
    BufferDesc,
    GroupKind,
    Paged,
    PageLayout,
    PerRequest,
    Piece,
    RankTopology,
    RoleLayout,
    grid_place,
)

__all__ = [
    "BufferRuns",
    "PagePairs",
    "Pairing",
    "PositionRule",
    "TransferWindow",
    "check_compatible",
    "fan",
    "layer_group_of",
    "map_buffers",
    "merge_runs",
    "page_runs",
    "pair_pages",
    "pair_slots",
    "paired_ordinals",
    "sent_bytes",
]


def fan(src_size: int, dst_size: int, dst_rank: int) -> range:
    """The source ranks, of ``src_size`` along one parallel dimension, whose equal share of it
    overlaps the share of destination rank ``dst_rank`` of ``dst_size``.

    ``fan(2, 4, 1) == range(0, 1)``: TP4 rank 1 holds a quarter lying within TP2 rank 0's half.
    ``fan(4, 2, 1) == range(2, 4)``: TP2 rank 1 holds the halves of TP4 ranks 2 and 3.
    """
    return range(dst_rank * src_size // dst_size, -(-(dst_rank + 1) * src_size // dst_size))


@dataclass(frozen=True)
class TransferWindow:
    """What one transfer of a request covers, in the request's token positions.

    A whole prompt of 100 tokens of which the destination reuses the first 32 is
    ``TransferWindow(0, 100, reused=32)``; the first of two pieces of it is
    ``TransferWindow(0, 50, reused=32)``.

    Attributes:
        start: First token position the transfer moves.
        end: Token position the transfer reaches, exclusive.
        reused: Token positions from the request's start that the destination already holds, as
            a prefix it reuses; their pages are not paired.
    """

    start: int
    end: int
    reused: int = 0


@dataclass(frozen=True)
class Pairing:
    """A source rank as it sends to one destination rank.

    Attributes:
        src: The source rank's place.
        fans: For each dimension of ``GRID_DIMENSIONS``, the source ranks along it that the
            destination pairs with (``fan``).
        key: Picks the sender among paired source ranks holding the same index: the one at
            ``key`` modulo their count (``elect``).
    """

    src: RankTopology
    fans: Mapping[str, range]
    key: int

    @classmethod
    def between(cls, src: RankTopology, dst: RankTopology, key: int) -> Pairing:
        """The source rank at ``src`` as it sends to the destination rank at ``dst``."""
        fans = {}
        for dim in GRID_DIMENSIONS:
            src_size, _ = grid_place(src, (dim,))
            dst_size, dst_rank = grid_place(dst, (dim,))
            fans[dim] = fan(src_size, dst_size, dst_rank)
        return cls(src, fans, key)

    def place(self, split: Sequence[str]) -> tuple[int, int]:
        """``(size, rank)`` of the source rank in the grid of the dimensions ``split``."""
        return grid_place(self.src, split)

    def paired(self, split: Sequence[str]) -> list[int]:
        """The source ranks the destination pairs with, by place in the grid of ``split``,
        ascending."""
        sizes = [grid_place(self.src, (dim,))[0] for dim in split]
        places = itertools.product(*(self.fans[dim] for dim in split))
        return sorted({_flat(coords, sizes) for coords in places})

    def elect(self, candidates: Sequence):
        """The one of ``candidates`` that sends: the one at ``key`` modulo their count."""
        return candidates[self.key % len(candidates)]

    def sends_replicas(self, split: Sequence[str]) -> bool:
        """Whether this source rank sends a buffer along the dimensions its layout does not split
        over (``split`` being those it does), whose ranks hold it alike: whether it is the elected
        one of the paired ranks there."""
        free = [dim for dim in GRID_DIMENSIONS if dim not in split]
        if not free:
            return True
        mine = tuple(grid_place(self.src, (dim,))[1] for dim in free)
        return self.elect(list(itertools.product(*(self.fans[dim] for dim in free)))) == mine


def _flat(coords: Sequence[int], sizes: Sequence[int]) -> int:
    rank = 0
    for c, n in zip(coords, sizes):
        rank = rank * n + c
    return rank


@dataclass(frozen=True)
class PositionRule:
    """How a buffer's runs move with the token positions a page holds: its layout on the source and
    on the destination. A position both sides hold of a page (``RoleLayout.positions``) moves from
    its index on the source to its index on the destination (``RoleLayout.index_of``), ``stride``
    bytes per index on each.

    In pages of 32 tokens, of page 1 (positions 32..63) written up to position 50: a source layout
    holding 34..49 and a destination layout holding 32..49 move 34..49, position 40 from the index
    the source's layout gives it to the index the destination's layout gives it.

    Attributes:
        src: The source's layout.
        dst: The destination's layout.
        src_stride: Bytes per index on the source.
        dst_stride: Bytes per index on the destination.
        tokens_per_page: Token positions of one page.
    """

    src: RoleLayout
    dst: RoleLayout
    src_stride: int
    dst_stride: int
    tokens_per_page: int

    def positions(self, page: int, window: TransferWindow) -> np.ndarray:
        """The token positions of page ``page`` (its position among a request's pages, from 0)
        that both sides hold once the request has written every position below ``window.end``,
        as int64."""
        src = self.src.positions(page, window.end, self.tokens_per_page)
        dst = self.dst.positions(page, window.end, self.tokens_per_page)
        return np.intersect1d(np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64))

    @property
    def kept(self) -> int:
        """The most positions of one page a transfer moves: those both hold of a page written
        whole."""
        return int(self.positions(0, TransferWindow(0, self.tokens_per_page)).size)

    def offsets(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Each position's byte offset from index 0, on the source and on the destination."""
        src_index = self.src.index_of(positions, self.tokens_per_page)
        dst_index = self.dst.index_of(positions, self.tokens_per_page)
        return src_index * self.src_stride, dst_index * self.dst_stride


@dataclass(frozen=True, eq=False)
class BufferRuns:
    """The bytes a source rank sends of one buffer the destination rank also holds: run ``i`` is
    ``sizes[i]`` bytes from ``src_offsets[i]`` of the source page's pool slot to ``dst_offsets[i]``
    of the destination's. For a buffer of token positions, these are the runs of the position at
    index 0 on both sides, and ``page_runs`` gives what a page moves.

    Attributes:
        src: The source rank's buffer.
        dst: The destination rank's buffer of the same layer and role.
        src_offsets: int64 offsets within the source buffer's pool slot.
        dst_offsets: int64 offsets within the destination buffer's pool slot.
        sizes: int64 run lengths in bytes.
        by_position: How the runs move with the token positions a page holds (``page_runs``):
            None for a buffer laid out alike for every position, and equal for buffers whose runs
            move alike.
    """

    src: BufferDesc
    dst: BufferDesc
    src_offsets: np.ndarray
    dst_offsets: np.ndarray
    sizes: np.ndarray
    by_position: PositionRule | None = None

    @property
    def fixed(self) -> bool:
        """Whether it moves the same runs of every page, which ``page_runs`` then returns."""
        return self.by_position is None

    @property
    def page_bytes(self) -> int:
        """The most bytes it moves of one page."""
        sent = int(self.sizes.sum())
        return sent if self.by_position is None else sent * self.by_position.kept


def page_runs(
    runs: BufferRuns, page: int, window: TransferWindow
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The runs that move page ``page`` (its position among a request's pages, from 0) in a
    transfer over ``window``: ``runs``' own for a buffer laid out alike for every position; else
    those of every position both sides hold, each from its index on the source to its index on
    the destination (``PositionRule``)."""
    rule = runs.by_position
    if rule is None:
        return runs.src_offsets, runs.dst_offsets, runs.sizes
    src_at, dst_at = rule.offsets(rule.positions(page, window))
    src = src_at[:, None] + runs.src_offsets[None, :]
    dst = dst_at[:, None] + runs.dst_offsets[None, :]
    sizes = np.broadcast_to(runs.sizes, src.shape).ravel()
    return merge_runs(src.ravel(), dst.ravel(), sizes)


def _arrays(pieces: Sequence[Piece]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``pieces`` as int64 arrays of coordinates, offsets and sizes, in order of coordinate."""
    ordered = sorted(pieces, key=lambda p: p.at)
    return tuple(
        np.fromiter((getattr(p, name) for p in ordered), dtype=np.int64, count=len(ordered))
        for name in ("at", "offset", "size")
    )


def _covering(pieces: tuple[np.ndarray, ...], lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """For each stretch ``[lo[i], hi[i])`` of the space, the index of the piece of ``pieces``
    (``_arrays``) holding it whole, -1 where none does."""
    at, _, size = pieces
    if at.size == 0:
        return np.full(lo.size, -1, dtype=np.int64)
    k = np.searchsorted(at, lo, side="right") - 1
    clamped = np.maximum(k, 0)
    held = (k >= 0) & (hi <= at[clamped] + size[clamped])
    return np.where(held, k, -1)


def _layout_runs(
    src: BufferDesc, dst: BufferDesc, pairing: Pairing, dst_topology: RankTopology
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """What the source sends of one buffer: the bytes of the space the destination holds that the
    source is the elected sender of, as offsets within each side's buffer and sizes; None when it
    sends none.

    The space is cut at every piece's bounds on either side, so each stretch lies whole in one
    piece of every rank holding it; each stretch the destination holds goes from the elected one
    of the paired source ranks holding it (``Pairing.elect``).

    Raises:
        ValueError: If the destination holds a byte no paired source rank holds.
    """
    split = src.layout.split
    if not pairing.sends_replicas(split):
        return None
    ranks, me = pairing.place(split)
    places = pairing.paired(split)
    held = [_arrays(src.layout.pieces(src.size, ranks, place)) for place in places]
    want = _arrays(dst.layout.pieces(dst.size, *grid_place(dst_topology, dst.layout.split)))
    edges = np.unique(
        np.concatenate(
            [want[0], want[0] + want[2], *(h[0] for h in held), *(h[0] + h[2] for h in held)]
        )
    )
    lo, hi = edges[:-1], edges[1:]
    into = _covering(want, lo, hi)
    lo, hi, into = lo[into >= 0], hi[into >= 0], into[into >= 0]
    cover = np.stack([_covering(h, lo, hi) for h in held])
    holders = cover >= 0
    count = holders.sum(axis=0)
    if (count == 0).any():
        i = int(np.argmax(count == 0))
        raise ValueError(
            f"buffer ({src.layer}, {src.role}): bytes {int(lo[i])}..{int(hi[i]) - 1} of its space, "
            "which the destination holds, are held by no source rank it pairs with"
        )
    if me not in places:
        return None
    # The elected holder of each stretch: the one at ``key`` modulo their count, in place order.
    order = np.cumsum(holders, axis=0) - 1
    elected = np.argmax(holders & (order == pairing.key % np.maximum(count, 1)), axis=0)
    mine = elected == places.index(me)
    if not mine.any():
        return None
    k = cover[places.index(me), mine]
    at, offset, _ = held[places.index(me)]
    lo, hi, into = lo[mine], hi[mine], into[mine]
    return offset[k] + lo - at[k], want[1][into] + lo - want[0][into], hi - lo


def merge_runs(
    src_offsets: np.ndarray, dst_offsets: np.ndarray, sizes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The runs with each that continues the one before it on both sides merged into it, in source
    order: runs (0 -> 100, 8 bytes) and (8 -> 108, 8 bytes) become (0 -> 100, 16 bytes)."""
    if src_offsets.size == 0:
        return src_offsets, dst_offsets, sizes
    order = np.argsort(src_offsets, kind="stable")
    src_offsets, dst_offsets, sizes = src_offsets[order], dst_offsets[order], sizes[order]
    starts = np.ones(src_offsets.size, dtype=bool)
    starts[1:] = (src_offsets[1:] != src_offsets[:-1] + sizes[:-1]) | (
        dst_offsets[1:] != dst_offsets[:-1] + sizes[:-1]
    )
    first = np.flatnonzero(starts)
    merged = np.add.reduceat(sizes, first)
    return src_offsets[first], dst_offsets[first], merged


def _space(layout: PageLayout, buffer: BufferDesc) -> tuple:
    """What the coordinates of the buffer's space mean on the rank of ``layout``
    (``RoleLayout.space``)."""
    return buffer.layout.space(buffer.size, grid_place(layout.topology, buffer.layout.split)[0])


def _position_rule(
    src: PageLayout,
    src_buffer: BufferDesc,
    dst: PageLayout,
    dst_buffer: BufferDesc,
    groups: tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]],
) -> PositionRule | None:
    """How a buffer both ranks transfer moves with token position, or None when it holds no token
    positions; ``groups`` are both layouts' ``layer_group_of``.

    Raises:
        ValueError: If one side's buffer holds token positions and the other's does not, or such a
            buffer lies in a layer group holding one slot per request.
    """
    if not src_buffer.positional and not dst_buffer.positional:
        return None
    where = f"buffer ({src_buffer.layer}, {src_buffer.role})"
    if src_buffer.positional != dst_buffer.positional:
        names = [b.layout.name if b.positional else None for b in (src_buffer, dst_buffer)]
        raise ValueError(f"{where}: token positions of {names[0]!r} against {names[1]!r}")
    for page, buffer, group_of in ((src, src_buffer, groups[0]), (dst, dst_buffer, groups[1])):
        group = page.layer_groups[group_of[buffer.pool_group, buffer.layer]]
        if group.kind.per_request:
            raise ValueError(
                f"{where}: a {buffer.layout.name} of token positions in a {group.kind} layer "
                "group, which holds one slot per request"
            )
    return PositionRule(
        src_buffer.layout,
        dst_buffer.layout,
        src_buffer.layout.stride(src_buffer.size),
        dst_buffer.layout.stride(dst_buffer.size),
        src.tokens_per_page,
    )


def map_buffers(src: PageLayout, dst: PageLayout, *, key: int | None = None) -> list[BufferRuns]:
    """What the rank of ``src`` sends the rank of ``dst``: for each buffer both transfer (same
    global layer and role), the runs of the bytes of its space the source is the elected sender
    of. Buffers it sends nothing of are left out. A buffer of token positions gives the runs of
    one position, which ``page_runs`` moves to each position of a page.

    Example: K of one layer, 8 heads of 256 bytes head-major, from TP2 rank 0 to TP4 rank 1: one
    ``BufferRuns`` of a single run, 512 bytes from byte ``s + 512`` of the source's slot to byte
    ``d`` of the destination's, ``s`` and ``d`` the buffer's offset in each slot.

    Args:
        src: The source rank's page layout.
        dst: The destination rank's page layout.
        key: Breaks ties between source ranks holding the same bytes; the destination's attention
            data-parallel rank by default.

    Raises:
        ValueError: If two matched buffers lay out different spaces (``RoleLayout.space``), one
            holds token positions and the other does not, such a buffer lies in a group holding
            one slot per request, or the destination holds a byte no source rank it pairs with
            holds.
    """
    pairing = Pairing.between(
        src.topology, dst.topology, dst.topology.dp_rank if key is None else key
    )
    groups = (layer_group_of(src), layer_group_of(dst))
    dst_buffers = {(b.layer, b.role): b for b in dst.buffers if b.transferred}
    out = []
    for src_buffer in src.buffers:
        dst_buffer = dst_buffers.get((src_buffer.layer, src_buffer.role))
        if dst_buffer is None or not src_buffer.transferred:
            continue
        rule = _position_rule(src, src_buffer, dst, dst_buffer, groups)
        spaces = _space(src, src_buffer), _space(dst, dst_buffer)
        if spaces[0] != spaces[1]:
            raise ValueError(
                f"buffer ({src_buffer.layer}, {src_buffer.role}): space {spaces[0]} against "
                f"{spaces[1]}"
            )
        runs = _layout_runs(src_buffer, dst_buffer, pairing, dst.topology)
        if runs is None:
            continue
        src_offsets, dst_offsets, sizes = runs
        merged = merge_runs(
            src_offsets + src_buffer.slot_offset, dst_offsets + dst_buffer.slot_offset, sizes
        )
        out.append(BufferRuns(src_buffer, dst_buffer, *merged, by_position=rule))
    return out


@dataclass(frozen=True, eq=False)
class PagePairs:
    """The slots one transfer of a request pairs in a layer group: source slot ``src_slots[i]``
    sends into destination slot ``dst_slots[i]``.

    Attributes:
        src_slots: int64 slot indices on the source.
        dst_slots: int64 slot indices on the destination.
        pages: int64 position of each pair's page among the request's pages, from 0, for runs
            that move by token position; None for a group holding one slot per request.
    """

    src_slots: np.ndarray
    dst_slots: np.ndarray
    pages: np.ndarray | None = None


def paired_ordinals(src_ids: np.ndarray, dst_ids: np.ndarray, start: int = 0) -> np.ndarray:
    """The indices of two block tables, both indexed by page and -1 where a side holds no block,
    at which both hold one, from index ``start`` on: tables ``[10, 11, -1, 13]`` and
    ``[20, -1, 22, 23]`` pair at indices 0 and 3.

    A block moves only where both sides hold it, so a sender that has not evicted yet, a receiver
    that already reused a prefix, a receiver with a wider speculative window, or a sender chunk
    that blanks everything outside its range all fall out of the same intersection.

    Raises:
        ValueError: If the tables differ in length.
    """
    if src_ids.size != dst_ids.size:
        raise ValueError(
            f"positional block tables differ in length: src={src_ids.size} dst={dst_ids.size}; "
            "both sides must span the same prompt"
        )
    mask = (src_ids >= 0) & (dst_ids >= 0)
    if start > 0:
        mask[:start] = False
    return np.flatnonzero(mask)


@functools.singledispatch
def pair_slots(
    kind: GroupKind,
    src: PageLayout,
    src_ids: np.ndarray,
    dst: PageLayout,
    dst_ids: np.ndarray,
    window: TransferWindow,
) -> PagePairs:
    """The slots of a layer group of kind ``kind`` that one transfer over ``window`` pairs, from
    the source's slot ids ``src_ids`` and the destination's ``dst_ids`` for the request. Each
    layer-group kind registers its own.

    Raises:
        ValueError: If ``kind`` registers no pairing.
    """
    raise ValueError(f"layer-group kind {kind.name!r} registers no pair_slots")


_NO_SLOTS = np.zeros(0, dtype=np.int64)


@pair_slots.register(Paged)
def _pair_paged(
    kind: Paged,
    src: PageLayout,
    src_ids: np.ndarray,
    dst: PageLayout,
    dst_ids: np.ndarray,
    window: TransferWindow,
) -> PagePairs:
    """A slot per page: pages both hold, past the destination's reused prefix. A Helix
    destination at CP rank ``r`` of ``n`` holds pages ``r, r + n, ...`` of a source holding them
    all; reuse is refused under Helix."""
    src_ids = np.asarray(src_ids, dtype=np.int64)
    dst_ids = np.asarray(dst_ids, dtype=np.int64)
    cp_size, cp_rank = dst.topology.cp_size, dst.topology.cp_rank
    helix = cp_size > 1 and src.topology.cp_size == 1
    if helix:
        src_ids = src_ids[cp_rank::cp_size]
    paired = paired_ordinals(src_ids, dst_ids, window.reused // src.tokens_per_page)
    pages = paired * cp_size + cp_rank if helix else paired
    return PagePairs(src_ids[paired], dst_ids[paired], pages)


@pair_slots.register(PerRequest)
def _pair_per_request(
    kind: PerRequest,
    src: PageLayout,
    src_ids: np.ndarray,
    dst: PageLayout,
    dst_ids: np.ndarray,
    window: TransferWindow,
) -> PagePairs:
    """One slot per request: the first slot of each side, when both hold one."""
    if len(src_ids) == 0 or len(dst_ids) == 0:
        return PagePairs(_NO_SLOTS, _NO_SLOTS)
    return PagePairs(
        np.asarray(src_ids[:1], dtype=np.int64), np.asarray(dst_ids[:1], dtype=np.int64)
    )


def pair_pages(
    src: PageLayout,
    src_group: int,
    src_ids: np.ndarray,
    dst: PageLayout,
    dst_group: int,
    dst_ids: np.ndarray,
    window: TransferWindow,
) -> PagePairs:
    """The slots one transfer over ``window`` pairs between layer group ``src_group`` of ``src``,
    holding the request in slots ``src_ids``, and ``dst_group`` of ``dst``, holding it in
    ``dst_ids``, as the groups' kind registers (``pair_slots``).

    Example: attention pages ``[10, 11, 12]`` and ``[20, 21, 22]`` with the first page reused
    pair slots 11 -> 21 and 12 -> 22, pages 1 and 2.

    Raises:
        ValueError: If the groups differ in kind, or as the kind's pairing does.
    """
    kind = src.layer_groups[src_group].kind
    if dst.layer_groups[dst_group].kind != kind:
        raise ValueError(
            f"layer group {src_group} of {kind} against {dst_group} of "
            f"{dst.layer_groups[dst_group].kind}"
        )
    return pair_slots(kind, src, src_ids, dst, dst_ids, window)


def layer_group_of(layout: PageLayout) -> dict[tuple[int, int], int]:
    """(pool group, global layer) -> index of the layer group holding that layer's buffers there.

    Raises:
        ValueError: If a layer lies in two layer groups of one pool group.
    """
    out: dict[tuple[int, int], int] = {}
    for index, group in enumerate(layout.layer_groups):
        for layer in group.layers:
            if out.setdefault((group.pool_group, layer), index) != index:
                raise ValueError(
                    f"layer {layer} lies in layer groups {out[group.pool_group, layer]} and "
                    f"{index} of pool group {group.pool_group}"
                )
    return out


def check_compatible(a: PageLayout, b: PageLayout) -> None:
    """Check that two ranks' pages correspond, whichever sends. Refused, each naming the
    difference:

    - KV of another element type or quantization.
    - On the layers both hold, a buffer one rank alone transfers, unless that rank rebuilds its
      role per request (``PageLayout.request_rebuilt_roles``: CSA2's generation workers rebuild
      the sliding caches a context-only worker keeps none of past its split); otherwise buffers
      of other sub-pages, spaces (``RoleLayout.space``), element types, layer-group kinds or
      model layers. Heads split over different rank counts lay out one space, and so may layouts
      of token positions each side sizes as it keeps.
    - A model layer's role whose buffers the two name by other global layers, or that one rank
      alone transfers of a model layer both hold, which would move one layer's bytes into
      another's or leave it unwritten.
    - Pages of different token counts where both transfer buffers of paged groups.

    How many tokens each side keeps of a windowed group is not compared: retention margins (reuse
    backoff, draft tokens) legitimately differ between the sides, and a transfer moves the blocks
    both still hold.

    Raises:
        ValueError: Naming every difference found.
    """
    groups = (layer_group_of(a), layer_group_of(b))
    problems = _kv_problems(a, b) + _buffer_problems(a, b, groups) + _naming_problems(a, b, groups)
    if _pages_pair(a, b, groups) and a.tokens_per_page != b.tokens_per_page:
        problems.append(f"pages of {a.tokens_per_page} tokens against {b.tokens_per_page}")
    if problems:
        raise ValueError("page layouts do not correspond: " + "; ".join(problems))


def _kv_problems(a: PageLayout, b: PageLayout) -> list[str]:
    problems = []
    if a.dtype != b.dtype:
        problems.append(f"KV elements of type {a.dtype} against {b.dtype}")
    if a.quant_mode != b.quant_mode:
        problems.append(f"KV quantization {a.quant_mode} against {b.quant_mode}")
    return problems


def _transferred(layout: PageLayout) -> dict[tuple[int, str], BufferDesc]:
    return {(b.layer, b.role): b for b in layout.buffers if b.transferred}


def _shared_buffers(a: PageLayout, b: PageLayout):
    """``(layer, role)``, ``a``'s buffer and ``b``'s, None where that rank does not transfer it,
    of every buffer either transfers on a layer both hold, in key order."""
    layers = {x.layer for x in a.buffers} & {x.layer for x in b.buffers}
    ta, tb = _transferred(a), _transferred(b)
    for key in sorted(k for k in set(ta) | set(tb) if k[0] in layers):
        yield key, ta.get(key), tb.get(key)


def _group_pair(a: PageLayout, x: BufferDesc, b: PageLayout, y: BufferDesc, groups):
    """The layer groups of ``a``'s buffer ``x`` and ``b``'s buffer ``y``, None where one lies in
    none."""
    index_a = groups[0].get((x.pool_group, x.layer))
    index_b = groups[1].get((y.pool_group, y.layer))
    if index_a is None or index_b is None:
        return None
    return a.layer_groups[index_a], b.layer_groups[index_b]


def _buffer_problems(a: PageLayout, b: PageLayout, groups) -> list[str]:
    """The differences of the buffers either rank transfers on the layers both hold."""
    problems = []
    for key, x, y in _shared_buffers(a, b):
        if x is None or y is None:
            if key[1] not in (a if y is None else b).request_rebuilt_roles:
                problems.append(f"buffer {key} is transferred by one rank only")
            continue
        if x.expansion != y.expansion:
            problems.append(f"buffer {key}: {x.expansion} sub-pages against {y.expansion}")
        problems += _layout_problems(key, a, x, b, y)
        pair = _group_pair(a, x, b, y, groups)
        if pair is None:
            problems.append(f"buffer {key} lies in no layer group of one rank")
            continue
        ga, gb = pair
        if ga.kind != gb.kind:
            problems.append(f"buffer {key}: a layer group of {ga.kind} against one of {gb.kind}")
        model_a = ga.model_layers[ga.layers.index(x.layer)]
        model_b = gb.model_layers[gb.layers.index(y.layer)]
        if model_a != model_b:
            # A layer named alike on both sides but of different model layers, as when one side
            # names its layers by pipeline stage, would move one layer's bytes into another's.
            problems.append(f"buffer {key}: of model layer {model_a} against {model_b}")
    return problems


def _layout_problems(
    key: tuple[int, str], a: PageLayout, x: BufferDesc, b: PageLayout, y: BufferDesc
) -> list[str]:
    """The differences of two ranks' layouts of one buffer: the spaces they lay out
    (``RoleLayout.space``) and the elements' types."""
    problems = []
    space_x, space_y = _space(a, x), _space(b, y)
    if space_x != space_y:
        problems.append(f"buffer {key}: space {space_x} against {space_y}")
    if x.layout.dtype != y.layout.dtype:
        problems.append(f"buffer {key}: elements of type {x.layout.dtype} against {y.layout.dtype}")
    return problems


def _pages_pair(a: PageLayout, b: PageLayout, groups) -> bool:
    """Whether both ranks transfer a buffer of paged groups (not ``GroupKind.per_request``), whose
    pages then pair one to one."""
    for _, x, y in _shared_buffers(a, b):
        pair = None if x is None or y is None else _group_pair(a, x, b, y, groups)
        if pair is not None and not pair[0].kind.per_request and not pair[1].kind.per_request:
            return True
    return False


def _naming_problems(a: PageLayout, b: PageLayout, groups) -> list[str]:
    """A model layer's role whose transferred buffers the ranks name by other global layers, or
    that one rank alone transfers of a model layer both hold, unless that rank rebuilds it per
    request."""
    problems = []
    names_a, names_b = _model_layer_names(a, groups[0]), _model_layer_names(b, groups[1])
    for model, role in sorted(set(names_a) & set(names_b)):
        if names_a[model, role] != names_b[model, role]:
            problems.append(
                f"model layer {model}'s {role} buffers in layers {sorted(names_a[model, role])} "
                f"against {sorted(names_b[model, role])}"
            )
    both = _held_model_layers(a, groups[0]) & _held_model_layers(b, groups[1])
    for model, role in sorted(set(names_a) ^ set(names_b)):
        holder = a if (model, role) in names_a else b
        if model in both and role not in holder.request_rebuilt_roles:
            problems.append(
                f"model layer {model}'s {role} buffers are transferred by one rank only, as a "
                "cache attached by what a pipeline stage holds or a role one peer lacks is; only "
                "a rank rebuilding them per request, as CSA2's replay does "
                "(PageTableSpec.request_rebuilt_roles), may hold them alone"
            )
    return problems


def _model_layer_names(
    layout: PageLayout, groups: dict[tuple[int, int], int]
) -> dict[tuple[int, str], set[int]]:
    """(Model layer, role) -> the global layers its transferred buffers of that role lie in."""
    names: dict[tuple[int, str], set[int]] = {}
    for buffer in layout.buffers:
        index = groups.get((buffer.pool_group, buffer.layer))
        if not buffer.transferred or index is None:
            continue
        group = layout.layer_groups[index]
        model = group.model_layers[group.layers.index(buffer.layer)]
        names.setdefault((model, buffer.role), set()).add(buffer.layer)
    return names


def _held_model_layers(layout: PageLayout, groups: dict[tuple[int, int], int]) -> set[int]:
    """The model layers of every buffer lying in a layer group, transferred or not."""
    out = set()
    for buffer in layout.buffers:
        index = groups.get((buffer.pool_group, buffer.layer))
        if index is not None:
            group = layout.layer_groups[index]
            out.add(group.model_layers[group.layers.index(buffer.layer)])
    return out


def sent_bytes(src: PageLayout, dst: PageLayout, *, key: int | None = None) -> dict[int, int]:
    """The most bytes the rank of ``src`` sends the rank of ``dst`` into one page of each of the
    destination's layer groups, a per-request group's slot counting as its page: ``map_buffers``'
    runs, summed by the index of the destination's layer group, a buffer of token positions' at
    the most positions a page moves (``BufferRuns.page_bytes``). Groups it sends nothing of are
    left out.

    Raises:
        ValueError: As ``map_buffers`` and ``layer_group_of`` do.
    """
    group_of = layer_group_of(dst)
    out: dict[int, int] = {}
    for runs in map_buffers(src, dst, key=key):
        group = group_of[runs.dst.pool_group, runs.dst.layer]
        out[group] = out.get(group, 0) + runs.page_bytes
    return out
