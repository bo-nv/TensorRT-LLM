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
"""One rank's page layout as immutable values, and which ranks of an instance hold the same bytes;
standard library only.

A buffer's layout (``RoleLayout``) says how its role's bytes spread over ranks. The buffers of one
layer and role on all ranks together hold one space of bytes, of which each rank's buffer holds
pieces (``RoleLayout.pieces``):

    ByHead(head_bytes=256, total=8)  # 8 heads of 256 B; TP2 rank 1 holds heads 4..7, bytes 1024..
    Sectioned((32, 16, 16))          # [x | B | C], each section split by byte over the ranks
    Replicated()                     # the whole buffer alike on every rank
    RingSlots(19)                    # token position p in slot p % 19, alike on every rank

How two ranks' pages correspond is derived from the pieces alone (``layout_mapping``), so a new way
of laying a role out is one ``RoleLayout`` subclass saying which pieces a rank holds; a layer
group's kind (``GroupKind``) says how it addresses a request's pages."""

from __future__ import annotations

import collections.abc
import dataclasses
import numbers
import types
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal, Mapping

if TYPE_CHECKING:
    from .page_table import BufferSite

__all__ = [
    "GRID_DIMENSIONS",
    "BufferDesc",
    "ByHead",
    "GroupKind",
    "LayerGroupDesc",
    "Own",
    "PageLayout",
    "Paged",
    "PerRequest",
    "Piece",
    "RankTopology",
    "Replicated",
    "RingSlots",
    "RoleLayout",
    "Sectioned",
    "ShardDesc",
    "check_pieces",
    "content_shard",
    "grid_place",
    "group_kind",
    "layout_from_dict",
    "layout_to_dict",
    "replica_ranks",
    "role_layout_from_dict",
]

_QUANT_MODES = ("fp8", "nvfp4", "fp8_ds_mla")


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        requirement = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be a {requirement} integer, got {value!r}")
    return int(value)


def _frozen(mapping: dict) -> Mapping:
    return types.MappingProxyType(mapping)


# The default of an ``unchecked`` constructor's mapping argument.
_NO_ENTRIES: Mapping = _frozen({})

GRID_DIMENSIONS = ("tp", "cp")
"""Parallel dimensions a layout may split over (``RoleLayout.split``): "tp" the tensor-parallel ranks of
one attention data-parallel group, "cp" the context-parallel ranks. Pipeline ranks hold other
layers and data-parallel ranks other requests, so neither splits a page."""


def grid_place(topology: RankTopology, dims: Sequence[str]) -> tuple[int, int]:
    """``(size, rank)`` of this rank in the grid of the parallel dimensions ``dims``, outer to
    inner, the last minor. Under attention data parallelism each tensor-parallel rank serves its own
    requests and holds its buffers whole, so "tp" counts this rank alone."""
    size, rank = 1, 0
    for dim in dims:
        if dim == "tp":
            n, r = (1, 0) if topology.attention_dp else (topology.tp_size, topology.tp_rank)
        elif dim == "cp":
            n, r = topology.cp_size, topology.cp_rank
        else:
            raise ValueError(f"{dim!r} is no grid dimension; one of {GRID_DIMENSIONS}")
        size, rank = size * n, rank * n + r
    return size, rank


_ROLE_LAYOUTS: dict[str, type[RoleLayout]] = {}


def _register(registry: dict, cls: type, what: str) -> None:
    """Enter ``cls`` in ``registry`` by its name; a class defined again in its own place (a module
    reloaded, a test run again) replaces itself, while another class of the same name is refused."""
    if not cls.name:
        return
    known = registry.get(cls.name)
    if known is not None and (known.__module__, known.__qualname__) != (
        cls.__module__,
        cls.__qualname__,
    ):
        raise ValueError(f"{what} {cls.name!r} is registered twice")
    registry[cls.name] = cls


def _split(split: Sequence[str], what: str) -> tuple[str, ...]:
    """``split`` as a tuple of distinct dimensions from ``GRID_DIMENSIONS``, at least one."""
    dims = tuple(str(dim) for dim in split)
    if not dims or len(set(dims)) != len(dims) or not set(dims) <= set(GRID_DIMENSIONS):
        raise ValueError(f"{what} splits over {dims}, not over distinct dims of {GRID_DIMENSIONS}")
    return dims


def _sparse(kind: object) -> dict:
    """The fields of a kind not at their defaults, as plain values (a tuple as a list, which the
    kind's constructor converts back)."""
    doc = {}
    for f in dataclasses.fields(kind):
        value = getattr(kind, f.name)
        if f.default is not dataclasses.MISSING and value == f.default:
            continue
        doc[f.name] = list(value) if isinstance(value, tuple) else value
    return doc


@dataclass(frozen=True)
class Piece:
    """Bytes of a role's space that a rank's buffer holds: ``size`` bytes from coordinate ``at`` of
    the space, at byte ``offset`` of the buffer.

    Attributes:
        at: Coordinate of the piece's first byte in the space of the role's buffers on all ranks.
        offset: Byte offset of the piece within the buffer on this rank.
        size: Positive number of bytes.
    """

    at: int
    offset: int
    size: int


@dataclass(frozen=True)
class RoleLayout:
    """How the bytes of a role's buffers spread over ranks.

    The buffers of one layer and role on all ranks together hold one space of bytes. A rank's
    buffer holds pieces of it (``pieces``), by the rank's place among the ranks of the parallel
    dimensions the layout splits over (``split``); a buffer every rank holds alike splits over
    none. Two ranks' buffers correspond where their layouts lay out the same space (``space``):
    each byte of it the destination holds comes from one of the source ranks holding it
    (``layout_mapping``).

    A subclass is one way of laying a role out, a frozen dataclass whose fields are its data,
    named (``name``) for peers and for ``layout_id``. A new way is one subclass, here or in a
    model's own module, giving ``pieces`` and ``space``; one of token positions (``positional``),
    as ``RingSlots``, also gives ``positions``, ``index_of`` and ``stride``. A manager's spec
    (``page_table.PageTableSpec``) names a layout per role; each buffer's is that layout resolved
    from the buffer's facts (``resolve``).

    Attributes:
        name: The layout's name, unique among role layouts.
        positional: Whether the buffer holds token positions, of which a page holds those it has
            written: its pieces are then those of one position, which lies at the index its
            position takes (``index_of``), ``stride`` bytes per index.

    Raises:
        TypeError: On defining a positional layout that does not give ``positions``,
            ``index_of`` and ``stride``.
    """

    name: ClassVar[str] = ""
    positional: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if cls.positional and (
            cls.positions is RoleLayout.positions
            or cls.index_of is RoleLayout.index_of
            or cls.stride is RoleLayout.stride
        ):
            raise TypeError(
                f"role layout {cls.__name__} holds token positions but does not say which a page "
                "holds (positions), where (index_of) or how far apart (stride)"
            )
        _register(_ROLE_LAYOUTS, cls, "role layout")

    @property
    def split(self) -> tuple[str, ...]:
        """The parallel dimensions the buffers spread over, outer to inner: none by default, every
        rank holding the whole space."""
        return ()

    @property
    def dtype(self) -> str | None:
        """The type of the elements by name, such as ``"bfloat16"``, or None where the layout
        states none; peers whose elements differ in type do not correspond."""
        return None

    def resolve(self, site: BufferSite) -> RoleLayout:
        """The layout of one buffer, filling what a manager's spec leaves to each buffer from its
        facts (``page_table.BufferSite``: its size, tokens, heads, and the dimensions its layer
        group splits over): this layout itself by default."""
        return self

    def check(self, size: int) -> None:
        """Raise ValueError if a buffer of ``size`` bytes cannot be laid out so."""

    def pieces(self, size: int, ranks: int, rank: int) -> tuple[Piece, ...]:
        """The pieces of the space a buffer of ``size`` bytes holds at place ``rank`` of the
        ``ranks`` ranks of its split (``grid_place``), lying back to back over the buffer
        (``check_pieces``); for a positional layout, those of one token position."""
        raise NotImplementedError(f"role layout {self.name!r} gives no pieces")

    def space(self, size: int, ranks: int) -> tuple:
        """What the coordinates of the space mean, for buffers of ``size`` bytes over ``ranks``
        ranks: a name and the space's extents, equal on two ranks whose buffers correspond. A
        layout laying out another's space, as in another order of its bytes, gives the other's."""
        raise NotImplementedError(f"role layout {self.name!r} gives no space")

    def even(self, ranks: int) -> bool:
        """Whether pieces over ``ranks`` ranks name the same bytes under every rank count that
        divides it or that it divides, so that ranks of those counts holding the same bytes may
        share names (``content_shard``): yes by default."""
        return True

    def positions(self, page: int, end: int, tokens_per_page: int) -> range:
        """For a positional layout: the token positions of page ``page`` (its position among a
        request's pages, from 0, of ``tokens_per_page`` positions each) that the buffer holds once
        the request has written every position below ``end``."""
        raise NotImplementedError(f"role layout {self.name!r} holds no token positions")

    def index_of(self, positions, tokens_per_page: int):
        """For a positional layout: the index holding each of ``positions`` (an integer or an
        integer array; the result is the same) in pages of ``tokens_per_page`` positions."""
        raise NotImplementedError(f"role layout {self.name!r} holds no token positions")

    def stride(self, size: int) -> int:
        """For a positional layout: the bytes between consecutive indices of a buffer of ``size``
        bytes."""
        raise NotImplementedError(f"role layout {self.name!r} holds no token positions")

    def to_dict(self) -> dict:
        """The layout as plain values for a peer, which ``role_layout_from_dict`` reads back."""
        return {"name": self.name, **_sparse(self)}

    def document(self) -> dict:
        """What ``layout_id`` hashes of the layout: its name and its fields not at their defaults,
        but the element type."""
        fields = _sparse(self)
        fields.pop("dtype", None)
        return {"layout": self.name, **fields}


def role_layout_from_dict(data: Mapping) -> RoleLayout:
    """The role layout ``RoleLayout.to_dict`` wrote, checked as its constructor checks it.

    Raises:
        ValueError: If it names no registered layout.
    """
    fields = dict(data)
    name = fields.pop("name", None)
    if name not in _ROLE_LAYOUTS:
        raise ValueError(f"unknown role layout {name!r}; known: {sorted(_ROLE_LAYOUTS)}")
    return _ROLE_LAYOUTS[name](**fields)


def _dtype(layout: RoleLayout) -> None:
    if layout.dtype is not None:
        object.__setattr__(layout, "dtype", str(layout.dtype))


@dataclass(frozen=True)
class Replicated(RoleLayout):
    """The whole buffer alike on every rank, as an index-K cache is."""

    name: ClassVar[str] = "replicated"

    def pieces(self, size: int, ranks: int, rank: int) -> tuple[Piece, ...]:
        return (Piece(0, 0, size),)

    def space(self, size: int, ranks: int) -> tuple:
        return ("replicated", size)


@dataclass(frozen=True)
class Own(RoleLayout):
    """Each rank's own share among the ranks of ``split``, which pairs only with the same rank's:
    the layout of a buffer whose split the manager does not describe further, or that never moves.

    Attributes:
        split: Parallel dimensions from ``GRID_DIMENSIONS``, outer to inner; the tensor-parallel
            ranks by default, those of the buffer's layer group once resolved. Normalized to a
            tuple.

    Raises:
        ValueError: If split is empty or names a dimension twice or one outside
            ``GRID_DIMENSIONS``.
    """

    name: ClassVar[str] = "own"
    split: tuple[str, ...] = ("tp",)

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", _split(self.split, "an owned layout"))

    def resolve(self, site: BufferSite) -> RoleLayout:
        return dataclasses.replace(self, split=site.split)

    def pieces(self, size: int, ranks: int, rank: int) -> tuple[Piece, ...]:
        return (Piece(rank * size, 0, size),)

    def space(self, size: int, ranks: int) -> tuple:
        return ("own", size, ranks)


@dataclass(frozen=True)
class Sectioned(RoleLayout):
    """Consecutive sections, each this rank's share of its own part of the space, split by byte
    over the ranks of ``split``, as Mamba2's conv state ``[x | B | C]`` is: with sections of 32,
    16 and 16 bytes on each of two ranks, rank 1 holds bytes 32..63, 80..95 and 112..127 of a
    space of 64 + 32 + 32 bytes.

    Attributes:
        section_bytes: Positive bytes of each section on this rank, in order; normalized to a
            tuple.
        split: Parallel dimensions from ``GRID_DIMENSIONS``, outer to inner; the tensor-parallel
            ranks by default, those of the buffer's layer group once resolved. Normalized to a
            tuple.
        dtype: The type of the elements by name, or None.

    Raises:
        ValueError: If there is no section, a section is not a positive integer, or split is empty
            or names a dimension twice or one outside ``GRID_DIMENSIONS``.
    """

    name: ClassVar[str] = "sectioned"
    section_bytes: tuple[int, ...]
    split: tuple[str, ...] = ("tp",)
    dtype: str | None = None

    def __post_init__(self) -> None:
        sections = tuple(_integer(n, "section bytes", 1) for n in self.section_bytes)
        if not sections:
            raise ValueError("a sectioned layout has one or more sections")
        object.__setattr__(self, "section_bytes", sections)
        object.__setattr__(self, "split", _split(self.split, "a sectioned layout"))
        _dtype(self)

    def resolve(self, site: BufferSite) -> RoleLayout:
        return dataclasses.replace(self, split=site.split)

    def check(self, size: int) -> None:
        if sum(self.section_bytes) != size:
            raise ValueError(
                f"sections of {list(self.section_bytes)} bytes do not cover its {size} bytes"
            )

    def pieces(self, size: int, ranks: int, rank: int) -> tuple[Piece, ...]:
        out, base, offset = [], 0, 0
        for n in self.section_bytes:
            out.append(Piece(base + rank * n, offset, n))
            base += ranks * n
            offset += n
        return tuple(out)

    def space(self, size: int, ranks: int) -> tuple:
        return ("sectioned", tuple(ranks * n for n in self.section_bytes))


@dataclass(frozen=True)
class ByHead(RoleLayout):
    """Split by head over the ranks of ``split``, head-major or, with ``token_major``,
    token-major.

    The space holds the heads of all ranks: ``[head][byte]`` head-major (HND), a head's bytes
    those of a page of it, and ``[token][head][byte]`` token-major (NHD). Place ``r`` of ``n``
    ranks holds the ``size // head_bytes`` heads from ``r * total // n``: 8 heads of 256 bytes
    over TP2, rank 1 holds heads 4..7, bytes 1024..2047 of the space; 6 heads over TP4, two per
    rank, ranks hold heads 0..1, 1..2, 3..4 and 4..5, overlapping where the heads they hold
    exceed the total.

    In a manager's spec, an attention buffer's heads are by default the layer's KV heads: K of a
    layer of 8 KV heads on TP2 holds 4 heads of ``size // 4`` bytes. A buffer of heads of its own
    states their bytes (``head_bytes``), and for attention their total over all ranks
    (``total``); recurrent state's total is its heads on this rank times the ranks. A buffer
    whose heads are unknown, or whose size they do not divide, is this rank's own share
    (``Own``).

    Attributes:
        token_major: Tokens outermost ([token, head, ...], NHD) rather than heads (HND).
        head_bytes: Positive bytes of one head of a page; None in a spec, for the layer's KV heads.
        total: Positive heads over the ranks of the split; None in a spec, for the layer's KV
            heads or for recurrent state.
        tokens: Positive token positions of a page of a token-major layout; resolved from the
            buffer.
        split: Parallel dimensions from ``GRID_DIMENSIONS``, outer to inner; the tensor-parallel
            ranks by default, those of the buffer's layer group once resolved. Normalized to a
            tuple.
        dtype: The type of the elements by name, or None.

    Raises:
        ValueError: If a count is not a positive integer, or split is empty or names a dimension
            twice or one outside ``GRID_DIMENSIONS``.
    """

    name: ClassVar[str] = "by_head"
    token_major: bool = False
    head_bytes: int | None = None
    total: int | None = None
    tokens: int | None = None
    split: tuple[str, ...] = ("tp",)
    dtype: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_major", bool(self.token_major))
        for name in ("head_bytes", "total", "tokens"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _integer(getattr(self, name), name, 1))
        object.__setattr__(self, "split", _split(self.split, "a head split"))
        _dtype(self)

    def resolve(self, site: BufferSite) -> RoleLayout:
        head_bytes, total = self.head_bytes, self.total
        if head_bytes is None and not site.state:
            kv_heads, kv_total = site.kv_heads
            if kv_heads > 0 and site.size % kv_heads == 0:
                head_bytes, total = site.size // kv_heads, kv_total
        elif head_bytes is not None and not site.state and total is None:
            raise ValueError(
                f"heads of {head_bytes} bytes are an attention buffer's own; state how many lie "
                "over all tensor ranks (total)"
            )
        if head_bytes is None or site.size % head_bytes:
            return Own(site.split)
        return dataclasses.replace(
            self,
            head_bytes=head_bytes,
            total=total or site.size // head_bytes * site.ranks,
            tokens=site.tokens if self.token_major else None,
            split=site.split,
        )

    def check(self, size: int) -> None:
        if self.head_bytes is None or self.total is None:
            raise ValueError("a head split states its head bytes and total (resolve)")
        if size % self.head_bytes:
            raise ValueError(
                f"{size} bytes is not a whole number of heads of {self.head_bytes} bytes"
            )
        if size // self.head_bytes > self.total:
            raise ValueError(f"{size // self.head_bytes} heads held of {self.total}")
        if self.token_major and (self.tokens is None or self.head_bytes % self.tokens):
            raise ValueError(
                f"heads of {self.head_bytes} bytes are not a whole number of rows of "
                f"{self.tokens} tokens"
            )

    def pieces(self, size: int, ranks: int, rank: int) -> tuple[Piece, ...]:
        heads = size // self.head_bytes
        first = rank * self.total // ranks
        if first + heads > self.total:
            raise ValueError(
                f"rank {rank} of {ranks} holds {heads} heads from head {first}, past the "
                f"{self.total} heads"
            )
        if not self.token_major:
            return (Piece(first * self.head_bytes, 0, size),)
        if heads == self.total:
            return (Piece(0, 0, size),)
        row = self.head_bytes // self.tokens
        return tuple(
            Piece((t * self.total + first) * row, t * heads * row, heads * row)
            for t in range(self.tokens)
        )

    def space(self, size: int, ranks: int) -> tuple:
        if self.token_major:
            return ("token_major", self.tokens, self.total, self.head_bytes // self.tokens)
        return ("head_major", self.total, self.head_bytes)

    def even(self, ranks: int) -> bool:
        return self.total % ranks == 0 or ranks % self.total == 0


@dataclass(frozen=True)
class RingSlots(RoleLayout):
    """A ring of ``slots`` token positions alike on every rank: of its page's tokens, position
    ``p`` lies in slot ``p % slots``, so the buffer keeps the latest ``slots`` positions the page
    has written. Each side sizes it as it keeps, so peers may differ in its slots.

    FP4 MLA's high-precision tail on a generation worker, 19 slots in pages of 32 tokens: of page
    1 (positions 32..63) written up to position 50, it holds 32..49, position 40 in slot 2.

    Attributes:
        slots: Positive number of positions the ring keeps on this rank.
        dtype: The type of the elements by name, or None.

    Raises:
        ValueError: If slots is not a positive integer.
    """

    name: ClassVar[str] = "ring"
    positional: ClassVar[bool] = True
    slots: int
    dtype: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "slots", _integer(self.slots, "ring slots", 1))
        _dtype(self)

    def check(self, size: int) -> None:
        if size % self.slots:
            raise ValueError(f"{size} bytes is not a whole number of {self.slots} slots")

    def pieces(self, size: int, ranks: int, rank: int) -> tuple[Piece, ...]:
        return (Piece(0, 0, size // self.slots),)

    def space(self, size: int, ranks: int) -> tuple:
        return ("ring", size // self.slots)

    def positions(self, page: int, end: int, tokens_per_page: int) -> range:
        first = page * tokens_per_page
        written = min(end, first + tokens_per_page)
        return range(max(first, written - self.slots), written)

    def index_of(self, positions, tokens_per_page: int):
        return positions % self.slots

    def stride(self, size: int) -> int:
        return size // self.slots


def check_pieces(layout: RoleLayout, size: int, ranks: int, rank: int) -> None:
    """Check that the pieces a ``size``-byte buffer laid out by ``layout`` holds at place ``rank``
    of ``ranks`` lie back to back over it (over one position's bytes for a positional layout),
    holding no byte of the space twice: bytes no piece describes would never move, or be moved
    twice.

    Raises:
        ValueError: Naming what does not fit.
    """
    extent = layout.stride(size) if layout.positional else size
    pieces = layout.pieces(size, ranks, rank)
    held = sum(p.size for p in pieces)
    if held != extent:
        raise ValueError(
            f"its pieces hold {held} bytes on this rank, not the {extent} bytes of its slot"
        )
    end = 0
    for p in sorted(pieces, key=lambda p: p.offset):
        if p.size <= 0 or p.offset != end:
            raise ValueError(f"its pieces do not lie back to back over its {extent} bytes")
        end += p.size
    spans = sorted((p.at, p.at + p.size) for p in pieces)
    if spans[0][0] < 0 or any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError("its pieces hold a byte of the space twice, or one before its start")


@dataclass(frozen=True)
class ShardDesc:
    """Which share of a layer group's content this rank holds.

    The default (count=1, index=0) is the whole content, named alike on every rank.

    Attributes:
        count: Positive number of content shards.
        index: Zero-based index of this rank's share, less than count.

    Raises:
        ValueError: If count is not a positive integer, index is not a non-negative integer, either
            is a boolean, or index lies outside count."""

    count: int = 1
    index: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "count", _integer(self.count, "shard count", 1))
        object.__setattr__(self, "index", _integer(self.index, "shard index"))
        if self.index >= self.count:
            raise ValueError(f"shard {self.index} of {self.count}")

    @classmethod
    def unchecked(cls, count: int = 1, index: int = 0) -> ShardDesc:
        """The constructor's shard of these values, without its checks and conversions; see
        ``PageLayout.unchecked``."""
        shard = object.__new__(cls)
        shard.__dict__.update(count=count, index=index)
        return shard


_GROUP_KINDS: dict[str, type[GroupKind]] = {}


@dataclass(frozen=True)
class GroupKind:
    """How a layer group addresses a request's pages, and so which of its slots one transfer pairs.

    A subclass is one such way. It states its data here; which slots pair is registered in
    ``layout_mapping`` (``pair_slots``), so a new way adds a subclass and edits no existing
    function. Each subclass names itself (``name``) for peers and for ``layout_id``; the kinds a
    layer group is built with may be given by name.

    Attributes:
        name: The kind's name, unique among layer-group kinds.
        per_request: Whether the group holds one slot per request, as recurrent state does, rather
            than one per page of its tokens; this also gives its kind in the backend contract.
    """

    name: ClassVar[str] = ""
    per_request: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        _register(_GROUP_KINDS, cls, "layer-group kind")

    def __str__(self) -> str:
        return self.name

    def to_dict(self) -> str | dict:
        """The kind as plain values for a peer and for ``layout_id``: its name, or with fields not
        at their defaults, its name and those fields; ``group_kind`` reads it back."""
        fields = _sparse(self)
        return {"name": self.name, **fields} if fields else self.name


def group_kind(value: GroupKind | str | Mapping) -> GroupKind:
    """The layer-group kind ``value`` names: a kind itself, its name, or ``GroupKind.to_dict``'s
    plain values.

    Raises:
        ValueError: If ``value`` names no registered kind.
    """
    if isinstance(value, GroupKind):
        return value
    fields = {"name": value} if isinstance(value, str) else dict(value)
    name = fields.pop("name", None)
    if name not in _GROUP_KINDS:
        raise ValueError(f"unknown layer-group kind {name!r}; known: {sorted(_GROUP_KINDS)}")
    return _GROUP_KINDS[name](**fields)


@dataclass(frozen=True)
class Paged(GroupKind):
    """A slot per page of the request's tokens, as attention KV is held."""

    name: ClassVar[str] = "attention"


@dataclass(frozen=True)
class PerRequest(GroupKind):
    """One slot per request, overwritten in place, as recurrent state is held."""

    name: ClassVar[str] = "state"
    per_request: ClassVar[bool] = True


@dataclass(frozen=True)
class LayerGroupDesc:
    """One local layer group's windows, pools, global layers and content shard.

    A block held only by the kept window may remain unwritten without preventing
    resumption when it lies outside the read window.

    Each buffer's ``layout`` says how it splits over the ranks; ``shard`` is the share
    of the group's content this rank holds, by the bytes its transferred buffers
    hold (``content_shard``), so ranks with equal shares hold equal bytes.

    Attributes:
        kind: How the group addresses a request's pages: ``Paged`` for attention KV data,
            ``PerRequest`` for recurrent state, or another ``GroupKind``; given by name, as
            "attention" or "state", it is normalized to the kind.
        kept_window: Positive number of tokens whose blocks the manager retains; None means full
            attention.
        num_sink_blocks: Non-negative number of sink blocks retained by the manager.
        pool_group: Non-negative index of the group's device pool group.
        layers: Global layer ids in this local group, normalized to a tuple.
        shard: The share of the group's content held on this rank.
        read_window: Positive number of tokens attention reads, defaulting to kept_window; None
            means full attention. A declared read window must lie within the kept window.
        model_layers: The model layer each of ``layers`` belongs to, in the same order, normalized
            to a tuple; None gives ``layers`` themselves. Layer ids numbering several caches of one
            model layer apart, such as one per attention type, all name that model layer.

    Raises:
        ValueError: If kind names no layer-group kind, an index, layer id or model layer is not a
            non-negative integer, a window is not a positive integer, any such value is a boolean,
            read_window exceeds kept_window or is given when kept_window is None, or model_layers
            does not name one model layer per layer."""

    kind: GroupKind
    kept_window: int | None
    num_sink_blocks: int
    pool_group: int
    layers: tuple[int, ...]
    shard: ShardDesc = ShardDesc()
    read_window: int | None = None
    model_layers: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", group_kind(self.kind))
        if self.kept_window is not None:
            object.__setattr__(self, "kept_window", _integer(self.kept_window, "kept_window", 1))
        object.__setattr__(
            self, "num_sink_blocks", _integer(self.num_sink_blocks, "num_sink_blocks")
        )
        object.__setattr__(self, "pool_group", _integer(self.pool_group, "pool_group"))
        object.__setattr__(self, "layers", tuple(_integer(n, "layer") for n in self.layers))
        if self.read_window is None:
            object.__setattr__(self, "read_window", self.kept_window)
        else:
            object.__setattr__(self, "read_window", _integer(self.read_window, "read_window", 1))
        if self.read_window is not None and (
            self.kept_window is None or self.read_window > self.kept_window
        ):
            raise ValueError(
                f"a read window of {self.read_window} tokens is not within the kept window of "
                f"{self.kept_window} tokens"
            )
        if self.model_layers is None:
            object.__setattr__(self, "model_layers", self.layers)
        else:
            model_layers = tuple(_integer(n, "model layer") for n in self.model_layers)
            if len(model_layers) != len(self.layers):
                raise ValueError(
                    f"{len(model_layers)} model layers for the {len(self.layers)} layers "
                    f"{self.layers}"
                )
            object.__setattr__(self, "model_layers", model_layers)

    @classmethod
    def unchecked(
        cls,
        kind: GroupKind,
        kept_window: int | None,
        num_sink_blocks: int,
        pool_group: int,
        layers: tuple[int, ...],
        shard: ShardDesc = ShardDesc(),
        read_window: int | None = None,
        model_layers: tuple[int, ...] | None = None,
    ) -> LayerGroupDesc:
        """The constructor's layer group of these values, without its checks and conversions; a
        read window of None still defaults to the kept window, and model layers of None to the
        layers. See ``PageLayout.unchecked``."""
        group = object.__new__(cls)
        group.__dict__.update(
            kind=kind,
            kept_window=kept_window,
            num_sink_blocks=num_sink_blocks,
            pool_group=pool_group,
            layers=layers,
            shard=shard,
            read_window=kept_window if read_window is None else read_window,
            model_layers=layers if model_layers is None else model_layers,
        )
        return group


@dataclass(frozen=True)
class BufferDesc:
    """One role of one global layer within a page's pool slot.

    Attributes:
        layer: Non-negative global layer id.
        role: Role name, normalized to a string.
        pool_group: Non-negative device pool group index.
        pool_index: Non-negative pool index within pool_group.
        slot_offset: Non-negative byte offset within that pool's slot.
        size: Positive byte size of the buffer.
        layout: How the role's bytes spread over ranks (``RoleLayout``); the whole buffer alike on
            every rank by default.
        transferred: Whether peers transfer this role. False marks local-only buffers, which still
            accompany copies of whole slots.
        expansion: Positive number of sub-pages within one page, each of
            PageLayout.tokens_per_page / expansion of the buffer's tokens; consumers accessing bytes
            within the buffer must support it.

    Raises:
        ValueError: If an index is not a non-negative integer, size or expansion is not a positive
            integer, any such value is a boolean, layout is no ``RoleLayout`` or does not fit the
            buffer (``RoleLayout.check``), or a layout of token positions lies in a page of
            sub-pages."""

    layer: int
    role: str
    pool_group: int
    pool_index: int
    slot_offset: int
    size: int
    layout: RoleLayout = Replicated()
    transferred: bool = True
    expansion: int = 1

    def __post_init__(self) -> None:
        for name in ("layer", "pool_group", "pool_index", "slot_offset"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        object.__setattr__(self, "size", _integer(self.size, "size", 1))
        object.__setattr__(self, "expansion", _integer(self.expansion, "expansion", 1))
        object.__setattr__(self, "role", str(self.role))
        if not isinstance(self.layout, RoleLayout):
            raise ValueError(
                f"buffer ({self.layer}, {self.role}): layout must be a RoleLayout, got "
                f"{self.layout!r}"
            )
        try:
            self.layout.check(self.size)
        except ValueError as error:
            raise ValueError(f"buffer ({self.layer}, {self.role}): {error}") from error
        if self.layout.positional and self.expansion != 1:
            raise ValueError(
                f"buffer ({self.layer}, {self.role}): a layout of token positions in a page of "
                f"{self.expansion} sub-pages"
            )

    @property
    def split(self) -> bool:
        """Whether its layout spreads over parallel dimensions; else every rank holds it alike."""
        return bool(self.layout.split)

    @property
    def positional(self) -> bool:
        """Whether it holds token positions (``RoleLayout.positional``)."""
        return self.layout.positional

    @classmethod
    def unchecked(
        cls,
        layer: int,
        role: str,
        pool_group: int,
        pool_index: int,
        slot_offset: int,
        size: int,
        layout: RoleLayout = Replicated(),
        transferred: bool = True,
        expansion: int = 1,
    ) -> BufferDesc:
        """The constructor's buffer of these values, without its checks and conversions; see
        ``PageLayout.unchecked``."""
        buffer = object.__new__(cls)
        buffer.__dict__.update(
            layer=layer,
            role=role,
            pool_group=pool_group,
            pool_index=pool_index,
            slot_offset=slot_offset,
            size=size,
            layout=layout,
            transferred=transferred,
            expansion=expansion,
        )
        return buffer


@dataclass(frozen=True)
class RankTopology:
    """This rank's parallel sizes and ranks, as native RankInfo carries them.

    Attributes:
        tp_size: Positive tensor-parallel group size.
        tp_rank: Zero-based tensor-parallel rank, less than tp_size.
        pp_size: Positive pipeline-parallel group size.
        pp_rank: Zero-based pipeline-parallel rank, less than pp_size.
        cp_size: Positive context-parallel group size.
        cp_rank: Zero-based context-parallel rank, less than cp_size.
        dp_size: Positive attention data-parallel group size.
        dp_rank: Zero-based attention data-parallel rank, less than dp_size.
        attention_dp: Whether attention uses data parallelism; each tensor-parallel rank then
            occupies its own attention data-parallel rank.

    Raises:
        ValueError: If a size is not a positive integer, a rank is not a non-negative integer, any
            such value is a boolean, or a rank lies outside its size."""

    tp_size: int = 1
    tp_rank: int = 0
    pp_size: int = 1
    pp_rank: int = 0
    cp_size: int = 1
    cp_rank: int = 0
    dp_size: int = 1
    dp_rank: int = 0
    attention_dp: bool = False

    def __post_init__(self) -> None:
        for name in ("tp", "pp", "cp", "dp"):
            size = _integer(getattr(self, f"{name}_size"), f"{name}_size", 1)
            rank = _integer(getattr(self, f"{name}_rank"), f"{name}_rank")
            if rank >= size:
                raise ValueError(f"{name} rank {rank} of {size}")
            object.__setattr__(self, f"{name}_size", size)
            object.__setattr__(self, f"{name}_rank", rank)

    @classmethod
    def unchecked(
        cls,
        tp_size: int = 1,
        tp_rank: int = 0,
        pp_size: int = 1,
        pp_rank: int = 0,
        cp_size: int = 1,
        cp_rank: int = 0,
        dp_size: int = 1,
        dp_rank: int = 0,
        attention_dp: bool = False,
    ) -> RankTopology:
        """The constructor's topology of these values, without its checks and conversions; see
        ``PageLayout.unchecked``."""
        topology = object.__new__(cls)
        topology.__dict__.update(
            tp_size=tp_size,
            tp_rank=tp_rank,
            pp_size=pp_size,
            pp_rank=pp_rank,
            cp_size=cp_size,
            cp_rank=cp_rank,
            dp_size=dp_size,
            dp_rank=dp_rank,
            attention_dp=attention_dp,
        )
        return topology


@dataclass(frozen=True)
class PageLayout:
    """How one rank's pages are built and where the rank stands, as immutable values.

    Values contain no addresses or manager ownership. A page uses one slot from each pool of its
    device pool group, in pool order. Buffers locate their bytes within those slots. Layer groups
    follow the manager's local order, indexed by LentLayerGroupRows.layer_group_index; their layer
    ids are global. Mapping inputs are copied into read-only mappings, and layer groups and buffers
    into tuples. Equal layouts hash alike and can serve as mapping keys.

    Region maps index Lender.regions, where staging regions precede device regions. A staging slot
    concatenates the group's pools' slots in pool order. Regions absent from a lender have no map
    entry; both maps are empty outside a lender.

    Attributes:
        tokens_per_page: Positive token count of one page. Under Helix, one manager block spans
            topology.cp_size pages, one per context-parallel rank.
        pool_slot_bytes: Device pool group index -> positive slot byte widths of its pools, in pool
            order.
        layer_groups: Tuple describing each local layer group's kind, windows, sinks, device pool
            group, global layer ids and content shard.
        buffers: Tuple of every role buffer of every layer within a page.
        topology: This rank's parallel sizes and ranks.
        total_kv_heads_by_layer: Global layer id -> positive KV head count over all tensor-parallel
            ranks, only for attention layers whose count the manager knows.
        dtype: Name of the manager's tensorrt_llm.bindings.DataType member for KV data, such as
            "HALF", "BF16", "FP8" or "NVFP4"; None if undeclared.
        quant_mode: None for unquantized KV, else "fp8", "nvfp4" or "fp8_ds_mla" (DeepSeek-V4's FP8
            format with its scales within the page).
        device_region_index: (pool group, pool index) -> device region index in Lender.regions.
        staging_region_index: Pool group -> staging region index in Lender.regions; empty when the
            lender stages nothing.
        rebuilds_untransferred: Whether some buffers peers do not transfer derive from the others,
            which the manager rebuilds after the native transfer imports a request's pages
            (``PageTableSpec.derived_roles``). The lender, which invokes no rebuild, then
            lends every buffer, and where the source holds none of the derived ones, the importer
            invokes the rebuild (``attach_lender``).
        request_rebuilt_roles: Roles whose buffers this rank, receiving, rebuilds for some requests
            rather than receiving them (``PageTableSpec.request_rebuilt_roles``), so a
            peer may hold none of them for a model layer it holds.

    Raises:
        ValueError: If tokens_per_page, a slot width or a KV head count is not a positive integer; a
            group, pool, layer or region index is not a non-negative integer; any such value is a
            boolean; a buffer's expansion does not divide tokens_per_page, or its pieces on this
            rank do not lie back to back over it (``check_pieces``); or quant_mode is not listed
            above."""

    tokens_per_page: int
    pool_slot_bytes: Mapping[int, tuple[int, ...]]
    layer_groups: tuple[LayerGroupDesc, ...]
    buffers: tuple[BufferDesc, ...]
    topology: RankTopology = RankTopology()
    total_kv_heads_by_layer: Mapping[int, int] = field(default_factory=dict)
    dtype: str | None = None
    quant_mode: Literal["fp8", "nvfp4", "fp8_ds_mla"] | None = None
    device_region_index: Mapping[tuple[int, int], int] = field(default_factory=dict)
    staging_region_index: Mapping[int, int] = field(default_factory=dict)
    rebuilds_untransferred: bool = False
    request_rebuilt_roles: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tokens_per_page", _integer(self.tokens_per_page, "tokens_per_page", 1)
        )
        pools = {
            _integer(g, "pool group"): tuple(_integer(w, "slot width", 1) for w in widths)
            for g, widths in self.pool_slot_bytes.items()
        }
        object.__setattr__(self, "pool_slot_bytes", _frozen(pools))
        heads = {
            _integer(layer, "layer"): _integer(count, "KV heads", 1)
            for layer, count in self.total_kv_heads_by_layer.items()
        }
        object.__setattr__(self, "total_kv_heads_by_layer", _frozen(heads))
        device = {
            (_integer(g, "pool group"), _integer(p, "pool index")): _integer(i, "region index")
            for (g, p), i in self.device_region_index.items()
        }
        object.__setattr__(self, "device_region_index", _frozen(device))
        staging = {
            _integer(g, "pool group"): _integer(i, "region index")
            for g, i in self.staging_region_index.items()
        }
        object.__setattr__(self, "staging_region_index", _frozen(staging))
        object.__setattr__(self, "rebuilds_untransferred", bool(self.rebuilds_untransferred))
        object.__setattr__(
            self, "request_rebuilt_roles", frozenset(str(r) for r in self.request_rebuilt_roles)
        )
        object.__setattr__(self, "layer_groups", tuple(self.layer_groups))
        object.__setattr__(self, "buffers", tuple(self.buffers))
        if self.quant_mode is not None and self.quant_mode not in _QUANT_MODES:
            raise ValueError(
                f"quant_mode must be None or one of {_QUANT_MODES}, got {self.quant_mode!r}"
            )
        for buffer in self.buffers:
            if self.tokens_per_page % buffer.expansion:
                raise ValueError(
                    f"buffer ({buffer.layer}, {buffer.role}): expansion {buffer.expansion} does not "
                    f"divide {self.tokens_per_page} tokens per page"
                )
            place = grid_place(self.topology, buffer.layout.split)
            try:
                check_pieces(buffer.layout, buffer.size, *place)
            except ValueError as error:
                raise ValueError(f"buffer ({buffer.layer}, {buffer.role}): {error}") from error

    @classmethod
    def unchecked(
        cls,
        tokens_per_page: int,
        pool_slot_bytes: Mapping[int, tuple[int, ...]],
        layer_groups: tuple[LayerGroupDesc, ...],
        buffers: tuple[BufferDesc, ...],
        topology: RankTopology = RankTopology(),
        total_kv_heads_by_layer: Mapping[int, int] = _NO_ENTRIES,
        dtype: str | None = None,
        quant_mode: Literal["fp8", "nvfp4", "fp8_ds_mla"] | None = None,
        device_region_index: Mapping[tuple[int, int], int] = _NO_ENTRIES,
        staging_region_index: Mapping[int, int] = _NO_ENTRIES,
        rebuilds_untransferred: bool = False,
        request_rebuilt_roles: frozenset[str] = frozenset(),
    ) -> PageLayout:
        """The constructor's layout of these values, without its checks and conversions.

        Mappings become read-only copies and layer groups and buffers tuples, as in the
        constructor; every other value, including those inside layer groups and buffers, is kept
        as given. For values the constructor keeps unchanged, such as another layout's fields, the
        result equals the constructor's. Values it would refuse or convert are not detected: they
        stay as given, the layout then breaks the constraints documented above, and code reading
        it may fail or misbehave. Build layouts from values of unknown origin with the constructor.
        """
        layout = object.__new__(cls)
        layout.__dict__.update(
            tokens_per_page=tokens_per_page,
            pool_slot_bytes=_frozen(dict(pool_slot_bytes)),
            layer_groups=tuple(layer_groups),
            buffers=tuple(buffers),
            topology=topology,
            total_kv_heads_by_layer=_frozen(dict(total_kv_heads_by_layer)),
            dtype=dtype,
            quant_mode=quant_mode,
            device_region_index=_frozen(dict(device_region_index)),
            staging_region_index=_frozen(dict(staging_region_index)),
            rebuilds_untransferred=rebuilds_untransferred,
            request_rebuilt_roles=frozenset(request_rebuilt_roles),
        )
        return layout

    def __hash__(self) -> int:
        values = (getattr(self, f.name) for f in dataclasses.fields(self))
        return hash(
            tuple(
                tuple(sorted(value.items()))
                if isinstance(value, collections.abc.Mapping)
                else value
                for value in values
            )
        )


def _same_group(layout: PageLayout, desc: LayerGroupDesc) -> LayerGroupDesc | None:
    """The layer group of ``layout`` holding exactly ``desc``'s layers, of its kind, or None."""
    layers = frozenset(desc.layers)
    for other in layout.layer_groups:
        if other.kind == desc.kind and frozenset(other.layers) == layers:
            return other
    return None


def _coverage(buffer: BufferDesc, topology: RankTopology) -> tuple[tuple[int, int], ...]:
    """The bytes of its space the buffer holds on the rank at ``topology``, as ``(start, end)``
    spans in order, adjacent ones merged."""
    ranks, rank = grid_place(topology, buffer.layout.split)
    spans: list[list[int]] = []
    for start, end in sorted(
        (p.at, p.at + p.size) for p in buffer.layout.pieces(buffer.size, ranks, rank)
    ):
        if spans and spans[-1][1] == start:
            spans[-1][1] = end
        else:
            spans.append([start, end])
    return tuple((start, end) for start, end in spans)


def _split_holdings(buffers: Sequence[BufferDesc], topology: RankTopology) -> tuple:
    """The bytes the rank at ``topology`` holds of the transferred buffers among ``buffers`` that
    split over ranks, as ((layer, role), spans) pairs in that key order."""
    out = [
        ((buffer.layer, buffer.role), _coverage(buffer, topology))
        for buffer in buffers
        if buffer.transferred and buffer.split
    ]
    return tuple(sorted(out, key=lambda item: item[0]))


def _held_indices(layout: PageLayout, desc: LayerGroupDesc) -> tuple | None:
    """The bytes this rank holds of the group's transferred buffers that split, by (layer, role);
    None when no transferred buffer splits."""
    buffers = [
        b for b in layout.buffers if b.pool_group == desc.pool_group and b.layer in desc.layers
    ]
    return _split_holdings(buffers, layout.topology) or None


def content_shard(buffers: Sequence[BufferDesc], topology: RankTopology) -> ShardDesc:
    """The share of a layer group's content the rank at ``topology`` holds, read from the group's
    ``buffers``: ranks of its grid hold the same share exactly when they hold the same bytes of
    every transferred buffer that splits, and shares are numbered in the order of the bytes they
    hold. The whole content when every rank holds alike.

    The numbering names the same bytes under every rank count that divides, or is divided by,
    each layout's split (``RoleLayout.even``), as attention heads and TP sizes do. Where one does
    not, each rank of the dimensions the group splits over is a share of its own, so that no other
    rank count can name other bytes alike."""

    def share(place: RankTopology) -> tuple:
        return _split_holdings(buffers, place)

    mine = share(topology)
    if not mine:
        return ShardDesc()
    split = [b for b in buffers if b.transferred and b.split]
    if any(not b.layout.even(grid_place(topology, b.layout.split)[0]) for b in split):
        dims = [dim for dim in GRID_DIMENSIONS if any(dim in b.layout.split for b in split)]
        size, rank = grid_place(topology, dims)
        return ShardDesc(size, rank) if size > 1 else ShardDesc()
    shares = {
        share(
            dataclasses.replace(
                topology,
                tp_rank=tp_rank,
                cp_rank=cp_rank,
                dp_rank=tp_rank if topology.attention_dp else topology.dp_rank,
            )
        )
        for tp_rank in range(topology.tp_size)
        for cp_rank in range(topology.cp_size)
    }
    if len(shares) == 1:
        return ShardDesc()
    return ShardDesc(len(shares), sorted(shares).index(mine))


def replica_ranks(
    ranks: Sequence[PageLayout],
    rank: int,
    group: int,
    *,
    replicated: bool = False,
) -> tuple[int, ...]:
    """The flat ranks of one instance holding the same bytes as layer group ``group`` of rank
    ``rank``, read from every rank's published layout.

    Ranks hold the same bytes of a layer group when they serve the same requests (the same
    attention data-parallel rank); hold the same pages (under Helix an attention group's pages split
    by context-parallel rank, while a state group's rows do not); hold the same layers, in a layer
    group of the same kind; and hold the same share of the group's content: a buffer every rank
    holds alike is the same on every such rank, and split buffers match when the ranks hold the
    same bytes of their spaces. Without split buffers, the published content shard is the only
    basis for matching.

    Args:
        ranks: Every rank's layout of one instance, by flat rank.
        rank: The flat rank whose layer group is matched.
        group: The index of the layer group in that rank's layout.
        replicated: Match the group's replicated buffers; otherwise its other buffers.

    Returns:
        The matching flat ranks, ascending, ``rank`` among them.

    Raises:
        IndexError: ``rank`` or ``group`` names no rank or layer group.
    """
    layout = ranks[rank]
    desc = layout.layer_groups[group]
    topology = layout.topology
    paged_by_context = not desc.kind.per_request
    held_indices = None if replicated else _held_indices(layout, desc)
    out = []
    for other_rank in range(len(ranks)):
        other = ranks[other_rank]
        other_topology = other.topology
        if other_topology.dp_rank != topology.dp_rank:
            continue
        if paged_by_context and other_topology.cp_rank != topology.cp_rank:
            continue
        match = _same_group(other, desc)
        if match is None:
            continue
        if not replicated:
            other_held = _held_indices(other, match)
            if held_indices is None and other_held is None:
                if match.shard != desc.shard:
                    continue
            elif other_held != held_indices:
                continue
        out.append(other_rank)
    return tuple(out)


def layout_to_dict(layout: PageLayout) -> dict:
    """``layout`` as plain values (dicts, lists, ints, strings) for a peer, which
    ``layout_from_dict`` reads back; the lender's region maps stay local and are left out."""
    return {
        "tokens_per_page": layout.tokens_per_page,
        "pool_slot_bytes": {int(g): list(w) for g, w in layout.pool_slot_bytes.items()},
        "layer_groups": [
            {
                "kind": d.kind.to_dict(),
                "kept_window": d.kept_window,
                "num_sink_blocks": d.num_sink_blocks,
                "pool_group": d.pool_group,
                "layers": list(d.layers),
                "shard": [d.shard.count, d.shard.index],
                "read_window": d.read_window,
                "model_layers": list(d.model_layers),
            }
            for d in layout.layer_groups
        ],
        "buffers": [
            {
                "layer": b.layer,
                "role": b.role,
                "pool_group": b.pool_group,
                "pool_index": b.pool_index,
                "slot_offset": b.slot_offset,
                "size": b.size,
                "layout": b.layout.to_dict(),
                "transferred": b.transferred,
                "expansion": b.expansion,
            }
            for b in layout.buffers
        ],
        "topology": dataclasses.asdict(layout.topology),
        "total_kv_heads_by_layer": {int(k): v for k, v in layout.total_kv_heads_by_layer.items()},
        "dtype": layout.dtype,
        "quant_mode": layout.quant_mode,
        "rebuilds_untransferred": layout.rebuilds_untransferred,
        "request_rebuilt_roles": sorted(layout.request_rebuilt_roles),
    }


def layout_from_dict(data: Mapping) -> PageLayout:
    """The layout ``layout_to_dict`` wrote, checked as the constructors check it."""
    return PageLayout(
        tokens_per_page=data["tokens_per_page"],
        pool_slot_bytes={int(g): tuple(w) for g, w in data["pool_slot_bytes"].items()},
        layer_groups=tuple(
            LayerGroupDesc(
                kind=d["kind"],
                kept_window=d["kept_window"],
                num_sink_blocks=d["num_sink_blocks"],
                pool_group=d["pool_group"],
                layers=tuple(d["layers"]),
                shard=ShardDesc(*d["shard"]),
                read_window=d["read_window"],
                model_layers=tuple(d["model_layers"]),
            )
            for d in data["layer_groups"]
        ),
        buffers=tuple(
            BufferDesc(
                layer=b["layer"],
                role=b["role"],
                pool_group=b["pool_group"],
                pool_index=b["pool_index"],
                slot_offset=b["slot_offset"],
                size=b["size"],
                layout=role_layout_from_dict(b["layout"]),
                transferred=b["transferred"],
                expansion=b["expansion"],
            )
            for b in data["buffers"]
        ),
        topology=RankTopology(**data["topology"]),
        total_kv_heads_by_layer={int(k): v for k, v in data["total_kv_heads_by_layer"].items()},
        dtype=data["dtype"],
        quant_mode=data["quant_mode"],
        rebuilds_untransferred=data["rebuilds_untransferred"],
        request_rebuilt_roles=frozenset(data["request_rebuilt_roles"]),
    )
