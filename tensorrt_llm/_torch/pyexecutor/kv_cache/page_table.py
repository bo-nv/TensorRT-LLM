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
"""A ``KVCacheManagerV2``'s page table as plain values: how its pages are laid out (``PageLayout``),
where its device pools lie and what each layer group is; and what a layer config or a rank's place
says about it.

``build_page_table`` builds the table from a manager: its runtime config and storage say which
buffers exist, of which layers and where; its ``PageTableSpec`` says what they do not, such as how
each role's bytes spread over ranks. Readers of the table, such as the lender and the native
transfer, never look into the manager."""

from __future__ import annotations

import dataclasses
import enum
import types
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Mapping

from .page_layout import (
    BufferDesc,
    GroupKind,
    LayerGroupDesc,
    Paged,
    PageLayout,
    PerRequest,
    RankTopology,
    check_pieces,
    content_shard,
    grid_place,
    layout_from_dict,
    layout_to_dict,
)

# The role layouts a manager's spec names (page_layout), for managers to import from here.
from .page_layout import ByHead as ByHead
from .page_layout import Own as Own
from .page_layout import Replicated as Replicated
from .page_layout import RingSlots as RingSlots
from .page_layout import RoleLayout as RoleLayout
from .page_layout import Sectioned as Sectioned

if TYPE_CHECKING:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig, SsmLayerConfig


@dataclass(frozen=True)
class DevicePool:
    """One device pool of pool group ``group``: slot ``s`` lives at ``base + s * slot_bytes``."""

    group: int
    index: int
    base: int
    slot_bytes: int
    num_slots: int


@dataclass(frozen=True, eq=False)
class ManagerLayout:
    """A manager's page table as plain values, without a manager reference; ``cp`` is (rank, size),
    (0, 1) without Helix. ``tokens_per_block`` spans ``size`` physical pages of
    ``tokens_per_block // size`` tokens."""

    page_layout: PageLayout
    tokens_per_block: int
    pool_group_of: tuple[int, ...]
    windows: tuple[int | None, ...]
    sink_blocks: tuple[int, ...]
    recurrent: tuple[bool, ...]
    sparse: tuple[bool, ...]
    layers: tuple[tuple[int, ...], ...]
    shards: tuple[tuple[int, int], ...]
    page_bytes: Mapping[int, int]
    device_pools: Mapping[int, tuple[DevicePool, ...]]
    cp: tuple[int, int] = (0, 1)
    # Per layer group, the window its attention reads, at most the window kept; ``()``: the same.
    read_windows: tuple[int | None, ...] = ()
    # Per layer group, whether it lends by name: none of its layers kept out of prefix reuse.
    named_groups: tuple[bool, ...] = field(kw_only=True)
    # Per layer group, whether every layer must cover a reused prefix; ``()``: every group's.
    prefix_required: tuple[bool, ...] = field(default=(), kw_only=True)
    # Per layer group, its (internal layer id, global layer id) pairs in storage order.
    local_layers: tuple[tuple[tuple[int, int], ...], ...] = field(default=(), kw_only=True)
    # Per layer group, the KV heads on this rank of its first layer.
    kv_heads: tuple[int, ...] = field(default=(), kw_only=True)
    # Whether every windowed layer group's stale range is the runtime's window formula of its
    # window and sink blocks at each history, as the manager answered for every history probed.
    stale_ranges_by_formula: bool = field(default=False, kw_only=True)

    @property
    def num_layer_groups(self) -> int:
        """The manager's layer groups."""
        return len(self.pool_group_of)

    @property
    def pool_groups(self) -> tuple[int, ...]:
        """The device pool groups, sorted; staging regions follow this order."""
        return tuple(sorted(self.device_pools))


def page_table_to_dict(table: ManagerLayout) -> dict:
    """``table`` as plain values (dicts, lists, ints, strings) for a peer, which
    ``page_table_from_dict`` reads back; its page layout as ``layout_to_dict`` writes it."""
    data: dict[str, Any] = {}
    for f in dataclasses.fields(table):
        value = getattr(table, f.name)
        if f.name == "page_layout":
            value = layout_to_dict(value)
        elif f.name == "device_pools":
            value = {int(g): [list(dataclasses.astuple(p)) for p in ps] for g, ps in value.items()}
        elif f.name == "page_bytes":
            value = {int(g): int(n) for g, n in value.items()}
        else:
            value = _lists(value)
        data[f.name] = value
    return data


def page_table_from_dict(data: Mapping) -> ManagerLayout:
    """The page table ``page_table_to_dict`` wrote."""
    values: dict[str, Any] = {}
    for f in dataclasses.fields(ManagerLayout):
        if f.name not in data:
            continue
        value = data[f.name]
        if f.name == "page_layout":
            value = layout_from_dict(value)
        elif f.name == "device_pools":
            value = {int(g): tuple(DevicePool(*p) for p in ps) for g, ps in value.items()}
        elif f.name == "page_bytes":
            value = {int(g): int(n) for g, n in value.items()}
        else:
            value = _tuples(value)
        values[f.name] = value
    return ManagerLayout(**values)


def _lists(value: Any) -> Any:
    return [_lists(v) for v in value] if isinstance(value, (tuple, list)) else value


def _tuples(value: Any) -> Any:
    return tuple(_tuples(v) for v in value) if isinstance(value, (tuple, list)) else value


def virtual_layers(
    attn_to_layer_id: Mapping[tuple[int, enum.Enum], int] | None,
) -> tuple[dict[int, tuple[int, int]], int] | None:
    """(internal id -> (model layer, attention type), type count) of a manager keeping several
    caches per model layer, from its (model layer, attention type) -> internal id map; None for an
    empty map. An internal id shared by several attention types of one model layer, as DeepSeek-V4
    keeps them, takes the first. A cache several model layers share enters the map only under the
    model layer that owns it, as CSA2's does, so its name does not depend on which layers a stage
    holds. Every member of the attention-type enum counts, so stages holding different attention
    types agree."""
    if not attn_to_layer_id:
        return None
    types_of: dict[int, tuple[int, int]] = {}
    for (model_layer, attn_type), layer in attn_to_layer_id.items():
        types_of.setdefault(int(layer), (int(model_layer), int(attn_type.value)))
    kinds = type(next(iter(attn_to_layer_id))[1])
    return types_of, max(int(member.value) for member in kinds) + 1


# -- what a layer config says ---------------------------------------------------------------------


def layer_configs_by_id(config: object) -> dict[int, AttentionLayerConfig | SsmLayerConfig]:
    """Internal layer id -> layer config of a runtime config; a layer without an explicit id uses
    its position in the config."""
    return {
        int(getattr(layer, "layer_id", position)): layer
        for position, layer in enumerate(config.layers)
    }


def is_recurrent_layer(layer_config: AttentionLayerConfig | SsmLayerConfig) -> bool:
    """Whether a layer config of the runtime describes a recurrent-state layer."""
    from tensorrt_llm.runtime import kv_cache_manager_v2 as runtime

    return isinstance(layer_config, runtime.SsmLayerConfig)


def layer_group_kind(layer_config: AttentionLayerConfig | SsmLayerConfig) -> GroupKind:
    """How the layer group of a layer config addresses a request's pages, which says which of its
    slots one transfer pairs (``layout_mapping.pair_slots``): one slot per request for recurrent
    state, else one per page of its tokens. A layer config of a new kind the runtime addresses
    otherwise takes a kind of its own here (``page_layout.GroupKind``)."""
    return PerRequest() if is_recurrent_layer(layer_config) else Paged()


def is_private_layer(layer_config: AttentionLayerConfig | SsmLayerConfig) -> bool:
    """Whether the native lifecycle excludes this layer from prefix reuse."""
    from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionReusePolicy

    return getattr(layer_config, "reuse_policy", None) == AttentionReusePolicy.PRIVATE


def is_non_required_layer(layer_config: AttentionLayerConfig | SsmLayerConfig) -> bool:
    """Whether prefix coverage is optional or private for this layer."""
    from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionReusePolicy

    return (
        getattr(layer_config, "reuse_policy", AttentionReusePolicy.REQUIRED)
        != AttentionReusePolicy.REQUIRED
    )


def has_sparse_buffers(layer_config: AttentionLayerConfig | SsmLayerConfig) -> bool:
    """Whether a layer config of the runtime has sparse buffers (``is_sparse``), whose read-only
    pages a cache can lock in host memory."""
    return any(getattr(b, "is_sparse", False) for b in getattr(layer_config, "buffers", ()) or ())


def buffer_expansions(
    layer_config: AttentionLayerConfig | SsmLayerConfig, tokens_per_block: int
) -> dict[str, int]:
    """Role -> sub-pages per block, from the layer's ``tokens_per_block_override``."""
    out = {}
    for buffer in getattr(layer_config, "buffers", ()) or ():
        override = getattr(buffer, "tokens_per_block_override", None)
        if override is not None:
            if tokens_per_block % int(override):
                raise ValueError(
                    f"buffer {buffer.role}: {override} tokens per block does not divide "
                    f"{tokens_per_block}"
                )
            out[str(buffer.role)] = tokens_per_block // int(override)
    return out


# -- where a rank stands and what it holds --------------------------------------------------------


def rank_topology(mapping: object) -> RankTopology:
    """The rank's place as native's ``RankInfo`` reads it from the mapping: under attention data
    parallelism each tensor-parallel rank is a data-parallel rank of its own."""
    if mapping is None:
        return RankTopology()
    adp = bool(getattr(mapping, "enable_attention_dp", False))
    tp_size, tp_rank = int(getattr(mapping, "tp_size", 1)), int(getattr(mapping, "tp_rank", 0))
    return RankTopology(
        tp_size=tp_size,
        tp_rank=tp_rank,
        pp_size=int(getattr(mapping, "pp_size", 1)),
        pp_rank=int(getattr(mapping, "pp_rank", 0)),
        cp_size=int(getattr(mapping, "cp_size", 1)),
        cp_rank=int(getattr(mapping, "cp_rank", 0)),
        dp_size=tp_size if adp else int(getattr(mapping, "dp_size", 1)),
        dp_rank=tp_rank if adp else 0,
        attention_dp=adp,
    )


# -- what a manager says of its pages: its PageTableSpec ------------------------------------------


@dataclass(frozen=True)
class BufferSite:
    """One buffer as a manager's spec leaves it to its role layout (``RoleLayout.resolve``).

    Attributes:
        layer: The manager's internal layer id.
        role: The buffer's role.
        size: Bytes of the buffer in one page's slot.
        tokens: Tokens of one page, or of one sub-page where the buffer has several per page.
        state: Whether the layer keeps recurrent state rather than attention KV.
        kv_heads: (KV heads on this rank, KV heads over all tensor ranks or None) of the layer.
        split: The parallel dimensions the layer's buffers split over: the tensor ranks for
            attention, those of the manager's recurrent state for state.
        ranks: Ranks in the grid of ``split``.
    """

    layer: int
    role: str
    size: int
    tokens: int
    state: bool
    kv_heads: tuple[int, int | None]
    split: tuple[str, ...]
    ranks: int


def _frozen_roles(roles) -> frozenset[str]:
    return frozenset(str(role) for role in roles)


@dataclass(frozen=True, eq=False)
class PageTableSpec:
    """What a manager says of its pages that its runtime config and storage do not.

    A manager returns it from ``page_table_spec``; ``build_page_table`` reads it with the
    manager's storage. The base manager's says K/V and their block scales split by KV head; a
    manager with other roles, or one of these laid out otherwise, returns its own, usually the
    base one with a few fields replaced (``dataclasses.replace``).

    Attributes:
        role_layouts: Role -> how its bytes spread over ranks.
        default_layout: The layout of a role ``role_layouts`` does not name, or None.
        untransferred_roles: Roles that hold nothing a peer needs: they stay in local copies of
            whole pages but never move.
        derived_roles: Roles derived from the transferred ones, which a transfer leaves for
            ``rebuild_disagg_imported_cache`` to rebuild after the import; the lender lends them
            with the rest.
        request_rebuilt_roles: Roles the manager, receiving, rebuilds for some requests rather
            than receiving them (``disagg_untransferred_buffers``), so a peer may hold none of
            them for a model layer it holds.
        layer_ids: Internal layer id -> (global layer id, model layer); None for those of the
            manager's model layers (``pp_layers``) or, keeping several caches per model layer,
            of its (model layer, attention type) map (``_layer_attn_to_layer_id``).
        quant_mode: How the KV data is quantized: None unquantized, else ``"fp8"``,
            ``"nvfp4"`` or ``"fp8_ds_mla"``.
    """

    role_layouts: Mapping[str, RoleLayout] = field(default_factory=dict)
    default_layout: RoleLayout | None = None
    untransferred_roles: frozenset[str] = frozenset()
    derived_roles: frozenset[str] = frozenset()
    request_rebuilt_roles: frozenset[str] = frozenset()
    layer_ids: Mapping[int, tuple[int, int]] | None = None
    quant_mode: str | None = None

    def __post_init__(self) -> None:
        layouts = {str(role): layout for role, layout in dict(self.role_layouts).items()}
        object.__setattr__(self, "role_layouts", types.MappingProxyType(layouts))
        for name in ("untransferred_roles", "derived_roles", "request_rebuilt_roles"):
            object.__setattr__(self, name, _frozen_roles(getattr(self, name)))
        if self.layer_ids is not None:
            ids = {int(lid): (int(g), int(m)) for lid, (g, m) in dict(self.layer_ids).items()}
            object.__setattr__(self, "layer_ids", types.MappingProxyType(ids))

    def layout_of(self, role: str) -> RoleLayout | None:
        """The layout of ``role``: its own, else the default, else this rank's own share for a
        role that never moves; None for a role that moves and that the spec does not lay out."""
        layout = self.role_layouts.get(role, self.default_layout)
        if layout is None and role in self.untransferred_roles | self.derived_roles:
            return Own()
        return layout


def kv_dtype_name(manager: object) -> str | None:
    """The element type of a manager's KV data by name, such as ``"BF16"``; None if it has
    none."""
    dtype = getattr(manager, "dtype", None)
    return None if dtype is None else str(getattr(dtype, "name", dtype))


def state_split(manager: object) -> tuple[str, ...]:
    """The parallel dimensions a manager's recurrent state splits over, as its pools are sized
    (``mamba_effective_tp_size``): tensor times context ranks under Helix, else tensor ranks,
    which attention data parallelism makes one rank alone (``grid_place``)."""
    if getattr(manager, "_has_cp_helix", False) and not getattr(
        manager.mapping, "enable_attention_dp", False
    ):
        return ("tp", "cp")
    return ("tp",)


def layer_ids_of(manager: object, spec: PageTableSpec | None = None) -> dict[int, tuple[int, int]]:
    """Internal layer id -> (global layer id, model layer) of each layer of ``manager``'s layer
    groups, as its page table names them: the ``layer_ids`` of its spec (``spec``, else its
    ``page_table_spec()``) or, without them, its model layers (``_layer_ids``).

    Raises:
        ValueError: If its model layers name no global layer for some layer.
    """
    spec = manager.page_table_spec() if spec is None else spec
    if spec.layer_ids is not None:
        return dict(spec.layer_ids)
    internal = sorted({int(lid) for group in manager.impl.layer_grouping for lid in group})
    return _layer_ids(manager, internal)


def _layer_ids(manager: object, internal: list[int]) -> dict[int, tuple[int, int]]:
    """Internal layer id -> (global layer id, model layer), the same under any pipeline split:
    the model layer itself, or ``model_layer * num_types + type`` for a manager keeping several
    caches per model layer (``virtual_layers``)."""
    pp_layers = list(manager.pp_layers)
    virtual = virtual_layers(getattr(manager, "_layer_attn_to_layer_id", None))
    if virtual is None:
        unnamed = [lid for lid in internal if not 0 <= lid < len(pp_layers)]
        if unnamed:
            raise ValueError(
                f"internal layers {unnamed} are no local layers, and the manager names no model "
                "layer for them (_layer_attn_to_layer_id, PageTableSpec.layer_ids)"
            )
        # Internal layer i is the stage's i-th model layer, as layer_offsets numbers it.
        offsets = getattr(manager, "layer_offsets", None)
        if offsets is not None:
            renumbered = [lid for lid in internal if offsets.get(int(pp_layers[lid])) != lid]
            if renumbered:
                raise ValueError(
                    f"internal layers {renumbered} are not the local index layer_offsets gives "
                    "their model layers; a manager numbering its layers otherwise names them "
                    "(_layer_attn_to_layer_id, PageTableSpec.layer_ids)"
                )
        return {lid: (int(pp_layers[lid]),) * 2 for lid in internal}
    types_of, num_types = virtual
    unnamed = [lid for lid in internal if lid not in types_of]
    if unnamed:
        raise ValueError(
            f"internal layers {unnamed} have no (model layer, type) in _layer_attn_to_layer_id"
        )
    return {
        lid: (types_of[lid][0] * num_types + types_of[lid][1], types_of[lid][0]) for lid in internal
    }


def _kv_heads(manager: object, model_layer: int) -> tuple[int, int | None]:
    """(KV heads on this rank, KV heads over all tensor ranks or None if unknown) of a model
    layer: 0 heads on this rank for one this pipeline stage does not hold."""
    pp_layers = [int(m) for m in manager.pp_layers]
    per_layer = list(manager.num_kv_heads_per_layer or ())
    index = pp_layers.index(model_layer) if model_layer in pp_layers else len(per_layer)
    per_rank = int(per_layer[index] or 0) if index < len(per_layer) else 0
    total = manager.num_kv_heads
    if isinstance(total, int):
        return per_rank, int(total)
    if total is None or model_layer >= len(total) or total[model_layer] is None:
        return per_rank, None
    return per_rank, int(total[model_layer])


def _definer(manager: object, name: str):
    """Where ``manager``'s ``name`` comes from: the manager itself for one bound to it, else the
    first class of its MRO defining it."""
    return next((c for c in (manager, *type(manager).__mro__) if name in vars(c)), None)


def _overrides(manager: object, name: str) -> bool:
    """Whether ``manager``'s method ``name``, bound to it or of its class, is its own rather than
    the base manager's, which rebuilds nothing."""
    from .kv_cache_manager_v2 import KVCacheManagerV2

    method = getattr(manager, name, None)
    if method is None:
        return False
    return getattr(method, "__func__", method) is not vars(KVCacheManagerV2)[name]


def _check_rebuilds(manager: object, spec: PageTableSpec, rebuilt: list[str]) -> None:
    """Refuse a manager deriving or rebuilding roles without the method that rebuilds them."""
    name = type(manager).__name__
    if rebuilt:
        if not _overrides(manager, "rebuild_disagg_imported_cache"):
            raise ValueError(
                f"{name} derives its {rebuilt} buffers from the transferred ones but does not "
                "override rebuild_disagg_imported_cache, so an import would leave them stale"
            )
        spec_owner = _definer(manager, "page_table_spec")
        rebuild_owner = _definer(manager, "rebuild_disagg_imported_cache")
        if (
            isinstance(spec_owner, type)
            and isinstance(rebuild_owner, type)
            and spec_owner is not rebuild_owner
            and issubclass(spec_owner, rebuild_owner)
        ):
            inherited = super(spec_owner, manager).page_table_spec().derived_roles
            added = sorted(set(rebuilt) - set(inherited))
            if added:
                raise ValueError(
                    f"{spec_owner.__name__} derives its {added} buffers, but its rebuild comes "
                    f"from {rebuild_owner.__name__}, which does not know them; override "
                    "rebuild_disagg_imported_cache with them"
                )
    per_request = sorted(spec.request_rebuilt_roles)
    if per_request and not _overrides(manager, "disagg_untransferred_buffers"):
        raise ValueError(
            f"{name} rebuilds its {per_request} buffers per request but does not override "
            "disagg_untransferred_buffers, which names the requests it rebuilds them for, so a "
            "peer lacking them would leave them unwritten"
        )


def build_page_table(manager: object) -> ManagerLayout:
    """``manager``'s page table: its page layout, device pools and the facts of its layer groups.

    Read from the manager's runtime config (``kv_cache_manager_py_config``: layers, their buffers
    and windows), its storage (``impl``: layer groups, pool groups, which buffers each slot
    holds), its parallel place and model (``mapping``, ``pp_layers``, KV heads, ``dtype``), and
    what it says beyond those (``page_table_spec``).

    Raises:
        ValueError: If its layer groups, layer configs and spec do not agree: a layer without a
            global id or of a model layer the stage does not hold, a role that moves and that the
            spec does not lay out, pieces that do not cover their buffer, two internal layers
            naming one buffer, a derived role without a rebuild, or a read window outside the
            kept window.
    """
    spec = manager.page_table_spec()
    impl = manager.impl
    tokens_per_block = int(manager.tokens_per_block)
    layer_cfg = layer_configs_by_id(manager.kv_cache_manager_py_config)
    grouping = [[int(layer) for layer in group] for group in impl.layer_grouping]
    pg_of_lg = [int(g) for g in impl.get_life_cycle_pool_group_indices()]
    if len(pg_of_lg) != len(grouping):
        raise ValueError(f"{len(grouping)} layer groups but {len(pg_of_lg)} pool-group indices")
    internal = sorted({lid for group in grouping for lid in group})
    layer_ids = layer_ids_of(manager, spec)
    missing = sorted(set(internal) - set(layer_ids))
    if missing:
        raise ValueError(f"the manager names no global layer for internal layers {missing}")
    global_of = {lid: int(ids[0]) for lid, ids in layer_ids.items()}
    model_of = {lid: int(ids[1]) for lid, ids in layer_ids.items()}
    stage_layers = {int(layer) for layer in manager.pp_layers}
    unheld = sorted({model_of[lid] for lid in internal} - stage_layers)
    if unheld:
        raise ValueError(
            f"the manager names model layers {unheld}, which this pipeline stage does not "
            f"hold (pp_layers {sorted(stage_layers)})"
        )
    kv_heads = {lid: _kv_heads(manager, model_of[lid]) for lid in internal}

    state_groups = set()
    for lg, group in enumerate(grouping):
        if not group or group[0] not in layer_cfg:
            raise ValueError(f"layer group {lg} has no layer config (layers {group})")
        if is_recurrent_layer(layer_cfg[group[0]]):
            state_groups.add(lg)

    derived = spec.derived_roles
    untransferred = spec.untransferred_roles | derived
    expansions = {lid: buffer_expansions(cfg, tokens_per_block) for lid, cfg in layer_cfg.items()}
    topology = rank_topology(manager.mapping)
    splits = {False: ("tp",), True: state_split(manager)}
    ranks = {state: grid_place(topology, split)[0] for state, split in splits.items()}
    cp_rank, cp_size = int(manager._helix_cp_rank), int(manager._helix_cp_size)

    device_pools: dict[int, tuple[DevicePool, ...]] = {}
    buffers: list[BufferDesc] = []
    # (global layer, role) -> the internal layer whose buffer it names.
    named: dict[tuple[int, str], int] = {}
    group_buffers: dict[int, list[BufferDesc]] = {lg: [] for lg in range(len(grouping))}
    for pg in impl.pool_group_descs:
        g = int(pg.pool_group_index)
        pools = tuple(
            DevicePool(
                g, int(p.pool_index), int(p.base_address), int(p.slot_bytes), int(pg.num_slots)
            )
            for p in pg.pools
        )
        device_pools[g] = tuple(sorted(pools, key=lambda p: p.index))
        # One variant per layer group drawing from this pool group; pool ``i`` of a slot holds
        # the ``i``-th coalesced buffer, whose members sit back to back.
        for variant in pg.slot_desc.variants:
            lg = int(variant.layer_group_id)
            state = lg in state_groups
            for pool_index, coalesced in enumerate(variant.coalesced_buffers):
                size = int(coalesced.single_buffer_size)
                for j, buffer_id in enumerate(coalesced.buffer_ids):
                    lid = int(buffer_id.layer_id)
                    role = str(buffer_id.role)
                    desc = _describe_buffer(
                        type(manager).__name__,
                        spec,
                        BufferSite(
                            layer=lid,
                            role=role,
                            size=size,
                            tokens=tokens_per_block // expansions.get(lid, {}).get(role, 1),
                            state=state,
                            kv_heads=kv_heads.get(lid, (0, None)),
                            split=splits[state],
                            ranks=ranks[state],
                        ),
                        topology,
                        global_layer=global_of.get(lid, lid),
                        pool=(g, pool_index, j * size),
                        transferred=role not in untransferred,
                        expansion=expansions.get(lid, {}).get(role, 1),
                    )
                    owner = named.setdefault((desc.layer, role), lid)
                    if owner != lid:
                        raise ValueError(
                            f"internal layers {owner} and {lid} both name the {role} buffer "
                            f"of global layer {desc.layer}"
                        )
                    buffers.append(desc)
                    group_buffers[lg].append(desc)

    rebuilt = sorted({b.role for b in buffers if b.role in derived})
    _check_rebuilds(manager, spec, rebuilt)

    layer_groups: list[LayerGroupDesc] = []
    total_kv_heads: dict[int, int] = {}
    for lg, group in enumerate(grouping):
        first = layer_cfg[group[0]]
        is_state = lg in state_groups
        window = read = None
        sink_tokens = 0
        if not is_state:
            window = getattr(first, "window_size", None)
            window = None if window is None else int(window)
            read = manager._attention_read_window(lg)
            read = None if read is None else int(read)
            if (read is None) != (window is None) or (read is not None and not 0 < read <= window):
                raise ValueError(
                    f"layer group {lg} reads a window of {read} tokens, which its kept window "
                    f"of {window} tokens does not hold"
                )
            sink_tokens = int(getattr(first, "num_sink_tokens", None) or 0)
            for lid in group:
                total = kv_heads[lid][1]
                if total:
                    total_kv_heads[global_of[lid]] = total
        # Layer groups with equal slot sizes may share a pool group, each drawing its own slots.
        g = pg_of_lg[lg]
        if g not in device_pools:
            raise ValueError(f"layer group {lg} maps to pool group {g}, which has no pools")
        names = sorted((global_of[lid], model_of[lid]) for lid in group)
        layer_groups.append(
            LayerGroupDesc(
                kind=layer_group_kind(first),
                kept_window=window,
                num_sink_blocks=-(-sink_tokens // tokens_per_block),
                pool_group=g,
                layers=tuple(layer for layer, _ in names),
                # The lender lends derived buffers too, so its names cover their bytes.
                shard=content_shard(
                    [
                        replace(b, transferred=True) if b.role in derived else b
                        for b in group_buffers[lg]
                    ],
                    topology,
                ),
                read_window=read,
                model_layers=tuple(model_layer for _, model_layer in names),
            )
        )

    page_layout = PageLayout(
        tokens_per_page=tokens_per_block,
        pool_slot_bytes={
            g: tuple(p.slot_bytes for p in pools) for g, pools in device_pools.items()
        },
        layer_groups=tuple(layer_groups),
        buffers=tuple(buffers),
        topology=topology,
        total_kv_heads_by_layer=total_kv_heads,
        dtype=kv_dtype_name(manager),
        quant_mode=spec.quant_mode,
        rebuilds_untransferred=bool(rebuilt),
        request_rebuilt_roles=spec.request_rebuilt_roles,
    )
    return ManagerLayout(
        page_layout=page_layout,
        tokens_per_block=tokens_per_block * cp_size,
        pool_group_of=tuple(pg_of_lg),
        windows=tuple(desc.kept_window for desc in layer_groups),
        sink_blocks=tuple(desc.num_sink_blocks for desc in layer_groups),
        recurrent=tuple(desc.kind.per_request for desc in layer_groups),
        sparse=tuple(has_sparse_buffers(layer_cfg[group[0]]) for group in grouping),
        layers=tuple(desc.layers for desc in layer_groups),
        shards=tuple((desc.shard.count, desc.shard.index) for desc in layer_groups),
        page_bytes={g: sum(p.slot_bytes for p in pools) for g, pools in device_pools.items()},
        device_pools=device_pools,
        cp=(cp_rank, cp_size),
        read_windows=tuple(desc.read_window for desc in layer_groups),
        named_groups=tuple(
            not any(is_private_layer(layer_cfg[lid]) for lid in group) for group in grouping
        ),
        prefix_required=tuple(
            not any(is_non_required_layer(layer_cfg[lid]) for lid in group) for group in grouping
        ),
        local_layers=tuple(tuple((lid, global_of[lid]) for lid in group) for group in grouping),
        kv_heads=tuple(kv_heads[group[0]][0] for group in grouping),
    )


def _describe_buffer(
    manager_name: str,
    spec: PageTableSpec,
    site: BufferSite,
    topology: RankTopology,
    *,
    global_layer: int,
    pool: tuple[int, int, int],
    transferred: bool,
    expansion: int,
) -> BufferDesc:
    """One buffer of the page layout, laid out as the spec says.

    Raises:
        ValueError: If the spec lays out no role that moves, the role's layout does not fit the
            buffer or its pieces on this rank do not lie back to back over it, or a buffer of
            several sub-pages per block that moves splits over ranks other than as each rank's
            own share.
    """
    where = f"buffer ({site.layer}, {site.role})"
    layout = spec.layout_of(site.role)
    if layout is None:
        raise ValueError(
            f"{where}: {manager_name} registers role {site.role!r} but does not declare how its "
            "bytes spread over ranks; name it in its PageTableSpec's role_layouts (or in "
            "untransferred_roles or derived_roles if it never moves)"
        )
    try:
        layout = layout.resolve(site)
        layout.check(site.size)
        check_pieces(layout, site.size, *grid_place(topology, layout.split))
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from error
    if (
        expansion != 1
        and transferred
        and site.role not in spec.derived_roles
        and layout.split
        and not isinstance(layout, Own)
    ):
        # Its pieces describe one page; ranks holding parts of several sub-pages would pair bytes
        # of different sub-pages. A rank's own share pairs only with the same rank's, whole.
        raise ValueError(
            f"{where}: {expansion} sub-pages per block that move are declarable only replicated "
            "or as each rank's own share"
        )
    group, index, offset = pool
    return BufferDesc(
        layer=global_layer,
        role=site.role,
        pool_group=group,
        pool_index=index,
        slot_offset=offset,
        size=site.size,
        layout=layout,
        transferred=transferred,
        expansion=expansion,
    )
