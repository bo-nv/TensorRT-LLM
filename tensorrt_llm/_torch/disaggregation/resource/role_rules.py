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

"""Shard rules the disaggregation page-table builder applies per buffer role.

The split of responsibilities:

* The **cache manager** says *where* the bytes are (``impl.pool_group_descs``,
  every buffer tagged ``(layer_id, role, offset, size)``), *how each role's
  bytes are arranged* (``get_disagg_role_mapper_kinds()`` →
  :class:`MapperKind` per role), and exposes its topology (``mapping``) and
  plain model geometry (``num_kv_heads``, ``head_dim``, ``dtype``,
  ``tokens_per_block``, ``ssm_state_shape``, ``conv_section_dims``, state
  dtypes).
* The **transceiver** derives everything else here: which rank axis shards a
  role (:class:`ShardAxis`), how many shards / replicas this rank's bytes
  correspond to, the re-split unit or section sizes, and the element type
  peers must agree on. :func:`build_role_layout` turns a rule, the manager's
  declared kind and the manager's geometry into the :class:`RoleLayout`
  shipped on every :class:`PoolView`.

A role the manager declares but this table does not know is still
transferable when it is ``REPLICATED`` (nothing to split) or lives in a paged
group (K/V head arithmetic applies). A per-request state role needs a row
here, because the shard axis and unit cannot be guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Dict, Optional, Tuple

from tensorrt_llm._torch.disaggregation.resource.page import MapperKind, RoleLayout
from tensorrt_llm._utils import get_size_in_bytes


class ShardAxis(IntEnum):
    """Which ranks split one role's bytes between them.

    ``NONE``: every rank of the TP group holds identical bytes (side caches
    computed redundantly per rank). ``TP``: split by unit across the TP
    group, with more ranks than units duplicating units (GQA). ``TP_X_CP``:
    split evenly across the TP group, or across ``tp * cp`` ranks under
    helix context parallelism, never duplicated (recurrent state). With
    attention DP every axis collapses to a single rank: each DP rank holds
    the whole role for its own requests.
    """

    NONE = 0
    TP = 1
    TP_X_CP = 2


def _tp_grid(mapping, axis: ShardAxis) -> Tuple[int, int]:
    """``(group_size, group_rank)`` of this rank along *axis*."""
    if mapping.enable_attention_dp:
        return 1, 0
    if axis == ShardAxis.TP_X_CP and mapping.cp_size > 1:
        has_helix = getattr(mapping, "has_cp_helix", None)
        if has_helix is None or has_helix():
            return (
                mapping.tp_size * mapping.cp_size,
                mapping.tp_rank * mapping.cp_size + mapping.cp_rank,
            )
    return mapping.tp_size, mapping.tp_rank


def shard_position(
    axis: ShardAxis, mapping, units_global: Optional[int] = None
) -> Tuple[int, int, int, int]:
    """Place this rank in the shard grid of a role.

    Returns ``(num_shards, shard_index, num_replicas, replica_index)``. The
    role is split into ``min(group_size, units_global)`` distinct shards;
    when the group is larger than that, consecutive ranks hold the same shard
    as replicas (GQA head duplication, replicated side caches, MLA latent
    K/V). ``units_global=None`` means one shard per rank of the grid.
    """
    group_size, group_rank = _tp_grid(mapping, axis)
    if group_size <= 0 or not (0 <= group_rank < group_size):
        raise ValueError(f"invalid group position rank={group_rank}, size={group_size}")
    if axis == ShardAxis.NONE:
        return 1, 0, group_size, group_rank
    num_shards = group_size if units_global is None else max(1, min(group_size, units_global))
    if group_size % num_shards != 0:
        raise ValueError(
            f"group size {group_size} is not a multiple of the shard count {num_shards}"
        )
    num_replicas = group_size // num_shards
    return num_shards, group_rank // num_replicas, num_replicas, group_rank % num_replicas


# ---------------------------------------------------------------------------
# Geometry readers (each reads one plain attribute set of the manager)
# ---------------------------------------------------------------------------


def _kv_heads_global(geom, model_layer: Optional[int]) -> int:
    """Model-wide (pre-TP) KV-head count for one model layer."""
    heads = geom.num_kv_heads
    if isinstance(heads, int):
        return int(heads)
    heads = list(heads)
    if model_layer is not None and 0 <= model_layer < len(heads) and heads[model_layer]:
        return int(heads[model_layer])
    return next((int(h) for h in heads if h), 1)


def _kv_head_bytes(geom, model_layer: Optional[int]) -> Optional[int]:  # noqa: ARG001
    """Bytes of one KV head for one block; ``None`` when not byte-aligned."""
    nbytes = get_size_in_bytes(geom.tokens_per_block * geom.head_dim, geom.dtype)
    return int(nbytes) if float(nbytes).is_integer() else None


def _kv_elem(geom) -> Tuple[str, Tuple[int, ...]]:
    return str(geom.dtype), (int(geom.head_dim),)


def _ssm_unit_bytes(geom, model_layer: Optional[int]) -> int:  # noqa: ARG001
    _, head_dim, d_state = geom.ssm_state_shape
    return int(head_dim) * int(d_state) * int(geom.ssm_state_dtype.itemsize)


def _ssm_elem(geom) -> Tuple[str, Tuple[int, ...]]:
    _, head_dim, d_state = geom.ssm_state_shape
    return str(geom.ssm_state_dtype), (int(head_dim), int(d_state))


def _conv_section_bytes(geom, model_layer: Optional[int]) -> Tuple[int, ...]:  # noqa: ARG001
    d_conv_m1 = int(geom.conv_state_shape[1])
    itemsize = int(geom.conv_state_dtype.itemsize)
    return tuple(int(dim) * d_conv_m1 * itemsize for dim in geom.conv_section_dims)


def _conv_elem(geom) -> Tuple[str, Tuple[int, ...]]:
    return str(geom.conv_state_dtype), (int(geom.conv_state_shape[1]), len(geom.conv_section_dims))


def _none(geom, model_layer: Optional[int]):  # noqa: ARG001
    return None


def _no_elem(geom) -> Tuple[str, Tuple[int, ...]]:  # noqa: ARG001
    return "", ()


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoleRule:
    """How one role is sharded and what its geometry is, in terms of manager attributes.

    Every callable takes the manager (the geometry provider); the ones taking
    a second argument also receive the model layer index when the caller
    knows it (``None`` otherwise). The byte arrangement itself
    (:class:`MapperKind`) is *not* part of the rule: the manager declares it
    via ``get_disagg_role_mapper_kinds``; ``default_kind`` only serves
    managers without that hook (the V1 builder).

    Fields:
        axis: Rank axis the role is sharded along.
        default_kind: Arrangement assumed when the manager declares none.
        units_global: Model-wide count of shard units (KV heads); ``None``
            for one shard per rank without duplication.
        shard_unit_bytes: Re-split granularity for ``HND``/``NHD``.
        section_bytes: Per-section sizes for ``SECTIONED``.
        elem: ``(dtype name, element shape)`` peers must agree on.
    """

    axis: ShardAxis
    default_kind: MapperKind
    units_global: Callable[[Any, Optional[int]], Optional[int]] = _none
    shard_unit_bytes: Callable[[Any, Optional[int]], Optional[int]] = _none
    section_bytes: Callable[[Any, Optional[int]], Optional[Tuple[int, ...]]] = _none
    elem: Callable[[Any], Tuple[str, Tuple[int, ...]]] = _no_elem


KV_RULE = RoleRule(
    axis=ShardAxis.TP,
    default_kind=MapperKind.HND,
    units_global=_kv_heads_global,
    shard_unit_bytes=_kv_head_bytes,
    elem=_kv_elem,
)
"""Attention K/V: head-sharded across TP, heads duplicated when TP exceeds them."""

REPLICATED_RULE = RoleRule(axis=ShardAxis.NONE, default_kind=MapperKind.REPLICATED)
"""Side caches identical on every TP rank (sparse-attention index keys)."""

SSM_STATE_RULE = RoleRule(
    axis=ShardAxis.TP_X_CP,
    default_kind=MapperKind.HND,
    shard_unit_bytes=_ssm_unit_bytes,
    elem=_ssm_elem,
)
"""SSM state ``(nheads, head_dim, d_state)``: one head per shard unit."""

CONV_STATE_RULE = RoleRule(
    axis=ShardAxis.TP_X_CP,
    default_kind=MapperKind.SECTIONED,
    section_bytes=_conv_section_bytes,
    elem=_conv_elem,
)
"""Convolution state ``(conv_dim, d_conv - 1)``: ``[x|B|C]`` or ``[q|k|v]`` sections."""

ROLE_RULES: Dict[str, RoleRule] = {
    "key": KV_RULE,
    "value": KV_RULE,
    "index_key": REPLICATED_RULE,
    "indexer_k": REPLICATED_RULE,
    "ssm_state": SSM_STATE_RULE,
    "conv_state": CONV_STATE_RULE,
}
"""Rule per buffer role name. Roles not listed fall back per :func:`resolve_role_rule`."""


def resolve_role_rule(
    role: str, kind: MapperKind, *, has_token_axis: bool, context: str = ""
) -> RoleRule:
    """Rule for *role*, given the arrangement the manager declared for it.

    Unlisted roles: ``REPLICATED`` needs no split arithmetic; a paged role
    shares the K/V head arrangement (block scales, DeepSeek-V4 per-type
    caches). A per-request state role that is neither has no safe shard
    axis and is rejected, so a new state type is never transferred with the
    wrong split.
    """
    rule = ROLE_RULES.get(role)
    if rule is not None:
        return rule
    if kind == MapperKind.REPLICATED:
        return REPLICATED_RULE
    if has_token_axis:
        return KV_RULE
    raise ValueError(
        f"{context}: per-request state role {role!r} is declared {kind.name} but has no "
        "entry in disaggregation ROLE_RULES; add one before enabling transfer"
    )


def build_role_layout(
    rule: RoleRule,
    geom,
    kind: Optional[MapperKind] = None,
    *,
    model_layer: Optional[int] = None,
    context: str = "",
) -> RoleLayout:
    """Instantiate *rule* for this rank with the manager's declared *kind* and geometry."""
    if kind is None:
        kind = rule.default_kind
    sections = rule.section_bytes(geom, model_layer)
    if (kind == MapperKind.SECTIONED) != (sections is not None):
        raise ValueError(
            f"{context}: mapper kind {kind.name} conflicts with the role's shard rule "
            f"({'has' if sections is not None else 'no'} section geometry)"
        )
    axis = ShardAxis.NONE if kind == MapperKind.REPLICATED else rule.axis
    units = None if axis == ShardAxis.NONE else rule.units_global(geom, model_layer)
    num_shards, shard_index, num_replicas, replica_index = shard_position(axis, geom.mapping, units)
    elem_dtype, elem_shape = rule.elem(geom)
    return RoleLayout(
        kind,
        num_shards=num_shards,
        shard_index=shard_index,
        num_replicas=num_replicas,
        replica_index=replica_index,
        shard_unit_bytes=(
            None if kind == MapperKind.REPLICATED else rule.shard_unit_bytes(geom, model_layer)
        ),
        section_bytes=sections,
        elem_dtype=elem_dtype,
        elem_shape=elem_shape,
    )
