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
"""The blob backend between two real staging lenders.

One manager publishes a request's KV blocks through its lender's staging slots; another fetches
them into a request of its own. A test lease holder drives the lender as the runtime does: lend,
wait for the view, submit it to the backend as it stands, wait until the attempt quiesces, mark
what arrived, release. The store is the in-memory one, or with ``TRTLLM_MOONCAKE_REAL=1`` a real
Mooncake pool (``TRTLLM_MOONCAKE_PROTOCOL``, default rdma). Allocates device pools.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import threading
from dataclasses import dataclass, replace

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.disaggregation.base.backend import MemoryKind
from tensorrt_llm._torch.disaggregation.blob.backend import BlobBackend, BlobConfig, RegionBinding
from tensorrt_llm._torch.disaggregation.blob.keys import encode_key
from tensorrt_llm._torch.disaggregation.blob.mooncake import MooncakeDriver
from tensorrt_llm._torch.disaggregation.blob.mooncake_setup import MooncakeConfig, open_sdk
from tensorrt_llm._torch.disaggregation.blob.store import DriverRejected
from tensorrt_llm._torch.disaggregation.testing.memory_store import (
    DEFAULT_CAPABILITIES,
    Commit,
    MemoryDriver,
    MemoryStore,
)
from tensorrt_llm._torch.disaggregation.testing.object_trailer import Checksum, encode_trailer
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import (
    LeaseView,
    ResumeBounds,
    StagingOptions,
    attach_lender,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing._layout import derive_layout, group_buffers

# An exception that escapes a thread fails the test.
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="allocates KV cache pools"),
    pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning"),
]

TPB = 32
PROMPT = list(range(1000, 1097))  # three whole blocks and one token
END = 96
BLOCKS = END // TPB
SOURCE, TARGET = 1, 2
SCOPE = b"blob-backend-round-trip"
KEY_PREFIX = "round-trip"
WAIT_S = 60.0
PROTOCOL = os.environ.get("TRTLLM_MOONCAKE_PROTOCOL", "rdma")
DEVICES = os.environ.get("TRTLLM_MOONCAKE_DEVICES", "")
ONE_ROW_PER_CALL = replace(DEFAULT_CAPABILITIES, max_rows_per_op=1)


class Wake:
    """The ``WakeFn`` a test hands the backend, with bounded waits on what it signals."""

    def __init__(self) -> None:
        self._woken = threading.Condition()

    def __call__(self) -> None:
        with self._woken:
            self._woken.notify_all()

    def settled(self, attempt):
        """The attempt's outcome, once it also quiesced."""
        with self._woken:
            done = self._woken.wait_for(
                lambda: attempt.poll() is not None and attempt.quiesced(), WAIT_S
            )
        assert done, "the attempt did not settle"
        return attempt.poll()


class Stores:
    """Drivers over one store: the in-memory one, or with ``master`` a real Mooncake pool. A test
    closes each driver it opened as its owner does, last; ``close`` closes those a failed test left
    open, only once every backend built over them shut down cleanly. ``cancel_effective=False``
    makes the in-memory drivers ignore a cancel."""

    def __init__(self, master: str | None = None, cancel_effective: bool = True) -> None:
        self._store = MemoryStore() if master is None else None
        self._master = master
        self._cancel_effective = cancel_effective
        self._drivers = []
        self.backends = []

    def open_driver(self):
        """A driver that moves every row on its own: the in-memory one takes one row per operation,
        each on a thread of its own; Mooncake cuts each operation into one-row parts for two
        workers, which run gets two at once and puts one at a time."""
        if self._store is not None:
            driver = MemoryDriver(self._store, capabilities=ONE_ROW_PER_CALL, background=True)
            driver.set_cancel_effective(self._cancel_effective)
        else:
            sdk = open_sdk(MooncakeConfig(self._master, 0, PROTOCOL, DEVICES, local_buffer_size=0))
            driver = MooncakeDriver(sdk, workers=2, rows_per_call=1)
        self._drivers.append(driver)
        return driver

    def commit(self, object_key: np.ndarray, data: bytes) -> None:
        """Commits ``data`` straight into the in-memory store, with the driver's trailer."""
        key = encode_key(KEY_PREFIX, object_key.tobytes())
        trailer = encode_trailer(key.encode(), len(data), checksum=Checksum.CRC32, payload=data)
        assert self._store.commit(key, data, trailer) is Commit.STORED

    def close(self) -> None:
        if all(backend.shutdown(WAIT_S) == frozenset() for backend in self.backends):
            for driver in self._drivers:
                with contextlib.suppress(DriverRejected):  # one a failed build never drained
                    driver.close()


@pytest.fixture(params=["memory", "mooncake"])
def stores(request, tmp_path):
    if request.param == "memory":
        yield Stores()
        return
    if os.environ.get("TRTLLM_MOONCAKE_REAL") != "1":
        pytest.skip("needs a real Mooncake pool; set TRTLLM_MOONCAKE_REAL=1")
    from test_common.mooncake_utils import running_master_on_free_ports, running_segment_provider

    with running_master_on_free_ports(str(tmp_path), protocol=PROTOCOL) as master:
        with running_segment_provider(
            master.address,
            str(tmp_path),
            segment_size=256 << 20,
            protocol=PROTOCOL,
            device_name=DEVICES,
        ):
            opened = Stores(master=master.address)
            yield opened
            opened.close()


def attach(mgr, share_count=None):
    return attach_lender(mgr, scope=SCOPE, staging=StagingOptions(END), share_count=share_count)


def ready_view(lease, mgr, tries: int = 3):
    for _ in range(tries):
        mgr._stream.synchronize()
        view = lease.poll()
        if view is not None:
            return view
    raise AssertionError(f"the lease never became ready: {lease.failure}")


@dataclass(frozen=True)
class HostRegion:
    """A lender region as the contract's ``SlotRegion``. The lender's own regions do not name their
    memory kind, which ``SlotRegion`` requires; the lender stages in host memory."""

    layout_label: str
    address: int
    nbytes: int
    slot_bytes: int
    slot_count: int
    memory_kind: MemoryKind = MemoryKind.HOST


def build(stores: Stores, lender, wake: Wake):
    """A backend over ``lender``'s regions, built on this thread: a driver, the lender's hold, the
    regions registered under it, and the backend. Returns the binding, whose ``release_hold``
    gives the hold back once the backend shut down, before its driver is closed, and the backend."""
    regions = [
        HostRegion(r.layout_label, r.address, r.nbytes, r.slot_bytes, r.slot_count)
        for r in lender.regions
    ]
    binding = RegionBinding(regions, stores.open_driver(), lender.hold_regions())
    binding.register()
    backend = BlobBackend(binding, wake, BlobConfig(key_prefix=KEY_PREFIX))
    stores.backends.append(backend)
    return binding, backend


def publish(stores, kit, mgr, select=None, *, share_count=None, rewrite=None):
    """Publishes the source request's blocks (those ``select`` keeps), named in ``share_count``
    shares, after ``rewrite(mgr, source)`` changed its pages; returns the view lent."""
    source = kit.published(mgr, SOURCE, PROMPT)
    if rewrite is not None:
        rewrite(mgr, source)
    lender, wake = attach(mgr, share_count), Wake()
    binding, backend = build(stores, lender, wake)
    lease = lender.lend_read(source, 0, END)
    view = ready_view(lease, mgr)
    submitted = view if select is None else LeaseView(tuple(g.select(select) for g in view.groups))
    outcome = wake.settled(backend.publish(submitted, None))
    assert outcome.fault is None
    assert all(mask.all() for mask in outcome.delivered_masks)
    lease.release()
    assert backend.shutdown(WAIT_S) == frozenset()
    binding.release_hold()
    binding.driver.close()
    return source, view


def fetch(stores, kit, mgr, *, share_count=None):
    """Fetches every block of the target request, named in ``share_count`` shares; returns the
    target, lender, lease, outcome, backend and binding, the lease not yet marked."""
    target = kit.admitted(mgr, TARGET, PROMPT)
    lender, wake = attach(mgr, share_count), Wake()
    binding, backend = build(stores, lender, wake)
    lease = lender.lend_write(target, 0, END)
    view = lease.poll()
    kit.fill_sentinel(mgr, target)
    outcome = wake.settled(backend.fetch(view, None))
    return target, lender, lease, outcome, backend, binding


def test_publish_then_fetch_restores_pages(kit, real_manager, stores) -> None:
    with real_manager() as mgr_a, real_manager() as mgr_b:
        source, _ = publish(stores, kit, mgr_a)
        target, lender, lease, outcome, backend, binding = fetch(stores, kit, mgr_b)
        assert outcome.fault is None
        lease.mark_arrived([np.array(mask) for mask in outcome.delivered_masks])
        mgr_b._stream.synchronize()
        assert lender.resume_bounds(target) == ResumeBounds(END, 0)
        for ordinal in range(BLOCKS):
            got = kit.digest(kit.page(mgr_b, target, 0, ordinal))
            assert got == kit.digest(kit.page(mgr_a, source, 0, ordinal)), "a fetched page differs"
        lease.release()
        assert backend.shutdown(WAIT_S) == frozenset()
        binding.release_hold()
        binding.driver.close()


def test_a_partial_fetch_resumes_only_up_to_its_first_undelivered_block(kit, real_manager) -> None:
    # Each row's call runs on a thread of its own, in any order. A later row's fault stops the
    # fetch, and the cancel it forwards could end the first row before its thread took it; the
    # driver ignores it, so the first row always lands.
    stores = Stores(cancel_effective=False)
    with real_manager() as mgr_a, real_manager() as mgr_b:
        first_block_only = np.arange(BLOCKS) == 0
        source, view = publish(stores, kit, mgr_a, select=first_block_only)
        for object_key in view.groups[0].object_keys[1:]:  # the later blocks are the wrong size
            stores.commit(object_key, b"short")
        target, lender, lease, outcome, backend, binding = fetch(stores, kit, mgr_b)
        assert outcome.delivered_masks[0].tolist() == [True, False, False]
        assert "object is 5 bytes" in outcome.fault
        lease.mark_arrived([np.array(mask) for mask in outcome.delivered_masks])
        mgr_b._stream.synchronize()
        assert lender.resume_bounds(target) == ResumeBounds(TPB, 0)  # usable up to the first gap
        sentinel = kit.digest(bytes([kit.SENTINEL]) * kit.DevicePages(mgr_b).page_bytes(0))
        pages = [kit.digest(kit.page(mgr_b, target, 0, ordinal)) for ordinal in range(BLOCKS)]
        assert pages == [kit.digest(kit.page(mgr_a, source, 0, 0)), sentinel, sentinel]
        lease.release()
        assert backend.shutdown(WAIT_S) == frozenset()
        binding.release_hold()
        binding.driver.close()


# Tensor-parallel ranks name their heads in SHARES head ranges, the lcm of the sizes below.
HEADS = 4
SHARES = 4


def tp(size: int, rank: int):
    """Rank ``rank`` of ``size`` tensor-parallel ranks."""
    from tensorrt_llm.mapping import Mapping

    return Mapping(world_size=size, tp_size=size, rank=rank)


def head_bytes(ordinal: int, layer: int, role: str, head: int, nbytes: int) -> bytes:
    """One KV head of one buffer of block ``ordinal``: the same bytes on every rank holding it."""
    name = f"{ordinal}/{layer}/{role}/{head}".encode()
    seed = int.from_bytes(hashlib.sha256(name).digest()[:8], "little")
    return np.random.default_rng(seed).integers(0, 256, nbytes, dtype=np.uint8).tobytes()


def rank_page(mgr, ordinal: int) -> bytes:
    """Block ``ordinal``'s page of layer group 0 on ``mgr``'s rank: of every buffer, in pool order,
    the heads that rank holds, head after head."""
    layout = derive_layout(mgr)
    assert layout.num_layer_groups == 1
    g = layout.pool_group_of[0]
    widths = layout.layout.pool_groups[g]
    page = bytearray(int(layout.page_bytes[g]))
    covered = 0
    for b in group_buffers(layout, 0):
        heads, nbytes = b.geometry.num_heads, b.geometry.bytes_per_head
        first = mgr.mapping.tp_rank * heads
        start = sum(widths[: b.pool[1]]) + b.offset
        for j in range(heads):
            page[start + j * nbytes : start + (j + 1) * nbytes] = head_bytes(
                ordinal, b.layer, b.role, first + j, nbytes
            )
        covered += b.size
    assert covered == len(page), "a page holds bytes no buffer covers"
    return bytes(page)


def write_heads(kit, mgr, request) -> None:
    """Overwrite the request's whole blocks with the heads ``mgr``'s rank holds (``rank_page``)."""
    dev = kit.DevicePages(mgr)
    for ordinal, slot in enumerate(kit.pages(kit.kv(mgr, request), 0)[:BLOCKS]):
        dev.write(0, slot, rank_page(mgr, ordinal))


def fetch_rank(stores, kit, mgr):
    """Fetches the target's blocks into ``mgr`` in ``SHARES`` shares and marks what arrived;
    returns the target, the lender, the delivered rows and the pages, then lets everything go."""
    target, lender, lease, outcome, backend, binding = fetch(stores, kit, mgr, share_count=SHARES)
    assert outcome.fault is None
    (rows,) = outcome.delivered_masks
    assert len(rows) == BLOCKS * SHARES // mgr.mapping.tp_size  # a row per share held
    lease.mark_arrived([np.array(mask) for mask in outcome.delivered_masks])
    mgr._stream.synchronize()
    pages = [kit.digest(kit.page(mgr, target, 0, ordinal)) for ordinal in range(BLOCKS)]
    bounds = lender.resume_bounds(target)
    lease.release()
    assert backend.shutdown(WAIT_S) == frozenset()
    binding.release_hold()
    binding.driver.close()
    return bounds, rows.tolist(), pages


@pytest.mark.parametrize("write_tp,read_tp", [(4, 2), (2, 4), (4, 1), (1, 4), (2, 2)])
def test_heads_published_at_one_tensor_parallel_size_are_fetched_at_another(
    kit, real_manager, stores, write_tp, read_tp
) -> None:
    """Every rank of one size publishes its heads named in SHARES head ranges; every rank of
    another fetches the ranges it holds and gets exactly its own heads of every block."""
    with contextlib.ExitStack() as stack:
        for rank in range(write_tp):
            mgr = stack.enter_context(real_manager(num_kv_heads=HEADS, mapping=tp(write_tp, rank)))
            publish(
                stores, kit, mgr, share_count=SHARES, rewrite=lambda m, r: write_heads(kit, m, r)
            )
        for rank in range(read_tp):
            mgr = stack.enter_context(real_manager(num_kv_heads=HEADS, mapping=tp(read_tp, rank)))
            bounds, rows, pages = fetch_rank(stores, kit, mgr)
            assert all(rows), f"read rank {rank} missed rows {rows}"
            assert bounds == ResumeBounds(END, 0)
            expected = [kit.digest(rank_page(mgr, ordinal)) for ordinal in range(BLOCKS)]
            assert pages == expected, f"read rank {rank} of {read_tp} got other heads"


def test_a_block_arrives_only_once_every_share_of_it_did(kit, real_manager, stores) -> None:
    """Rank 3 of four never publishes: the TP2 rank holding its heads gets the other range's rows
    but no block, the other TP2 rank every block."""
    with contextlib.ExitStack() as stack:
        for rank in range(3):
            mgr = stack.enter_context(real_manager(num_kv_heads=HEADS, mapping=tp(4, rank)))
            publish(
                stores, kit, mgr, share_count=SHARES, rewrite=lambda m, r: write_heads(kit, m, r)
            )
        whole = stack.enter_context(real_manager(num_kv_heads=HEADS, mapping=tp(2, 0)))
        bounds, rows, pages = fetch_rank(stores, kit, whole)
        assert all(rows) and bounds == ResumeBounds(END, 0)
        assert pages == [kit.digest(rank_page(whole, ordinal)) for ordinal in range(BLOCKS)]
        torn = stack.enter_context(real_manager(num_kv_heads=HEADS, mapping=tp(2, 1)))
        bounds, rows, pages = fetch_rank(stores, kit, torn)
        assert rows == [True, False] * BLOCKS  # shares 2 and 3 of each block, block by block
        assert bounds.max_position == 0
        sentinel = kit.digest(bytes([kit.SENTINEL]) * kit.DevicePages(torn).page_bytes(0))
        assert pages == [sentinel] * BLOCKS
