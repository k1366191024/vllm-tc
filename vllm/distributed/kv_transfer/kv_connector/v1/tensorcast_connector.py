# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar

import torch

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

from .base import KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole

if TYPE_CHECKING:
    from tensorcast.api.store import Store
    from tensorcast.api.store.types import FallbackOptions

    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

EngineId = str
ReqId = str
RetryResult = TypeVar("RetryResult")


@dataclass
class RecvReqMeta:
    local_block_ids: list[int]
    remote_engine_id: str
    remote_request_id: str
    remote_block_ids: list[int]
    tensorcast_key_prefix: str | None = None


class TensorcastConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.reqs_to_recv: dict[ReqId, RecvReqMeta] = {}
        self.reqs_to_send: dict[ReqId, list[int]] = {}

    def add_new_req(
        self,
        request_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
        *,
        load_remote_cache: bool = True,
    ) -> None:
        if load_remote_cache:
            self.reqs_to_recv[request_id] = RecvReqMeta(
                local_block_ids=local_block_ids,
                remote_engine_id=kv_transfer_params["remote_engine_id"],
                remote_request_id=kv_transfer_params["remote_request_id"],
                remote_block_ids=kv_transfer_params["remote_block_ids"],
                tensorcast_key_prefix=kv_transfer_params.get("tensorcast_key_prefix"),
            )
        else:
            self.reqs_to_send[request_id] = local_block_ids


class TensorcastConnector(KVConnectorBase_V1):
    """KV connector backed by TensorCast Store.

    Integration shape intentionally mirrors MooncakeConnector: a thin shim that
    delegates scheduler vs. worker responsibilities to separate objects.

    This connector uses TensorCast materialization with
    `FallbackOptions(prefer="p2p")` so that GPU P2P is preferred when available.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: TensorcastConnectorScheduler | None = (
                TensorcastConnectorScheduler(vllm_config, self.engine_id)
            )
            self.connector_worker: TensorcastConnectorWorker | None = None
        else:
            self.connector_scheduler = None
            self.connector_worker = TensorcastConnectorWorker(vllm_config, self.engine_id)

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    # ==============================
    # Worker-side methods
    # ==============================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        _ = forward_context
        _ = kwargs
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, TensorcastConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        _ = layer_name
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        _ = layer_name
        _ = kv_layer
        _ = attn_metadata
        _ = kwargs
        return

    def wait_for_save(self):
        return

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        assert self.connector_worker is not None
        return self.connector_worker.get_finished(finished_req_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()


class TensorcastConnectorScheduler:
    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self.vllm_config = vllm_config
        self.engine_id: EngineId = engine_id
        assert vllm_config.kv_transfer_config is not None
        self.kv_role = vllm_config.kv_transfer_config.kv_role

        self._reqs_need_recv: dict[ReqId, tuple[Request, list[int]]] = {}
        self._reqs_need_send: dict[ReqId, list[int]] = {}

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        params = request.kv_transfer_params
        if params is not None and params.get("do_remote_prefill"):
            token_ids = request.prompt_token_ids or []
            count = len(token_ids) - num_computed_tokens
            if count > 0:
                return count, True
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        params = request.kv_transfer_params
        if not params:
            return

        if params.get("do_remote_prefill"):
            assert self.kv_role != "kv_producer"
            required = ("remote_engine_id", "remote_request_id", "remote_block_ids")
            if all(k in params for k in required):
                local_block_ids = (
                    blocks.get_unhashed_block_ids() if num_external_tokens > 0 else []
                )
                self._reqs_need_recv[request.request_id] = (request, local_block_ids)
            else:
                logger.warning(
                    "Got invalid KVTransferParams: %s. This request will not utilize KVTransfer",
                    params,
                )
            params["do_remote_prefill"] = False

        elif params.get("do_remote_decode"):
            # No-op placeholder to keep parity with other connectors.
            self._reqs_need_send.setdefault(request.request_id, [])

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        _ = scheduler_output
        meta = TensorcastConnectorMetadata()

        if self.kv_role != "kv_producer":
            for req_id, (req, block_ids) in self._reqs_need_recv.items():
                assert req.kv_transfer_params is not None
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                    load_remote_cache=True,
                )
            self._reqs_need_recv.clear()

        if self.kv_role != "kv_consumer":
            for req_id, block_ids in self._reqs_need_send.items():
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params={},
                    load_remote_cache=False,
                )
            self._reqs_need_send.clear()

        return meta

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        params = request.kv_transfer_params
        if not params:
            return False, None

        if params.get("do_remote_prefill"):
            # Request aborted before update_state_after_alloc. Trigger a best-effort
            # recv path on worker so it can clean up any remote state if needed.
            assert self.kv_role != "kv_producer"
            self._reqs_need_recv[request.request_id] = (request, [])
            params["do_remote_prefill"] = False
            return False, None

        if (
            not params.get("do_remote_decode")
            or request.status != RequestStatus.FINISHED_LENGTH_CAPPED
        ):
            return False, None

        assert self.kv_role != "kv_consumer"
        delay_free_blocks = len(block_ids) > 0
        if delay_free_blocks:
            self._reqs_need_send[request.request_id] = block_ids

        extra = self.vllm_config.kv_transfer_config.kv_connector_extra_config
        key_prefix = extra.get("tensorcast_key_prefix", "vllm/kv")
        return delay_free_blocks, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_block_ids=block_ids,
            remote_engine_id=self.engine_id,
            remote_request_id=request.request_id,
            # TensorCast-specific hints (optional, ignored by other connectors)
            tensorcast_key_prefix=key_prefix,
            tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
        )


@dataclass
class _SendState:
    key: str
    artifact_id: str
    expire_at_s: float


class TensorcastConnectorWorker:
    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self.vllm_config = vllm_config
        self.engine_id: EngineId = engine_id
        assert vllm_config.kv_transfer_config is not None
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config

        self._key_prefix: str = extra.get("tensorcast_key_prefix", "vllm/kv")
        self._ttl_ms: int = int(extra.get("tensorcast_ttl_ms", 60_000))
        self._prefer: str = str(extra.get("tensorcast_fallback_prefer", "p2p"))
        self._put_policy: str = str(extra.get("tensorcast_put_policy", "pinned"))
        self._tensorcast_address: str | None = extra.get("tensorcast_address")
        self._key_visibility_timeout_s = (
            max(0, int(extra.get("tensorcast_key_visibility_timeout_ms", 5_000)))
            / 1000.0
        )
        self._key_visibility_retry_interval_s = (
            max(1, int(extra.get("tensorcast_key_visibility_retry_interval_ms", 100)))
            / 1000.0
        )

        self._tp_rank = get_tensor_model_parallel_rank()
        self._kv_caches: dict[str, torch.Tensor] = {}

        self._finished_sending: set[str] = set()
        self._finished_recving: set[str] = set()
        self._invalid_block_ids: set[int] = set()
        self._send_states: dict[str, _SendState] = {}

        self._store, self._fallback = self._init_tensorcast()

    def _init_tensorcast(self) -> tuple["Store", "FallbackOptions"]:
        try:
            import tensorcast
            from tensorcast.api.store.types import FallbackOptions
        except ImportError as e:
            raise ImportError(
                "TensorcastConnector requires the 'tensorcast' package to be installed "
                "and importable in the vLLM worker environment."
            ) from e

        # Connect to an existing daemon session. If tensorcast_address is None,
        # TensorCast will auto-discover a local daemon session.
        tensorcast.init(mode="connect", address=self._tensorcast_address)
        store = tensorcast.store()
        fallback = FallbackOptions(prefer=self._prefer, allow_p2p=True)
        return store, fallback

    def _make_key(self, key_prefix: str, engine_id: str, request_id: str) -> str:
        return f"{key_prefix}/{engine_id}/{request_id}/tp{self._tp_rank}"

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._kv_caches = kv_caches

    def get_block_ids_with_load_errors(self) -> set[int]:
        return set(self._invalid_block_ids)

    def _is_retryable_tensorcast_error(self, exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, str) and status_code.upper() in {
            "NOT_FOUND",
            "FAILED_PRECONDITION",
            "UNAVAILABLE",
            "DEADLINE_EXCEEDED",
        }:
            return True

        with contextlib.suppress(Exception):
            import grpc

            if isinstance(exc, grpc.RpcError):
                grpc_code = exc.code()
                if grpc_code in {
                    grpc.StatusCode.NOT_FOUND,
                    grpc.StatusCode.FAILED_PRECONDITION,
                    grpc.StatusCode.UNAVAILABLE,
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                }:
                    return True

        lowered = str(exc).lower()
        return any(
            message in lowered
            for message in (
                "key not found",
                "artifact row missing",
                "artifact/index not ready",
                "failed_precondition",
            )
        )

    def _retry_transient_tensorcast_op(
        self,
        *,
        req_id: str,
        key: str,
        operation_name: str,
        fn: Callable[[], RetryResult],
    ) -> RetryResult:
        deadline = time.perf_counter() + self._key_visibility_timeout_s
        attempt = 0

        while True:
            try:
                return fn()
            except Exception as exc:
                if (
                    self._key_visibility_timeout_s <= 0
                    or not self._is_retryable_tensorcast_error(exc)
                ):
                    raise

                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise

                attempt += 1
                if attempt == 1:
                    logger.warning(
                        "TensorcastConnector retrying %s for req_id=%s key=%s: %s",
                        operation_name,
                        req_id,
                        key,
                        exc,
                    )
                time.sleep(min(self._key_visibility_retry_interval_s, remaining))

    def _get_tensorcast_client(self) -> Any | None:
        runtime = getattr(self._store, "_runtime", None)
        if runtime is None:
            return None
        ensure_client = getattr(runtime, "ensure_client", None)
        if not callable(ensure_client):
            return None
        return ensure_client()

    def _ensure_key_mapping_ready(
        self,
        *,
        req_id: str,
        key: str,
        artifact_id: str,
        descriptor: Any | None,
    ) -> None:
        client = self._get_tensorcast_client()

        def _resolve_key() -> None:
            resolved_artifact_id = self._store.artifact(
                key=key,
                fallback=self._fallback,
            ).artifact_id
            if resolved_artifact_id != artifact_id:
                raise RuntimeError(
                    "TensorcastConnector key resolved to unexpected artifact: "
                    f"key={key}, expected={artifact_id}, actual={resolved_artifact_id}"
                )

        def _publish_and_resolve() -> None:
            if client is not None and descriptor is not None:
                client.publish_replica_key(key=key, descriptor=descriptor)
            _resolve_key()

        self._retry_transient_tensorcast_op(
            req_id=req_id,
            key=key,
            operation_name="key publish/resolve",
            fn=_publish_and_resolve,
        )

    def start_load_kv(self, metadata: TensorcastConnectorMetadata) -> None:
        if self.kv_role != "kv_producer":
            for req_id, meta in metadata.reqs_to_recv.items():
                self._recv_one(req_id, meta)

        if self.kv_role != "kv_consumer":
            for req_id, block_ids in metadata.reqs_to_send.items():
                self._send_one(req_id, block_ids)

        self._gc_send_states()

    def _recv_one(self, req_id: str, meta: RecvReqMeta) -> None:
        if not meta.local_block_ids:
            # Full prefix-cache hit: nothing to materialize into local blocks.
            self._finished_recving.add(req_id)
            return

        if not self._kv_caches:
            raise RuntimeError("TensorcastConnectorWorker.register_kv_caches not called")

        num_blocks = min(len(meta.local_block_ids), len(meta.remote_block_ids))
        if num_blocks <= 0:
            self._finished_recving.add(req_id)
            return

        key_prefix = meta.tensorcast_key_prefix or self._key_prefix
        key = self._make_key(key_prefix, meta.remote_engine_id, meta.remote_request_id)
        try:
            def _materialize_remote_kv() -> dict[str, torch.Tensor]:
                artifact = self._store.artifact(key=key, fallback=self._fallback)

                total_remote_blocks = len(meta.remote_block_ids)
                start = max(0, total_remote_blocks - num_blocks)
                stop = start + num_blocks
                slices = {
                    layer_name: [(1, slice(start, stop))]
                    for layer_name in self._kv_caches
                }
                narrowed_artifact = artifact.view(slices=slices)

                packed_tensors: dict[str, torch.Tensor] = {}
                for layer_name, kv_cache in self._kv_caches.items():
                    if not kv_cache.is_cuda:
                        raise RuntimeError(
                            "TensorcastConnector currently requires CUDA KV caches for P2P"
                        )
                    packed_tensors[layer_name] = torch.empty(
                        (kv_cache.shape[0], num_blocks, *kv_cache.shape[2:]),
                        device=kv_cache.device,
                        dtype=kv_cache.dtype,
                    )

                narrowed_artifact.tensor_dict_into(packed_tensors)
                return packed_tensors

            packed = self._retry_transient_tensorcast_op(
                req_id=req_id,
                key=key,
                operation_name="remote KV materialization",
                fn=_materialize_remote_kv,
            )

            from vllm.platforms import current_platform

            src_idx = torch.arange(num_blocks, device=next(iter(packed.values())).device)
            dst_idx = torch.tensor(meta.local_block_ids[:num_blocks], device=src_idx.device)
            for layer_name, kv_cache in self._kv_caches.items():
                current_platform.insert_blocks_to_device(
                    packed[layer_name], kv_cache, src_idx, dst_idx
                )

        except Exception as e:
            logger.exception(
                "TensorcastConnector failed to materialize KV for req_id=%s key=%s: %s",
                req_id,
                key,
                e,
            )
            self._invalid_block_ids.update(meta.local_block_ids)
        finally:
            self._finished_recving.add(req_id)

    def _send_one(self, req_id: str, block_ids: list[int]) -> None:
        # update_state_after_alloc() may emit an early placeholder notification
        # with empty block_ids before request_finished() provides the actual
        # finalized blocks to publish. This should not be reported as a
        # completed send to the scheduler.
        if not block_ids:
            return

        if not self._kv_caches:
            raise RuntimeError("TensorcastConnectorWorker.register_kv_caches not called")

        try:
            tensors: dict[str, torch.Tensor] = {}
            for layer_name, kv_cache in self._kv_caches.items():
                if not kv_cache.is_cuda:
                    raise RuntimeError(
                        "TensorcastConnector currently requires CUDA KV caches for P2P"
                    )
                # Pack requested blocks into a compact contiguous tensor so that:
                # 1) we can safely free vLLM blocks after registration, and
                # 2) TensorCast can LIP-register the compact buffer.
                tensors[layer_name] = kv_cache[:, block_ids].contiguous()

            key = self._make_key(self._key_prefix, self.engine_id, req_id)
            registered = self._store.put(
                tensors,
                key=key,
                policy=self._put_policy,
            )
            descriptor = getattr(
                getattr(registered, "registration_result", None),
                "descriptor",
                None,
            )
            self._ensure_key_mapping_ready(
                req_id=req_id,
                key=key,
                artifact_id=registered.artifact_id,
                descriptor=descriptor,
            )

            expire_at = time.perf_counter() + (self._ttl_ms / 1000.0)
            self._send_states[req_id] = _SendState(
                key=key,
                artifact_id=registered.artifact_id,
                expire_at_s=expire_at,
            )
        except Exception:
            logger.exception("TensorcastConnector failed to register KV for req_id=%s", req_id)
        finally:
            # Signal to scheduler that it can free the original request blocks.
            self._finished_sending.add(req_id)

    def _gc_send_states(self) -> None:
        now = time.perf_counter()
        expired = [req_id for req_id, st in self._send_states.items() if st.expire_at_s <= now]
        for req_id in expired:
            st = self._send_states.pop(req_id)
            try:
                self._store.deregister_artifact(
                    st.artifact_id,
                    wait=False,
                )
            except Exception:
                logger.warning(
                    "TensorcastConnector failed to deregister KV artifact for req_id=%s key=%s artifact_id=%s",
                    req_id,
                    st.key,
                    st.artifact_id,
                    exc_info=True,
                )

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        # Conform to base contract: only return finished_sending IDs that were
        # reported as finished to us in current or previous calls.
        if finished_req_ids:
            self._finished_sending.intersection_update(finished_req_ids)

        finished_sending = self._finished_sending or None
        finished_recving = self._finished_recving or None

        self._finished_sending = set()
        self._finished_recving = set()
        return finished_sending, finished_recving
