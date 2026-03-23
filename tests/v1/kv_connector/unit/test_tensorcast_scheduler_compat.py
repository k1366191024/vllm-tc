# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from types import SimpleNamespace

import pytest
import torch
from transformers import OPTConfig

from vllm.distributed.kv_transfer.kv_connector.v1.tensorcast_connector import (
    TensorcastConnectorWorker,
)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, KVConnectorOutput
from vllm.v1.request import RequestStatus

from .utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

pytestmark = pytest.mark.cpu_test


def _make_local_opt_model(tmp_path) -> str:
    model_dir = tmp_path / "opt-tiny"
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg = OPTConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        ffn_dim=256,
        max_position_embeddings=2048,
    )
    # vLLM checks model architectures during config validation.
    cfg.architectures = ["OPTForCausalLM"]
    cfg.save_pretrained(model_dir)
    return str(model_dir)


def test_remote_decode_kv_transfer_params_schema(tmp_path):
    """TensorcastConnector should emit NIXL-compatible kv_transfer_params."""

    vllm_config = create_vllm_config(
        model=_make_local_opt_model(tmp_path),
        kv_connector="TensorcastConnector",
    )
    scheduler = create_scheduler(vllm_config)

    request = create_request(request_id=1, do_remote_decode=True)
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(reqs=[request])
    engine_core_outputs = scheduler.update_from_output(scheduler_output, model_runner_output)

    output = engine_core_outputs[0].outputs[0]
    params = output.kv_transfer_params
    assert params is not None
    assert params["do_remote_prefill"] is True
    assert params["do_remote_decode"] is False
    assert "remote_block_ids" in params
    assert "remote_engine_id" in params
    assert "remote_request_id" in params


def test_remote_prefill_async_wait_state_machine(tmp_path):
    """Basic remote prefill lifecycle should work with TensorcastConnector."""

    vllm_config = create_vllm_config(
        model=_make_local_opt_model(tmp_path),
        kv_connector="TensorcastConnector",
    )
    scheduler = create_scheduler(vllm_config)

    request = create_request(request_id=1, do_remote_prefill=True)
    scheduler.add_request(request)
    request_id = request.request_id

    # Step 1: remote KV not arrived -> WAITING_FOR_REMOTE_KVS
    scheduler_output = scheduler.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.update_from_output(scheduler_output, EMPTY_MODEL_RUNNER_OUTPUT)

    # Step 2: remote KV arrived -> move back to WAITING
    scheduler_output = scheduler.schedule()
    model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    model_runner_output.kv_connector_output = KVConnectorOutput(finished_recving={request_id})
    scheduler.update_from_output(scheduler_output, model_runner_output)

    # Step 3: scheduler can now run it
    scheduler_output = scheduler.schedule()
    assert len(scheduler_output.scheduled_new_reqs) == 1


def test_remote_decode_unknown_finished_send_is_ignored(tmp_path):
    vllm_config = create_vllm_config(
        model=_make_local_opt_model(tmp_path),
        kv_connector="TensorcastConnector",
    )
    scheduler = create_scheduler(vllm_config)
    scheduler_output = scheduler.schedule()

    model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    model_runner_output.kv_connector_output = KVConnectorOutput(
        finished_sending={"missing-request-id"}
    )

    scheduler.update_from_output(scheduler_output, model_runner_output)


def test_tensorcast_worker_retries_key_publish_until_visible(tmp_path, monkeypatch):
    class FakeKVCache:
        is_cuda = True
        device = torch.device("cuda:0")
        dtype = torch.float16
        shape = (2, 4, 1, 1)

        def __getitem__(self, item):
            _ = item
            return self

        def contiguous(self):
            return self

    class FakeClient:
        def __init__(self):
            self.publish_calls = 0

        def publish_replica_key(self, *, key, descriptor):
            _ = key
            _ = descriptor
            self.publish_calls += 1
            return True

    class FakeStore:
        def __init__(self):
            self._runtime = SimpleNamespace(ensure_client=lambda: client)
            self.resolve_calls = 0

        def put(self, tensors, *, key, policy):
            _ = tensors
            _ = key
            _ = policy
            return SimpleNamespace(
                artifact_id="artifact-1",
                registration_result=SimpleNamespace(descriptor=object()),
            )

        def deregister_artifact(self, artifact_id, *, wait):
            _ = artifact_id
            _ = wait

        def artifact(self, *, key, fallback):
            _ = key
            _ = fallback
            store = self

            class _Artifact:
                @property
                def artifact_id(self):
                    store.resolve_calls += 1
                    if store.resolve_calls < 3:
                        raise RuntimeError("key not found")
                    return "artifact-1"

            return _Artifact()

    client = FakeClient()
    store = FakeStore()

    monkeypatch.setattr(
        TensorcastConnectorWorker,
        "_init_tensorcast",
        lambda self: (store, object()),
    )
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.tensorcast_connector.get_tensor_model_parallel_rank",
        lambda: 0,
    )

    vllm_config = create_vllm_config(
        model=_make_local_opt_model(tmp_path),
        kv_connector="TensorcastConnector",
        kv_connector_extra_config={
            "tensorcast_key_visibility_timeout_ms": 500,
            "tensorcast_key_visibility_retry_interval_ms": 1,
        },
    )
    worker = TensorcastConnectorWorker(vllm_config, engine_id="engine-1")
    worker.register_kv_caches({"layer": FakeKVCache()})

    worker._send_one("req-1", [0, 1])

    finished_sending, _ = worker.get_finished({"req-1"})
    assert finished_sending == {"req-1"}
    assert client.publish_calls >= 2
    assert store.resolve_calls >= 3


def test_tensorcast_worker_ignores_empty_send_placeholder(tmp_path, monkeypatch):
    class FakeStore:
        def __init__(self):
            self._runtime = SimpleNamespace(ensure_client=lambda: None)

    monkeypatch.setattr(
        TensorcastConnectorWorker,
        "_init_tensorcast",
        lambda self: (FakeStore(), object()),
    )
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.tensorcast_connector.get_tensor_model_parallel_rank",
        lambda: 0,
    )

    vllm_config = create_vllm_config(
        model=_make_local_opt_model(tmp_path),
        kv_connector="TensorcastConnector",
    )
    worker = TensorcastConnectorWorker(vllm_config, engine_id="engine-1")

    worker._send_one("req-placeholder", [])

    finished_sending, finished_recving = worker.get_finished({"req-placeholder"})
    assert finished_sending is None
    assert finished_recving is None
