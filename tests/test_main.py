from collections import deque
import json
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app, get_settings
from app.models import (
    ExecutionQueue,
    ExecutionQueues,
    QueueID,
    RetrievedShader,
    ShaderID,
)
from vulkan_platform_py import *
from pydantic.json import pydantic_encoder
from dummy_data import *


def test_execution_queue_created():
    with TestClient(app) as client:
        execution_queues: ExecutionQueues = ExecutionQueues(
            **client.get("/queues").json()
        )
        assert execution_queues.queues == {}
        queue_id: QueueID = client.put(
            "/queues", data=json.dumps(dummy_executor1, default=pydantic_encoder)
        ).json()
        execution_queues: ExecutionQueues = ExecutionQueues(
            **client.get("/queues").json()
        )
        assert queue_id in execution_queues.queues


def test_multiple_execution_queues_created():
    with TestClient(app) as client:
        queue1_id: QueueID = client.put(
            "/queues", data=json.dumps(dummy_executor1, default=pydantic_encoder)
        ).json()
        queue2_id: QueueID = client.put(
            "/queues", data=json.dumps(dummy_executor2, default=pydantic_encoder)
        ).json()
        execution_queues: ExecutionQueues = ExecutionQueues(
            **client.get("/queues").json()
        )
        assert queue1_id in execution_queues.queues
        assert queue2_id in execution_queues.queues


def test_execution_queue_creation_idempotence():
    with TestClient(app) as client:
        queue1_id: QueueID = client.put(
            "/queues", data=json.dumps(dummy_executor1, default=pydantic_encoder)
        ).json()
        queue2_id: QueueID = client.put(
            "/queues", data=json.dumps(dummy_executor1, default=pydantic_encoder)
        ).json()
        assert queue1_id == queue2_id


def test_single_shader_submitted_to_single_execution_queue():
    with TestClient(app) as client:
        queue_id: QueueID = client.put(
            "/queues", data=json.dumps(dummy_executor1, default=pydantic_encoder)
        ).json()
        shader_id: ShaderID = client.post(
            "/shaders",
            data=json.dumps(dummy_shader_submission1, default=pydantic_encoder),
        ).json()
        execution_queue: ExecutionQueue = ExecutionQueue(
            **client.get(f"/queues/{queue_id}").json()
        )
        assert shader_id in set(map(lambda s: s.shader_id, execution_queue.queue))


def test_single_shader_submitted_to_multiple_execution_queues():
    with TestClient(app) as client:
        queue1_id: QueueID = client.put(
            "/queues", data=json.dumps(dummy_executor1, default=pydantic_encoder)
        ).json()
        queue2_id: QueueID = client.put(
            "/queues", data=json.dumps(dummy_executor1, default=pydantic_encoder)
        ).json()
        shader_id: ShaderID = client.post(
            "/shaders",
            data=json.dumps(dummy_shader_submission1, default=pydantic_encoder),
        ).json()
        execution_queue1: ExecutionQueue = ExecutionQueue(
            **client.get(f"/queues/{queue1_id}").json()
        )
        execution_queue2: ExecutionQueue = ExecutionQueue(
            **client.get(f"/queues/{queue2_id}").json()
        )
        assert shader_id in set(map(lambda s: s.shader_id, execution_queue1.queue))
        assert shader_id in set(map(lambda s: s.shader_id, execution_queue2.queue))


def test_multiple_shaders_submitted_to_single_execution_queue():
    with TestClient(app) as client:
        queue_id: QueueID = client.put(
            "/queues", data=json.dumps(dummy_executor1, default=pydantic_encoder)
        ).json()
        shader_id1: ShaderID = client.post(
            "/shaders",
            data=json.dumps(dummy_shader_submission1, default=pydantic_encoder),
        ).json()
        shader_id2: ShaderID = client.post(
            "/shaders",
            data=json.dumps(dummy_shader_submission2, default=pydantic_encoder),
        ).json()
        execution_queue: ExecutionQueue = ExecutionQueue(
            **client.get(f"/queues/{queue_id}").json()
        )
        shaders_ids_in_queue: set[ShaderID] = set(
            map(lambda s: s.shader_id, execution_queue.queue)
        )
        assert shader_id1 in shaders_ids_in_queue
        assert shader_id2 in shaders_ids_in_queue


def test_multiple_shaders_submitted_to_multiple_execution_queues():
    with TestClient(app) as client:
        queue1_id: QueueID = client.put(
            "/queues", data=json.dumps(dummy_executor1, default=pydantic_encoder)
        ).json()
        queue2_id: QueueID = client.put(
            "/queues", data=json.dumps(dummy_executor1, default=pydantic_encoder)
        ).json()
        shader_id1: ShaderID = client.post(
            "/shaders",
            data=json.dumps(dummy_shader_submission1, default=pydantic_encoder),
        ).json()
        shader_id2: ShaderID = client.post(
            "/shaders",
            data=json.dumps(dummy_shader_submission2, default=pydantic_encoder),
        ).json()
        execution_queue1: ExecutionQueue = ExecutionQueue(
            **client.get(f"/queues/{queue1_id}").json()
        )
        execution_queue2: ExecutionQueue = ExecutionQueue(
            **client.get(f"/queues/{queue2_id}").json()
        )
        shaders_ids_in_queue1: set[ShaderID] = set(
            map(lambda s: s.shader_id, execution_queue1.queue)
        )
        shaders_ids_in_queue2: set[ShaderID] = set(
            map(lambda s: s.shader_id, execution_queue2.queue)
        )
        assert shader_id1 in shaders_ids_in_queue1
        assert shader_id2 in shaders_ids_in_queue1
        assert shader_id1 in shaders_ids_in_queue2
        assert shader_id2 in shaders_ids_in_queue2


def test_new_queue_is_populated_from_buffer_queue():
    with TestClient(app) as client:
        client.put(
            "/queues", data=json.dumps(dummy_executor1, default=pydantic_encoder)
        ).json()
        shader_id1: ShaderID = client.post(
            "/shaders",
            data=json.dumps(dummy_shader_submission1, default=pydantic_encoder),
        ).json()
        shader_id2: ShaderID = client.post(
            "/shaders",
            data=json.dumps(dummy_shader_submission2, default=pydantic_encoder),
        ).json()
        queue_id: QueueID = client.put(
            "/queues", data=json.dumps(dummy_executor1, default=pydantic_encoder)
        ).json()
        execution_queue: ExecutionQueue = ExecutionQueue(
            **client.get(f"/queues/{queue_id}").json()
        )
        shaders_ids_in_queue: set[ShaderID] = set(
            map(lambda s: s.shader_id, execution_queue.queue)
        )
        assert shader_id1 in shaders_ids_in_queue
        assert shader_id2 in shaders_ids_in_queue


def test_shader_popped_from_queue():
    with TestClient(app) as client:
        queue_id: QueueID = client.put(
            "/queues", data=json.dumps(dummy_executor1, default=pydantic_encoder)
        ).json()
        shader_id: ShaderID = client.post(
            "/shaders",
            data=json.dumps(dummy_shader_submission1, default=pydantic_encoder),
        ).json()
        retrieved_shader: RetrievedShader = RetrievedShader(
            **client.post(
                "/shaders/next",
                data=json.dumps(dummy_executor1, default=pydantic_encoder),
            ).json()
        )
        execution_queue: ExecutionQueue = ExecutionQueue(
            **client.get(f"/queues/{queue_id}").json()
        )
        assert retrieved_shader.shader_id == shader_id
        assert shader_id not in map(lambda s: s.shader_id, execution_queue.queue)


def test_priority_shader_gets_popped_first():
    with TestClient(app) as client:
        client.put(
            "/queues", data=json.dumps(dummy_executor1, default=pydantic_encoder)
        ).json()
        shader_id_unprioritized: ShaderID = client.post(
            "/shaders",
            data=json.dumps(dummy_shader_submission1, default=pydantic_encoder),
        ).json()
        shader_id_prioritized: ShaderID = client.post(
            "/shaders",
            data=json.dumps(
                dummy_shader_submission_prioritized, default=pydantic_encoder
            ),
        ).json()
        retrieved_shader: RetrievedShader = RetrievedShader(
            **client.post(
                "/shaders/next",
                data=json.dumps(dummy_executor1, default=pydantic_encoder),
            ).json()
        )
        assert retrieved_shader.shader_id == shader_id_prioritized
        retrieved_shader: RetrievedShader = RetrievedShader(
            **client.post(
                "/shaders/next",
                data=json.dumps(dummy_executor1, default=pydantic_encoder),
            ).json()
        )
        assert retrieved_shader.shader_id == shader_id_unprioritized


def test_shaders_submitted_before_executors_are_buffered():
    with TestClient(app) as client:
        shader_id1: ShaderID = client.post(
            "/shaders",
            data=json.dumps(dummy_shader_submission1, default=pydantic_encoder),
        ).json()
        shader_id2: ShaderID = client.post(
            "/shaders",
            data=json.dumps(dummy_shader_submission2, default=pydantic_encoder),
        ).json()
        client.put(
            "/queues", data=json.dumps(dummy_executor1, default=pydantic_encoder)
        ).json()
        queue_id: QueueID = client.put(
            "/queues", data=json.dumps(dummy_executor1, default=pydantic_encoder)
        ).json()
        execution_queue: ExecutionQueue = ExecutionQueue(
            **client.get(f"/queues/{queue_id}").json()
        )
        shaders_ids_in_queue: set[ShaderID] = set(
            map(lambda s: s.shader_id, execution_queue.queue)
        )
        assert shader_id1 in shaders_ids_in_queue
        assert shader_id2 in shaders_ids_in_queue
