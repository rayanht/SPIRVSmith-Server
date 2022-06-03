import copy
from collections import deque
from typing import Optional, TypeAlias
from uuid import uuid4

from fastapi import HTTPException
from pydantic import BaseModel, Extra, Field, PrivateAttr
from vulkan_platform_py import ExecutionPlatform

GeneratorID: TypeAlias = str
ShaderID: TypeAlias = str
QueueID: TypeAlias = str


class BaseModelWithConfig(BaseModel):
    class Config:
        orm_mode: bool = True
        extra: Extra = Extra.ignore


class MetadataTag(BaseModel):
    name: str
    description: str


class ShaderData(BaseModelWithConfig):
    shader_id: ShaderID
    shader_assembly: str
    generator_version: str


class ShaderSubmission(ShaderData):
    prioritize: bool
    n_buffers: int


class ExecutionQueue(BaseModelWithConfig):
    queue: deque[ShaderSubmission]


class BufferSubmission(BaseModelWithConfig):
    executor: ExecutionPlatform
    buffer_dump: str


class ExecutionQueues(BaseModelWithConfig):
    queues: dict[QueueID, ExecutionQueue] = Field(default_factory=dict)
    _executor_lookup: dict[ExecutionPlatform, QueueID] = PrivateAttr(
        default_factory=dict
    )
    _buffer_queue: ExecutionQueue = PrivateAttr(
        default_factory=lambda: ExecutionQueue(queue=deque())
    )
    _mismatch_queue: ExecutionQueue = PrivateAttr(
        default_factory=lambda: ExecutionQueue(queue=deque())
    )

    def new_execution_queue(self, executor: ExecutionPlatform) -> QueueID:
        if executor in self._executor_lookup:
            return self._executor_lookup[executor]
        queue_initializer: ExecutionQueue = ExecutionQueue(queue=deque())
        try:
            queue_initializer.queue = copy.deepcopy(self._buffer_queue.queue)
        except ValueError:
            pass
        self._executor_lookup[executor] = str(uuid4())
        self.queues[self._executor_lookup[executor]] = queue_initializer
        return self._executor_lookup[executor]

    def get_execution_queue(
        self, executor: ExecutionPlatform
    ) -> deque[ShaderSubmission]:
        queue_id: QueueID = self._executor_lookup[executor]
        return self.queues[queue_id].queue

    def enqueue(self, shader: ShaderSubmission) -> None:
        self._buffer_queue.queue.append(shader)
        for execution_queue in self.queues.values():
            if shader.prioritize:
                execution_queue.queue.appendleft(shader)
            else:
                execution_queue.queue.append(shader)

    def get_next_shader(self, executor: ExecutionPlatform) -> ShaderSubmission:
        if executor not in self._executor_lookup:
            self.new_execution_queue(executor)
        queue_id: QueueID = self._executor_lookup[executor]
        try:
            return self.queues[queue_id].queue.popleft()
        except IndexError:
            raise HTTPException(status_code=404)

    def flush(self) -> None:
        for execution_queue in self.queues.values():
            execution_queue.queue.clear()
