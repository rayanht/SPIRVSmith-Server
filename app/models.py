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


class BoundedInt(BaseModel):
    min: int
    max: int


class MutationsConfig(BaseModelWithConfig):
    w_memory_operation: BoundedInt
    w_logical_operation: BoundedInt
    w_arithmetic_operation: BoundedInt
    w_control_flow_operation: BoundedInt
    w_function_operation: BoundedInt
    w_bitwise_operation: BoundedInt
    w_conversion_operation: BoundedInt
    w_composite_operation: BoundedInt

    w_scalar_type: BoundedInt
    w_container_type: BoundedInt

    w_composite_constant: BoundedInt
    w_scalar_constant: BoundedInt


class FuzzingStrategy(BaseModelWithConfig):
    mutations_config: MutationsConfig

    enable_ext_glsl_std_450: bool

    w_memory_operation: int
    w_logical_operation: int
    w_arithmetic_operation: int
    w_control_flow_operation: int
    w_function_operation: int
    w_bitwise_operation: int
    w_conversion_operation: int
    w_composite_operation: int

    w_scalar_type: int
    w_container_type: int

    w_composite_constant: int
    w_scalar_constant: int

    p_statement: float
    p_picking_statement_operand: float

    mutation_rate: float


class GeneratorInfo(BaseModelWithConfig):
    id: GeneratorID
    fuzzer_version: str
    strategy: Optional[FuzzingStrategy] = None


class ShaderData(BaseModelWithConfig):
    shader_id: ShaderID
    shader_assembly: str
    generator_info: GeneratorInfo


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
