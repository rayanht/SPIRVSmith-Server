import copy
from typing import Optional, TypeAlias
from uuid import uuid4
from pydantic import BaseModel, PrivateAttr
from collections import deque
from vulkan_platform_py import ExecutionPlatform


GeneratorID: TypeAlias = str
ShaderID: TypeAlias = str
QueueID: TypeAlias = str


class BaseModelORM(BaseModel):
    class Config:
        orm_mode: bool = True


class MutationsConfig(BaseModelORM):
    w_memory_operation: tuple[int, int]
    w_logical_operation: tuple[int, int]
    w_arithmetic_operation: tuple[int, int]
    w_control_flow_operation: tuple[int, int]
    w_function_operation: tuple[int, int]
    w_bitwise_operation: tuple[int, int]
    w_conversion_operation: tuple[int, int]
    w_composite_operation: tuple[int, int]

    w_scalar_type: tuple[int, int]
    w_container_type: tuple[int, int]

    w_composite_constant: tuple[int, int]
    w_scalar_constant: tuple[int, int]


class FuzzingStrategy(BaseModelORM):
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


class GeneratorInfo(BaseModelORM):
    id: GeneratorID
    fuzzer_version: str
    strategy: Optional[FuzzingStrategy] = None


class BaseShader(BaseModelORM):
    shader_id: ShaderID
    shader_assembly: str
    generator_info: GeneratorInfo


class ShaderSubmission(BaseShader):
    prioritize: bool
    n_buffers: int


class RetrievedShader(BaseShader):
    ...


class ExecutionQueue(BaseModelORM):
    queue: deque[ShaderSubmission]


class BufferSubmission(BaseModelORM):
    executor: ExecutionPlatform
    buffer_dump: str


class ExecutionQueues(BaseModelORM):
    queues: dict[QueueID, ExecutionQueue]
    _executor_lookup: dict[ExecutionPlatform, QueueID] = PrivateAttr(
        default_factory=dict
    )

    def new_execution_queue(self, executor: ExecutionPlatform) -> QueueID:
        if executor in self._executor_lookup:
            return self._executor_lookup[executor]
        queue_initializer: ExecutionQueue = ExecutionQueue(queue=deque())
        try:
            queue_initializer.queue = copy.deepcopy(
                self.queues[
                    max(
                        self.queues,
                        key=lambda x: len(self.queues[x].queue),
                    )
                ].queue
            )
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
        for execution_queue in self.queues.values():
            if shader.prioritize:
                execution_queue.queue.appendleft(shader)
            else:
                execution_queue.queue.append(shader)

    def get_next_shader(self, executor: ExecutionPlatform) -> ShaderSubmission:
        queue_id: QueueID = self._executor_lookup[executor]
        return self.queues[queue_id].queue.popleft()

    def flush(self) -> None:
        for execution_queue in self.queues.values():
            execution_queue.queue.clear()
