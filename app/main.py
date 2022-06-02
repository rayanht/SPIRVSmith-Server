from functools import lru_cache

from fastapi import BackgroundTasks, Depends, FastAPI, Path
import contextlib
from typing import Any, AsyncGenerator, AsyncIterator, Callable
import fastapi
from fastapi.routing import APIRoute
import firebase_admin
from firebase_admin import firestore
from google.cloud import bigquery
from google.cloud import storage
from app.config import Settings
from app.models import (
    BufferSubmission,
    ExecutionQueue,
    ExecutionQueues,
    GeneratorInfo,
    MetadataTag,
    QueueID,
    RetrievedShader,
    ShaderID,
    ShaderSubmission,
)
from vulkan_platform_py import ExecutionPlatform

shaders_tag = MetadataTag(
    name="shaders", description="Endpoints related to the brokerage of shaders."
)

queues_tag = MetadataTag(
    name="queues",
    description="Endpoints related to the management of execution queues.",
)

generators_tag = MetadataTag(
    name="generators", description="Endpoints related to generators."
)

buffers_tag = MetadataTag(
    name="buffers", description="Endpoints related to the management of buffers dumps."
)

metadata_tags = [shaders_tag, queues_tag, generators_tag, buffers_tag]


app = FastAPI(
    title="spirvsmith-server",
    version="0.1.0",
    openapi_tags=[tag.dict() for tag in metadata_tags],
    contact={
        "name": "Rayan Hatout",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)


@lru_cache()
def get_settings():
    return Settings()


def singleton(app: fastapi.FastAPI) -> Any:
    """
    Decorator that can be used to define a resource with the same lifetime as the
    fastapi application.
    Example::

        from typing import AsyncIterator

        import fastapi

        # For demo purposes
        class Database:
            async def start(): ...
            async def shutdown(): ...

        app = fastapi.FastAPI()

        @singleton(app)
        async def database() -> AsyncIterator[Database]:
            db = Database()
            await db.start()
            yield db
            await db.shutdown()

        @app.get("/users")
        def get_users(db: Database = database):
            ...
    """

    def decorator(func: Callable[[], AsyncGenerator[Any, None]]) -> Any:
        cm_factory = contextlib.asynccontextmanager(func)
        stack = contextlib.AsyncExitStack()
        sentinel_start = object()
        sentinel_shutdown = object()
        value: Any = sentinel_start

        @app.on_event("startup")
        async def _startup() -> None:
            nonlocal value
            value = await stack.enter_async_context(cm_factory())

        @app.on_event("shutdown")
        async def _shutdown() -> None:
            nonlocal value
            await stack.pop_all().aclose()
            value = sentinel_shutdown

        def get_value() -> Any:
            if value is sentinel_start:
                raise RuntimeError("Application not started yet.")
            if value is sentinel_shutdown:
                raise RuntimeError("Application already shut down.")
            return value

        return fastapi.Depends(get_value)

    return decorator


class ShaderBroker:
    GCS_client: storage.Client
    GCS_bucket: storage.Bucket
    BQ_client: bigquery.Client
    FS_client: firestore.firestore.Client
    execution_queues: ExecutionQueues

    async def start(self, settings: Settings = get_settings()):
        if settings.app_env != "dev":
            firebase_admin.initialize_app()
            self.GCS_client = storage.Client()
            self.GCS_bucket = self.GCS_client.get_bucket("spirv_shaders_bucket")
            self.BQ_client = bigquery.Client()
            self.FS_client = firestore.client()
        self.execution_queues = ExecutionQueues()

    async def shutdown(self, settings: Settings = get_settings()):
        if settings.app_env != "dev":
            firebase_admin.delete_app(firebase_admin.get_app())

    def GCS_upload_shader(self, shader: ShaderSubmission) -> None:
        self.GCS_bucket.blob(f"{shader.shader_id}.spasm").upload_from_string(
            shader.shader_assembly, checksum="md5"
        )

    def GCS_download_shader(self, shader_id: ShaderID) -> str:
        return self.GCS_bucket.blob(f"{shader_id}.spasm").download_as_string()

    def BQ_insert_shader_with_buffer_dump(
        self,
        shader_id: ShaderID,
        execution_platform: ExecutionPlatform,
        buffer_submission: BufferSubmission,
    ) -> None:
        insert_query = f"""
        INSERT INTO `spirvsmith.spirv.shader_data`
        VALUES (
            "{shader_id}",
            "{buffer_submission.buffer_dump}",
            "{execution_platform.operating_system.value}",
            "{execution_platform.get_active_hardware().hardware_type.value}",
            "{execution_platform.get_active_hardware().hardware_vendor.value}",
            "{execution_platform.get_active_hardware().hardware_model}",
            "{execution_platform.get_active_hardware().driver_version}",
            "{execution_platform.vulkan_backend.value}",
            CURRENT_TIMESTAMP()
        )
        """
        self.BQ_client.query(insert_query).result()


@singleton(app)
async def get_broker() -> AsyncIterator[ShaderBroker]:
    broker = ShaderBroker()
    await broker.start()
    yield broker
    await broker.shutdown()


@app.post("/shaders", tags=["shaders"])
def submit_shader(
    shader: ShaderSubmission,
    background_tasks: BackgroundTasks,
    broker: ShaderBroker = get_broker,
    settings: Settings = Depends(get_settings),
):
    broker.execution_queues.enqueue(shader)
    if settings.app_env != "dev":
        background_tasks.add_task(broker.GCS_upload_shader, shader)
    return shader.shader_id


@app.get("/shaders/{shader_id}", response_model=RetrievedShader, tags=["shaders"])
async def get_hader(
    shader_id: ShaderID = Path(title="The ID of the shader to get"),
    broker: ShaderBroker = get_broker,
):
    shader_assembly: str = await broker.GCS_download_shader(shader_id)
    return RetrievedShader(shader_id=shader_id, shader_assembly=shader_assembly)


@app.post("/shaders/next", response_model=RetrievedShader, tags=["shaders"])
def get_next_shader(executor: ExecutionPlatform, broker: ShaderBroker = get_broker):
    shader: ShaderSubmission = broker.execution_queues.get_next_shader(executor)
    return RetrievedShader(
        shader_id=shader.shader_id,
        shader_assembly=shader.shader_assembly,
        generator_info=shader.generator_info,
    )


@app.post("/generators", tags=["generators"])
def register_generator(
    generator: GeneratorInfo,
    broker: ShaderBroker = get_broker,
    settings: Settings = Depends(get_settings),
):
    if settings.app_env != "dev":
        document_reference = broker.FS_client.collection("configurations").document(
            generator.id
        )
        document_reference.set(generator.strategy.dict())
    return 200


@app.put("/queues", tags=["queues"])
def register_executor(
    executor: ExecutionPlatform,
    broker: ShaderBroker = get_broker,
):
    return broker.execution_queues.new_execution_queue(executor)


@app.delete("/queues", tags=["queues"])
def delete_queues(broker: ShaderBroker = get_broker):
    broker.execution_queues.flush()
    return 200


@app.get("/queues", response_model=ExecutionQueues, tags=["queues"])
def get_queues(broker: ShaderBroker = get_broker):
    return broker.execution_queues


@app.get("/queues/{queue_id}", response_model=ExecutionQueue, tags=["queues"])
def get_queue(
    queue_id: QueueID = Path(title="The ID of the queue to get"),
    broker: ShaderBroker = get_broker,
):
    return broker.execution_queues.queues[queue_id]


@app.post("/buffers/{shader_id}", tags=["buffers"])
def post_buffers(
    buffer_submission: BufferSubmission,
    shader_id: ShaderID = Path(title="The ID of the shader to get"),
    broker: ShaderBroker = get_broker,
    settings: Settings = Depends(get_settings),
):
    if settings.app_env != "dev":
        broker.BQ_insert_shader_with_buffer_dump(
            shader_id, buffer_submission.executor, buffer_submission
        )
    return 200


def use_route_names_as_operation_ids(app: FastAPI) -> None:
    """
    Simplify operation IDs so that generated API clients have simpler function
    names.

    Should be called only after all routes have been added.
    """
    for route in app.routes:
        if isinstance(route, APIRoute):
            route.operation_id = route.name


use_route_names_as_operation_ids(app)
