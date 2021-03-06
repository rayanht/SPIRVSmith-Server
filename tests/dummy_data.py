from vulkan_platform_py import *

from app.models import BufferSubmission, ShaderSubmission

dummy_executor1: ExecutionPlatform = ExecutionPlatform(
    VulkanBackend.Vulkan,
    OperatingSystem.Linux,
    {
        HardwareType.GPU: [
            HardwareInformation(
                HardwareType.GPU, HardwareVendor.Nvidia, "Tesla P100", "v1.0"
            )
        ]
    },
)

dummy_executor2: ExecutionPlatform = ExecutionPlatform(
    VulkanBackend.Vulkan,
    OperatingSystem.Darwin,
    {
        HardwareType.GPU: [
            HardwareInformation(
                HardwareType.GPU, HardwareVendor.Nvidia, "Tesla P100", "v1.0"
            )
        ]
    },
)

dummy_shader_submission1: ShaderSubmission = ShaderSubmission(
    generator_version="v1.0",
    shader_id="dummy_shader1",
    shader_assembly="<<SPIR-V>>",
    prioritize=False,
    n_buffers=1,
)

dummy_shader_submission2: ShaderSubmission = ShaderSubmission(
    generator_version="v1.0",
    shader_id="dummy_shader2",
    shader_assembly="<<SPIR-V>>",
    prioritize=False,
    n_buffers=1,
)

dummy_shader_submission_prioritized: ShaderSubmission = ShaderSubmission(
    generator_version="v1.0",
    shader_id="dummy_shader2",
    shader_assembly="<<SPIR-V>>",
    prioritize=True,
    n_buffers=1,
)


def create_dummy_buffer_submission(
    executor: ExecutionPlatform, value: str
) -> BufferSubmission:
    return BufferSubmission(executor=executor, buffer_dump=value)
