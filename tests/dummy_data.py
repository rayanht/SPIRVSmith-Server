from vulkan_platform_py import *

from app.models import GeneratorInfo, ShaderSubmission

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

dummy_generator_info: GeneratorInfo = GeneratorInfo(
    id="dummy_generator", fuzzer_version="v1.0"
)

dummy_shader_submission1: ShaderSubmission = ShaderSubmission(
    generator_info=dummy_generator_info,
    shader_id="dummy_shader1",
    shader_assembly="<<SPIR-V>>",
    prioritize=False,
    n_buffers=1,
)

dummy_shader_submission2: ShaderSubmission = ShaderSubmission(
    generator_info=dummy_generator_info,
    shader_id="dummy_shader2",
    shader_assembly="<<SPIR-V>>",
    prioritize=False,
    n_buffers=1,
)

dummy_shader_submission_prioritized: ShaderSubmission = ShaderSubmission(
    generator_info=dummy_generator_info,
    shader_id="dummy_shader2",
    shader_assembly="<<SPIR-V>>",
    prioritize=True,
    n_buffers=1,
)
