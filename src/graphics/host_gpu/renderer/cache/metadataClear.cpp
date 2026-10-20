#include "graphics/host_gpu/renderer/cache/metadataClear.h"

#include "common/assert.h"
#include "gpu_tiler_shaders/metadata_clear_check_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"

#include <algorithm>
#include <array>

namespace Libs::Graphics {

namespace {

constexpr uint64_t PredicateBufferSize = uint64_t {256} << 10u;

struct CheckParams {
	uint32_t metadata_offset;
	uint32_t slice_words;
	uint32_t predicate_offset;
	uint32_t code_count;
	uint32_t codes_lo;
	uint32_t codes_hi;
	uint32_t write_back;
};

} // namespace

MetadataClearChecker::MetadataClearChecker(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics),
      m_predicates(graphics, scheduler, MemoryUsage::DeviceLocal, 0,
	               vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
	                   (graphics.conditional_rendering_enabled
                            ? vk::BufferUsageFlagBits::eConditionalRenderingEXT
							: vk::BufferUsageFlags {}),
	               PredicateBufferSize) {
	if (!graphics.conditional_rendering_enabled) {
		return;
	}
	SetVulkanObjectNameF(m_graphics.device, m_predicates.Handle(), "Metadata Clear Predicates");

	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = std::size(bindings);
	layout_info.pBindings    = bindings;
	RequireVulkanSuccess(
	    m_graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &m_desc_layout),
	    "create metadata-clear descriptor layout");

	const vk::PushConstantRange  push_range {vk::ShaderStageFlagBits::eCompute, 0,
	                                         sizeof(CheckParams)};
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &m_desc_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_range;
	RequireVulkanSuccess(
	    m_graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr, &m_pipeline_layout),
	    "create metadata-clear pipeline layout");

	const auto module = CompileSPV(METADATA_CLEAR_CHECK_SPV, m_graphics.device);
	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage  = vk::ShaderStageFlagBits::eCompute;
	stage.module = module;
	stage.pName  = "main";
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = m_pipeline_layout;
	const auto result =
	    m_graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &m_pipeline);
	m_graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create metadata-clear pipeline");
	SetVulkanObjectNameF(m_graphics.device, m_pipeline, "Metadata Clear Check");
}

MetadataClearChecker::~MetadataClearChecker() {
	if (m_pipeline) {
		m_graphics.device.destroyPipeline(m_pipeline, nullptr);
	}
	if (m_pipeline_layout) {
		m_graphics.device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	}
	if (m_desc_layout) {
		m_graphics.device.destroyDescriptorSetLayout(m_desc_layout, nullptr);
	}
}

uint64_t MetadataClearChecker::Check(vk::CommandBuffer command, const Buffer& metadata,
                                     uint64_t offset, uint64_t slice_size, uint32_t count,
                                     std::span<const uint8_t> codes, bool write_back) {
	EXIT_IF(!Enabled() || count == 0 || codes.empty() || codes.size() > PredicatesPerSlice ||
	        slice_size % 4 != 0 || offset % 4 != 0);
	// The ring outlives the GPU work that reads it: it holds thousands of frames of checks.
	const uint64_t predicate_bytes = uint64_t {count} * PredicatesPerSlice * sizeof(uint32_t);
	EXIT_IF(predicate_bytes > PredicateBufferSize);
	if (m_cursor + predicate_bytes > PredicateBufferSize) {
		m_cursor = 0;
	}
	const auto predicate_offset = m_cursor;
	m_cursor += predicate_bytes;

	const auto alignment = std::max<uint64_t>(
	    m_graphics.physical_device_properties.limits.minStorageBufferOffsetAlignment, 4);
	const auto binding_offset = offset / alignment * alignment;
	const auto binding_size   = offset - binding_offset + slice_size * count;

	CheckParams params {};
	params.metadata_offset  = static_cast<uint32_t>((offset - binding_offset) / 4);
	params.slice_words      = static_cast<uint32_t>(slice_size / 4);
	params.predicate_offset = static_cast<uint32_t>(predicate_offset / 4);
	params.code_count       = static_cast<uint32_t>(codes.size());
	for (size_t i = 0; i < codes.size(); i++) {
		auto& packed = i < 4 ? params.codes_lo : params.codes_hi;
		packed |= static_cast<uint32_t>(codes[i]) << ((i & 3u) * 8u);
	}
	params.write_back = write_back ? 1u : 0u;

	const vk::DescriptorBufferInfo infos[] {
	    {metadata.Handle(), binding_offset, binding_size},
	    {m_predicates.Handle(), predicate_offset, predicate_bytes},
	};
	// The push constant addresses predicates from the start of the binding.
	params.predicate_offset = 0;
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t index = 0; index < writes.size(); ++index) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}

	vk::MemoryBarrier2 pre_barrier {};
	pre_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	pre_barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	pre_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	pre_barrier.dstAccessMask =
	    vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite;
	vk::MemoryBarrier2 post_barrier {};
	post_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	post_barrier.srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite;
	post_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eConditionalRenderingEXT |
	                             vk::PipelineStageFlagBits2::eAllCommands;
	post_barrier.dstAccessMask = vk::AccessFlagBits2::eConditionalRenderingReadEXT |
	                             vk::AccessFlagBits2::eMemoryRead |
	                             vk::AccessFlagBits2::eMemoryWrite;

	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &pre_barrier;
	command.pipelineBarrier2(dependency);
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0, writes);
	command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(params),
	                      &params);
	command.dispatch(count, 1, 1);
	dependency.pMemoryBarriers = &post_barrier;
	command.pipelineBarrier2(dependency);
	return predicate_offset;
}

} // namespace Libs::Graphics
