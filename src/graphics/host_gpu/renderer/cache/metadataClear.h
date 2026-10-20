#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_METADATACLEAR_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_METADATACLEAR_H_

#include "common/common.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <span>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;

// Decides on the GPU whether color metadata slices hold a fast clear, so the clear can be
// applied with conditional rendering instead of waiting for the metadata to be read back.
class MetadataClearChecker {
public:
	static constexpr uint32_t PredicatesPerSlice = 8;

	MetadataClearChecker(GraphicContext& graphics, CommandScheduler& scheduler);
	~MetadataClearChecker();
	KYTY_CLASS_NO_COPY(MetadataClearChecker);

	[[nodiscard]] bool Enabled() const noexcept { return m_pipeline != nullptr; }

	// Records the check of count slices of slice_size bytes starting at offset in metadata.
	// Predicate k of slice s is the word at the returned offset + (s * PredicatesPerSlice + k)
	// * 4 of PredicateBuffer(): it is set when every byte of the slice is codes[k]. With
	// write_back, a matched slice is overwritten with 0xff (expanded) on the GPU.
	[[nodiscard]] uint64_t Check(vk::CommandBuffer command, const Buffer& metadata, uint64_t offset,
	                             uint64_t slice_size, uint32_t count,
	                             std::span<const uint8_t> codes, bool write_back);
	[[nodiscard]] vk::Buffer PredicateBuffer() const noexcept { return m_predicates.Handle(); }

private:
	GraphicContext&         m_graphics;
	Buffer                  m_predicates;
	uint64_t                m_cursor          = 0;
	vk::DescriptorSetLayout m_desc_layout     = nullptr;
	vk::PipelineLayout      m_pipeline_layout = nullptr;
	vk::Pipeline            m_pipeline        = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_METADATACLEAR_H_
