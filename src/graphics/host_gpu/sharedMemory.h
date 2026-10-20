#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_SHAREDMEMORY_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_SHAREDMEMORY_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

namespace Libs::Graphics {

struct GraphicContext;

struct SharedMemory {
	vk::Buffer buffer = nullptr;
	// The direct-memory chunk the buffer is bound into; chunks are shared between buffers.
	uint64_t chunk = UINT64_MAX;
	// Set instead of a chunk for ranges backed by scattered pieces of direct memory: an import
	// of the guest mapping's window that holds the range, shared between buffers.
	uint64_t         window        = UINT64_MAX;
	vk::DeviceMemory window_memory = nullptr;
	// Set instead for ranges neither a chunk nor a window holds.
	vk::DeviceMemory dedicated_memory = nullptr;

	[[nodiscard]] bool IsValid() const {
		return buffer && (chunk != UINT64_MAX || window != UINT64_MAX || dedicated_memory);
	}
};

// Creates a buffer over the same physical pages as guest direct memory, so CPU and GPU see each
// other's writes without copies. Fails when the range is not fully backed by direct memory or
// the host cannot share it (udmabuf missing, too large, or no dma-buf import).
[[nodiscard]] bool ImportGuestMemory(GraphicContext& graphics, uint64_t vaddr, uint64_t size,
                                     vk::BufferUsageFlags usage, SharedMemory& shared);
void               ReleaseSharedMemory(GraphicContext& graphics, SharedMemory& shared);
// Live imported allocations; each one adds work to every queue submission.
[[nodiscard]] uint64_t SharedMemoryImportCount();

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_SHAREDMEMORY_H_
