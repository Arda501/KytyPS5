#include "graphics/host_gpu/sharedMemory.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "kernel/memory.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX && !defined(__APPLE__)
#include <cstdio>
#include <fcntl.h>
#include <linux/udmabuf.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace Libs::Graphics {

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX && !defined(__APPLE__)

namespace {

constexpr uint32_t UdmabufListLimit = 1024;

struct Chunk {
	vk::DeviceMemory memory = nullptr;
	uint32_t         users  = 0;
};

struct Window {
	vk::DeviceMemory                           memory = nullptr;
	uint32_t                                   users  = 0;
	uint64_t                                   size   = 0;
	std::vector<std::pair<uint64_t, uint64_t>> segments;
};

std::mutex                g_chunk_mutex;
std::map<uint64_t, Chunk> g_chunks;
// Keyed by guest base address; a growing mapping can leave several windows at one base.
std::multimap<uint64_t, Window> g_windows;
std::atomic<uint64_t>           g_import_count {0};

int UdmabufDevice() {
	static const int device = [] {
		const int fd = open("/dev/udmabuf", O_RDWR | O_CLOEXEC);
		if (fd < 0) {
			Log::WriteToConsoleAndLog(
			    "GPU: /dev/udmabuf is unavailable, guest buffers are copied to the GPU.\n");
		}
		return fd;
	}();
	return device;
}

uint64_t UdmabufSizeLimit() {
	static const uint64_t limit = [] {
		uint64_t megabytes = 64;
		if (auto* file = std::fopen("/sys/module/udmabuf/parameters/size_limit_mb", "r")) {
			unsigned long long value = 0;
			if (std::fscanf(file, "%llu", &value) == 1 && value != 0) {
				megabytes = value;
			}
			std::fclose(file);
		}
		return megabytes << 20u;
	}();
	return limit;
}

// Every imported allocation is listed in each queue submission, so buffers share one import
// per chunk of direct memory. A chunk pins all of its pages, which caps its size; larger
// buffers get imports of their own, up to the udmabuf limit.
uint64_t ChunkSize() {
	constexpr uint64_t MaxChunkSize = uint64_t {256} << 20u;
	return std::min(UdmabufSizeLimit(), MaxChunkSize);
}

uint32_t FindMemoryType(uint32_t bits) {
	for (uint32_t type = 0; type < 32; type++) {
		if ((bits & (1u << type)) != 0) {
			return type;
		}
	}
	return UINT32_MAX;
}

// Imports the concatenated (offset, size) pieces of the direct-memory file as one allocation.
vk::DeviceMemory ImportBacking(GraphicContext& graphics, int memfd,
                               const std::vector<std::pair<uint64_t, uint64_t>>& segments,
                               uint64_t size, uint32_t memory_type_bits) {
	const int device = UdmabufDevice();
	if (device < 0 || segments.empty() || segments.size() > UdmabufListLimit ||
	    size > UdmabufSizeLimit()) {
		return nullptr;
	}
	std::vector<uint8_t> storage(sizeof(udmabuf_create_list) +
	                             segments.size() * sizeof(udmabuf_create_item));
	auto*                list = reinterpret_cast<udmabuf_create_list*>(storage.data());
	list->flags               = UDMABUF_FLAGS_CLOEXEC;
	list->count               = static_cast<uint32_t>(segments.size());
	for (size_t i = 0; i < segments.size(); i++) {
		list->list[i].memfd  = static_cast<uint32_t>(memfd);
		list->list[i].offset = segments[i].first;
		list->list[i].size   = segments[i].second;
	}
	const int dma_buf = ioctl(device, UDMABUF_CREATE_LIST, list);
	if (dma_buf < 0) {
		return nullptr;
	}

	constexpr auto            handle_type = vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT;
	vk::MemoryFdPropertiesKHR fd_properties {};
	const auto                type = graphics.device.getMemoryFdPropertiesKHR(
	                                     handle_type, dma_buf, &fd_properties) == vk::Result::eSuccess
	                                     ? FindMemoryType(memory_type_bits & fd_properties.memoryTypeBits)
	                                     : UINT32_MAX;
	if (type == UINT32_MAX) {
		close(dma_buf);
		return nullptr;
	}
	vk::ImportMemoryFdInfoKHR import_info {};
	import_info.handleType = handle_type;
	import_info.fd         = dma_buf;
	vk::MemoryAllocateFlagsInfo flags_info {};
	flags_info.pNext = &import_info;
	flags_info.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
	vk::MemoryAllocateInfo allocate_info {};
	allocate_info.pNext           = &flags_info;
	allocate_info.allocationSize  = size;
	allocate_info.memoryTypeIndex = type;
	vk::DeviceMemory memory;
	// A successful import takes ownership of the file descriptor.
	if (graphics.device.allocateMemory(&allocate_info, nullptr, &memory) != vk::Result::eSuccess) {
		close(dma_buf);
		return nullptr;
	}
	g_import_count++;
	return memory;
}

// The (offset, size) pieces of window bytes [offset, offset + size).
std::vector<std::pair<uint64_t, uint64_t>> WindowPieces(const Window& window, uint64_t offset,
                                                        uint64_t size) {
	std::vector<std::pair<uint64_t, uint64_t>> pieces;
	uint64_t                                   position = 0;
	for (const auto& [backing, length]: window.segments) {
		const auto from = std::max(offset, position);
		const auto to   = std::min(offset + size, position + length);
		if (from < to) {
			const auto piece = backing + (from - position);
			if (!pieces.empty() && pieces.back().first + pieces.back().second == piece) {
				pieces.back().second += to - from;
			} else {
				pieces.emplace_back(piece, to - from);
			}
		}
		position += length;
	}
	return pieces;
}

bool ImportWindow(GraphicContext& graphics, int memfd, uint64_t vaddr, uint64_t size,
                  const std::vector<std::pair<uint64_t, uint64_t>>& buffer_segments,
                  const vk::MemoryRequirements& requirements, SharedMemory& shared) {
	uint64_t base        = 0;
	uint64_t window_size = 0;
	if (!LibKernel::Memory::GetGuestMappingWindow(vaddr, size, ChunkSize(), base, window_size) ||
	    (vaddr - base) % requirements.alignment != 0) {
		return false;
	}
	std::vector<std::pair<uint64_t, uint64_t>> pieces;
	for (const auto& [backing, length]: buffer_segments) {
		if (!pieces.empty() && pieces.back().first + pieces.back().second == backing) {
			pieces.back().second += length;
		} else {
			pieces.emplace_back(backing, length);
		}
	}
	std::lock_guard lock(g_chunk_mutex);
	// Reuse a window whose pages still back the range.
	auto [first, last] = g_windows.equal_range(base);
	auto entry         = g_windows.end();
	for (auto it = first; it != last; ++it) {
		if (vaddr + size <= base + it->second.size &&
		    WindowPieces(it->second, vaddr - base, size) == pieces) {
			entry = it;
			break;
		}
	}
	if (entry == g_windows.end()) {
		int      window_fd = -1;
		uint64_t backing   = 0;
		Window   window;
		if (!LibKernel::Memory::GetDirectMemoryBacking(base, window_size, window_fd, backing,
		                                               window.segments) ||
		    window_fd != memfd) {
			return false;
		}
		window.memory = ImportBacking(graphics, memfd, window.segments, window_size,
		                              requirements.memoryTypeBits);
		if (!window.memory) {
			return false;
		}
		window.size = window_size;
		entry       = g_windows.emplace(base, std::move(window));
	}
	if (graphics.device.bindBufferMemory(shared.buffer, entry->second.memory, vaddr - base) !=
	    vk::Result::eSuccess) {
		if (entry->second.users == 0) {
			graphics.device.freeMemory(entry->second.memory);
			g_import_count--;
			g_windows.erase(entry);
		}
		return false;
	}
	entry->second.users++;
	shared.window        = base;
	shared.window_memory = entry->second.memory;
	return true;
}

} // namespace

bool ImportGuestMemory(GraphicContext& graphics, uint64_t vaddr, uint64_t size,
                       vk::BufferUsageFlags usage, SharedMemory& shared) {
	static const bool disabled = std::getenv("KYTY_NO_SHARED_MEMORY") != nullptr;
	if (!graphics.dma_buf_import_enabled || size == 0 || disabled) {
		return false;
	}
	int                                        memfd        = -1;
	uint64_t                                   backing_size = 0;
	std::vector<std::pair<uint64_t, uint64_t>> segments;
	if (!LibKernel::Memory::GetDirectMemoryBacking(vaddr, size, memfd, backing_size, segments)) {
		return false;
	}

	vk::ExternalMemoryBufferCreateInfo external_info {};
	external_info.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT;
	vk::BufferCreateInfo buffer_info {};
	buffer_info.pNext = &external_info;
	buffer_info.size  = size;
	buffer_info.usage = usage;
	if (graphics.device.createBuffer(&buffer_info, nullptr, &shared.buffer) !=
	    vk::Result::eSuccess) {
		return false;
	}
	const auto requirements = graphics.device.getBufferMemoryRequirements(shared.buffer);

	const auto offset     = segments[0].first;
	const auto chunk      = offset / ChunkSize();
	const auto chunk_base = chunk * ChunkSize();
	const auto chunk_size = std::min(ChunkSize(), backing_size - chunk_base);
	if (segments.size() != 1 || offset + size > chunk_base + chunk_size ||
	    (offset - chunk_base) % requirements.alignment != 0) {
		// Every import is listed in each queue submission, so ranges no chunk holds share an
		// import of their guest mapping's window when they can.
		if (ImportWindow(graphics, memfd, vaddr, size, segments, requirements, shared)) {
			return true;
		}
		// No chunk or window holds the whole range; give it an import of its own.
		shared.dedicated_memory =
		    ImportBacking(graphics, memfd, segments, size, requirements.memoryTypeBits);
		if (!shared.dedicated_memory ||
		    graphics.device.bindBufferMemory(shared.buffer, shared.dedicated_memory, 0) !=
		        vk::Result::eSuccess) {
			ReleaseSharedMemory(graphics, shared);
			return false;
		}
		return true;
	}

	std::lock_guard lock(g_chunk_mutex);
	auto&           entry = g_chunks[chunk];
	if (!entry.memory) {
		entry.memory = ImportBacking(graphics, memfd, {{chunk_base, chunk_size}}, chunk_size,
		                             requirements.memoryTypeBits);
	}
	if (!entry.memory ||
	    graphics.device.bindBufferMemory(shared.buffer, entry.memory, offset - chunk_base) !=
	        vk::Result::eSuccess) {
		if (entry.users == 0) {
			if (entry.memory) {
				graphics.device.freeMemory(entry.memory);
				g_import_count--;
			}
			g_chunks.erase(chunk);
		}
		graphics.device.destroyBuffer(shared.buffer);
		shared = {};
		return false;
	}
	entry.users++;
	shared.chunk = chunk;
	return true;
}

void ReleaseSharedMemory(GraphicContext& graphics, SharedMemory& shared) {
	if (shared.buffer) {
		graphics.device.destroyBuffer(shared.buffer);
	}
	if (shared.dedicated_memory) {
		graphics.device.freeMemory(shared.dedicated_memory);
		g_import_count--;
	}
	if (shared.window != UINT64_MAX) {
		std::lock_guard lock(g_chunk_mutex);
		auto [first, last] = g_windows.equal_range(shared.window);
		auto entry         = std::find_if(first, last, [&](const auto& window) {
			return window.second.memory == shared.window_memory;
		});
		EXIT_IF(entry == last || entry->second.users == 0);
		if (--entry->second.users == 0) {
			graphics.device.freeMemory(entry->second.memory);
			g_windows.erase(entry);
			g_import_count--;
		}
	}
	if (shared.chunk != UINT64_MAX) {
		std::lock_guard lock(g_chunk_mutex);
		auto            entry = g_chunks.find(shared.chunk);
		EXIT_IF(entry == g_chunks.end() || entry->second.users == 0);
		if (--entry->second.users == 0) {
			graphics.device.freeMemory(entry->second.memory);
			g_chunks.erase(entry);
			g_import_count--;
		}
	}
	shared = {};
}

uint64_t SharedMemoryImportCount() {
	return g_import_count.load();
}

#else

bool ImportGuestMemory(GraphicContext&, uint64_t, uint64_t, vk::BufferUsageFlags, SharedMemory&) {
	return false;
}

uint64_t SharedMemoryImportCount() {
	return 0;
}

void ReleaseSharedMemory(GraphicContext& graphics, SharedMemory& shared) {
	if (shared.buffer) {
		graphics.device.destroyBuffer(shared.buffer);
	}
	shared = {};
}

#endif

} // namespace Libs::Graphics
