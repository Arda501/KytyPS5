#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <map>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class TextureCache;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = (LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemorySize) >> CACHING_PAGEBITS;
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);

	static constexpr uint64_t PageIndex(uint64_t address) {
		return (address < LOWER_ADDRESS_SIZE
		            ? address
		            : address - LibKernel::Memory::kExtendedMemoryBase + LOWER_ADDRESS_SIZE) >>
		       CACHING_PAGEBITS;
	}
	static constexpr uint64_t GuestAddress(uint64_t offset) {
		return offset < LOWER_ADDRESS_SIZE
		           ? offset
		           : offset - LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemoryBase;
	}

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer = false,
	                                                        BufferId id              = {});
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return m_stream_buffer;
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// The value of the last constant fill covering the range, while nothing else has written it.
	// Lets readers of fill-cleared surface metadata skip waiting for the GPU.
	[[nodiscard]] bool FindFill(uint64_t vaddr, uint64_t size, uint32_t& value, uint64_t& tick);
	void               ForgetFills(uint64_t vaddr, uint64_t size);
	// Records GPU work that leaves value in the range, as a fill does.
	void RecordFill(uint64_t vaddr, uint64_t size, uint32_t value);
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool IsRegionRegistered(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	void               ProcessFaultBuffer();
	void               SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size);
	// Uploads the CPU-modified pages of every buffer inside mapped. Unless full is set, only
	// ranges that changed since the previous call are visited.
	void SynchronizeCpuModifiedBuffers(const RangeSet& mapped, bool full);
	void RunGarbageCollector();

	// Buffers the CPU rewrites every frame are shared: they alias guest direct memory, so CPU
	// writes reach the GPU without faults or uploads, and GPU writes need no download, only a
	// wait. The GPU accesses them when it runs, so command-processor writes into them are
	// recorded in GPU order.
	[[nodiscard]] Buffer* FindSharedBuffer(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool    IsRegionShared(uint64_t vaddr, uint64_t size);
	// Writes a completion label without faulting: guest memory through the backing alias, and
	// a GPU-modified device copy through an ordered update, so neither needs a download.
	// Returns false when the caller should write guest memory directly.
	[[nodiscard]] bool WriteLabel(uint64_t vaddr, const void* data, uint32_t size);
	// The command processor reads draw arguments here: GPU writes to the buffer then submit
	// early, so the read only waits for work up to the write.
	void NoteCommandReadback(uint64_t vaddr, uint64_t size);
	// Drops shared buffers over a guest range that is being unmapped.
	void UnmapSharedBuffers(uint64_t vaddr, uint64_t size);

private:
	friend struct BufferCacheTestAccess;

	bool IsBufferInvalid(BufferId id) const {
		const auto* buffer = m_slot_buffers.try_get(id);
		return buffer == nullptr || buffer->is_deleted;
	}

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 44, 20>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	// Takes ownership of shared pages for GPU-ordered writes, as shader writes do: CPU and
	// command-processor reads then wait for them instead of reading stale bytes.
	void                        MarkSharedGpuWrite(Buffer& buffer, uint64_t vaddr, uint64_t size);
	void                        TouchBuffer(const Buffer& buffer);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void Unregister(BufferId id);
	template <bool insert>
	void ChangeRegister(BufferId id);
	void DeleteBuffer(BufferId id);
	[[nodiscard]] bool SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                     bool is_written, bool is_texel_buffer);
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      uint64_t total_size);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// Synchronous downloads publish before returning; asynchronous callers wait before reuse.
	template <bool async>
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size);
	void               ForgetFillsLocked(uint64_t vaddr, uint64_t size);
	void               NoteCpuAccess(Buffer& buffer);
	void               NoteCpuDownload(Buffer& buffer);
	void               UpdateSharing();
	void               ShareBuffer(BufferId id);
	void               ReportFaultOwners();

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	FaultManager                                      m_fault_manager;
	Buffer                                            m_gds_buffer;
	Buffer                                            m_bda_pagetable_buffer;
	Common::SlotVector<Buffer>                        m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                          m_buffers;
	PageTable                                          m_page_table;
	RangeSet                                           m_gpu_modified_ranges;
	struct FillRecord {
		uint64_t end   = 0;
		uint32_t value = 0;
		uint64_t tick  = 0;
	};
	std::mutex                                 m_fill_mutex;
	std::map<uint64_t, FillRecord>             m_fill_records;
	MemoryTracker                              m_memory_tracker;
	StreamBuffer                               m_staging_buffer;
	StreamBuffer                               m_stream_buffer;
	StreamBuffer                               m_download_buffer;
	StreamBuffer                               m_device_buffer;
	TextureCache&                              m_texture_cache;
	uint64_t                                   m_total_used_memory = 0;
	std::vector<std::pair<uint64_t, uint64_t>> m_cpu_modified_ranges;
	// Buffer addresses waiting for the next frame boundary to change sharing.
	std::vector<uint64_t> m_share_candidates;
	size_t                m_shared_buffer_count = 0;
	// Downloads per buffer address since the last report.
	std::map<uint64_t, uint32_t> m_download_stats;
	struct CreateStats {
		uint32_t creates = 0;
		uint32_t joins   = 0;
		uint32_t retires = 0;
	};
	std::map<uint64_t, CreateStats> m_create_stats;
	uint64_t                        m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t                        m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	uint64_t                        m_gc_tick            = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
