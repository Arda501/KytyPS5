#ifndef KYTY_COMMON_PROFILER_H_
#define KYTY_COMMON_PROFILER_H_

#include "common/common.h"

#include <cstdint>
#include <string>
#include <tracy/Tracy.hpp> // IWYU pragma: export
#include <vector>

namespace profiler::colors {

inline constexpr uint32_t RedA100        = 0xff8a80;
inline constexpr uint32_t Blue300        = 0x64b5f6;
inline constexpr uint32_t CyanA700       = 0x00b8d4;
inline constexpr uint32_t Green100       = 0xc8e6c9;
inline constexpr uint32_t Green200       = 0xa5d6a7;
inline constexpr uint32_t Green300       = 0x81c784;
inline constexpr uint32_t Green400       = 0x66bb6a;
inline constexpr uint32_t Amber300       = 0xffd54f;
inline constexpr uint32_t DeepOrangeA200 = 0xff6e40;

} // namespace profiler::colors

namespace Profiler {

void SetThreadName(const char* name);

enum class Counter {
	GpuReadFaults,
	GpuWriteFaults,
	GpuThreadWriteFaults,
	BdaSynchronizations,
	BdaSyncRanges,
	BdaFullScans,
	WriteProtectionRuns,
	SubmitsTotal,
	SubmitsForWait,
	SubmitsBatchNewWait,
	SubmitsBatchFull,
	SubmitsBatchIdle,
	SubmitsBatchAllBlocked,
	ReadbacksReadFault,
	ReadbacksWriteFault,
	ReadbacksBacking,
	ReadbacksMetadata,
	ReadbacksIndirect,
	MetadataFromFill,
	MetadataOnGpu,
	RenderPasses,
	SubmitsForProgress,
	SubmitsEarly,
	NativeIndirectDraws,
	PipelinesCreated,
	PermutationsCompiled,
	Count,
};

// Async-signal-safe: counts events for the per-frame plots of PlotCounters.
void Count(Counter counter, uint64_t value = 1) noexcept;
// Async-signal-safe: samples a faulting page for the distinct-page plot.
void RecordWriteFault(uint64_t vaddr) noexcept;
// Names the guest memory range of a sampled fault page for the periodic fault report.
void SetWriteFaultDescriber(std::string (*describe)(uint64_t vaddr));
// Appends the sampled fault page numbers recorded since cursor, then advances it.
void CopyWriteFaultPages(uint64_t& cursor, std::vector<uint64_t>& pages);
void PlotCounters();

void Initialize();
void Shutdown();

struct Lifecycle {
	static constexpr const char* name               = "Profiler";
	static constexpr auto        initialize         = Profiler::Initialize;
	static constexpr auto        shutdown           = Profiler::Shutdown;
	static constexpr auto        emergency_shutdown = Profiler::Shutdown;
};

} // namespace Profiler

#define KYTY_PROFILER_CONCAT_IMPL(a, b) a##b
#define KYTY_PROFILER_CONCAT(a, b)      KYTY_PROFILER_CONCAT_IMPL(a, b)
#define KYTY_PROFILER_COLOR_OR_DEFAULT(default_color, ...)                                         \
	KYTY_PROFILER_COLOR_OR_DEFAULT_IMPL(default_color __VA_OPT__(, ) __VA_ARGS__, default_color)
#define KYTY_PROFILER_COLOR_OR_DEFAULT_IMPL(default_color, color, ...) color

#define KYTY_PROFILER_BLOCK(name, ...)                                                             \
	KYTY_PROFILER_BLOCK_IMPL(__LINE__, name __VA_OPT__(, ) __VA_ARGS__)
#define KYTY_PROFILER_BLOCK_IMPL(line, name, ...)                                                  \
	static constexpr tracy::SourceLocationData KYTY_PROFILER_CONCAT(                               \
	    kyty_profiler_source_location_, line) {name, TracyFunction, TracyFile,                     \
		                                       static_cast<uint32_t>(line),                        \
		                                       KYTY_PROFILER_COLOR_OR_DEFAULT(0, __VA_ARGS__)};    \
	tracy::ScopedZone KYTY_PROFILER_CONCAT(kyty_profiler_block_, line)(                            \
	    &KYTY_PROFILER_CONCAT(kyty_profiler_source_location_, line), TRACY_CALLSTACK,              \
	    tracy::ProfilerAvailable())

#define KYTY_PROFILER_FUNCTION(...) KYTY_PROFILER_BLOCK(nullptr __VA_OPT__(, ) __VA_ARGS__)

#define KYTY_PROFILER_THREAD(name) Profiler::SetThreadName(name)

#endif /* KYTY_COMMON_PROFILER_H_ */
