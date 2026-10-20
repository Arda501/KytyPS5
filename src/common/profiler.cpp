#include "common/profiler.h"

#include "common/emulatorConfig.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <common/TracyProtocol.hpp>
#include <common/TracyVersion.hpp>
#include <cstdio>
#include <iterator>
#include <map>
#include <tracy/Tracy.hpp>
#include <vector>

namespace Profiler {

void SetThreadName(const char* name) {
	if (tracy::ProfilerAvailable() && name != nullptr) {
		tracy::SetThreadName(name);
	}
}

static constexpr const char* COUNTER_NAMES[] = {
    "GPU read faults",
    "GPU write faults",
    "GPU-thread write faults",
    "BDA synchronizations",
    "BDA sync ranges",
    "BDA full scans",
    "Write protection runs",
    "Submits: total",
    "Submits: wait",
    "Submits: batch new wait",
    "Submits: batch full",
    "Submits: batch idle",
    "Submits: batch all blocked",
    "Readbacks: read fault",
    "Readbacks: write fault",
    "Readbacks: backing read",
    "Readbacks: metadata",
    "Readbacks: indirect args",
    "Metadata reads from fills",
    "Metadata checks on GPU",
    "Render passes",
    "Submits: progress",
    "Submits: early",
    "Native indirect draws",
    "Pipelines created",
    "Permutations compiled",
};
static_assert(std::size(COUNTER_NAMES) == static_cast<size_t>(Counter::Count));

static std::atomic<uint64_t> g_counters[static_cast<size_t>(Counter::Count)] = {};

void Count(Counter counter, uint64_t value) noexcept {
	g_counters[static_cast<size_t>(counter)].fetch_add(value, std::memory_order_relaxed);
}

static constexpr size_t      FAULT_SAMPLE_COUNT                  = 65536;
static std::atomic<uint64_t> g_fault_samples[FAULT_SAMPLE_COUNT] = {};
static std::atomic<uint64_t> g_fault_sample_next {0};

void RecordWriteFault(uint64_t vaddr) noexcept {
	const auto index = g_fault_sample_next.fetch_add(1, std::memory_order_relaxed);
	g_fault_samples[index % FAULT_SAMPLE_COUNT].store(vaddr >> 12u, std::memory_order_relaxed);
}

void CopyWriteFaultPages(uint64_t& cursor, std::vector<uint64_t>& pages) {
	const auto next  = g_fault_sample_next.load(std::memory_order_relaxed);
	const auto count = std::min<uint64_t>(next - cursor, FAULT_SAMPLE_COUNT);
	for (uint64_t i = next - count; i < next; i++) {
		pages.push_back(g_fault_samples[i % FAULT_SAMPLE_COUNT].load(std::memory_order_relaxed));
	}
	cursor = next;
}

static std::atomic<std::string (*)(uint64_t)> g_fault_describer {nullptr};

void SetWriteFaultDescriber(std::string (*describe)(uint64_t vaddr)) {
	g_fault_describer.store(describe, std::memory_order_release);
}

static void ReportFaultPages(const std::vector<uint64_t>& pages) {
	static uint32_t frame    = 0;
	auto*           describe = g_fault_describer.load(std::memory_order_acquire);
	if (++frame % 300 != 0 || describe == nullptr) {
		return;
	}
	std::map<std::string, uint64_t> ranges;
	for (const auto page: pages) {
		ranges[describe(page << 12u)]++;
	}
	std::vector<std::pair<uint64_t, std::string>> sorted;
	for (const auto& [name, count]: ranges) {
		sorted.emplace_back(count, name);
	}
	std::sort(sorted.rbegin(), sorted.rend());
	std::printf("Write-fault pages in one frame (%zu distinct):\n", pages.size());
	for (size_t i = 0; i < std::min<size_t>(sorted.size(), 12); i++) {
		std::printf("  %6" PRIu64 "  %s\n", sorted[i].first, sorted[i].second.c_str());
	}
}

static void PlotDistinctFaultPages() {
	static uint64_t       last  = 0;
	const auto            next  = g_fault_sample_next.load(std::memory_order_relaxed);
	const auto            count = std::min<uint64_t>(next - last, FAULT_SAMPLE_COUNT);
	std::vector<uint64_t> pages;
	pages.reserve(count);
	for (uint64_t i = next - count; i < next; i++) {
		pages.push_back(g_fault_samples[i % FAULT_SAMPLE_COUNT].load(std::memory_order_relaxed));
	}
	std::sort(pages.begin(), pages.end());
	pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
	TracyPlot("Distinct write-fault pages", static_cast<int64_t>(pages.size()));
	ReportFaultPages(pages);
	last = next;
}

void PlotCounters() {
	if (!tracy::ProfilerAvailable()) {
		return;
	}
	PlotDistinctFaultPages();
	static uint64_t last[static_cast<size_t>(Counter::Count)] = {};
	for (size_t i = 0; i < static_cast<size_t>(Counter::Count); i++) {
		const auto value = g_counters[i].load(std::memory_order_relaxed);
		TracyPlot(COUNTER_NAMES[i], static_cast<int64_t>(value - last[i]));
		last[i] = value;
	}
}

void Initialize() {
	if (Config::ProfilerEnabled() && !tracy::ProfilerAvailable()) {
		tracy::StartupProfiler();
		TracySetProgramName("KytyPS5");
		::printf("Tracy profiler enabled: client %d.%d.%d, protocol %u, "
		         "broadcast %u, connect to 127.0.0.1:8086\n",
		         tracy::Version::Major, tracy::Version::Minor, tracy::Version::Patch,
		         tracy::ProtocolVersion, tracy::BroadcastVersion);
	}
}

void Shutdown() {
	if (tracy::ProfilerAvailable()) {
		tracy::ShutdownProfiler();
	}
}

} // namespace Profiler
