#pragma once

#include "MDAA/Core/Assert.h"
#include "MDAA/Core/ProgressBar.h"
#include "MDAA/Core/Types.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <string_view>
#include <thread>
#include <vector>

namespace MDAA {

// The worker count every parallel helper of this library uses.
[[nodiscard]] inline u32 ThreadCount() {
    return std::max(1U, std::thread::hardware_concurrency());
}

// Splits [0, total) into one contiguous chunk per thread and runs body on
// every position. body(threadIndex, position) gets threadIndex in
// [0, ThreadCount()) and position in [0, total). The chunks are static, so a
// body may keep per thread state in an array indexed by threadIndex. Blocks
// until every worker is done and paints the bar from the calling thread.
template <typename Body>
void RunParallel(std::string_view label, u64 total, Body body) {
    const auto  count = static_cast<usize>(total);
    const auto  workers = static_cast<usize>(ThreadCount());
    const usize chunkSize = (count + workers - 1) / workers;

    std::atomic<u64> done {0};
    std::atomic<i32> pending {0};

    std::vector<std::jthread> pool;
    pool.reserve(workers);
    for (usize t = 0; t < workers; t++) {
        const usize begin = std::min(t * chunkSize, count);
        const usize end = std::min(begin + chunkSize, count);
        pending.fetch_add(1, std::memory_order_relaxed);
        pool.emplace_back([&done, &pending, &body, begin, end, t] {
            for (usize position = begin; position < end; position++) {
                body(static_cast<i32>(t), position);
                done.fetch_add(1, std::memory_order_relaxed);
            }
            pending.fetch_sub(1, std::memory_order_release);
        });
    }

    ProgressBar progress(label, total);
    while (pending.load(std::memory_order_acquire) > 0) {
        progress.Report(done.load(std::memory_order_relaxed));
        std::this_thread::sleep_for(std::chrono::milliseconds(ProgressBar::DefaultIntervalMilliseconds));
    }
    for (std::jthread &worker : pool) {
        worker.join();
    }
    progress.Finish();
}

template <typename Body>
void RunChunks(std::string_view label, i32 chunkCount, Body body) {
    MDAA_CHECKF(chunkCount > 0, "a parallel run needs at least one chunk, got {}", chunkCount);

    const auto total = static_cast<u64>(chunkCount);
    const auto workers = std::min(static_cast<usize>(ThreadCount()),
                                  static_cast<usize>(chunkCount));

    std::atomic<u64> next {0};
    std::atomic<u64> done {0};
    std::atomic<i32> pending {0};

    std::vector<std::jthread> pool;
    pool.reserve(workers);
    for (usize t = 0; t < workers; t++) {
        pending.fetch_add(1, std::memory_order_relaxed);
        pool.emplace_back([&next, &done, &pending, &body, total, t] {
            for (u64 index = next.fetch_add(1, std::memory_order_relaxed); index < total;
                 index = next.fetch_add(1, std::memory_order_relaxed)) {
                body(static_cast<i32>(t), static_cast<i32>(index));
                done.fetch_add(1, std::memory_order_relaxed);
            }
            pending.fetch_sub(1, std::memory_order_release);
        });
    }

    ProgressBar progress(label, total);
    while (pending.load(std::memory_order_acquire) > 0) {
        progress.Report(done.load(std::memory_order_relaxed));
        std::this_thread::sleep_for(std::chrono::milliseconds(ProgressBar::DefaultIntervalMilliseconds));
    }
    for (std::jthread &worker : pool) {
        worker.join();
    }
    progress.Finish();
}

} // namespace MDAA
