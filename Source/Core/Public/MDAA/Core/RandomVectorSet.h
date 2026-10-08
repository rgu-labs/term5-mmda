#pragma once

#include "MDAA/Core/Assert.h"
#include "MDAA/Core/Parallel.h"
#include "MDAA/Core/Random.h"
#include "MDAA/Core/Types.h"

#include <span>

namespace MDAA {

struct UniformVectorBlock final {
    u64 Seed = 0;
    i32 First = 0;
    i32 Count = 0;
    i32 Dimension = 0;
    f64 ComponentMin = -1.0;
    f64 ComponentMax = 1.0;

    usize Values() const {
        return static_cast<usize>(Count) * static_cast<usize>(Dimension);
    }
};

inline void FillUniformVectorBlock(const UniformVectorBlock &block, std::span<f64> out) {
    MDAA_CHECKF(block.Dimension > 0, "a vector needs a positive dimension, got {}", block.Dimension);
    MDAA_CHECKF(block.Count >= 0, "a block needs a non negative count, got {}", block.Count);
    MDAA_CHECKF(block.First >= 0, "a block starts at a non negative index, got {}", block.First);
    MDAA_CHECKF(
        block.ComponentMin < block.ComponentMax,
        "the component range [{}, {}] is empty",
        block.ComponentMin,
        block.ComponentMax);
    MDAA_CHECKF(
        out.size() >= block.Values(),
        "the block holds {} values, the buffer holds {}",
        block.Values(),
        out.size());

    const auto dimension = static_cast<usize>(block.Dimension);

    Random random(block.Seed + static_cast<u64>(block.First));
    for (i32 index = 0; index < block.Count; index++) {
        for (usize component = 0; component < dimension; component++) {
            const usize offset = (static_cast<usize>(index) * dimension) + component;
            out[offset] = random.Uniform(block.ComponentMin, block.ComponentMax);
        }
    }
}

// The parallel form of FillUniformVectorBlock for a whole set of vectors.
// Vector i is always seeded with Seed + First + i on its own, so the values
// come out the same whatever ThreadCount() is and however the work was split.
// Do not seed once per chunk instead: mt19937_64 is not a counter based
// generator and a chunk seeded stream does not reproduce the per vector one.
inline void FillUniformVectorSet(const UniformVectorBlock &block, std::span<f64> out) {
    MDAA_CHECKF(block.Dimension > 0, "a vector needs a positive dimension, got {}", block.Dimension);
    MDAA_CHECKF(block.Count >= 0, "a set needs a non negative count, got {}", block.Count);
    MDAA_CHECKF(block.First >= 0, "a set starts at a non negative index, got {}", block.First);
    MDAA_CHECKF(
        block.ComponentMin < block.ComponentMax,
        "the component range [{}, {}] is empty",
        block.ComponentMin,
        block.ComponentMax);
    MDAA_CHECKF(
        out.size() >= block.Values(),
        "the set holds {} values, the buffer holds {}",
        block.Values(),
        out.size());
    if (block.Count == 0) {
        return;
    }

    const auto width = static_cast<usize>(block.Dimension);
    RunParallel("generate vectors", static_cast<u64>(block.Count), [block, out, width](i32, usize position) {
        FillUniformVectorBlock(
            {
                .Seed = block.Seed,
                .First = block.First + static_cast<i32>(position),
                .Count = 1,
                .Dimension = block.Dimension,
                .ComponentMin = block.ComponentMin,
                .ComponentMax = block.ComponentMax,
            },
            out.subspan(position * width, width));
    });
}

} // namespace MDAA
