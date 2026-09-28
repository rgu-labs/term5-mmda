#pragma once

#include "MDAA/Core/Assert.h"
#include "MDAA/Core/Random.h"
#include "MDAA/Core/Types.h"

#include <span>
#include <vector>

namespace MDAA {

struct RandomSpdMatrix final {
    u64 Seed = 0;
    i32 Size = 0;
    f64 ComponentMin = -1.0;
    f64 ComponentMax = 1.0;

    usize Values() const {
        return static_cast<usize>(Size) * static_cast<usize>(Size);
    }
};

inline void FillRandomSpdMatrix(const RandomSpdMatrix &params, std::span<f64> out) {
    MDAA_CHECKF(params.Size > 0, "a matrix needs a positive size, got {}", params.Size);
    MDAA_CHECKF(
        params.ComponentMin < params.ComponentMax,
        "the component range [{}, {}] is empty",
        params.ComponentMin,
        params.ComponentMax);
    MDAA_CHECKF(
        out.size() >= params.Values(),
        "the matrix holds {} values, the buffer holds {}",
        params.Values(),
        out.size());

    const i32  size = params.Size;
    const auto width = static_cast<usize>(size);

    Random           random(params.Seed);
    std::vector<f64> basis(params.Values());
    for (f64 &value : basis) {
        value = random.Uniform(params.ComponentMin, params.ComponentMax);
    }

    for (i32 row = 0; row < size; row++) {
        for (i32 column = 0; column < size; column++) {
            f64 product = 0.0;
            for (i32 k = 0; k < size; k++) {
                product += basis[(static_cast<usize>(row) * width) + static_cast<usize>(k)] *
                           basis[(static_cast<usize>(column) * width) + static_cast<usize>(k)];
            }
            out[(static_cast<usize>(row) * width) + static_cast<usize>(column)] =
                row == column ? product + static_cast<f64>(size) : product;
        }
    }
}

} // namespace MDAA
