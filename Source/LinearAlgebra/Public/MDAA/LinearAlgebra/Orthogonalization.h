#pragma once

#include "MDAA/Core/Types.h"

#include <utility>
#include <vector>

namespace MDAA {

template <typename Vector, typename InnerProduct>
std::vector<Vector>
Orthogonalize(const std::vector<Vector> &vectors, InnerProduct innerProduct) {
    constexpr f64 tolerance = 1.0e-12;

    std::vector<Vector> basis;

    for (const Vector &vector : vectors) {
        Vector    orthogonal = vector;
        const f64 originalSquared = innerProduct(vector, vector);

        for (const Vector &previous : basis) {
            const f64 previousSquared = innerProduct(previous, previous);

            if (previousSquared <= 0.0) {
                continue;
            }

            const f64 projection = innerProduct(orthogonal, previous) / previousSquared;
            orthogonal -= projection * previous;
        }

        if (innerProduct(orthogonal, orthogonal) <= tolerance * originalSquared) {
            orthogonal *= 0.0;
        }

        basis.push_back(std::move(orthogonal));
    }

    return basis;
}

} // namespace MDAA
