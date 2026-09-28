#pragma once

#include "MDAA/Core/Assert.h"
#include "MDAA/Core/KdTree.h"
#include "MDAA/Core/Types.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <span>
#include <vector>

namespace MDAA {

template <i32 Dimension>
class Sphere final {
    static_assert(Dimension > 0, "a sphere needs a positive dimension");

  public:
    void Build(std::span<const f64> points, usize count, i32 leafSize) {
        constexpr auto width = static_cast<usize>(Dimension);
        MDAA_CHECKF(count > 0, "the sphere needs points, got {}", count);
        MDAA_CHECKF(
            points.size() >= count * width,
            "{} points of {} components need {} values, the buffer holds {}",
            count,
            width,
            count * width,
            points.size());

        std::vector<f64> unit(count * width);
        for (usize index = 0; index < count; index++) {
            const std::span<const f64> point = points.subspan(index * width, width);

            f64 sum = 0.0;
            for (usize d = 0; d < width; d++) {
                sum += point[d] * point[d];
            }
            const f64 norm = std::sqrt(sum);
            MDAA_CHECKF(
                norm > 0.0,
                "the point {} turned into a zero vector and lost its direction",
                index);

            const std::span<f64> target = {unit.data() + (index * width), width};
            for (usize d = 0; d < width; d++) {
                target[d] = point[d] / norm;
            }
        }

        m_Tree.Build(unit, count, leafSize);
    }

    [[nodiscard]] usize Size() const {
        return m_Tree.Size();
    }

    [[nodiscard]] std::span<const f64> Point(usize index) const {
        return m_Tree.Point(index);
    }

    [[nodiscard]] f64 SquaredDistanceTo(usize index, std::span<const f64> query) const {
        return m_Tree.SquaredDistanceTo(index, query);
    }

    template <typename Visitor>
    void ForEachInShell(std::span<const f64> query, f64 angle, f64 tolerance, f64 slack, const Visitor &visit) const {
        MDAA_CHECKF(
            angle - tolerance >= 0.0,
            "the window of {} reaches below zero",
            angle);
        MDAA_CHECKF(
            angle + tolerance <= std::numbers::pi,
            "the window of {} reaches past pi, the chord is no longer monotone",
            angle);

        m_Tree.ForEachInShell(
            query,
            SquaredChord(angle - tolerance) - slack,
            SquaredChord(angle + tolerance) + slack,
            visit);
    }

    [[nodiscard]] static f64 SquaredChord(f64 angle) {
        return 2.0 - (2.0 * std::cos(angle));
    }

    [[nodiscard]] static f64 AngleBetween(std::span<const f64> x, std::span<const f64> y) {
        constexpr auto width = static_cast<usize>(Dimension);
        MDAA_CHECKF(
            x.size() >= width && y.size() >= width,
            "an angle needs two vectors of {} components, got {} and {}",
            width,
            x.size(),
            y.size());

        f64 cosine = 0.0;
        for (usize d = 0; d < width; d++) {
            cosine += x[d] * y[d];
        }
        cosine = std::clamp(cosine, -1.0, 1.0);
        return std::atan2(std::sqrt(std::max(0.0, 1.0 - (cosine * cosine))), cosine);
    }

  private:
    KdTree<Dimension> m_Tree;
};

} // namespace MDAA
