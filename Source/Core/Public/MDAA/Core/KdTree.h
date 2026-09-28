#pragma once

#include "MDAA/Core/Assert.h"
#include "MDAA/Core/ProgressBar.h"
#include "MDAA/Core/Types.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace MDAA {

template <i32 Dimension>
class KdTree final {
    static_assert(Dimension > 0, "a kd tree needs a positive dimension");

  public:
    static constexpr usize Width = static_cast<usize>(Dimension);

    void Build(std::span<const f64> points, usize count, i32 leafSize) {
        MDAA_CHECKF(count > 0, "the tree needs points, got {}", count);
        MDAA_CHECKF(leafSize > 0, "a leaf has to hold at least one point, got {}", leafSize);
        MDAA_CHECKF(
            count <= static_cast<usize>(std::numeric_limits<u32>::max()),
            "the tree indexes at most {} points, got {}",
            std::numeric_limits<u32>::max(),
            count);
        MDAA_CHECKF(
            points.size() >= count * Width,
            "{} points of {} components need {} values, the buffer holds {}",
            count,
            Width,
            count * Width,
            points.size());

        m_Points.assign(points.begin(), points.begin() + static_cast<std::ptrdiff_t>(count * Width));
        m_Count = count;

        m_Order.resize(count);
        for (usize index = 0; index < count; index++) {
            m_Order[index] = static_cast<u32>(index);
        }

        m_Nodes.clear();
        m_Nodes.reserve((count / static_cast<usize>(leafSize)) + 2);

        const u64 total = count * Levels(count, leafSize);
        m_Scanned = 0;
        ProgressBar progress("build the tree", total);
        m_Progress = &progress;
        BuildNode(0, static_cast<u32>(count), leafSize);
        m_Progress = nullptr;
        progress.Finish();
    }

    [[nodiscard]] usize Size() const {
        return m_Count;
    }

    [[nodiscard]] std::span<const f64> Point(usize index) const {
        MDAA_CHECKF(index < m_Count, "no point {}, there are {}", index, m_Count);
        return {m_Points.data() + (index * Width), Width};
    }

    [[nodiscard]] f64 SquaredDistanceTo(usize index, std::span<const f64> query) const {
        const std::span<const f64> point = Point(index);
        f64                        sum = 0.0;
        for (usize d = 0; d < Width; d++) {
            const f64 delta = query[d] - point[d];
            sum += delta * delta;
        }
        return sum;
    }

    template <typename Visitor>
    void ForEachInShell(std::span<const f64> query, f64 inner, f64 outer, const Visitor &visit) const {
        MDAA_CHECKF(inner <= outer, "the shell [{}, {}] is empty or reversed", inner, outer);
        if (m_Nodes.empty()) {
            return;
        }
        QueryNode(0, query, inner, outer, visit);
    }

  private:
    struct Node final {
        std::array<f64, Width> Min {};
        std::array<f64, Width> Max {};
        u32                    Begin = 0;
        u32                    End = 0;
        i32                    Left = -1;
        i32                    Right = -1;
    };

    [[nodiscard]] static u64 Levels(usize count, i32 leafSize) {
        const u64 leaves = static_cast<u64>(leafSize);
        u64       span = count;
        u64       depth = 1;
        while (span > leaves) {
            span /= 2;
            depth++;
        }
        return depth;
    }

    [[nodiscard]] f64 Coordinate(u32 index, usize d) const {
        return m_Points[(static_cast<usize>(index) * Width) + d];
    }

    // NOLINTNEXTLINE(misc-no-recursion) the depth is log2(count / leafSize), a kd tree is recursive by nature
    i32 BuildNode(u32 begin, u32 end, i32 leafSize) {
        const i32 self = static_cast<i32>(m_Nodes.size());
        m_Nodes.emplace_back();

        for (usize d = 0; d < Width; d++) {
            m_Nodes[static_cast<usize>(self)].Min[d] = std::numeric_limits<f64>::infinity();
            m_Nodes[static_cast<usize>(self)].Max[d] = -std::numeric_limits<f64>::infinity();
        }
        for (u32 position = begin; position < end; position++) {
            for (usize d = 0; d < Width; d++) {
                const f64 value = Coordinate(m_Order[position], d);
                m_Nodes[static_cast<usize>(self)].Min[d] =
                    std::min(m_Nodes[static_cast<usize>(self)].Min[d], value);
                m_Nodes[static_cast<usize>(self)].Max[d] =
                    std::max(m_Nodes[static_cast<usize>(self)].Max[d], value);
            }
            ++m_Scanned;
            if (m_Progress != nullptr && (m_Scanned & 0x3FFU) == 0U) {
                m_Progress->Report(m_Scanned);
            }
        }
        m_Nodes[static_cast<usize>(self)].Begin = begin;
        m_Nodes[static_cast<usize>(self)].End = end;

        if (end - begin <= static_cast<u32>(leafSize)) {
            return self;
        }

        usize axis = 0;
        f64   widest = -1.0;
        for (usize d = 0; d < Width; d++) {
            const f64 spread = m_Nodes[static_cast<usize>(self)].Max[d] -
                               m_Nodes[static_cast<usize>(self)].Min[d];
            if (spread > widest) {
                widest = spread;
                axis = d;
            }
        }
        if (!(widest > 0.0)) {
            return self;
        }

        const u32  middle = begin + ((end - begin) / 2);
        const auto split = static_cast<std::ptrdiff_t>(axis);
        std::nth_element(
            m_Order.begin() + static_cast<std::ptrdiff_t>(begin),
            m_Order.begin() + static_cast<std::ptrdiff_t>(middle),
            m_Order.begin() + static_cast<std::ptrdiff_t>(end),
            [this, split](u32 lhs, u32 rhs) {
                return Coordinate(lhs, static_cast<usize>(split)) <
                       Coordinate(rhs, static_cast<usize>(split));
            });

        const i32 left = BuildNode(begin, middle, leafSize);
        const i32 right = BuildNode(middle, end, leafSize);
        m_Nodes[static_cast<usize>(self)].Left = left;
        m_Nodes[static_cast<usize>(self)].Right = right;
        return self;
    }

    template <typename Visitor>
    // NOLINTNEXTLINE(misc-no-recursion) the depth is log2(count / leafSize), a kd tree is recursive by nature
    void QueryNode(i32 index, std::span<const f64> query, f64 inner, f64 outer, const Visitor &visit) const {
        const Node &node = m_Nodes[static_cast<usize>(index)];

        f64 nearest = 0.0;
        f64 farthest = 0.0;
        for (usize d = 0; d < Width; d++) {
            const f64 value = query[d];
            const f64 below = std::max(0.0, node.Min[d] - value);
            const f64 above = std::max(0.0, value - node.Max[d]);
            nearest += (below * below) + (above * above);

            const f64 reach = std::max(std::abs(value - node.Min[d]), std::abs(value - node.Max[d]));
            farthest += reach * reach;
        }
        if (nearest > outer || farthest < inner) {
            return;
        }

        if (node.Left < 0) {
            for (u32 position = node.Begin; position < node.End; position++) {
                const u32 other = m_Order[position];
                const f64 chord = SquaredDistanceTo(other, query);
                if (chord >= inner && chord <= outer) {
                    visit(other);
                }
            }
            return;
        }

        QueryNode(node.Left, query, inner, outer, visit);
        QueryNode(node.Right, query, inner, outer, visit);
    }

    std::vector<f64>  m_Points;
    std::vector<u32>  m_Order;
    std::vector<Node> m_Nodes;
    usize             m_Count = 0;
    u64               m_Scanned = 0;
    ProgressBar      *m_Progress = nullptr;
};

} // namespace MDAA
