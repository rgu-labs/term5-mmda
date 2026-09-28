#include "MDAA/Core/Assert.h"
#include "MDAA/Core/Parallel.h"
#include "MDAA/Core/Random.h"
#include "MDAA/Core/RandomSpdMatrix.h"
#include "MDAA/Core/RandomVectorSet.h"
#include "MDAA/Core/Sphere.h"
#include "MDAA/Core/Timer.h"
#include "MDAA/Core/Types.h"
#include "MDAA/LinearAlgebra/Matrix.h"
#include "MDAA/LinearAlgebra/PointSet.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <limits>
#include <numbers>
#include <queue>
#include <vector>

namespace {

using MDAA::f64;
using MDAA::i32;
using MDAA::RunParallel;
using MDAA::Sphere;
using MDAA::ThreadCount;
using MDAA::u32;
using MDAA::u64;
using MDAA::u8;
using MDAA::usize;

constexpr i32 VectorCount = 1'000'000;
constexpr i32 Dimension = 3;
constexpr f64 ComponentMin = -1.0;
constexpr f64 ComponentMax = 1.0;
constexpr f64 LambdaMin = 0.5;
constexpr f64 LambdaMax = 2.0;
constexpr u64 Seed = 88005553535;
constexpr u64 CrossCheckSeed = 0x5DEECE66DULL;
constexpr f64 Tolerance = 1.0e-9;
constexpr f64 ShellSlack = 1.0e-12;
constexpr i32 BestPairCount = 10;
constexpr i32 CrossCheckPairs = 100'000;
constexpr i32 LeafSize = 16;

using Vector = MDAA::ColumnVector<Dimension>;
using Matrix = MDAA::SquareMatrix<Dimension>;
using PointMatrix = MDAA::PointMatrix<Dimension>;

constexpr std::array<f64, 4> TargetAngles = {
    std::numbers::pi / 6.0,
    std::numbers::pi / 4.0,
    std::numbers::pi / 3.0,
    std::numbers::pi / 2.0,
};

constexpr std::array<const char *, 4> TargetNames = {"pi/6", "pi/4", "pi/3", "pi/2"};

enum class Product : u8 { Euclidean,
                          Lambda,
                          MatrixA };

struct AngleContext final {
    Vector Weights = Vector::Ones();
    Matrix Positive = Matrix::Identity();
};

struct VectorPair final {
    u32 Left = 0;
    u32 Right = 0;
    f64 Angle = 0.0;
    f64 Deviation = 0.0;
};

struct DeviationLess final {
    bool operator()(const VectorPair &lhs, const VectorPair &rhs) const {
        if (lhs.Deviation != rhs.Deviation) {
            return lhs.Deviation < rhs.Deviation;
        }
        if (lhs.Left != rhs.Left) {
            return lhs.Left < rhs.Left;
        }
        return lhs.Right < rhs.Right;
    }
};

using ClosestPairs = std::priority_queue<VectorPair, std::vector<VectorPair>, DeviationLess>;

struct AngleStats final {
    u64          Count = 0;
    f64          MinDeviation = std::numeric_limits<f64>::infinity();
    ClosestPairs Closest;

    void Consider(const VectorPair &pair) {
        if (static_cast<i32>(Closest.size()) < BestPairCount) {
            Closest.push(pair);
            return;
        }
        if (DeviationLess {}(pair, Closest.top())) {
            Closest.pop();
            Closest.push(pair);
        }
    }

    void Observe(const VectorPair &pair) {
        ++Count;
        MinDeviation = std::min(MinDeviation, pair.Deviation);
        Consider(pair);
    }

    void Merge(AngleStats &other) {
        Count += other.Count;
        MinDeviation = std::min(MinDeviation, other.MinDeviation);
        while (!other.Closest.empty()) {
            Consider(other.Closest.top());
            other.Closest.pop();
        }
    }
};

f64 AngleBetween(const Vector &x, const Vector &y) {
    const f64 cosine = std::clamp(x.dot(y), -1.0, 1.0);
    return std::atan2(std::sqrt(std::max(0.0, 1.0 - (cosine * cosine))), cosine);
}

f64 CosineByTransform(const Vector &x, const Vector &y) {
    return std::cos(AngleBetween(x, y));
}

f64 CosineByDot(const Vector &x, const Vector &y) {
    return std::clamp(x.dot(y) / (x.norm() * y.norm()), -1.0, 1.0);
}

f64 CosineByWeights(const Vector &x, const Vector &y, const Vector &weights) {
    const f64 numerator = (weights.array() * x.array() * y.array()).sum();
    const f64 denominator = std::sqrt((weights.array() * x.array().square()).sum()) *
                            std::sqrt((weights.array() * y.array().square()).sum());
    return std::clamp(numerator / denominator, -1.0, 1.0);
}

f64 CosineByMatrix(const Vector &x, const Vector &y, const Matrix &a) {
    const f64 numerator = x.dot(a * y);
    const f64 denominator = std::sqrt(x.dot(a * x)) * std::sqrt(y.dot(a * y));
    return std::clamp(numerator / denominator, -1.0, 1.0);
}

f64 CosineByDefinition(Product product, const AngleContext &context, const Vector &x, const Vector &y) {
    switch (product) {
        case Product::Euclidean:
            return CosineByDot(x, y);
        case Product::Lambda:
            return CosineByWeights(x, y, context.Weights);
        case Product::MatrixA:
            return CosineByMatrix(x, y, context.Positive);
    }
    return 0.0;
}

f64 AngleByDefinition(Product product, const AngleContext &context, const Vector &x, const Vector &y) {
    return std::acos(CosineByDefinition(product, context, x, y));
}

const char *ProductLabel(Product product) {
    switch (product) {
        case Product::Euclidean:
            return "(a) (x, y)";
        case Product::Lambda:
            return "(b) (x, y) with Lambda";
        case Product::MatrixA:
            return "(c) (x, y) with A";
    }
    return "";
}

void SearchChunk(
    const Sphere<Dimension> &sphere,
    f64                      target,
    f64                      tolerance,
    usize                    begin,
    usize                    end,
    AngleStats              &stats) {
    for (usize position = begin; position < end; position++) {
        const u32                  origin = static_cast<u32>(position);
        const std::span<const f64> query = sphere.Point(origin);
        sphere.ForEachInShell(query, target, tolerance, ShellSlack, [&](u32 other) {
            if (other <= origin) {
                return;
            }
            const f64 angle = Sphere<Dimension>::AngleBetween(query, sphere.Point(other));
            const f64 deviation = std::abs(angle - target);
            if (deviation > tolerance) {
                return;
            }
            stats.Observe(
                {.Left = origin,
                 .Right = other,
                 .Angle = angle,
                 .Deviation = deviation});
        });
    }
}

u64 CountByBruteForce(const Sphere<Dimension> &sphere, u32 origin, f64 target, f64 tolerance) {
    u64                        count = 0;
    const std::span<const f64> query = sphere.Point(origin);
    for (u32 other = origin + 1; other < sphere.Size(); other++) {
        const f64 angle = Sphere<Dimension>::AngleBetween(query, sphere.Point(other));
        if (std::abs(angle - target) <= tolerance) {
            ++count;
        }
    }
    return count;
}

void CheckCompleteness(const Sphere<Dimension> &sphere) {
    constexpr usize samples = 200;
    constexpr f64   testTolerance = 1.0e-3;

    const auto stride = static_cast<u32>(sphere.Size() / samples);
    const auto checks = TargetAngles.size() * samples;

    MDAA_CHECKF(stride > 0, "the dataset holds {} vectors, the check needs more", sphere.Size());

    RunParallel(
        "verify against brute force",
        static_cast<u64>(checks),
        [&sphere, stride, testTolerance](i32, usize position) {
            const auto which = position / samples;
            const auto sample = position % samples;
            const f64  target = TargetAngles[which];
            const u32  origin = static_cast<u32>(sample) * stride;

            AngleStats stats;
            SearchChunk(sphere, target, testTolerance, origin, origin + 1, stats);

            const u64 expected = CountByBruteForce(sphere, origin, target, testTolerance);
            MDAA_CHECKF(
                stats.Count == expected,
                "the tree found {} pairs for i = {}, brute force found {}, theta = {}",
                stats.Count,
                origin,
                expected,
                target);
        });
}

AngleStats SearchAngle(const Sphere<Dimension> &sphere, f64 target, f64 tolerance, const char *name) {
    const u32 threadCount = ThreadCount();

    std::vector<AngleStats> blocks(threadCount);
    RunParallel(
        std::format("search theta = {}", name),
        static_cast<u64>(VectorCount),
        [&blocks, &sphere, target, tolerance](i32 t, usize position) {
            SearchChunk(sphere, target, tolerance, position, position + 1, blocks[t]);
        });

    AngleStats stats = blocks.front();
    for (u32 t = 1; t < threadCount; t++) {
        stats.Merge(blocks[t]);
    }
    return stats;
}

std::vector<VectorPair> Drain(ClosestPairs &closest) {
    std::vector<VectorPair> pairs;
    pairs.reserve(closest.size());
    while (!closest.empty()) {
        pairs.push_back(closest.top());
        closest.pop();
    }
    std::ranges::reverse(pairs);
    return pairs;
}

struct CrossCheck final {
    f64 Cosine = 0.0;
    f64 Angle = 0.0;
    u64 Sampled = 0;
    u64 Found = 0;
};

CrossCheck RunCrossCheck(
    Product                                     product,
    const AngleContext                         &context,
    const PointMatrix                          &vectors,
    const PointMatrix                          &points,
    const std::vector<std::vector<VectorPair>> &closest) {
    MDAA::Random random(CrossCheckSeed);

    CrossCheck result;
    const auto compare = [&](u32 left, u32 right) {
        const f64 byDefinition = CosineByDefinition(product, context, vectors.col(left), vectors.col(right));
        const f64 byTransform = CosineByTransform(points.col(left), points.col(right));
        result.Cosine = std::max(result.Cosine, std::abs(byDefinition - byTransform));
        ++result.Sampled;
    };

    for (const std::vector<VectorPair> &pairs : closest) {
        for (const VectorPair &pair : pairs) {
            compare(pair.Left, pair.Right);
            const f64 byDefinition =
                AngleByDefinition(product, context, vectors.col(pair.Left), vectors.col(pair.Right));
            result.Angle = std::max(result.Angle, std::abs(byDefinition - pair.Angle));
            ++result.Found;
        }
    }
    for (i32 sample = 0; sample < CrossCheckPairs; sample++) {
        const u32 left = random.Next<u32>() % static_cast<u32>(VectorCount);
        const u32 right = random.Next<u32>() % static_cast<u32>(VectorCount);
        if (left != right) {
            compare(left, right);
        }
    }

    MDAA_CHECKF(
        result.Cosine < 1.0e-12,
        "the change of basis and the definition give different cosines, the worst gap is {}",
        result.Cosine);
    MDAA_CHECKF(
        result.Angle < 1.0e-9,
        "the change of basis and the definition give different angles, the worst gap is {}",
        result.Angle);
    return result;
}

void PrintVector(const char *label, const Vector &vector) {
    std::print("        {} = [", label);
    for (i32 d = 0; d < Dimension; d++) {
        std::print("{}{:13.8f}", d == 0 ? "" : ", ", vector[d]);
    }
    std::println("]");
}

void PrintMatrix(const char *label, const Matrix &matrix) {
    std::println("{} = {}x{}", label, matrix.rows(), matrix.cols());
    for (Eigen::Index row = 0; row < matrix.rows(); row++) {
        std::print("  [");
        for (Eigen::Index column = 0; column < matrix.cols(); column++) {
            std::print("{}{:12.6f}", column == 0 ? "" : " ", matrix(row, column));
        }
        std::println("]");
    }
}

void ReportProduct(
    Product             product,
    const Matrix       &transform,
    const AngleContext &context,
    const PointMatrix  &vectors,
    f64                 tolerance) {
    MDAA::Timer timer;
    timer.Start();

    const PointMatrix points = transform * vectors;

    Sphere<Dimension> sphere;
    sphere.Build(
        {points.data(), static_cast<usize>(points.size())},
        VectorCount,
        LeafSize);
    CheckCompleteness(sphere);

    std::array<AngleStats, TargetAngles.size()> stats;
    std::array<f64, TargetAngles.size()>        seconds {};
    for (usize target = 0; target < TargetAngles.size(); target++) {
        MDAA::Timer angleTimer;
        angleTimer.Start();
        stats[target] =
            SearchAngle(sphere, TargetAngles[target], tolerance, TargetNames[target]);
        angleTimer.Stop();
        seconds[target] = angleTimer.ElapsedSeconds();
    }

    timer.Stop();

    std::vector<std::vector<VectorPair>> closest;
    closest.reserve(TargetAngles.size());
    for (AngleStats &angleStats : stats) {
        closest.push_back(Drain(angleStats.Closest));
    }

    const CrossCheck check = RunCrossCheck(product, context, vectors, points, closest);

    std::println("\n{}, searched in {:.2f} s", ProductLabel(product), timer.ElapsedSeconds());
    std::println("    {:<8} {:>8} {:>8} {:>14} {:>8}", "theta", "value", "pairs", "min dev", "time");
    for (usize target = 0; target < TargetAngles.size(); target++) {
        std::println(
            "    {:<8} {:>8.5f} {:>8} {:>14.3e} {:>7.2f} s",
            TargetNames[target],
            TargetAngles[target],
            stats[target].Count,
            stats[target].MinDeviation,
            seconds[target]);
    }

    for (usize target = 0; target < TargetAngles.size(); target++) {
        if (closest[target].empty()) {
            std::println("\n    no pair of the dataset is within the tolerance of {}", TargetNames[target]);
            continue;
        }
        std::println(
            "\n    the {} closest pairs to {} = {:.16f}",
            static_cast<i32>(closest[target].size()),
            TargetNames[target],
            TargetAngles[target]);
        for (const VectorPair &pair : closest[target]) {
            const f64 angle =
                AngleByDefinition(product, context, vectors.col(pair.Left), vectors.col(pair.Right));
            std::println("      i = {}, j = {}", pair.Left, pair.Right);
            PrintVector("x", vectors.col(pair.Left));
            PrintVector("y", vectors.col(pair.Right));
            std::println(
                "        angle = {:.16f}   deviation = {:.3e}",
                angle,
                std::abs(angle - TargetAngles[target]));
        }
    }

    std::println(
        "\n    cross check, max |cos by the definition - cos after the change of basis| = {:.3e} over {} pairs",
        check.Cosine,
        check.Sampled);
    std::println(
        "    cross check, max |angle by the definition - angle after the change of basis| ="
        " {:.3e} over the {} found pairs",
        check.Angle,
        check.Found);
}

} // namespace

int main() {
    Eigen::setNbThreads(1);

    const f64 tolerance = Tolerance;
    const u32 threadCount = ThreadCount();

    std::vector<f64> storage(static_cast<usize>(VectorCount) * static_cast<usize>(Dimension));
    MDAA::Timer      generation;
    generation.Start();
    MDAA::FillUniformVectorSet(
        {.Seed = Seed,
         .First = 0,
         .Count = VectorCount,
         .Dimension = Dimension,
         .ComponentMin = ComponentMin,
         .ComponentMax = ComponentMax},
        storage);
    generation.Stop();

    const Eigen::Map<const PointMatrix> vectors(storage.data(), Dimension, VectorCount);
    MDAA_CHECKF(
        vectors.colwise().norm().minCoeff() > 0.0,
        "a null vector of the dataset has no direction, there is one");

    MDAA::Random               random(Seed);
    std::array<f64, Dimension> weights {};
    for (f64 &weight : weights) {
        weight = random.Uniform(LambdaMin, LambdaMax);
    }
    const Vector lambda = Vector::Map(weights.data());
    const Matrix lambdaRoot = MDAA::DiagonalSquareRoot(lambda);

    std::array<f64, static_cast<usize>(Dimension) * static_cast<usize>(Dimension)> matrixStorage {};
    MDAA::FillRandomSpdMatrix(
        {.Seed = Seed,
         .Size = Dimension,
         .ComponentMin = ComponentMin,
         .ComponentMax = ComponentMax},
        matrixStorage);
    const Matrix a = Eigen::Map<const Matrix, Eigen::RowMajor>(matrixStorage.data());

    const Matrix root = MDAA::SpdMatrixRoot(a);

    std::println("Lab01.02");
    std::println(
        "{} vectors, N = {}, seed = {}, {} threads, tolerance = {:.3e} rad",
        VectorCount,
        Dimension,
        Seed,
        threadCount,
        tolerance);
    std::println("vectors generated in {:.2f} s", generation.ElapsedSeconds());
    std::println();
    PrintMatrix("Lambda", lambda.asDiagonal());
    std::println();
    PrintMatrix("A", a);
    std::println();
    PrintMatrix("A^1/2", root);
    std::println();
    std::println(
        "the vectors are continuous and random, so an exact angle has probability 0,"
        " every pair below is within the tolerance");

    const AngleContext          context {.Weights = lambda, .Positive = a};
    const std::array<Matrix, 3> transforms = {
        Matrix::Identity(),
        lambdaRoot,
        root,
    };
    const std::array<Product, 3> products = {Product::Euclidean, Product::Lambda, Product::MatrixA};

    for (usize index = 0; index < products.size(); index++) {
        ReportProduct(products[index], transforms[index], context, vectors, tolerance);
    }

    return 0;
}
