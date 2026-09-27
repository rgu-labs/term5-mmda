#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <limits>
#include <numbers>
#include <queue>
#include <span>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace {

using MDAA::f64;
using MDAA::i32;
using MDAA::u32;
using MDAA::u64;
using MDAA::u8;
using MDAA::usize;

using Vector = Eigen::Vector2d;
using Matrix = Eigen::Matrix2d;
using PointMatrix = Eigen::Matrix<f64, 2, Eigen::Dynamic>;

constexpr i32 VectorCount = 1'000'000;
constexpr i32 Dimension = 2;
constexpr f64 ComponentMin = -1.0;
constexpr f64 ComponentMax = 1.0;
constexpr f64 LambdaMin = 0.5;
constexpr f64 LambdaMax = 2.0;
constexpr u64 Seed = 88005553535;
constexpr u64 CrossCheckSeed = 0x5DEECE66DULL;
constexpr f64 DefaultTolerance = 1.0e-9;
constexpr f64 WindowSlack = 1.0e-6;
constexpr f64 WindowFloor = 1.0e-15;
constexpr i32 BestPairCount = 10;
constexpr i32 CrossCheckPairs = 100'000;

constexpr f64 TwoPi = 2.0 * std::numbers::pi;

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

struct Directions final {
    std::vector<f64> ByIndex;
    std::vector<f64> Sorted;
    std::vector<u32> Order;
};

i32 ThreadCount() {
    return static_cast<i32>(std::max(static_cast<u32>(1), std::thread::hardware_concurrency()));
}

f64 ParseTolerance(i32 argc, char **argv) {
    if (argc == 1) {
        return DefaultTolerance;
    }
    MDAA_CHECKF(argc == 2, "the only argument is the tolerance in radians, got {} arguments", argc - 1);

    const std::string_view text = argv[1];
    f64                    tolerance = 0.0;
    const auto             result = std::from_chars(text.data(), text.data() + text.size(), tolerance);
    MDAA_CHECKF(
        result.ec == std::errc() && result.ptr == text.data() + text.size() && tolerance > 0.0,
        "'{}' is not a positive tolerance in radians",
        text);
    return tolerance;
}

f64 NormalizeAngle(f64 angle) {
    f64 normalized = std::fmod(angle, TwoPi);
    if (normalized < 0.0) {
        normalized += TwoPi;
    }
    return normalized;
}

f64 AngleBetween(const Vector &x, const Vector &y) {
    return std::atan2(std::abs((x.x() * y.y()) - (x.y() * y.x())), x.dot(y));
}

f64 CosineByTransform(const Vector &x, const Vector &y) {
    return std::clamp(x.dot(y) / (x.norm() * y.norm()), -1.0, 1.0);
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

void GenerateVectors(std::span<f64> storage) {
    const i32 threadCount = ThreadCount();
    const i32 blockSize = (VectorCount + threadCount - 1) / threadCount;

    std::vector<std::jthread> workers;
    workers.reserve(threadCount);
    for (i32 t = 0; t < threadCount; t++) {
        const i32 begin = std::min(t * blockSize, VectorCount);
        const i32 end = std::min(begin + blockSize, VectorCount);
        workers.emplace_back([storage, begin, end] {
            MDAA::FillUniformVectorBlock(
                {.Seed = Seed,
                 .First = begin,
                 .Count = end - begin,
                 .Dimension = Dimension,
                 .ComponentMin = ComponentMin,
                 .ComponentMax = ComponentMax},
                storage.subspan(
                    static_cast<usize>(begin) * static_cast<usize>(Dimension),
                    static_cast<usize>(end - begin) * static_cast<usize>(Dimension)));
        });
    }
    for (std::jthread &worker : workers) {
        worker.join();
    }
}

Directions MakeDirections(const PointMatrix &points) {
    Directions dirs;
    dirs.ByIndex.resize(VectorCount);
    dirs.Sorted.resize(VectorCount);
    dirs.Order.resize(VectorCount);

    std::vector<std::pair<f64, u32>> entries;
    entries.reserve(VectorCount);
    for (i32 index = 0; index < VectorCount; index++) {
        const Vector point = points.col(static_cast<Eigen::Index>(index));
        dirs.ByIndex[index] = NormalizeAngle(std::atan2(point.y(), point.x()));
        entries.emplace_back(dirs.ByIndex[index], static_cast<u32>(index));
    }

    std::ranges::sort(entries);
    for (usize position = 0; position < entries.size(); position++) {
        dirs.Sorted[position] = entries[position].first;
        dirs.Order[position] = entries[position].second;
    }
    return dirs;
}

template <typename Visitor>
void ForEachInRange(const Directions &dirs, f64 low, f64 high, const Visitor &visit) {
    const auto first = std::ranges::lower_bound(dirs.Sorted, low);
    for (auto current = first; current != dirs.Sorted.end() && *current < high; current++) {
        visit(static_cast<u32>(current - dirs.Sorted.begin()));
    }
}

template <typename Visitor>
void ForEachInWindow(const Directions &dirs, f64 center, f64 window, const Visitor &visit) {
    const f64 low = center - window;
    const f64 high = center + window;
    if (low < 0.0) {
        ForEachInRange(dirs, low + TwoPi, TwoPi, visit);
        ForEachInRange(dirs, 0.0, high, visit);
    } else if (high > TwoPi) {
        ForEachInRange(dirs, low, TwoPi, visit);
        ForEachInRange(dirs, 0.0, high - TwoPi, visit);
    } else {
        ForEachInRange(dirs, low, high, visit);
    }
}

void SearchChunk(
    const PointMatrix &points,
    const Directions  &dirs,
    f64                target,
    f64                tolerance,
    f64                window,
    usize              begin,
    usize              end,
    AngleStats        &stats) {
    for (usize index = begin; index < end; index++) {
        const Vector point = points.col(static_cast<Eigen::Index>(index));
        const f64    phi = dirs.ByIndex[index];
        for (i32 sign = -1; sign <= 1; sign += 2) {
            const f64 center = NormalizeAngle(phi + (static_cast<f64>(sign) * target));
            ForEachInWindow(dirs, center, window, [&](u32 position) {
                const u32 other = dirs.Order[position];
                if (other <= static_cast<u32>(index)) {
                    return;
                }
                const f64 angle = AngleBetween(point, points.col(other));
                const f64 deviation = std::abs(angle - target);
                if (deviation > tolerance) {
                    return;
                }
                stats.Observe(
                    {.Left = static_cast<u32>(index),
                     .Right = other,
                     .Angle = angle,
                     .Deviation = deviation});
            });
        }
    }
}

AngleStats SearchAngle(const PointMatrix &points, const Directions &dirs, f64 target, f64 tolerance) {
    const f64 window = (tolerance * (1.0 + WindowSlack)) + WindowFloor;

    const i32 threadCount = ThreadCount();
    const i32 chunkSize = (VectorCount + threadCount - 1) / threadCount;

    std::vector<AngleStats>   blocks(threadCount);
    std::vector<std::jthread> workers;
    workers.reserve(threadCount);
    for (i32 t = 0; t < threadCount; t++) {
        const i32 begin = std::min(t * chunkSize, VectorCount);
        const i32 end = std::min(begin + chunkSize, VectorCount);
        workers.emplace_back([&blocks, &points, &dirs, target, tolerance, window, begin, end, t] {
            SearchChunk(
                points,
                dirs,
                target,
                tolerance,
                window,
                static_cast<usize>(begin),
                static_cast<usize>(end),
                blocks[t]);
        });
    }
    for (std::jthread &worker : workers) {
        worker.join();
    }

    AngleStats stats = blocks.front();
    for (i32 t = 1; t < threadCount; t++) {
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
    std::println("        {} = [{:13.8f}, {:13.8f}]", label, vector.x(), vector.y());
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
    MDAA_CHECKF(
        points.cwiseAbs().minCoeff() > 0.0,
        "a vector turned into a zero vector and lost its direction");

    const Directions dirs = MakeDirections(points);

    std::array<AngleStats, TargetAngles.size()> stats;
    std::array<f64, TargetAngles.size()>        seconds {};
    for (usize target = 0; target < TargetAngles.size(); target++) {
        MDAA::Timer angleTimer;
        angleTimer.Start();
        stats[target] = SearchAngle(points, dirs, TargetAngles[target], tolerance);
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

int main(i32 argc, char **argv) {
    Eigen::setNbThreads(1);
    MDAA_CHECKF(
        Dimension == 2,
        "the search sweeps the directions of a plane, it needs N = 2, got N = {}",
        Dimension);

    const f64 tolerance = ParseTolerance(argc, argv);
    const i32 threadCount = ThreadCount();

    std::vector<f64> storage(static_cast<usize>(VectorCount) * static_cast<usize>(Dimension));
    MDAA::Timer      generation;
    generation.Start();
    GenerateVectors(storage);
    generation.Stop();

    const Eigen::Map<const PointMatrix> vectors(storage.data(), Dimension, VectorCount);
    MDAA_CHECKF(
        vectors.cwiseAbs().minCoeff() > 0.0,
        "a null vector of the dataset has no direction, there is one");

    MDAA::Random               random(Seed);
    std::array<f64, Dimension> weights {};
    for (f64 &weight : weights) {
        weight = random.Uniform(LambdaMin, LambdaMax);
    }
    const Vector lambda = Vector::Map(weights.data());
    MDAA_CHECKF(
        lambda.minCoeff() > 0.0,
        "the weights of Lambda have to be positive, the smallest is {}",
        lambda.minCoeff());

    constexpr usize               matrixValues = static_cast<usize>(Dimension) * static_cast<usize>(Dimension);
    std::array<f64, matrixValues> matrixStorage {};
    MDAA::FillRandomSpdMatrix(
        {.Seed = Seed,
         .Size = Dimension,
         .ComponentMin = ComponentMin,
         .ComponentMax = ComponentMax},
        matrixStorage);
    const Matrix a = Eigen::Map<const Matrix, Eigen::RowMajor>(matrixStorage.data());
    MDAA_CHECKF(a.isApprox(a.transpose()), "M * transpose(M) + n * I has to stay symmetric");

    const Eigen::SelfAdjointEigenSolver<Matrix> solver(a);
    MDAA_CHECKF(solver.info() == Eigen::Success, "the eigen decomposition of A failed");
    const Vector &eigenvalues = solver.eigenvalues();
    MDAA_CHECKF(
        eigenvalues.minCoeff() > 0.0,
        "A has to be positive definite, the smallest eigenvalue is {}",
        eigenvalues.minCoeff());
    const Matrix root = solver.eigenvectors() * eigenvalues.array().sqrt().matrix().asDiagonal() *
                        solver.eigenvectors().transpose();
    MDAA_CHECKF(
        (root.transpose() * root - a).norm() < 1.0e-9,
        "A^1/2 squared has to be A, the gap is {}",
        (root.transpose() * root - a).norm());

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
        Matrix(Vector(lambda.array().sqrt()).asDiagonal()),
        root,
    };
    const std::array<Product, 3> products = {Product::Euclidean, Product::Lambda, Product::MatrixA};

    for (usize index = 0; index < products.size(); index++) {
        ReportProduct(products[index], transforms[index], context, vectors, tolerance);
    }

    return 0;
}
