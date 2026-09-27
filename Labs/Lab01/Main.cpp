#include <Eigen/Core>

#include <algorithm>
#include <limits>
#include <print>
#include <thread>

namespace {

using MDAA::f64;
using MDAA::i32;
using MDAA::u32;
using MDAA::u64;

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

constexpr i32 VectorCount = 1'000'000;
constexpr i32 Dimension = 10;
constexpr i32 HalfPower = 2;
constexpr f64 ComponentMin = -1.0;
constexpr f64 ComponentMax = 1.0;
constexpr u64 Seed = 88005553535;

struct Extrema final {
    f64    Min = std::numeric_limits<f64>::infinity();
    f64    Max = -std::numeric_limits<f64>::infinity();
    Vector MinVector = Vector::Zero(Dimension);
    Vector MaxVector = Vector::Zero(Dimension);

    void Add(f64 value, const Vector &vector) {
        if (value < Min) {
            Min = value;
            MinVector = vector;
        }
        if (value > Max) {
            Max = value;
            MaxVector = vector;
        }
    }

    void Merge(const Extrema &other) {
        if (other.Min < Min) {
            Min = other.Min;
            MinVector = other.MinVector;
        }
        if (other.Max > Max) {
            Max = other.Max;
            MaxVector = other.MaxVector;
        }
    }
};

struct Norms final {
    Extrema Infinity = {};
    Extrema L1 = {};
    Extrema L2L = {};
    Extrema SPDMatrix = {};

    void Merge(const Norms &other) {
        Infinity.Merge(other.Infinity);
        L1.Merge(other.L1);
        L2L.Merge(other.L2L);
        SPDMatrix.Merge(other.SPDMatrix);
    }
};

Matrix CreateRandomSPDMatrix(i32 n, u64 seed) {
    MDAA_CHECKF(n > 0, "a matrix needs a positive size, got n = {}", n);

    MDAA::Random random(seed);

    Matrix basis(n, n);
    for (i32 row = 0; row < n; row++) {
        for (i32 column = 0; column < n; column++) {
            basis(row, column) = random.Uniform(ComponentMin, ComponentMax);
        }
    }

    Matrix a = basis * basis.transpose();
    a.diagonal().array() += static_cast<f64>(n);

    MDAA_CHECKF(a.isApprox(a.transpose()), "M * transpose(M) + n * I has to stay symmetric");
    return a;
}

void Scan(i32 begin, i32 end, const Matrix &a, Norms &norms) {
    MDAA_CHECKF(begin <= end, "the block [{}, {}) is empty or reversed", begin, end);
    MDAA_CHECKF(
        a.rows() == Dimension && a.cols() == Dimension,
        "expected a {}x{} matrix, got {}x{}",
        Dimension,
        Dimension,
        a.rows(),
        a.cols());

    MDAA::Random random(Seed + static_cast<u64>(begin));
    Vector       x = Vector::Zero(Dimension);

    for (i32 index = begin; index < end; index++) {
        for (f64 &value : x) {
            value = random.Uniform(ComponentMin, ComponentMax);
        }

        norms.Infinity.Add(x.cwiseAbs().maxCoeff(), x);
        norms.L1.Add(x.lpNorm<1>(), x);
        norms.L2L.Add(x.lpNorm<2 * HalfPower>(), x);
        norms.SPDMatrix.Add((a * x).dot(x), x);
    }
}

void PrintVector(const char *label, const Vector &vector) {
    std::print("  {} = [", label);
    for (Eigen::Index i = 0; i < vector.size(); i++) {
        std::print("{}{:.6f}", i == 0 ? "" : ", ", vector[i]);
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

void Report(const char *symbol, const Extrema &extrema) {
    MDAA_CHECK(extrema.Min <= extrema.Max);

    std::println("{}   min = {:.12g}   max = {:.12g}", symbol, extrema.Min, extrema.Max);
    PrintVector("shortest", extrema.MinVector);
    PrintVector("longest ", extrema.MaxVector);
    std::println();
}

} // namespace

int main() {
    Eigen::setNbThreads(1);

    const Matrix a = CreateRandomSPDMatrix(Dimension, Seed);

    PrintMatrix("A", a);
    std::println();

    const i32 threadCount = static_cast<i32>(std::max(u32(1), std::thread::hardware_concurrency()));
    const i32 blockSize = (VectorCount + threadCount - 1) / threadCount;

    MDAA::Timer timer;
    timer.Start();

    std::vector<Norms>        blocks(threadCount);
    std::vector<std::jthread> workers;
    workers.reserve(threadCount);
    for (i32 t = 0; t < threadCount; t++) {
        const i32 begin = std::min(t * blockSize, VectorCount);
        const i32 end = std::min(begin + blockSize, VectorCount);
        workers.emplace_back([&blocks, &a, t, begin, end] { Scan(begin, end, a, blocks[t]); });
    }
    for (std::jthread &worker : workers) {
        worker.join();
    }

    Norms norms = blocks.front();
    for (i32 t = 1; t < threadCount; t++) {
        norms.Merge(blocks[t]);
    }

    timer.Stop();

    MDAA_CHECKF(
        norms.Infinity.Min != std::numeric_limits<f64>::infinity(),
        "nothing was scanned, {} blocks of {} over {} vectors",
        threadCount,
        blockSize,
        VectorCount);
    MDAA_CHECKF(
        norms.SPDMatrix.Min > 0.0,
        "every x^T A x has to be positive for a positive definite A, the smallest is {}",
        norms.SPDMatrix.Min);

    std::println(
        "{} vectors of N = {}, 2l = {}, {} threads, {:.2f} s\n",
        VectorCount,
        Dimension,
        2 * HalfPower,
        threadCount,
        timer.ElapsedSeconds());

    Report("||X||_inf     ", norms.Infinity);
    Report("||X||_1       ", norms.L1);
    Report("||X||_2l      ", norms.L2L);
    Report("||X||_A       ", norms.SPDMatrix);

    return 0;
}
