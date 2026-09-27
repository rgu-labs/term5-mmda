#include <Eigen/Core>

#include <algorithm>
#include <limits>
#include <print>
#include <thread>

namespace {

using MDAA::f64;
using MDAA::i32;
using MDAA::u64;

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

constexpr i32 VectorCount = 1'000'000;
constexpr i32 Dimension = 100;
constexpr i32 HalfPower = 2; // the l of the ||X||_{2l} norm
constexpr f64 ComponentMin = -1.0;
constexpr f64 ComponentMax = 1.0;
constexpr u64 Seed = 88005553535;

// Smallest and largest value seen, with the vector that produced it
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
        Add(other.Min, other.MinVector);
        Add(other.Max, other.MaxVector);
    }
};

struct Norms final {
    Extrema infinity = {};
    Extrema l1 = {};
    Extrema lp = {};
    Extrema quadratic = {};

    void Merge(const Norms &other) {
        infinity.Merge(other.infinity);
        l1.Merge(other.l1);
        lp.Merge(other.lp);
        quadratic.Merge(other.quadratic);
    }
};

Matrix MakeSymmetricPositiveDefinite(i32 n, u64 seed) {
    MDAA::Random random(seed);

    Matrix basis(n, n);
    for (i32 row = 0; row < n; ++row) {
        for (i32 column = 0; column < n; ++column) {
            basis(row, column) = random.Uniform(ComponentMin, ComponentMax);
        }
    }

    Matrix a = basis * basis.transpose();
    a.diagonal().array() += static_cast<f64>(n);
    return a;
}

void Scan(i32 begin, i32 end, const Matrix &a, Norms &norms) {
    MDAA::Random random(Seed + static_cast<u64>(begin));
    Vector       x = Vector::Zero(Dimension);

    for (i32 index = begin; index < end; ++index) {
        for (f64 &value : x) {
            value = random.Uniform(ComponentMin, ComponentMax);
        }

        norms.infinity.Add(x.cwiseAbs().maxCoeff(), x);
        norms.l1.Add(x.lpNorm<1>(), x);
        norms.lp.Add(x.lpNorm<2 * HalfPower>(), x);
        norms.quadratic.Add((a * x).dot(x), x);
    }
}

void PrintVector(const char *label, const Vector &vector) {
    std::print("  {} = [", label);
    for (Eigen::Index i = 0; i < vector.size(); ++i) {
        std::print("{}{:.6f}", i == 0 ? "" : ", ", vector[i]);
    }
    std::println("]");
}

void Report(const char *symbol, const Extrema &extrema) {
    std::println("{}   min = {:.12g}   max = {:.12g}", symbol, extrema.Min, extrema.Max);
    PrintVector("shortest", extrema.MinVector);
    PrintVector("longest ", extrema.MaxVector);
    std::println();
}

} // namespace

int main() {
    Eigen::setNbThreads(1);

    const Matrix a = MakeSymmetricPositiveDefinite(Dimension, Seed);

    const i32 threadCount = static_cast<i32>(std::max(1u, std::thread::hardware_concurrency()));
    const i32 blockSize = (VectorCount + threadCount - 1) / threadCount;

    MDAA::Timer timer;
    timer.Start();

    std::vector<Norms>       blocks(threadCount);
    std::vector<std::thread> workers;
    workers.reserve(threadCount);
    for (i32 t = 0; t < threadCount; ++t) {
        const i32 begin = std::min(t * blockSize, VectorCount);
        const i32 end = std::min(begin + blockSize, VectorCount);
        workers.emplace_back([&blocks, &a, t, begin, end] { Scan(begin, end, a, blocks[t]); });
    }
    for (std::thread &worker : workers) {
        worker.join();
    }

    Norms norms = blocks.front();
    for (i32 t = 1; t < threadCount; ++t) {
        norms.Merge(blocks[t]);
    }

    timer.Stop();

    std::println(
        "{} vectors of N = {}, 2l = {}, {} threads, {:.2f} s\n",
        VectorCount,
        Dimension,
        2 * HalfPower,
        threadCount,
        timer.ElapsedSeconds());

    Report("||X||_inf     ", norms.infinity);
    Report("||X||_1       ", norms.l1);
    Report("||X||_2l      ", norms.lp);
    Report("||X||_A       ", norms.quadratic);

    return 0;
}
