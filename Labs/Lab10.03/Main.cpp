#include "MDAA/Core/Assert.h"
#include "MDAA/Core/Types.h"
#include "MDAA/LinearAlgebra/LinearAlgebra.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <functional>
#include <print>
#include <string>
#include <vector>

namespace {

using MDAA::f64;
using MDAA::usize;

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using Vectors = std::vector<Vector>;
using Product = std::function<f64(const Vector &, const Vector &)>;

constexpr f64 Tolerance = 1.0e-10;

f64 EuclideanProduct(const Vector &a, const Vector &b) {
    return a.dot(b);
}

Product MatrixProduct(const Matrix &a) {
    return [a](const Vector &x, const Vector &y) -> f64 {
        return x.dot(a * y);
    };
}

Vector MakeVector(std::initializer_list<f64> values) {
    Vector       result(static_cast<Eigen::Index>(values.size()));
    Eigen::Index index = 0;

    for (const f64 value : values) {
        result[index++] = value;
    }

    return result;
}

void PrintVector(const Vector &vector) {
    std::print("(");

    for (Eigen::Index index = 0; index < vector.size(); index++) {
        std::print("{}{:.6f}", index == 0 ? "" : ", ", vector[index]);
    }

    std::print(")");
}

void PrintVectors(const char *name, const Vectors &vectors) {
    for (usize index = 0; index < vectors.size(); index++) {
        std::print("  {}{} = ", name, index + 1);
        PrintVector(vectors[index]);
        std::println();
    }
}

f64 MaxOffDiagonal(const Vectors &vectors, const Product &product) {
    f64 result = 0.0;

    for (usize row = 0; row < vectors.size(); row++) {
        for (usize column = row + 1; column < vectors.size(); column++) {
            result = std::max(result, std::abs(product(vectors[row], vectors[column])));
        }
    }

    return result;
}

void Show(const std::string &title, const Vectors &input, const Product &product) {
    const Vectors output = MDAA::Orthogonalize(input, product);
    const f64     residual = MaxOffDiagonal(output, product);

    std::println("{}", title);
    std::println("input:");
    PrintVectors("a", input);
    std::println("output:");
    PrintVectors("b", output);
    std::println("max |(b_i, b_j)|, i != j = {:.3e}", residual);

    MDAA_CHECKF(
        input.size() == output.size(),
        "the result has to keep the number of vectors");
    MDAA_CHECKF(
        residual < Tolerance,
        "the vectors have to be orthogonal, the largest product is {}",
        residual);

    std::println();
}

} // namespace

int main() {
    Eigen::setNbThreads(1);

    const Vectors task1A {
        MakeVector({1, 1, 0, 0}),
        MakeVector({1, 0, 1, 0}),
        MakeVector({1, 0, 0, 1}),
        MakeVector({0, 0, 1, 1}),
    };

    const Vectors task1B {
        MakeVector({1, 1, 0, 0, 0}),
        MakeVector({1, 0, 1, 0, 0}),
        MakeVector({1, 0, 0, 1, 0}),
        MakeVector({1, 0, 0, 0, 1}),
    };

    const Vectors task1C {
        MakeVector({2, 0, 1, 0}),
        MakeVector({1, 1, 0, 1}),
        MakeVector({0, 2, 1, 1}),
    };

    Matrix matrix2A(4, 4);
    matrix2A << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 3, 0,
        0, 0, 0, 2;

    Matrix matrix2B(5, 5);
    matrix2B << 5, 1, 0, 0, 0,
        1, 5, 1, 0, 0,
        0, 1, 5, 1, 0,
        0, 0, 1, 5, 1,
        0, 0, 0, 1, 5;

    Matrix matrix2C(4, 4);
    matrix2C << 2, 1, 0, 0,
        1, 2, 1, 0,
        0, 1, 2, 1,
        0, 0, 1, 2;

    std::println("Task 1: Euclidean inner product");
    std::println();

    Show("1a", task1A, EuclideanProduct);
    Show("1b", task1B, EuclideanProduct);
    Show("1c", task1C, EuclideanProduct);

    std::println("Task 2: inner product [x, y]_A = x^T A y");
    std::println();

    Show("2a", task1A, MatrixProduct(matrix2A));
    Show("2b", task1B, MatrixProduct(matrix2B));
    Show("2c", task1C, MatrixProduct(matrix2C));

    return 0;
}
