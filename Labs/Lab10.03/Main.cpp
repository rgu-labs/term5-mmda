#include "MDAA/Core/Assert.h"
#include "MDAA/Core/Types.h"
#include "MDAA/LinearAlgebra/LinearAlgebra.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <print>
#include <vector>

namespace {

using MDAA::f64;
using MDAA::usize;

using Vector = Eigen::VectorXd;
using Vectors = std::vector<Vector>;

constexpr f64 Tolerance = 1.0e-10;

f64 InnerProduct(const Vector &a, const Vector &b) {
    return a.dot(b);
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

void PrintGram(const Vectors &vectors, const char *title) {
    std::println("{}", title);

    for (const Vector &row : vectors) {
        std::print("  [");

        for (const Vector &column : vectors) {
            std::print("{:>12.6f}", InnerProduct(row, column));
        }

        std::println("]");
    }
}

f64 MaxOffDiagonal(const Vectors &vectors) {
    f64 result = 0.0;

    for (usize row = 0; row < vectors.size(); row++) {
        for (usize column = row + 1; column < vectors.size(); column++) {
            result = std::max(result, std::abs(InnerProduct(vectors[row], vectors[column])));
        }
    }

    return result;
}

void Show(const char *title, const Vectors &input) {
    const Vectors output = MDAA::Orthogonalize(input, InnerProduct);
    const f64     residual = MaxOffDiagonal(output);

    std::println("{}", title);
    std::println("input:");
    PrintVectors("v", input);
    PrintGram(input, "Gram matrix before:");
    std::println("output:");
    PrintVectors("q", output);
    PrintGram(output, "Gram matrix after:");
    std::println("max |(q_i, q_j)|, i != j = {:.3e}", residual);

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

    const Vectors independent {
        Eigen::Vector4d(1.0, 1.0, 1.0, 1.0),
        Eigen::Vector4d(1.0, 2.0, 3.0, 4.0),
        Eigen::Vector4d(1.0, 4.0, 9.0, 16.0),
        Eigen::Vector4d(1.0, 8.0, 27.0, 64.0),
    };

    const Vectors dependent {
        Eigen::Vector3d(1.0, 2.0, 3.0),
        Eigen::Vector3d(2.0, 4.0, 6.0),
        Eigen::Vector3d(0.0, 1.0, 0.0),
    };

    std::println("Orthogonalization");
    std::println("inner product: (x, y) = sum x_i * y_i");
    std::println();

    Show("linearly independent vectors", independent);
    Show("linearly dependent vectors: v2 = 2 * v1", dependent);

    std::println("a dependent vector becomes zero and adds no new direction");

    return 0;
}
