#include "MDAA/Core/Assert.h"
#include "MDAA/Core/Types.h"
#include "MDAA/LinearAlgebra/LinearAlgebra.h"

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <print>
#include <string>

namespace {

using MDAA::f64;
using MDAA::i32;

constexpr i32 SystemCount = 2;
constexpr i32 Size = 5;
constexpr f64 Tolerance = 1.0e-9;

struct System final {
    std::array<f64, Size> Lower {};
    std::array<f64, Size> Diagonal {};
    std::array<f64, Size> Upper {};
    std::array<f64, Size> Right {};
};

System MakeFirst() {
    return System {
        .Lower = {0.0, 5.0, -8.0, 6.0, 3.0},
        .Diagonal = {-11.0, -15.0, 11.0, -15.0, 6.0},
        .Upper = {-9.0, -2.0, -3.0, 4.0, 0.0},
        .Right = {-122.0, -48.0, -14.0, -50.0, 42.0},
    };
}

System MakeSecond() {
    return System {
        .Lower = {0.0, 3.0, 2.0, 5.0, -8.0},
        .Diagonal = {10.0, 10.0, -9.0, 16.0, 16.0},
        .Upper = {5.0, -2.0, -5.0, -4.0, 0.0},
        .Right = {-120.0, -91.0, 5.0, -74.0, -56.0},
    };
}

Eigen::VectorXd ToVector(const std::array<f64, Size> &values) {
    Eigen::VectorXd vector(Size);

    for (i32 index = 0; index < Size; index++) {
        vector[index] = values[static_cast<std::size_t>(index)];
    }

    return vector;
}

std::string Equation(const System &system, const i32 row) {
    std::string text;

    for (i32 column = std::max(0, row - 1); column <= std::min(Size - 1, row + 1); column++) {
        f64 coefficient = system.Diagonal[static_cast<std::size_t>(column)];

        if (column < row) {
            coefficient = system.Lower[static_cast<std::size_t>(row)];
        } else if (column > row) {
            coefficient = system.Upper[static_cast<std::size_t>(row)];
        }

        const std::string term =
            std::format("{}*x{}", std::abs(coefficient), column + 1);

        if (text.empty()) {
            text = coefficient < 0.0 ? "-" + term : term;
        } else {
            text += coefficient < 0.0 ? " - " + term : " + " + term;
        }
    }

    return std::format("{} = {}", text, system.Right[static_cast<std::size_t>(row)]);
}

void PrintSystem(const System &system) {
    for (i32 row = 0; row < Size; row++) {
        std::println("  {}", Equation(system, row));
    }
}

std::string Text(const Eigen::VectorXd &x) {
    std::string text = "(";

    for (i32 index = 0; index < static_cast<i32>(x.size()); index++) {
        text += std::format("{}{:.6f}", index == 0 ? "" : ", ", x[index]);
    }

    return text + ")";
}

} // namespace

int main() {
    Eigen::setNbThreads(1);

    const std::array<System, SystemCount> systems {
        MakeFirst(),
        MakeSecond(),
    };

    for (i32 index = 0; index < SystemCount; index++) {
        const System &system = systems[static_cast<std::size_t>(index)];

        const Eigen::VectorXd solution = MDAA::SolveTridiagonal(
            ToVector(system.Lower),
            ToVector(system.Diagonal),
            ToVector(system.Upper),
            ToVector(system.Right));

        std::println("system {}", index + 1);
        PrintSystem(system);
        std::println("  x = {}", Text(solution));
    }

    return 0;
}
