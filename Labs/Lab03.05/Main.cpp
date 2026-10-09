#include "MDAA/Core/Assert.h"
#include "MDAA/Core/Types.h"
#include "MDAA/LinearAlgebra/LinearAlgebra.h"

#include <Eigen/Core>
#include <Eigen/LU>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <print>

namespace {

using MDAA::f64;
using MDAA::i32;

constexpr i32 Segments = 10;
constexpr f64 Phi = 1.0;
constexpr f64 Psi = std::numbers::e;
constexpr f64 Tolerance = 1.0e-8;

f64 Coefficient(const f64 t) {
    return 1.0 + t;
}

f64 Source(const f64 t) {
    return std::exp(t);
}

f64 Difference(const f64 previous, const f64 middle, const f64 next, const f64 h,
               const f64 t) {
    const f64 decay = Coefficient(t) * Coefficient(t) * middle;
    const f64 scheme = (next - (2.0 * middle) + previous) / (h * h);

    return scheme - decay - Source(t);
}

} // namespace

int main() {
    Eigen::setNbThreads(1);

    const f64 h = 1.0 / static_cast<f64>(Segments);
    const f64 factor = 1.0 / (h * h);
    const i32 interior = Segments - 1;

    Eigen::VectorXd lower(interior);
    Eigen::VectorXd diagonal(interior);
    Eigen::VectorXd upper(interior);
    Eigen::VectorXd right(interior);

    for (i32 index = 0; index < interior; index++) {
        const f64 t = static_cast<f64>(index + 1) * h;

        lower[index] = index > 0 ? factor : 0.0;
        diagonal[index] = -(2.0 * factor) - (Coefficient(t) * Coefficient(t));
        upper[index] = index + 1 < interior ? factor : 0.0;
        right[index] = Source(t);
    }

    right[0] -= Phi * factor;
    right[interior - 1] -= Psi * factor;

    const Eigen::VectorXd solution = MDAA::SolveTridiagonal(lower, diagonal, upper, right);

    Eigen::VectorXd table(Segments + 1);
    table[0] = Phi;
    table[Segments] = Psi;

    for (i32 index = 1; index < Segments; index++) {
        table[index] = solution[index - 1];
    }

    std::println("x'' - (1 + t)^2 x = e^t, 0 < t < 1, x(0) = 1, x(1) = e");
    std::println("N = {}, h = {:.6f}", Segments, h);
    std::println("    n      t_n             x_n");

    for (i32 index = 0; index <= Segments; index++) {
        const f64 t = static_cast<f64>(index) * h;
        std::println("{:>4} {:>10.6f} {:>18.10f}", index, t, table[index]);
    }

    return 0;
}
