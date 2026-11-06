#include "MDAA/Core/Types.h"
#include "MDAA/Interpolation/Interpolation.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <print>
#include <vector>

namespace {

using MDAA::f64;
using MDAA::usize;

using MDAA::DividedDifferences;
using MDAA::LagrangePolynomial;
using MDAA::LagrangeValue;
using MDAA::NewtonPolynomial;
using MDAA::NewtonValue;
using MDAA::PrintPolynomial;

}

int main(int argc, char **argv) {
    const bool hasXi = argc > 1;
    const f64  xi = hasXi ? std::strtod(argv[1], nullptr) : 0.0;

    const std::vector<std::vector<f64>> xsList = {
        {-1.0, 0.0, 1.0, 2.0, 3.0, 4.0},
        {-0.9, 0.0, 0.9, 1.8, 2.7, 3.6},
        {1.0, 1.9, 2.8, 3.7, 4.6, 5.5},
    };

    const std::vector<std::vector<f64>> ysList = {
        {0.86603, 1.0, 0.86603, 0.5, 0.0, -0.5},
        {-0.36892, 0.0, 0.36892, 0.85408, 1.7856, 6.3138},
        {2.4142, 1.0818, 0.50953, 0.11836, -0.24008, -0.66818},
    };

    for (usize index = 0; index < xsList.size(); index++) {
        const auto &xs = xsList[index];
        const auto &ys = ysList[index];
        const f64   point = hasXi ? xi : xs.front() + 0.371234 * (xs.back() - xs.front());
        const auto  lagrange = LagrangePolynomial(xs, ys);
        const auto  coefficients = DividedDifferences(xs, ys);
        const auto  newton = NewtonPolynomial(xs, coefficients);

        std::println("I.{}: xi = {:.6f}", index + 1, point);

        std::print("  Lagrange polynomial: ");
        PrintPolynomial(lagrange);

        std::print("  Newton polynomial:   ");
        PrintPolynomial(newton);

        std::print("  Newton form: {:.6f}", coefficients[0]);

        for (usize k = 1; k < coefficients.size(); k++) {
            const f64 coefficient = coefficients[k];
            std::print(" {} ({:.6f})", coefficient < 0.0 ? "-" : "+", std::abs(coefficient));

            for (usize j = 0; j < k; j++) {
                const f64 node = xs[j];
                std::print("(x {} {:.6f})", node < 0.0 ? "+" : "-", std::abs(node));
            }
        }

        std::println();

        const f64 lagrangeValue = LagrangeValue(xs, ys, point);
        const f64 newtonValue = NewtonValue(xs, coefficients, point);
        f64       residual = 0.0;

        for (usize i = 0; i < xs.size(); i++) {
            residual = std::max(residual, std::abs(LagrangeValue(xs, ys, xs[i]) - ys[i]));
        }

        std::println("  P_L = {:.10f}, P_N = {:.10f}, diff = {:.3e}", lagrangeValue,
                     newtonValue, std::abs(lagrangeValue - newtonValue));
        std::println("  max |P(x_i) - y_i| = {:.3e}", residual);
    }

    return 0;
}
