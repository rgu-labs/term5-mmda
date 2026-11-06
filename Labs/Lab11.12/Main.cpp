#include "MDAA/Core/Types.h"
#include "MDAA/Interpolation/Interpolation.h"

#include <cmath>
#include <print>
#include <vector>

int main() {
    using MDAA::f64;

    const std::vector<f64> xs = {1.08, 1.13, 1.20, 1.27, 1.31};
    const std::vector<f64> ys = {1.302, 1.386, 1.509, 1.217, 1.284};
    const f64              target = 1.14;

    std::println("III.12: target = {:.2f}", target);

    for (int degree = 1; degree <= 3; degree++) {
        const auto [selectedXs, selectedYs] =
            MDAA::NearestWindow(xs, ys, target, degree);
        const auto coefficients = MDAA::DividedDifferences(selectedXs, selectedYs);
        const f64  lagrange = MDAA::LagrangeValue(selectedXs, selectedYs, target);
        const f64  newton = MDAA::NewtonValue(selectedXs, coefficients, target);

        std::print("  degree {} nodes ", degree);
        MDAA::PrintNodes(selectedXs);
        std::println(", Lagrange = {:.10f}, Newton = {:.10f}, diff = {:.3e}", lagrange,
                     newton, std::abs(lagrange - newton));
    }

    return 0;
}
