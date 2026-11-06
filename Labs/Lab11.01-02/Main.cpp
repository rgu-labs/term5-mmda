#include "MDAA/Core/Types.h"
#include "MDAA/Interpolation/Interpolation.h"

#include <cmath>
#include <numbers>
#include <print>
#include <span>
#include <string_view>
#include <vector>

namespace {

using MDAA::f64;

using MDAA::LagrangeValue;
using MDAA::NearestWindow;
using MDAA::PrintNodes;

void RunApproximation(std::string_view name, std::span<const f64> xs, f64 target,
                      f64 (*function)(f64)) {
    std::vector<f64> ys;

    for (const f64 x : xs) {
        ys.push_back(function(x));
    }

    std::println("{}: target = {:.6f}", name, target);

    for (int degree = 1; degree <= 2; degree++) {
        const auto [selectedXs, selectedYs] = NearestWindow(xs, ys, target, degree);
        const f64 value = LagrangeValue(selectedXs, selectedYs, target);
        const f64 exact = function(target);

        std::print("  degree {} nodes ", degree);
        PrintNodes(selectedXs);
        std::println(", P = {:.10f}, f = {:.10f}, error = {:.3e}", value, exact,
                     std::abs(value - exact));
    }
}

}

int main() {
    const std::vector<f64> firstXs = {0.0, 0.6, 0.9};

    RunApproximation("III.1a cos x", firstXs, 0.45, [](f64 x) { return std::cos(x); });
    RunApproximation("III.1b sqrt(1+x)", firstXs, 0.45,
                     [](f64 x) { return std::sqrt(1.0 + x); });
    RunApproximation("III.1c ln(x+1)", firstXs, 0.45,
                     [](f64 x) { return std::log(1.0 + x); });
    RunApproximation("III.1d tan x", firstXs, 0.45, [](f64 x) { return std::tan(x); });

    const std::vector<f64> secondXs = {1.0, 1.25, 1.6};

    RunApproximation("III.2a sin(pi x)", secondXs, 1.4,
                     [](f64 x) { return std::sin(std::numbers::pi * x); });
    RunApproximation("III.2b cbrt(x-1)", secondXs, 1.4,
                     [](f64 x) { return std::cbrt(x - 1.0); });
    RunApproximation("III.2c lg(3x-1)", secondXs, 1.4,
                     [](f64 x) { return std::log10(3.0 * x - 1.0); });
    RunApproximation("III.2d exp(2x)-x", secondXs, 1.4,
                     [](f64 x) { return std::exp(2.0 * x) - x; });

    return 0;
}
