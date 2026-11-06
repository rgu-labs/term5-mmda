#include "MDAA/Core/Types.h"
#include "MDAA/Interpolation/Interpolation.h"

#include <cmath>
#include <print>
#include <string_view>
#include <vector>

namespace {

using MDAA::f64;

using MDAA::DividedDifferences;
using MDAA::LagrangeValue;
using MDAA::NearestWindow;
using MDAA::NewtonValue;
using MDAA::PrintNodes;

struct TableCase {
    std::string_view name;
    std::vector<f64> xs;
    std::vector<f64> ys;
    f64              target;
};

void RunTable(const TableCase &item) {
    std::println("{}: target = {:.6f}", item.name, item.target);

    for (int degree = 1; degree <= 3; degree++) {
        const auto [selectedXs, selectedYs] =
            NearestWindow(item.xs, item.ys, item.target, degree);
        const auto coefficients = DividedDifferences(selectedXs, selectedYs);
        const f64  lagrange = LagrangeValue(selectedXs, selectedYs, item.target);
        const f64  newton = NewtonValue(selectedXs, coefficients, item.target);

        std::print("  degree {} nodes ", degree);
        PrintNodes(selectedXs);
        std::println(", P_L = {:.10f}, P_N = {:.10f}, diff = {:.3e}", lagrange, newton,
                     std::abs(lagrange - newton));
    }
}

}

int main() {
    const std::vector<TableCase> cases = {
        {"III.3a", {8.1, 8.3, 8.6, 8.7}, {16.94410, 17.56492, 18.50515, 18.820913}, 8.4},
        {"III.3b",
         {-0.75, -0.5, -0.25, 0.0},
         {-0.07181250, -0.02475000, 0.33493750, 1.10100000},
         -1.0 / 3.0},
        {"III.3c",
         {0.1, 0.2, 0.3, 0.4},
         {0.62049958, -0.28398668, 0.00660095, 0.24842440},
         0.25},
        {"III.3d",
         {0.6, 0.7, 0.8, 1.0},
         {-0.17694460, 0.01375227, 0.22363362, 0.65809197},
         0.9},
        {"III.4a", {0.0, 0.25, 0.5, 0.75}, {1.0, 1.64872, 2.71828, 4.48169}, 0.43},
        {"III.4b",
         {-0.5, -0.25, 0.25, 0.5},
         {1.93750, 1.33203, 0.800781, 0.687500},
         0.0},
        {"III.4c",
         {0.1, 0.2, 0.3, 0.4},
         {-0.29004986, -0.56079734, -0.81401972, -1.0526302},
         0.18},
        {"III.4d",
         {-1.0, -0.5, 0.0, 0.5},
         {0.86199480, 0.95802009, 1.0986123, 1.2943767},
         0.25},
    };

    for (const auto &item : cases) {
        RunTable(item);
    }

    return 0;
}
