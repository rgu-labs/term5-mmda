#include "MDAA/Core/Types.h"
#include "MDAA/Interpolation/Interpolation.h"

#include <cmath>
#include <print>
#include <span>
#include <string_view>
#include <vector>

namespace {

using MDAA::f64;

using MDAA::LagrangeValue;

void Show(std::string_view label, std::span<const f64> xs, f64 x) {
    const auto       function = [](f64 value) { return std::exp(value); };
    std::vector<f64> ys;

    for (const f64 node : xs) {
        ys.push_back(function(node));
    }

    const f64 value = LagrangeValue(xs, ys, x);

    std::println("III.8{}: P({:.2f}) = {:.10f}, f = {:.10f}, error = {:.10f}", label, x,
                 value, function(x), std::abs(value - function(x)));
}

}

int main() {
    const std::vector<f64> firstLinear = {0.0, 0.5};
    const std::vector<f64> secondLinear = {0.5, 1.0};
    const std::vector<f64> quadratic = {0.0, 1.0, 2.0};

    Show("a", firstLinear, 0.25);
    Show("b", secondLinear, 0.75);
    Show("c", quadratic, 0.25);
    Show("c", quadratic, 0.75);

    return 0;
}
