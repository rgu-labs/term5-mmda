#include "MDAA/Core/Types.h"
#include "MDAA/Interpolation/Interpolation.h"

#include <cmath>
#include <print>
#include <vector>

int main() {
    using MDAA::f64;

    const auto function = [](f64 x) { return std::sqrt(x - x * x); };

    const f64 x1 = (3.0 + std::sqrt(5.0)) / 6.0;

    const std::vector<f64> xs = {0.0, x1, 1.0};
    const std::vector<f64> ys = {function(0.0), function(x1), function(1.0)};

    const f64 value = MDAA::LagrangeValue(xs, ys, 0.5);

    std::println("III.6: x1 = {:.10f}, f(0.5) - P2(0.5) = {:.10f}", x1,
                 function(0.5) - value);

    return 0;
}
