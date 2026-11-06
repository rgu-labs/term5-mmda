#include "MDAA/Core/Types.h"
#include "MDAA/Interpolation/Interpolation.h"

#include <print>
#include <vector>

int main() {
    using MDAA::f64;

    const std::vector<f64> xs = {0.0, 0.5, 1.0, 2.0};

    const auto coefficient = [&xs](f64 y) {
        const std::vector<f64> ys = {0.0, y, 3.0, 2.0};
        return MDAA::DividedDifferences(xs, ys).back();
    };

    const f64 first = coefficient(0.0);
    const f64 second = coefficient(1.0);
    const f64 y = (6.0 - first) / (second - first);

    std::println("III.5: y = {:.6f}, coefficient = {:.10f}", y, coefficient(y));

    return 0;
}
