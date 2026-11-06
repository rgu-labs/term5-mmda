#include "MDAA/Core/Types.h"
#include "MDAA/Interpolation/Interpolation.h"

#include <print>
#include <span>
#include <string_view>
#include <vector>

namespace {

using MDAA::f64;

using MDAA::Evaluate;
using MDAA::LagrangePolynomial;
using MDAA::PrintPolynomial;

void Run(std::string_view name, std::span<const f64> days, std::span<const f64> weights) {
    const auto polynomial = LagrangePolynomial(days, weights);

    std::print("III.9 {} polynomial: ", name);
    PrintPolynomial(polynomial);

    f64       bestX = days.front();
    f64       bestY = Evaluate(polynomial, bestX);
    const int samples = 1000000;

    for (int i = 0; i <= samples; i++) {
        const f64 x = days.front() + (days.back() - days.front()) *
                                         static_cast<f64>(i) / static_cast<f64>(samples);
        const f64 value = Evaluate(polynomial, x);

        if (value > bestY) {
            bestY = value;
            bestX = x;
        }
    }

    std::println("  max = {:.10f} at day = {:.10f}", bestY, bestX);
}

}

int main() {
    const std::vector<f64> days = {0.0, 6.0, 10.0, 13.0, 17.0, 20.0, 28.0};
    const std::vector<f64> sampleOne = {6.67, 17.33, 42.67, 37.33, 30.10, 29.31, 28.74};
    const std::vector<f64> sampleTwo = {6.67, 16.11, 18.89, 15.00, 10.56, 9.44, 8.89};

    Run("sample 1", days, sampleOne);
    Run("sample 2", days, sampleTwo);

    return 0;
}
