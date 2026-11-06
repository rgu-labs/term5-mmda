#include "MDAA/Core/Types.h"
#include "MDAA/Interpolation/Interpolation.h"

#include <algorithm>
#include <cmath>
#include <print>
#include <string_view>
#include <vector>

namespace {

using MDAA::f64;
using MDAA::usize;

using MDAA::LagrangePolynomial;
using MDAA::LagrangeValue;
using MDAA::PrintPolynomial;

struct BoundCase {
    std::string_view name;
    f64 (*function)(f64);
    f64 (*derivative)(f64);
    std::vector<f64> nodes;
};

}

int main() {
    const std::vector<BoundCase> cases = {
        {"III.7a",
         [](f64 x) { return std::exp(2.0 * x) * std::cos(3.0 * x); },
         [](f64 x) {
             return std::exp(2.0 * x) *
                    (-46.0 * std::cos(3.0 * x) - 9.0 * std::sin(3.0 * x));
         },
         {0.0, 0.3, 0.6}},
        {"III.7b",
         [](f64 x) { return std::sin(std::log(x)); },
         [](f64 x) {
             return (3.0 * std::sin(std::log(x)) + std::cos(std::log(x))) /
                    (x * x * x);
         },
         {2.0, 2.4, 2.6}},
        {"III.7c",
         [](f64 x) { return std::log(x); },
         [](f64 x) { return -6.0 / (x * x * x * x); },
         {1.0, 1.1, 1.3, 1.4}},
        {"III.7d",
         [](f64 x) { return std::cos(x) + std::sin(x); },
         [](f64 x) { return std::cos(x) + std::sin(x); },
         {0.0, 0.25, 0.5, 1.0}},
    };

    const int samples = 1000000;

    for (const auto &item : cases) {
        std::vector<f64> ys;

        for (const f64 x : item.nodes) {
            ys.push_back(item.function(x));
        }

        const auto polynomial = LagrangePolynomial(item.nodes, ys);

        std::print("{} polynomial: ", item.name);
        PrintPolynomial(polynomial);

        const f64 low = item.nodes.front();
        const f64 high = item.nodes.back();
        f64       derivativeMax = 0.0;
        f64       omegaMax = 0.0;
        f64       actualMax = 0.0;

        for (int i = 0; i <= samples; i++) {
            const f64 x =
                low + (high - low) * static_cast<f64>(i) / static_cast<f64>(samples);
            derivativeMax = std::max(derivativeMax, std::abs(item.derivative(x)));

            f64 omega = 1.0;

            for (const f64 node : item.nodes) {
                omega *= x - node;
            }

            omegaMax = std::max(omegaMax, std::abs(omega));
            actualMax =
                std::max(actualMax,
                         std::abs(item.function(x) - LagrangeValue(item.nodes, ys, x)));
        }

        f64 factorial = 1.0;

        for (usize k = 2; k <= item.nodes.size(); k++) {
            factorial *= static_cast<f64>(k);
        }

        const f64 bound = derivativeMax / factorial * omegaMax;

        std::println("  max |f^(n+1)| = {:.10e}", derivativeMax);
        std::println("  max |omega| = {:.10e}", omegaMax);
        std::println("  bound = {:.10e}, actual = {:.10e}", bound, actualMax);
    }

    return 0;
}
