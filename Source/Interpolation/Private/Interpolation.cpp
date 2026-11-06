#include "MDAA/Interpolation/Interpolation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <print>

namespace MDAA {

namespace {

Polynomial Multiply(const Polynomial &a, const Polynomial &b) {
    Polynomial result(a.size() + b.size() - 1, 0.0);

    for (usize i = 0; i < a.size(); i++) {
        for (usize j = 0; j < b.size(); j++) {
            result[i + j] += a[i] * b[j];
        }
    }

    return result;
}

Polynomial Add(const Polynomial &a, const Polynomial &b) {
    Polynomial result(std::max(a.size(), b.size()), 0.0);

    for (usize i = 0; i < a.size(); i++) {
        result[i] += a[i];
    }

    for (usize i = 0; i < b.size(); i++) {
        result[i] += b[i];
    }

    return result;
}

Polynomial Scale(const Polynomial &a, f64 factor) {
    Polynomial result = a;

    for (f64 &value : result) {
        value *= factor;
    }

    return result;
}

}

f64 LagrangeValue(std::span<const f64> xs, std::span<const f64> ys, f64 x) {
    f64 result = 0.0;

    for (usize i = 0; i < xs.size(); i++) {
        f64 term = ys[i];

        for (usize j = 0; j < xs.size(); j++) {
            if (j != i) {
                term *= (x - xs[j]) / (xs[i] - xs[j]);
            }
        }

        result += term;
    }

    return result;
}

std::vector<f64> DividedDifferences(std::span<const f64> xs, std::span<const f64> ys) {
    const usize n = xs.size();

    std::vector<f64> row(ys.begin(), ys.end());
    std::vector<f64> coefficients(n);
    coefficients[0] = row[0];

    for (usize order = 1; order < n; order++) {
        for (usize i = 0; i + order < n; i++) {
            row[i] = (row[i + 1] - row[i]) / (xs[i + order] - xs[i]);
        }

        coefficients[order] = row[0];
    }

    return coefficients;
}

f64 NewtonValue(std::span<const f64> xs, std::span<const f64> coefficients, f64 x) {
    f64 result = coefficients.back();

    for (usize k = coefficients.size() - 1; k > 0; k--) {
        result = coefficients[k - 1] + (x - xs[k - 1]) * result;
    }

    return result;
}

f64 Evaluate(const Polynomial &polynomial, f64 x) {
    f64 result = 0.0;

    for (usize k = polynomial.size(); k > 0; k--) {
        result = result * x + polynomial[k - 1];
    }

    return result;
}

Polynomial LagrangePolynomial(std::span<const f64> xs, std::span<const f64> ys) {
    Polynomial result {0.0};

    for (usize i = 0; i < xs.size(); i++) {
        Polynomial basis {1.0};
        f64        denominator = 1.0;

        for (usize j = 0; j < xs.size(); j++) {
            if (j != i) {
                basis = Multiply(basis, Polynomial {-xs[j], 1.0});
                denominator *= xs[i] - xs[j];
            }
        }

        result = Add(result, Scale(basis, ys[i] / denominator));
    }

    return result;
}

Polynomial NewtonPolynomial(std::span<const f64> xs, std::span<const f64> coefficients) {
    Polynomial result {0.0};
    Polynomial term {1.0};

    for (usize k = 0; k < coefficients.size(); k++) {
        result = Add(result, Scale(term, coefficients[k]));
        term = Multiply(term, Polynomial {-xs[k], 1.0});
    }

    return result;
}

void PrintPolynomial(std::span<const f64> coefficients) {
    bool first = true;

    for (usize k = coefficients.size(); k > 0; k--) {
        const usize power = k - 1;
        const f64   value = coefficients[power];

        if (std::abs(value) < 1e-14) {
            continue;
        }

        if (first) {
            if (value < 0.0) {
                std::print("-");
            }
        } else {
            std::print("{}", value < 0.0 ? " - " : " + ");
        }

        const f64 magnitude = std::abs(value);

        if (power == 0) {
            std::print("{:.6f}", magnitude);
        } else if (power == 1) {
            std::print("{:.6f}x", magnitude);
        } else {
            std::print("{:.6f}x^{}", magnitude, power);
        }

        first = false;
    }

    if (first) {
        std::print("0");
    }

    std::println();
}

void PrintNodes(std::span<const f64> xs) {
    std::print("[");

    for (usize i = 0; i < xs.size(); i++) {
        std::print("{}{:.6f}", i == 0 ? "" : ", ", xs[i]);
    }

    std::print("]");
}

std::pair<std::vector<f64>, std::vector<f64>> NearestWindow(std::span<const f64> xs,
                                                            std::span<const f64> ys,
                                                            f64 target, i32 degree) {
    const usize k = static_cast<usize>(degree + 1);

    usize best = 0;
    f64   bestMax = std::numeric_limits<f64>::infinity();
    bool  bestContains = false;

    for (usize i = 0; i + k <= xs.size(); i++) {
        f64 maximum = 0.0;

        for (usize j = i; j < i + k; j++) {
            maximum = std::max(maximum, std::abs(xs[j] - target));
        }

        const bool contains = xs[i] <= target && target <= xs[i + k - 1];

        bool better = false;

        if (maximum < bestMax - 1e-12) {
            better = true;
        } else if (std::abs(maximum - bestMax) <= 1e-12 && contains && !bestContains) {
            better = true;
        }

        if (better) {
            best = i;
            bestMax = maximum;
            bestContains = contains;
        }
    }

    std::vector<f64> selectedXs;
    std::vector<f64> selectedYs;

    for (usize i = best; i < best + k; i++) {
        selectedXs.push_back(xs[i]);
        selectedYs.push_back(ys[i]);
    }

    return {selectedXs, selectedYs};
}

}
