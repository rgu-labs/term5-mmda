#pragma once

#include "MDAA/Core/Types.h"

#include <span>
#include <utility>
#include <vector>

namespace MDAA {

using Polynomial = std::vector<f64>;

f64 LagrangeValue(std::span<const f64> xs, std::span<const f64> ys, f64 x);

std::vector<f64> DividedDifferences(std::span<const f64> xs, std::span<const f64> ys);

f64 NewtonValue(std::span<const f64> xs, std::span<const f64> coefficients, f64 x);

f64 Evaluate(const Polynomial &polynomial, f64 x);

Polynomial LagrangePolynomial(std::span<const f64> xs, std::span<const f64> ys);

Polynomial NewtonPolynomial(std::span<const f64> xs, std::span<const f64> coefficients);

void PrintPolynomial(std::span<const f64> coefficients);

void PrintNodes(std::span<const f64> xs);

std::pair<std::vector<f64>, std::vector<f64>> NearestWindow(std::span<const f64> xs,
                                                            std::span<const f64> ys,
                                                            f64 target, i32 degree);

}
