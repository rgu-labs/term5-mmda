#include "MDAA/Optimization/LineSearch.h"

#include "MDAA/Optimization/Norms.h"

namespace MDAA {

namespace {

constexpr i32 Limit = 100;
constexpr i32 Steps = 80;

} // namespace

f64 ExactLineSearch(const Objective &objective, const Eigen::VectorXd &x, const Eigen::VectorXd &direction) {
    const f64 size = InfinityNorm(direction);

    if (size <= 0.0) {
        return 0.0;
    }

    const auto value = [&](const f64 step) { return objective.Value(x + (step * direction)); };

    const auto slope = [&](const f64 step) {
        return objective.Gradient(x + (step * direction)).dot(direction);
    };

    f64 left = 1.0 / size;
    f64 right = left;

    for (i32 index = 0; index < Limit && slope(left) > 0.0; index++) {
        left /= 2.0;
    }

    for (i32 index = 0; index < Limit && slope(right) < 0.0; index++) {
        right *= 2.0;
    }

    for (i32 index = 0; index < Steps; index++) {
        const f64 middle = 0.5 * (left + right);

        if (slope(middle) < 0.0) {
            left = middle;
        } else {
            right = middle;
        }
    }

    const f64 step = 0.5 * (left + right);

    return value(step) < value(0.0) ? step : 0.0;
}

} // namespace MDAA
