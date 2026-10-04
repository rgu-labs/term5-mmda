#pragma once

#include "MDAA/Core/Types.h"
#include "MDAA/Optimization/Objective.h"

#include <Eigen/Core>

#include <functional>

namespace MDAA {

enum class StopReason : u8 {
    Accuracy,
    Iterations,
    Stationary,
};

struct SteepestDescentOptions final {
    i32 MaxIterations;
};

struct SteepestDescentResult final {
    Eigen::VectorXd Point;
    f64             Value = 0.0;
    f64             GradientNorm = 0.0;
    i32             Iterations = 0;
    StopReason      Reason = StopReason::Accuracy;
};

using SteepestDescentObserver =
    std::function<bool(i32, const Eigen::VectorXd &, f64, const Eigen::VectorXd &)>;

SteepestDescentResult RunSteepestDescent(const Objective               &objective,
                                         const Eigen::VectorXd         &start,
                                         const SteepestDescentOptions  &options,
                                         const SteepestDescentObserver &observer);

} // namespace MDAA
