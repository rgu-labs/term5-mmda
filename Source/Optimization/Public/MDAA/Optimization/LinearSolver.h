#pragma once

#include "MDAA/Core/Types.h"
#include "MDAA/Optimization/Norms.h"
#include "MDAA/Optimization/StopReason.h"

#include <Eigen/Core>

#include <functional>

namespace MDAA {

enum class IterativeMethod : u8 {
    ResidualMinimization,
    SteepestDescent,
};

struct LinearSolverOptions final {
    IterativeMethod Method = IterativeMethod::SteepestDescent;
    VectorNorm      Norm {};
    f64             Epsilon = 1.0e-6;
    i32             MaxIterations = 1'000'000;
};

struct LinearSolverResult final {
    Eigen::VectorXd Point;
    Eigen::VectorXd Step;
    f64             StepNorm = 0.0;
    i32             Iterations = 0;
    StopReason      Reason = StopReason::Iterations;
};

using LinearSolverObserver =
    std::function<bool(i32, const Eigen::VectorXd &, const Eigen::VectorXd &, f64)>;

LinearSolverResult SolveLinearSystem(const Eigen::MatrixXd &a, const Eigen::VectorXd &b,
                                     const Eigen::VectorXd      &start,
                                     const LinearSolverOptions  &options,
                                     const LinearSolverObserver &observer = {});

} // namespace MDAA
