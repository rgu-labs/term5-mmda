#include "MDAA/Optimization/SteepestDescent.h"

#include "MDAA/Optimization/LineSearch.h"

namespace MDAA {

SteepestDescentResult RunSteepestDescent(const Objective &objective, const Eigen::VectorXd &start,
                                         const SteepestDescentOptions  &options,
                                         const SteepestDescentObserver &observer) {
    SteepestDescentResult result;

    Eigen::VectorXd point = start;

    result.Point = point;
    result.Value = objective.Value(point);
    result.GradientNorm = objective.Gradient(point).norm();

    for (i32 iteration = 1; iteration <= options.MaxIterations; iteration++) {
        const Eigen::VectorXd direction = -objective.Gradient(point);
        const f64             step = ExactLineSearch(objective, point, direction);
        const Eigen::VectorXd displacement = step * direction;
        const Eigen::VectorXd next = point + displacement;

        result.Point = next;
        result.Value = objective.Value(next);
        result.GradientNorm = direction.norm();
        result.Iterations = iteration;

        const bool stop =
            static_cast<bool>(observer) && observer(iteration, next, result.Value, displacement);

        if (step == 0.0) {
            result.Reason = StopReason::Stationary;
            return result;
        }

        if (stop) {
            return result;
        }

        point = next;
    }

    result.Reason = StopReason::Iterations;
    return result;
}

} // namespace MDAA
