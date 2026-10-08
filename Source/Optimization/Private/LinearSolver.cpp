#include "MDAA/Optimization/LinearSolver.h"

#include "MDAA/Core/Assert.h"

namespace MDAA {

namespace {

void CheckRequest(const Eigen::MatrixXd &a, const Eigen::VectorXd &b,
                  const Eigen::VectorXd &start, const LinearSolverOptions &options) {
    MDAA_CHECKF(
        a.rows() == a.cols(),
        "a square matrix is required, got {}x{}",
        a.rows(),
        a.cols());
    MDAA_CHECKF(a.rows() == b.size(), "b needs {} numbers, got {}", a.rows(), b.size());
    MDAA_CHECKF(
        start.size() == b.size(),
        "the start needs {} numbers, got {}",
        b.size(),
        start.size());
    MDAA_CHECKF(options.Epsilon > 0.0, "epsilon has to be positive, got {}", options.Epsilon);
    MDAA_CHECKF(
        options.MaxIterations > 0,
        "the iteration limit has to be positive, got {}",
        options.MaxIterations);
}

f64 StepRate(const LinearSolverOptions &options, const Eigen::VectorXd &direction,
             const Eigen::VectorXd &applied) {
    if (options.Method == IterativeMethod::ResidualMinimization) {
        return direction.dot(applied) / applied.squaredNorm();
    }

    return direction.squaredNorm() / direction.dot(applied);
}

} // namespace

LinearSolverResult SolveLinearSystem(const Eigen::MatrixXd &a, const Eigen::VectorXd &b,
                                     const Eigen::VectorXd      &start,
                                     const LinearSolverOptions  &options,
                                     const LinearSolverObserver &observer) {
    CheckRequest(a, b, start, options);

    const bool symmetric = a.isApprox(a.transpose());

    LinearSolverResult result;
    result.Point = start;

    for (i32 iteration = 1; iteration <= options.MaxIterations; iteration++) {
        const Eigen::VectorXd residual = b - (a * result.Point);
        const Eigen::VectorXd direction = symmetric ? residual : a.transpose() * residual;
        const Eigen::VectorXd applied =
            symmetric ? Eigen::VectorXd(a * direction)
                      : Eigen::VectorXd(a.transpose() * (a * direction));

        if (direction.isZero(0.0) || applied.isZero(0.0)) {
            result.Reason = StopReason::Stationary;
            return result;
        }

        const f64 rate = StepRate(options, direction, applied);

        result.Step = rate * direction;
        result.Point += result.Step;
        result.StepNorm = options.Norm(result.Step);
        result.Iterations = iteration;

        const bool stop = static_cast<bool>(observer) &&
                          observer(iteration, result.Point, result.Step, result.StepNorm);

        if (stop || result.StepNorm < options.Epsilon) {
            result.Reason = StopReason::Accuracy;
            return result;
        }
    }

    return result;
}

} // namespace MDAA
