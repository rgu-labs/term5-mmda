#include "MDAA/LinearAlgebra/Tridiagonal.h"

#include "MDAA/Core/Assert.h"

namespace MDAA {

Eigen::VectorXd SolveTridiagonal(const Eigen::VectorXd &lower, const Eigen::VectorXd &main,
                                 const Eigen::VectorXd &upper, const Eigen::VectorXd &right) {
    const Eigen::Index count = main.size();

    MDAA_CHECKF(count > 0, "a system needs at least one equation, got {}", count);
    MDAA_CHECKF(
        lower.size() == count && upper.size() == count && right.size() == count,
        "the diagonals need {} numbers each",
        count);
    MDAA_CHECKF(main[0] != 0.0, "the first pivot is zero");

    Eigen::VectorXd cross(count);
    Eigen::VectorXd offset(count);

    cross[0] = upper[0] / main[0];
    offset[0] = right[0] / main[0];

    for (Eigen::Index index = 1; index < count; index++) {
        const f64 pivot = main[index] - (lower[index] * cross[index - 1]);
        MDAA_CHECKF(pivot != 0.0, "the pivot of equation {} is zero", index);

        cross[index] = upper[index] / pivot;
        offset[index] = (right[index] - (lower[index] * offset[index - 1])) / pivot;
    }

    Eigen::VectorXd solution = offset;

    for (Eigen::Index index = count - 2; index >= 0; index--) {
        solution[index] -= cross[index] * solution[index + 1];
    }

    return solution;
}

} // namespace MDAA
