#pragma once

#include "MDAA/Core/Types.h"

#include <Eigen/Core>

namespace MDAA {

Eigen::VectorXd SolveTridiagonal(const Eigen::VectorXd &lower, const Eigen::VectorXd &main,
                                 const Eigen::VectorXd &upper, const Eigen::VectorXd &right);

} // namespace MDAA
