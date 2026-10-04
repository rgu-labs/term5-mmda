#pragma once

#include "MDAA/Core/Types.h"
#include "MDAA/Optimization/Objective.h"

#include <Eigen/Core>

namespace MDAA {

f64 ExactLineSearch(const Objective &objective, const Eigen::VectorXd &x, const Eigen::VectorXd &direction);

} // namespace MDAA
