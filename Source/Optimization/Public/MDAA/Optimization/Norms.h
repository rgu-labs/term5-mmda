#pragma once

#include "MDAA/Core/Types.h"

#include <Eigen/Core>

namespace MDAA {

f64 InfinityNorm(const Eigen::VectorXd &x);
f64 L1Norm(const Eigen::VectorXd &x);
f64 L2LNorm(const Eigen::VectorXd &x, i32 halfPower);

} // namespace MDAA
