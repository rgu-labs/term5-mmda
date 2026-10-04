#pragma once

#include "MDAA/Core/Types.h"

#include <Eigen/Core>

#include <functional>

namespace MDAA {

struct Objective final {
    std::function<f64(const Eigen::VectorXd &)>             Value;
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> Gradient;
};

} // namespace MDAA
