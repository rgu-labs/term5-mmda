#pragma once

#include "MDAA/Core/Types.h"
#include "MDAA/LinearAlgebra/Matrix.h"

#include <Eigen/Core>

namespace MDAA {

template <i32 Size>
using PointMatrix = Eigen::Matrix<f64, Size, Eigen::Dynamic>;

} // namespace MDAA
