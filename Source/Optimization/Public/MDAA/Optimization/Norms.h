#pragma once

#include "MDAA/Core/Types.h"

#include <Eigen/Core>

namespace MDAA {

enum class NormKind : u8 {
    Infinity,
    One,
    L2L,
};

struct VectorNorm final {
    NormKind Kind = NormKind::Infinity;
    i32      HalfPower = 1;

    f64 operator()(const Eigen::VectorXd &x) const;
};

f64 InfinityNorm(const Eigen::VectorXd &x);
f64 L1Norm(const Eigen::VectorXd &x);
f64 L2LNorm(const Eigen::VectorXd &x, i32 halfPower);

} // namespace MDAA
