#include "MDAA/Optimization/Norms.h"

#include <cmath>

namespace MDAA {

f64 VectorNorm::operator()(const Eigen::VectorXd &x) const {
    switch (Kind) {
        case NormKind::Infinity:
            return InfinityNorm(x);
        case NormKind::One:
            return L1Norm(x);
        case NormKind::L2L:
            return L2LNorm(x, HalfPower);
    }

    return 0.0;
}

f64 InfinityNorm(const Eigen::VectorXd &x) {
    return x.cwiseAbs().maxCoeff();
}

f64 L1Norm(const Eigen::VectorXd &x) {
    return x.cwiseAbs().sum();
}

f64 L2LNorm(const Eigen::VectorXd &x, const i32 halfPower) {
    const f64 power = 2.0 * static_cast<f64>(halfPower);
    return std::pow(x.cwiseAbs().array().pow(power).sum(), 1.0 / power);
}

} // namespace MDAA
