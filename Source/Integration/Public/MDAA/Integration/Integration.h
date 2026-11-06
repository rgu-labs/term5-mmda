#pragma once

#include "MDAA/Core/Types.h"

#include <functional>
#include <string_view>

namespace MDAA {

using Integrand = std::function<f64(f64)>;

enum class Method : u8 {
    Rectangle,
    Trapezoid,
    Simpson,
};

f64 Rectangle(const Integrand &f, f64 a, f64 b, i64 n);

f64 Trapezoid(const Integrand &f, f64 a, f64 b, i64 n);

// NOTE(koftamainee): n must be even
f64 Simpson(const Integrand &f, f64 a, f64 b, i64 n);

f64 Integrate(const Integrand &f, f64 a, f64 b, Method method, i64 n);

i32 Order(Method method);

std::string_view MethodName(Method method);

struct RungeResult {
    f64  value = 0.0;   // I_2n
    f64  refined = 0.0; // I_2n + (I_2n - I_n) / (2^p - 1)
    f64  error = 0.0;   // |I_2n - I_n| / (2^p - 1)
    i64  n = 0;
    bool converged = false;
};

RungeResult IntegrateRunge(const Integrand &f, f64 a, f64 b, Method method,
                           f64 eps, i64 n0 = 2, i64 maxN = i64 {67'108'864});

} // namespace MDAA
