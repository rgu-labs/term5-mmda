#include "MDAA/Integration/Integration.h"

#include <cmath>

namespace MDAA {

f64 Rectangle(const Integrand &f, f64 a, f64 b, i64 n) {
    const f64 h = (b - a) / static_cast<f64>(n);
    f64       sum = 0.0;

    for (i64 i = 0; i < n; i++) {
        sum += f(a + ((static_cast<f64>(i) + 0.5) * h));
    }

    return h * sum;
}

f64 Trapezoid(const Integrand &f, f64 a, f64 b, i64 n) {
    const f64 h = (b - a) / static_cast<f64>(n);
    f64       sum = 0.5 * (f(a) + f(b));

    for (i64 i = 1; i < n; i++) {
        sum += f(a + (static_cast<f64>(i) * h));
    }

    return h * sum;
}

f64 Simpson(const Integrand &f, f64 a, f64 b, i64 n) {
    const f64 h = (b - a) / static_cast<f64>(n);
    f64       odd = 0.0;
    f64       even = 0.0;

    for (i64 i = 1; i < n; i++) {
        const f64 value = f(a + (static_cast<f64>(i) * h));

        if (i % 2 == 1) {
            odd += value;
        } else {
            even += value;
        }
    }

    return h / 3.0 * (f(a) + (4.0 * odd) + (2.0 * even) + f(b));
}

f64 Integrate(const Integrand &f, f64 a, f64 b, Method method, i64 n) {
    switch (method) {
        case Method::Rectangle:
            return Rectangle(f, a, b, n);
        case Method::Trapezoid:
            return Trapezoid(f, a, b, n);
        case Method::Simpson:
            return Simpson(f, a, b, n);
    }

    return 0.0;
}

i32 Order(Method method) {
    return method == Method::Simpson ? 4 : 2;
}

std::string_view MethodName(Method method) {
    switch (method) {
        case Method::Rectangle:
            return "Rectangle";
        case Method::Trapezoid:
            return "Trapezoid";
        case Method::Simpson:
            return "Simpson";
    }

    return "Unknown";
}

RungeResult IntegrateRunge(const Integrand &f, f64 a, f64 b, Method method,
                           f64 eps, i64 n0, i64 maxN) {
    i64 n = n0 < 2 ? 2 : n0;

    if (method == Method::Simpson && n % 2 != 0) {
        n++;
    }

    const f64 denominator = std::pow(2.0, Order(method)) - 1.0;

    RungeResult result;
    f64         previous = Integrate(f, a, b, method, n);

    while (2 * n <= maxN) {
        n *= 2;
        const f64 current = Integrate(f, a, b, method, n);
        const f64 difference = (current - previous) / denominator;

        result.value = current;
        result.refined = current + difference;
        result.error = std::abs(difference);
        result.n = n;

        if (result.error < eps) {
            result.converged = true;
            break;
        }

        previous = current;
    }

    return result;
}

} // namespace MDAA
