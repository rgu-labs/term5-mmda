#include "MDAA/Core/Types.h"
#include "MDAA/Integration/Integration.h"

#include <array>
#include <charconv>
#include <cmath>
#include <numbers>
#include <optional>
#include <span>
#include <print>
#include <string_view>
#include <vector>

namespace {

using MDAA::f64;
using MDAA::i32;
using MDAA::i64;
using MDAA::Integrand;
using MDAA::Method;

constexpr f64 Pi = std::numbers::pi;

struct Task {
    std::string_view   title;
    std::string_view   note;
    Integrand          f;
    f64                a;
    f64                b;
    std::optional<f64> exact;
    i64                n0 = 2;
};

i32 ParsePrecision(std::span<char *const> args) {
    constexpr i32 defaultPrecision = 6;

    if (args.size() < 2) {
        return defaultPrecision;
    }

    const std::string_view arg = args[1];
    i32                    value = 0;
    const auto [ptr, ec] = std::from_chars(arg.data(), arg.data() + arg.size(), value);

    if (ec != std::errc {} || ptr != arg.data() + arg.size() || value < 1 || value > 14) {
        std::println("bad precision '{}', expected integer in [1, 14], using {}", arg, defaultPrecision);
        return defaultPrecision;
    }

    return value;
}

std::vector<Task> MakeTasks() {
    std::vector<Task> tasks;

    {
        // F(x) = x(2x^2 + 9)sqrt(9 + x^2)/8 - 81/8 ln(x + sqrt(9 + x^2))
        const auto antiderivative = [](f64 x) {
            const f64 root = std::sqrt(9.0 + (x * x));
            return (x * ((2.0 * x * x) + 9.0) * root / 8.0) - (81.0 / 8.0 * std::log(x + root));
        };

        tasks.push_back({
            .title = "1. int_1^5 x^2 sqrt(9 + x^2) dx",
            .note = "no singularities",
            .f = [](f64 x) { return x * x * std::sqrt(9.0 + (x * x)); },
            .a = 1.0,
            .b = 5.0,
            .exact = antiderivative(5.0) - antiderivative(1.0),
        });
    }

    tasks.push_back({
        .title = "2. int_0^1 x^15 sqrt(1 + 3x^8) dx",
        .note = "no singularities, exact = 29/270",
        .f = [](f64 x) { return std::pow(x, 15) * std::sqrt(1.0 + (3.0 * std::pow(x, 8))); },
        .a = 0.0,
        .b = 1.0,
        .exact = 29.0 / 270.0,
    });

    tasks.push_back({
        .title = "3. int_0^2pi dx / (sin^4 x + cos^4 x)",
        .note = "no singularities, exact = 2 sqrt(2) pi, n0 = 16 to avoid false stop on nodes k pi/2",
        .f = [](f64 x) {
            const f64 s = std::sin(x);
            const f64 c = std::cos(x);
            return 1.0 / ((s * s * s * s) + (c * c * c * c));
        },
        .a = 0.0,
        .b = 2.0 * Pi,
        .exact = 2.0 * std::numbers::sqrt2 * Pi,
        .n0 = 16,
    });

    tasks.push_back({
        .title = "4. int_0^pi x sin x / (1 + cos^2 x) dx",
        .note = "no singularities, exact = pi^2 / 4",
        .f = [](f64 x) {
            const f64 c = std::cos(x);
            return x * std::sin(x) / (1.0 + (c * c));
        },
        .a = 0.0,
        .b = Pi,
        .exact = Pi * Pi / 4.0,
    });

    tasks.push_back({
        .title = "5. int_0^pi cos^2(3x) / x^(2/7) dx",
        .note = "singular at x = 0, substitution x = t^7: 7 int_0^(pi^(1/7)) t^4 cos^2(3t^7) dt",
        .f = [](f64 t) {
            const f64 c = std::cos(3.0 * std::pow(t, 7));
            return 7.0 * std::pow(t, 4) * c * c;
        },
        .a = 0.0,
        .b = std::pow(Pi, 1.0 / 7.0),
        .exact = std::nullopt,
    });

    tasks.push_back({
        .title = "6. int_0^(pi/4) sin^3(4x) / x^(1/3) dx",
        .note = "0/0 at x = 0, substitution x = t^3: 3 int_0^((pi/4)^(1/3)) t sin^3(4t^3) dt",
        .f = [](f64 t) {
            const f64 s = std::sin(4.0 * t * t * t);
            return 3.0 * t * s * s * s;
        },
        .a = 0.0,
        .b = std::cbrt(Pi / 4.0),
        .exact = std::nullopt,
    });

    return tasks;
}

void Solve(const Task &task, f64 eps, i32 precision) {
    constexpr std::array methods = {Method::Rectangle, Method::Trapezoid, Method::Simpson};

    std::println("{}", task.title);
    std::println("   {}", task.note);

    if (task.exact) {
        std::println("   exact = {:.{}f}", *task.exact, precision + 4);
    }

    std::println("   {:<10} {:>10} {:>22} {:>12} {:>12}",
                 "method", "n", "I", "runge", "|I - exact|");

    for (const Method method : methods) {
        const MDAA::RungeResult result = MDAA::IntegrateRunge(task.f, task.a, task.b, method, eps, task.n0);

        std::print("   {:<10} {:>10} {:>22.{}f} {:>12.3e}",
                   MDAA::MethodName(method), result.n,
                   result.value, precision + 4,
                   result.error);

        if (task.exact) {
            std::print(" {:>12.3e}", std::abs(result.value - *task.exact));
        } else {
            std::print(" {:>12}", "-");
        }

        std::println("");
    }

    std::println("");
}

} // namespace

int main(int argc, char *argv[]) {
    const i32 precision = ParsePrecision(std::span<char *const>(argv, static_cast<MDAA::usize>(argc)));
    const f64 eps = std::pow(10.0, -precision);

    std::println("eps = 1e-{}", precision);
    std::println("");

    for (const Task &task : MakeTasks()) {
        Solve(task, eps, precision);
    }

    return 0;
}
