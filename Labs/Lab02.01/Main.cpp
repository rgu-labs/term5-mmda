#include "MDAA/Core/Assert.h"
#include "MDAA/Core/Timer.h"
#include "MDAA/Core/Types.h"
#include "MDAA/Optimization/Optimization.h"

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <format>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <print>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

using MDAA::f64;
using MDAA::i32;
using MDAA::usize;

constexpr i32 NormCount = 3;
constexpr i32 ThresholdCount = 7;
constexpr i32 ProblemCount = 7;
constexpr i32 DefaultMaxIterations = 1'000'000;
constexpr i32 DefaultHalfPower = 2;
constexpr i32 MaxHalfPower = 3;
constexpr f64 DefaultK = 1.0;
constexpr f64 Tolerance = 1.0e-6;

const std::array<f64, ThresholdCount> Thresholds {
    1.0e-2,
    1.0e-3,
    1.0e-4,
    1.0e-5,
    1.0e-10,
    1.0e-12,
    1.0e-15,
};

const std::array<std::string_view, NormCount> NormLabels {
    "||.||_inf",
    "||.||_1",
    "||.||_2l",
};

std::string Text(const Eigen::VectorXd &x) {
    std::string text = "(";

    for (i32 index = 0; index < static_cast<i32>(x.size()); index++) {
        text += std::format("{}{:.6f}", index == 0 ? "" : ", ", x[index]);
    }

    return text + ")";
}

Eigen::VectorXd Vector(const usize size, const std::initializer_list<f64> values) {
    MDAA_CHECKF(std::cmp_equal(values.size(), size), "a point of size {} needs {} numbers", size, values.size());

    Eigen::VectorXd x(size);

    for (i32 index = 0; std::cmp_less(index, size); index++) {
        x[index] = *std::next(values.begin(), index);
    }

    return x;
}

struct Problem final {
    std::string_view             Formula;
    MDAA::Objective              Objective;
    std::vector<Eigen::VectorXd> Starts;
    f64                          Minimum = 0.0;
};

Problem MakeFirst() {
    MDAA::Objective objective;

    objective.Value = [](const Eigen::VectorXd &x) {
        return (x[0] * x[0]) + (x[1] * x[1]) - (2.0 * x[0]) - (4.0 * x[1]) + 5.0;
    };
    objective.Gradient = [](const Eigen::VectorXd &x) {
        return Eigen::Vector2d((2.0 * x[0]) - 2.0, (2.0 * x[1]) - 4.0);
    };

    return {
        .Formula = "x1^2 + x2^2 - 2x1 - 4x2 + 5",
        .Objective = objective,
        .Starts = {Vector(2, {-1.0, 2.0}), Vector(2, {5.0, -3.0}), Vector(2, {0.0, 0.0})},
        .Minimum = 0.0,
    };
}

Problem MakeSecond() {
    MDAA::Objective objective;

    objective.Value = [](const Eigen::VectorXd &x) {
        return (2.0 * x[0] * x[0]) - (2.0 * x[0] * x[1]) + (x[1] * x[1]) + (2.0 * x[0]) -
               (2.0 * x[1]);
    };
    objective.Gradient = [](const Eigen::VectorXd &x) {
        return Eigen::Vector2d((4.0 * x[0]) - (2.0 * x[1]) + 2.0, (-2.0 * x[0]) + (2.0 * x[1]) - 2.0);
    };

    return {
        .Formula = "2x1^2 - 2x1x2 + x2^2 + 2x1 - 2x2",
        .Objective = objective,
        .Starts = {Vector(2, {-1.0, -1.0}), Vector(2, {2.0, 1.0}), Vector(2, {4.0, 3.0})},
        .Minimum = -1.0,
    };
}

Problem MakeThird() {
    MDAA::Objective objective;

    objective.Value = [](const Eigen::VectorXd &x) {
        return (x[0] * x[0]) + (2.0 * x[1] * x[1]) + (x[2] * x[2]) - (2.0 * x[0] * x[1]) - x[0] +
               (2.0 * x[2]);
    };
    objective.Gradient = [](const Eigen::VectorXd &x) {
        return Eigen::Vector3d((2.0 * x[0]) - (2.0 * x[1]) - 1.0, (4.0 * x[1]) - (2.0 * x[0]), (2.0 * x[2]) + 2.0);
    };

    return {
        .Formula = "x1^2 + 2x2^2 + x3^2 - 2x1x2 - x1 + 2x3",
        .Objective = objective,
        .Starts = {
            Vector(3, {3.0, 3.0, 3.0}),
            Vector(3, {-1.0, 2.0, 0.0}),
            Vector(3, {0.0, 0.0, 0.0}),
        },
        .Minimum = -1.5,
    };
}

Problem MakeFourth() {
    MDAA::Objective objective;

    objective.Value = [](const Eigen::VectorXd &x) {
        return (x[0] * x[0]) + (x[1] * x[1]) + (x[2] * x[2]) - (x[0] * x[1]) + x[0] -
               (2.0 * x[2]);
    };
    objective.Gradient = [](const Eigen::VectorXd &x) {
        return Eigen::Vector3d((2.0 * x[0]) - x[1] + 1.0, (2.0 * x[1]) - x[0], (2.0 * x[2]) - 2.0);
    };

    return {
        .Formula = "x1^2 + x2^2 + x3^2 - x1x2 + x1 - 2x3",
        .Objective = objective,
        .Starts = {
            Vector(3, {3.0, 3.0, 3.0}),
            Vector(3, {-1.0, 2.0, 0.0}),
            Vector(3, {0.0, 0.0, 0.0}),
        },
        .Minimum = -1.3333333333333333,
    };
}

Problem MakeFifth() {
    MDAA::Objective objective;

    objective.Value = [](const Eigen::VectorXd &x) {
        return (x[0] * x[0]) + (x[1] * x[1]) - (0.2 * x[0] * x[1]) - (2.2 * x[0]) + (2.2 * x[1]) +
               2.2;
    };
    objective.Gradient = [](const Eigen::VectorXd &x) {
        return Eigen::Vector2d((2.0 * x[0]) - (0.2 * x[1]) - 2.2, (2.0 * x[1]) - (0.2 * x[0]) + 2.2);
    };

    return {
        .Formula = "x1^2 + x2^2 - 0.2x1x2 - 2.2x1 + 2.2x2 + 2.2",
        .Objective = objective,
        .Starts = {Vector(2, {1.0, 2.0}), Vector(2, {-1.0, -1.0}), Vector(2, {0.0, 0.0})},
        .Minimum = 0.0,
    };
}

Problem MakeSixth() {
    MDAA::Objective objective;

    objective.Value = [](const Eigen::VectorXd &x) {
        return (5.0 * x[0] * x[0]) + (4.075 * x[1] * x[1]) - (9.0 * x[0] * x[1]) + x[0] + 2.0;
    };
    objective.Gradient = [](const Eigen::VectorXd &x) {
        return Eigen::Vector2d((10.0 * x[0]) - (9.0 * x[1]) + 1.0, (8.15 * x[1]) - (9.0 * x[0]));
    };

    return {
        .Formula = "5x1^2 + 4.075x2^2 - 9x1x2 + x1 + 2",
        .Objective = objective,
        .Starts = {Vector(2, {-1.0, 2.0}), Vector(2, {1.0, 0.0}), Vector(2, {2.0, -2.0})},
        .Minimum = -6.15,
    };
}

Problem MakeSeventh(const f64 k) {
    MDAA::Objective objective;

    objective.Value = [k](const Eigen::VectorXd &x) {
        const f64 first = x[0] + (10.0 * x[1]);
        const f64 second = x[2] - x[3] + (2.0 * k);
        const f64 third = x[1] - (2.0 * x[2]);
        const f64 fourth = x[0] - x[3] + (11.0 * k);
        return (first * first) + (5.0 * second * second) + (third * third * third * third) +
               (100.0 * fourth * fourth * fourth * fourth);
    };
    objective.Gradient = [k](const Eigen::VectorXd &x) {
        const f64 first = x[0] + (10.0 * x[1]);
        const f64 second = x[2] - x[3] + (2.0 * k);
        const f64 third = x[1] - (2.0 * x[2]);
        const f64 fourth = x[0] - x[3] + (11.0 * k);
        return Eigen::Vector4d(
            (2.0 * first) + (400.0 * fourth * fourth * fourth),
            (20.0 * first) + (4.0 * third * third * third),
            (10.0 * second) - (8.0 * third * third * third),
            (-10.0 * second) - (400.0 * fourth * fourth * fourth));
    };

    return {
        .Formula = "(x1 + 10x2)^2 + 5(x3 - x4 + 2k)^2 + (x2 - 2x3)^4 + 100(x1 - x4 + 11k)^4",
        .Objective = objective,
        .Starts = {
            Vector(4, {0.0, 0.0, 0.0, 0.0}),
            Vector(4, {10.0, 10.0, 10.0, 10.0}),
            Vector(4, {-20.0, 15.0, -7.0, 3.0}),
        },
        .Minimum = 0.0,
    };
}

struct Settings final {
    i32                             Problem = 0;
    i32                             MaxIterations = DefaultMaxIterations;
    i32                             HalfPower = DefaultHalfPower;
    f64                             K = DefaultK;
    std::optional<std::vector<f64>> Start = std::nullopt;
    bool                            Trace = false;
    bool                            Points = false;
};

f64 ParseNumber(const std::string_view text) {
    f64        value = 0.0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);

    MDAA_FATAL(
        result.ec != std::errc {} || result.ptr != text.data() + text.size(),
        "the flag needs a number, got '{}'",
        text);

    return value;
}

i32 ParseInteger(const std::string_view text) {
    return static_cast<i32>(ParseNumber(text));
}

std::vector<f64> ParseList(const std::string_view text) {
    std::vector<f64> values;

    usize begin = 0;

    while (begin <= text.size()) {
        const usize            end = text.find(',', begin);
        const std::string_view piece = text.substr(begin, end - begin);

        MDAA_FATAL(piece.empty(), "the flag needs comma separated numbers, got '{}'", text);
        values.push_back(ParseNumber(piece));

        if (end == std::string_view::npos) {
            break;
        }

        begin = end + 1;
    }

    return values;
}

void PrintUsage(const std::string_view program) {
    std::println("usage: {} [options]", program);
    std::println("  --function N        minimise only the problem N, 1..{}, default all", ProblemCount);
    std::println("  --start a,b[,c[,d]] one starting point instead of the built in ones");
    std::println("  --k K               the parameter k of problem 7, default 1");
    std::println("  --l L               the exponent of ||X||_2l is 2L, default 2");
    std::println("  --max-iterations N  the limit of the iterations, default {}", DefaultMaxIterations);
    std::println("  --trace             print every iterate");
    std::println("  --points            print the point of every iteration");
}

bool WantsHelp(const int argc, const char *const *argv) {
    for (i32 index = 1; index < argc; index++) {
        const std::string_view flag = argv[index];

        if (flag == "--help" || flag == "-h") {
            return true;
        }
    }

    return false;
}

std::string_view FlagValue(
    const std::string_view   flag,
    const char *const *const argv,
    const int                argc,
    i32                     &index) {
    MDAA_FATAL(index >= argc, "'{}' needs a value", flag);

    const std::string_view value = argv[index];
    index++;

    return value;
}

void ReadFlag(
    Settings                &settings,
    const std::string_view   flag,
    const char *const *const argv,
    const int                argc,
    i32                     &index) {
    if (flag == "--trace") {
        settings.Trace = true;
    } else if (flag == "--points") {
        settings.Points = true;
    } else if (flag == "--function") {
        settings.Problem = ParseInteger(FlagValue(flag, argv, argc, index));
    } else if (flag == "--start") {
        settings.Start = ParseList(FlagValue(flag, argv, argc, index));
    } else if (flag == "--k") {
        settings.K = ParseNumber(FlagValue(flag, argv, argc, index));
    } else if (flag == "--l") {
        settings.HalfPower = ParseInteger(FlagValue(flag, argv, argc, index));
    } else if (flag == "--max-iterations") {
        settings.MaxIterations = ParseInteger(FlagValue(flag, argv, argc, index));
    } else {
        MDAA_FATAL(true, "unknown flag '{}', try --help", flag);
    }
}

Settings ParseSettings(const int argc, const char *const *argv) {
    Settings settings;

    i32 index = 1;

    while (index < argc) {
        const std::string_view flag = argv[index];
        index++;

        ReadFlag(settings, flag, argv, argc, index);
    }

    MDAA_FATAL(
        settings.Problem < 0 || settings.Problem > ProblemCount,
        "--function takes 1..{}, got {}",
        ProblemCount,
        settings.Problem);
    MDAA_FATAL(
        settings.HalfPower < 1 || settings.HalfPower > MaxHalfPower,
        "--l takes 1..{}, got {}",
        MaxHalfPower,
        settings.HalfPower);
    MDAA_FATAL(settings.MaxIterations < 1, "--max-iterations takes a positive number");

    return settings;
}

std::vector<Eigen::VectorXd> StartsFor(const Problem &problem, const Settings &settings) {
    if (!settings.Start.has_value()) {
        return problem.Starts;
    }

    const i32 size = static_cast<i32>(problem.Starts.front().size());

    MDAA_CHECKF(
        std::cmp_equal(settings.Start->size(), static_cast<usize>(size)),
        "the starting point of problem {} needs {} numbers, got {}",
        settings.Problem,
        size,
        settings.Start->size());

    Eigen::VectorXd start(size);

    for (i32 index = 0; index < size; index++) {
        start[index] = settings.Start->at(static_cast<usize>(index));
    }

    return {start};
}

struct Cell final {
    i32  Iteration = -1;
    bool Reached = false;
    f64  Value = 0.0;
};

using Table = std::array<std::array<Cell, ThresholdCount>, NormCount>;

void Record(
    Table                 &table,
    const i32              iteration,
    const f64              value,
    const Eigen::VectorXd &displacement,
    const i32              halfPower) {
    const std::array<f64, NormCount> sizes {
        MDAA::InfinityNorm(displacement),
        MDAA::L1Norm(displacement),
        MDAA::L2LNorm(displacement, halfPower),
    };

    for (i32 norm = 0; norm < NormCount; norm++) {
        const auto which = static_cast<usize>(norm);

        for (i32 index = 0; index < ThresholdCount; index++) {
            const auto position = static_cast<usize>(index);
            Cell      &cell = table[which][position];

            if (!cell.Reached && sizes[which] < Thresholds[position]) {
                cell.Reached = true;
                cell.Iteration = iteration;
                cell.Value = value;
            }
        }
    }
}

bool Complete(const Table &table) {
    for (const auto &row : table) {
        for (const Cell &cell : row) {
            if (!cell.Reached) {
                return false;
            }
        }
    }

    return true;
}

void PrintTable(const Table &table) {
    std::println(
        "      {:>8} | {:>9} {:>15} {:>9} {:>15} {:>9} {:>15}",
        "epsilon",
        NormLabels[0],
        "f(x)",
        NormLabels[1],
        "f(x)",
        NormLabels[2],
        "f(x)");

    for (i32 index = 0; index < ThresholdCount; index++) {
        const auto position = static_cast<usize>(index);
        std::print("      {:>8.0e} |", Thresholds[position]);

        for (i32 norm = 0; norm < NormCount; norm++) {
            const auto  which = static_cast<usize>(norm);
            const Cell &cell = table[which][position];

            if (cell.Reached) {
                std::print(" {:>9} {:>15.6e}", cell.Iteration, cell.Value);
            } else {
                std::print(" {:>9} {:>15}", "> limit", "-");
            }
        }

        std::println();
    }
}

void RunFromStart(const i32 number, const Problem &problem, const Eigen::VectorXd &start,
                  const Settings &settings) {
    std::println(
        "  start {}, f(x0) = {:.10e}", Text(start), problem.Objective.Value(start));

    if (settings.Trace) {
        std::println("      {:>6} {:>14} {:>18}", "iter", "step", "f(x)");
    }

    const MDAA::SteepestDescentOptions options {.MaxIterations = settings.MaxIterations};

    Table       table {};
    f64         previousValue = problem.Objective.Value(start);
    f64         largestGrowth = 0.0;
    MDAA::Timer timer;
    timer.Start();

    const auto observe = [&](const i32 iteration, const Eigen::VectorXd &point, const f64 value,
                             const Eigen::VectorXd &displacement) {
        largestGrowth = std::max(largestGrowth, value - previousValue);
        previousValue = value;

        Record(table, iteration, value, displacement, settings.HalfPower);

        if (settings.Trace) {
            std::println(
                "      {:>6} {:>14.6e} {:>18.10e}",
                iteration,
                MDAA::InfinityNorm(displacement),
                value);
        }

        if (settings.Points) {
            std::println("      iter {}, x = {}, f = {:.16e}", iteration, Text(point), value);
        }

        return Complete(table);
    };

    const MDAA::SteepestDescentResult result =
        MDAA::RunSteepestDescent(problem.Objective, start, options, observe);

    timer.Stop();

    PrintTable(table);

    std::println(
        "      iterations {}, f = {:.16e}, x = {}, ||grad(x)||_2 = {:.6e}, {:.3f} s",
        result.Iterations,
        result.Value,
        Text(result.Point),
        result.GradientNorm,
        timer.ElapsedSeconds());

    if (result.Reason == MDAA::StopReason::Iterations) {
        std::println();
        return;
    }

    MDAA_CHECKF(
        largestGrowth <= 1.0e-9 * (1.0 + std::abs(result.Value)),
        "problem {} from {} has raised f by {}",
        number,
        Text(start),
        largestGrowth);
    MDAA_CHECKF(
        std::abs(result.Value - problem.Minimum) < Tolerance,
        "problem {} from {} has to reach f* = {}, the gap is {}",
        number,
        Text(start),
        problem.Minimum,
        std::abs(result.Value - problem.Minimum));

    std::println();
}

void RunProblem(const i32 number, const Problem &problem, const Settings &settings) {
    std::println("{} {}, f* = {:.16e}", number, problem.Formula, problem.Minimum);

    for (const Eigen::VectorXd &start : StartsFor(problem, settings)) {
        RunFromStart(number, problem, start, settings);
    }
}

void RunAll(const Settings &settings) {
    const std::vector<Problem> problems {
        MakeFirst(),
        MakeSecond(),
        MakeThird(),
        MakeFourth(),
        MakeFifth(),
        MakeSixth(),
        MakeSeventh(settings.K),
    };

    const i32 first = settings.Problem == 0 ? 1 : settings.Problem;
    const i32 last = settings.Problem == 0 ? ProblemCount : settings.Problem;

    for (i32 number = first; number <= last; number++) {
        RunProblem(number, problems[static_cast<usize>(number) - 1], settings);
    }
}

} // namespace

int main(const int argc, char *const *argv) {
    Eigen::setNbThreads(1);

    if (WantsHelp(argc, argv)) {
        PrintUsage(argv[0]);
        return 0;
    }

    const Settings settings = ParseSettings(argc, argv);

    MDAA_FATAL(
        settings.Start.has_value() && settings.Problem == 0,
        "--start needs --function, the problems have different dimensions");

    RunAll(settings);

    return 0;
}
