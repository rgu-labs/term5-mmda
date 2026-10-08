#include "MDAA/Core/Assert.h"
#include "MDAA/Core/Types.h"
#include "MDAA/Optimization/Optimization.h"

#include <Eigen/Core>
#include <Eigen/LU>

#include <array>
#include <charconv>
#include <cmath>
#include <format>
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

constexpr i32 Size = 6;
constexpr i32 MethodCount = 2;
constexpr i32 NormCount = 3;
constexpr i32 ThresholdCount = 7;
constexpr i32 DefaultMaxIterations = 1'000'000;
constexpr i32 DefaultHalfPower = 1;
constexpr i32 MaxHalfPower = 8;
constexpr f64 Tolerance = 1.0e-9;

constexpr std::array<f64, ThresholdCount> Thresholds {
    1.0e-2,
    1.0e-3,
    1.0e-4,
    1.0e-5,
    1.0e-10,
    1.0e-12,
    1.0e-15,
};

constexpr std::array<std::string_view, NormCount> NormLabels {
    "||.||_inf",
    "||.||_1",
    "||.||_2l",
};

constexpr std::array<std::string_view, MethodCount> MethodLabels {
    "minimal residuals",
    "steepest descent",
};

constexpr std::array<MDAA::NormKind, NormCount> NormKinds {
    MDAA::NormKind::Infinity,
    MDAA::NormKind::One,
    MDAA::NormKind::L2L,
};

struct Settings final {
    i32                             Method = 0;
    i32                             Norm = 0;
    i32                             HalfPower = DefaultHalfPower;
    i32                             MaxIterations = DefaultMaxIterations;
    std::optional<std::vector<f64>> Start = std::nullopt;
    bool                            Trace = false;
};

struct Cell final {
    i32 Iteration = -1;
    f64 Value = 0.0;
};

using Table = std::array<std::array<Cell, ThresholdCount>, NormCount>;

std::string Text(const Eigen::VectorXd &x) {
    std::string text = "(";

    for (i32 index = 0; index < static_cast<i32>(x.size()); index++) {
        text += std::format("{}{:.6f}", index == 0 ? "" : ", ", x[index]);
    }

    return text + ")";
}

Eigen::VectorXd ToVector(const std::vector<f64> &values) {
    Eigen::VectorXd x(static_cast<Eigen::Index>(values.size()));

    for (usize index = 0; index < values.size(); index++) {
        x[static_cast<Eigen::Index>(index)] = values[index];
    }

    return x;
}

f64 Value(const Eigen::MatrixXd &a, const Eigen::VectorXd &f, const Eigen::VectorXd &x) {
    return x.dot(a * x) - (2.0 * f.dot(x));
}

void PrintMatrix(const Eigen::MatrixXd &a) {
    for (Eigen::Index row = 0; row < a.rows(); row++) {
        std::print("  [");

        for (Eigen::Index column = 0; column < a.cols(); column++) {
            std::print("{}{:8.6f}", column == 0 ? "" : " ", a(row, column));
        }

        std::println("]");
    }
}

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
    std::println("  --method NAME      mn or sd, default both");
    std::println("  --norm NAME        inf, 1 or 2l, default all");
    std::println("  --l L              the exponent of ||X||_2l is 2L, default {}", DefaultHalfPower);
    std::println("  --start a,b[,c...] one starting point instead of the built-in ones");
    std::println("  --max-iterations N the limit of the iterations, default {}", DefaultMaxIterations);
    std::println("  --trace            print every iterate");
}

bool WantsHelp(const int argc, const char *const *argv) {
    for (i32 index = 1; index < argc; index++) {
        if (std::string_view(argv[index]) == "--help" || std::string_view(argv[index]) == "-h") {
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

i32 ParseMethod(const std::string_view value) {
    if (value == "mn") {
        return 1;
    }

    if (value == "sd") {
        return 2;
    }

    MDAA_FATAL(true, "--method takes mn or sd, got '{}'", value);

    return 0;
}

i32 ParseNorm(const std::string_view value) {
    if (value == "inf") {
        return 1;
    }

    if (value == "1") {
        return 2;
    }

    if (value == "2l") {
        return 3;
    }

    MDAA_FATAL(true, "--norm takes inf, 1 or 2l, got '{}'", value);

    return 0;
}

void ReadFlag(
    Settings                &settings,
    const std::string_view   flag,
    const char *const *const argv,
    const int                argc,
    i32                     &index) {
    if (flag == "--trace") {
        settings.Trace = true;
    } else if (flag == "--method") {
        settings.Method = ParseMethod(FlagValue(flag, argv, argc, index));
    } else if (flag == "--norm") {
        settings.Norm = ParseNorm(FlagValue(flag, argv, argc, index));
    } else if (flag == "--l") {
        settings.HalfPower = ParseInteger(FlagValue(flag, argv, argc, index));
    } else if (flag == "--start") {
        settings.Start = ParseList(FlagValue(flag, argv, argc, index));
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
        settings.Method < 0 || settings.Method > MethodCount,
        "--method takes mn or sd, got {}",
        settings.Method);
    MDAA_FATAL(
        settings.Norm < 0 || settings.Norm > NormCount,
        "--norm takes inf, 1 or 2l, got {}",
        settings.Norm);
    MDAA_FATAL(
        settings.HalfPower < 1 || settings.HalfPower > MaxHalfPower,
        "--l takes 1..{}, got {}",
        MaxHalfPower,
        settings.HalfPower);
    MDAA_FATAL(settings.MaxIterations < 1, "--max-iterations takes a positive number");

    if (settings.Start.has_value()) {
        MDAA_FATAL(
            settings.Start->size() != Size,
            "--start needs {} numbers, got {}",
            Size,
            settings.Start->size());
    }

    return settings;
}

void PrintTable(const Table &table, const std::array<bool, NormCount> &active) {
    std::print("      {:>8} |", "epsilon");

    for (i32 norm = 0; norm < NormCount; norm++) {
        if (active[static_cast<usize>(norm)]) {
            std::print(" {:>9} {:>14}", NormLabels[static_cast<usize>(norm)], "f(x)");
        }
    }

    std::println();

    for (i32 index = 0; index < ThresholdCount; index++) {
        std::print("      {:>8.0e} |", Thresholds[static_cast<usize>(index)]);

        for (i32 norm = 0; norm < NormCount; norm++) {
            const auto which = static_cast<usize>(norm);

            if (!active[which]) {
                continue;
            }

            const Cell &cell = table[which][static_cast<usize>(index)];
            if (cell.Iteration >= 0) {
                std::print(" {:>9} {:>14.6e}", cell.Iteration, cell.Value);
            } else {
                std::print(" {:>9} {:>14}", "> limit", "-");
            }
        }

        std::println();
    }
}

MDAA::LinearSolverResult RunColumn(
    const Eigen::MatrixXd &a,
    const Eigen::VectorXd &f,
    const Eigen::VectorXd &start,
    const i32              method,
    const i32              norm,
    const Settings        &settings,
    Table                 &table) {
    const auto which = static_cast<usize>(norm - 1);

    const MDAA::LinearSolverOptions options {
        .Method = method == 1 ? MDAA::IterativeMethod::ResidualMinimization
                              : MDAA::IterativeMethod::SteepestDescent,
        .Norm = {.Kind = NormKinds[which], .HalfPower = settings.HalfPower},
        .Epsilon = Thresholds.back(),
        .MaxIterations = settings.MaxIterations,
    };

    if (settings.Trace) {
        std::println("    {}", NormLabels[which]);
        std::println(
            "      {:>6} {:>14} {:>16} {}",
            "iter",
            "step",
            "f(x)",
            "x");
    }

    const MDAA::LinearSolverObserver observer =
        [&](const i32 iteration, const Eigen::VectorXd &point, const Eigen::VectorXd &,
            const f64 size) {
            const f64 value = Value(a, f, point);

            if (settings.Trace) {
                std::println(
                    "      {:>6} {:>14.6e} {:>16.10e} {}",
                    iteration,
                    size,
                    value,
                    Text(point));
            }

            for (i32 index = 0; index < ThresholdCount; index++) {
                Cell &cell = table[which][static_cast<usize>(index)];

                if (cell.Iteration < 0 && size < Thresholds[static_cast<usize>(index)]) {
                    cell.Iteration = iteration;
                    cell.Value = value;
                }
            }

            return table[which].back().Iteration >= 0;
        };

    return MDAA::SolveLinearSystem(a, f, start, options, observer);
}

void PrintResult(
    const std::string_view          method,
    const std::string_view          norm,
    const MDAA::LinearSolverResult &result,
    const f64                       gap,
    const f64                       value,
    const f64                       star) {
    std::println(
        "      {:>16} x = {}  f = {:.10e}  gap = {:.3e}  iterations = {}",
        norm,
        Text(result.Point),
        value,
        gap,
        result.Iterations);

    if (result.Reason != MDAA::StopReason::Iterations) {
        MDAA_CHECKF(
            gap < Tolerance,
            "{} {} has to reach the minimizer, the gap is {}",
            method,
            norm,
            gap);
        MDAA_CHECKF(
            std::abs(value - star) < Tolerance,
            "{} {} has to reach the minimum, f - f* = {}",
            method,
            norm,
            value - star);
    }
}

void RunStart(
    const Eigen::MatrixXd &a,
    const Eigen::VectorXd &f,
    const Eigen::VectorXd &reference,
    const f64              star,
    const Eigen::VectorXd &start,
    const Settings        &settings) {
    std::println("  start = {}, f(x) = {:.10e}", Text(start), Value(a, f, start));

    const i32 firstMethod = settings.Method > 0 ? settings.Method : 1;
    const i32 lastMethod = settings.Method > 0 ? settings.Method : MethodCount;
    const i32 firstNorm = settings.Norm > 0 ? settings.Norm : 1;
    const i32 lastNorm = settings.Norm > 0 ? settings.Norm : NormCount;

    for (i32 method = firstMethod; method <= lastMethod; method++) {
        Table                                           table {};
        std::array<bool, NormCount>                     active {};
        std::array<MDAA::LinearSolverResult, NormCount> results {};

        std::println("  {}", MethodLabels[static_cast<usize>(method - 1)]);

        for (i32 norm = firstNorm; norm <= lastNorm; norm++) {
            const auto which = static_cast<usize>(norm - 1);
            active[which] = true;
            results[which] = RunColumn(a, f, start, method, norm, settings, table);
        }

        PrintTable(table, active);

        for (i32 norm = firstNorm; norm <= lastNorm; norm++) {
            const auto                      which = static_cast<usize>(norm - 1);
            const MDAA::LinearSolverResult &result = results[which];
            const f64                       gap = (result.Point - reference).cwiseAbs().maxCoeff();
            const f64                       value = Value(a, f, result.Point);

            PrintResult(
                MethodLabels[static_cast<usize>(method - 1)],
                NormLabels[which],
                result,
                gap,
                value,
                star);
        }
    }

    std::println();
}

} // namespace

int main(const int argc, char *const *argv) {
    Eigen::setNbThreads(1);

    if (WantsHelp(argc, argv)) {
        PrintUsage(argv[0]);
        return 0;
    }

    const Settings settings = ParseSettings(argc, argv);

    Eigen::MatrixXd a(Size, Size);
    a << 4, -1, 0, -1, 0, 0, //
        -1, 4, -1, 0, -1, 0, //
        0, -1, 4, 0, 0, -1,  //
        -1, 0, 0, 4, -1, 0,  //
        0, -1, 0, -1, 4, -1, //
        0, 0, -1, 0, -1, 4;

    Eigen::VectorXd f(Size);
    f << 0, 5, 0, 6, -2, 6;

    const Eigen::VectorXd reference = a.partialPivLu().solve(f);
    const f64             star = Value(a, f, reference);

    std::println("f(x) = (Ax, x) - 2(f, x)");
    std::println("A =");
    PrintMatrix(a);
    std::println("f = {}", Text(f));
    std::println("minimizer = {}", Text(reference));
    std::println("minimum f* = {:.16e}", star);
    std::println();

    std::vector<Eigen::VectorXd> starts;

    if (settings.Start.has_value()) {
        starts.emplace_back(ToVector(*settings.Start));
    } else {
        starts.emplace_back(Eigen::VectorXd::Zero(Size));
        starts.emplace_back(ToVector({1, 1, 1, 1, 1, 1}));
        starts.emplace_back(ToVector({5, -5, 5, -5, 5, -5}));
    }

    for (const Eigen::VectorXd &start : starts) {
        RunStart(a, f, reference, star, start, settings);
    }

    return 0;
}
