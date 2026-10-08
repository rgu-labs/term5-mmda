#include "MDAA/Core/Assert.h"
#include "MDAA/Core/Types.h"
#include "MDAA/Optimization/Optimization.h"

#include <Eigen/Core>
#include <Eigen/LU>

#include <array>
#include <charconv>
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

constexpr i32 SystemCount = 6;
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

struct System final {
    Eigen::MatrixXd A;
    Eigen::VectorXd B;
};

struct Settings final {
    i32                             System = 0;
    i32                             Method = 0;
    i32                             Norm = 0;
    i32                             HalfPower = DefaultHalfPower;
    i32                             MaxIterations = DefaultMaxIterations;
    std::optional<std::vector<f64>> Start = std::nullopt;
    bool                            Trace = false;
};

using Table = std::array<std::array<i32, ThresholdCount>, NormCount>;

System MakeFirst() {
    System system;

    system.A.resize(4, 4);
    system.A << 2, 2, -1, 1, 4, 3, -1, 2, 8, 5, -3, 4, 3, 3, -2, 2;
    system.B.resize(4);
    system.B << 4, 6, 12, 6;

    return system;
}

System MakeSecond() {
    System system;

    system.A.resize(4, 4);
    system.A << 4, 1, -1, 1, 1, 4, -1, -1, -1, -1, 5, 1, 1, -1, 1, 3;
    system.B.resize(4);
    system.B << -2, -1, 0, 1;

    return system;
}

System MakeThird() {
    System system;

    system.A.resize(4, 4);
    system.A << 2.8, 2.1, -1.3, 0.3, -1.4, 4.5, -7.7, 1.3, 0.6, 2.1, -5.8, 2.4, 3.5, -6.5, 3.2,
        -7.9;
    system.B.resize(4);
    system.B << 1, 1, 1, 1;

    return system;
}

System MakeFourth() {
    System system;

    system.A.resize(5, 5);
    system.A << 4, 1, 1, 0, 1, 1, 3, 1, 1, 0, 1, 1, 5, -1, -1, 0, 1, -1, 4, 0, 1, 0, -1, 0, 4;
    system.B.resize(5);
    system.B << 6, 6, 6, 6, 6;

    return system;
}

System MakeFifth() {
    System system;

    system.A.resize(6, 6);
    system.A << 4, -1, 0, -1, 0, 0, -1, 4, -1, 0, -1, 0, 0, -1, 4, 0, 0, -1, -1, 0, 0, 4, -1, 0,
        0, -1, 0, -1, 4, -1, 0, 0, -1, 0, -1, 4;
    system.B.resize(6);
    system.B << 0, 5, 0, 6, -2, 6;

    return system;
}

System MakeSixth() {
    System system;

    system.A.resize(6, 6);
    system.A << 4, -1, 0, 0, 0, 0, -1, 4, -1, 0, 0, 0, 0, -1, 4, 0, 0, 0, 0, 0, 0, 4, -1, 0, 0,
        0, 0, -1, 4, -1, 0, 0, 0, 0, -1, 4;
    system.B.resize(6);
    system.B << 0, 5, 0, 6, -2, 6;

    return system;
}

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

Eigen::VectorXd StartFor(const Settings &settings, const Eigen::Index size) {
    if (!settings.Start.has_value()) {
        return Eigen::VectorXd::Zero(size);
    }

    MDAA_CHECKF(
        settings.Start->size() == static_cast<usize>(size),
        "the start needs {} numbers, got {}",
        size,
        settings.Start->size());

    return ToVector(*settings.Start);
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
    std::println("  --system N        solve only the system N, 1..{}, default all", SystemCount);
    std::println("  --method NAME     mn or sd, default both");
    std::println("  --norm NAME       inf, 1 or 2l, default all");
    std::println("  --l L             the exponent of ||X||_2l is 2L, default {}", DefaultHalfPower);
    std::println("  --start a,b[,c...] one starting point instead of zero");
    std::println("  --max-iterations N the limit of the iterations, default {}", DefaultMaxIterations);
    std::println("  --trace           print every iterate");
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
    } else if (flag == "--system") {
        settings.System = ParseInteger(FlagValue(flag, argv, argc, index));
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
        settings.System < 0 || settings.System > SystemCount,
        "--system takes 1..{}, got {}",
        SystemCount,
        settings.System);
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

    return settings;
}

void PrintTable(const Table &table, const std::array<bool, NormCount> &active) {
    std::print("      {:>8} |", "epsilon");

    for (i32 norm = 0; norm < NormCount; norm++) {
        if (active[static_cast<usize>(norm)]) {
            std::print(" {:>14}", NormLabels[static_cast<usize>(norm)]);
        }
    }

    std::println();

    for (i32 index = 0; index < ThresholdCount; index++) {
        std::print("      {:>8.0e} |", Thresholds[static_cast<usize>(index)]);

        for (i32 norm = 0; norm < NormCount; norm++) {
            if (!active[static_cast<usize>(norm)]) {
                continue;
            }

            const i32 cell = table[static_cast<usize>(norm)][static_cast<usize>(index)];
            if (cell >= 0) {
                std::print(" {:>14}", cell);
            } else {
                std::print(" {:>14}", "> limit");
            }
        }

        std::println();
    }
}

MDAA::LinearSolverResult RunColumn(
    const System          &system,
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
        std::println("      {:>6} {:>14} {}", "iter", "step", "x");
    }

    const MDAA::LinearSolverObserver observer =
        [&](const i32 iteration, const Eigen::VectorXd &point, const Eigen::VectorXd &,
            const f64 size) {
            if (settings.Trace) {
                std::println("      {:>6} {:>14.6e} {}", iteration, size, Text(point));
            }

            for (i32 index = 0; index < ThresholdCount; index++) {
                i32 &cell = table[which][static_cast<usize>(index)];

                if (cell < 0 && size < Thresholds[static_cast<usize>(index)]) {
                    cell = iteration;
                }
            }

            return table[which].back() >= 0;
        };

    return MDAA::SolveLinearSystem(system.A, system.B, start, options, observer);
}

void RunSystem(const System &system, const i32 index, const Settings &settings) {
    const Eigen::VectorXd reference = system.A.partialPivLu().solve(system.B);
    const Eigen::VectorXd start = StartFor(settings, system.A.rows());

    std::println("system {}", index + 1);
    std::println("  start = {}", Text(start));
    std::println("  solution = {}", Text(reference));

    const i32 firstMethod = settings.Method > 0 ? settings.Method : 1;
    const i32 lastMethod = settings.Method > 0 ? settings.Method : MethodCount;
    const i32 firstNorm = settings.Norm > 0 ? settings.Norm : 1;
    const i32 lastNorm = settings.Norm > 0 ? settings.Norm : NormCount;

    for (i32 method = firstMethod; method <= lastMethod; method++) {
        Table table {};
        for (auto &row : table) {
            row.fill(-1);
        }

        std::array<bool, NormCount>                     active {};
        std::array<MDAA::LinearSolverResult, NormCount> results {};

        std::println("  {}", MethodLabels[static_cast<usize>(method - 1)]);

        for (i32 norm = firstNorm; norm <= lastNorm; norm++) {
            const auto which = static_cast<usize>(norm - 1);
            active[which] = true;
            results[which] = RunColumn(system, start, method, norm, settings, table);
        }

        PrintTable(table, active);

        for (i32 norm = firstNorm; norm <= lastNorm; norm++) {
            const auto                      which = static_cast<usize>(norm - 1);
            const MDAA::LinearSolverResult &result = results[which];
            const f64                       gap = (result.Point - reference).cwiseAbs().maxCoeff();

            std::println(
                "      {:>16} x = {}  gap = {:.3e}  iterations = {}",
                NormLabels[which],
                Text(result.Point),
                gap,
                result.Iterations);

            if (result.Reason != MDAA::StopReason::Iterations) {
                MDAA_CHECKF(
                    gap < Tolerance,
                    "system {} {} {} has to reach the solution, the gap is {}",
                    index + 1,
                    MethodLabels[static_cast<usize>(method - 1)],
                    NormLabels[which],
                    gap);
            }
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

    MDAA_FATAL(
        settings.Start.has_value() && settings.System == 0,
        "--start needs --system, the systems have different sizes");

    const std::array<System, SystemCount> systems {
        MakeFirst(),
        MakeSecond(),
        MakeThird(),
        MakeFourth(),
        MakeFifth(),
        MakeSixth(),
    };

    for (i32 index = 0; index < SystemCount; index++) {
        if (settings.System > 0 && settings.System != index + 1) {
            continue;
        }

        RunSystem(systems[static_cast<usize>(index)], index, settings);
    }

    return 0;
}
