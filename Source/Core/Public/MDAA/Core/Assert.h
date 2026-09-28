#pragma once

#include "MDAA/Core/Macro.h"

#include <cstdlib>
#include <format>
#include <iostream>
#include <print>
#include <utility>

namespace MDAA {

inline void Abort() {
    std::abort();
}

#ifdef MDAA_ENABLE_ASSERTS

inline void CheckAssertFailed(const char *expr, const char *where) {
    std::println(std::cerr, "[CHECK] assertion failed: {}\nat {} ", expr, where);
    ::MDAA::Abort();
}

template <typename... Args>
inline void CheckfAssertFailed(
    const char                 *expr,
    const char                 *where,
    std::format_string<Args...> fmt,
    Args &&...args) {
    std::println(std::cerr, "[CHECKF] assertion failed: {}\nat {} ", expr, where);
    std::println(std::cerr, fmt, std::forward<Args>(args)...);
    ::MDAA::Abort();
}

#define MDAA_CHECK(expr)                                     \
    do {                                                     \
        if (!(expr)) {                                       \
            ::MDAA::CheckAssertFailed(                       \
                #expr,                                       \
                __FILE__ ":" MDAA_STRINGIFY(__LINE__) ":1"); \
        }                                                    \
    } while (0)

#define MDAA_CHECKF(expr, fmt, ...)                         \
    do {                                                    \
        if (!(expr)) {                                      \
            ::MDAA::CheckfAssertFailed(                     \
                #expr,                                      \
                __FILE__ ":" MDAA_STRINGIFY(__LINE__) ":1", \
                fmt __VA_OPT__(, )                          \
                    __VA_ARGS__);                           \
        }                                                   \
    } while (0)

#else

#define MDAA_CHECK(expr) ((void)0)
#define MDAA_CHECKF(expr, fmt, ...) ((void)0)

#endif

template <typename... Args>
inline void FatalAssertFailed(
    const char                 *expr,
    const char                 *where,
    std::format_string<Args...> fmt,
    Args &&...args) {
    std::println(std::cerr, "Fatal error {}\nat {}", expr, where);
    std::println(std::cerr, fmt, std::forward<Args>(args)...);
    ::MDAA::Abort();
}

#define MDAA_FATAL(expr, fmt, ...)                          \
    do {                                                    \
        if (expr) {                                         \
            ::MDAA::FatalAssertFailed(                      \
                #expr,                                      \
                __FILE__ ":" MDAA_STRINGIFY(__LINE__) ":1", \
                fmt __VA_OPT__(, )                          \
                    __VA_ARGS__);                           \
        }                                                   \
    } while (0)

} // namespace MDAA
