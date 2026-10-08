#pragma once

#include "MDAA/Core/Types.h"

#include <concepts>
#include <limits>
#include <random>

namespace MDAA {

class Random final {
  public:
    using Engine = std::mt19937_64;

    static constexpr u64 DefaultSeed = 0x2545F4914F6CDD1DULL;

    Random() noexcept
        // NOLINTNEXTLINE(bugprone-random-generator-seed,cert-msc32-c,cert-msc51-cpp)
        : m_Engine(DefaultSeed) {
    }

    explicit Random(u64 seed) noexcept
        : m_Engine(seed) {
    }

    void Seed(u64 seed) noexcept {
        m_Engine.seed(seed);
    }

    template <typename T>
    [[nodiscard]] T Next() {
        if constexpr (std::floating_point<T>) {
            std::uniform_real_distribution<T> dist;
            return dist(m_Engine);
        } else {
            std::uniform_int_distribution<T> dist {
                std::numeric_limits<T>::lowest(),
                std::numeric_limits<T>::max(),
            };
            return dist(m_Engine);
        }
    }

    template <std::floating_point T>
    [[nodiscard]] T Uniform(T min, T max) {
        std::uniform_real_distribution<T> dist(min, max);
        return dist(m_Engine);
    }

    template <std::floating_point T>
    [[nodiscard]] T Normal(T mean, T deviation) {
        std::normal_distribution<T> dist(mean, deviation);
        return dist(m_Engine);
    }

  private:
    Engine m_Engine;
};

} // namespace MDAA
