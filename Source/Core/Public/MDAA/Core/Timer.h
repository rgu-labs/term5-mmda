#pragma once

#include "MDAA/Core/Types.h"

#include <chrono>

namespace MDAA {

class Timer final {
  public:
    using Clock = std::chrono::steady_clock;

    void Start() {
        m_Start = Clock::now();
        m_Stop = m_Start;
        m_IsRunning = true;
    }

    void Stop() {
        if (!m_IsRunning) {
            return;
        }
        m_Stop = Clock::now();
        m_IsRunning = false;
    }

    void Reset() {
        m_Start = Clock::now();
        m_Stop = m_Start;
        m_IsRunning = false;
    }

    [[nodiscard]] bool IsRunning() const {
        return m_IsRunning;
    }

    [[nodiscard]] f64 ElapsedNanoseconds() const {
        return static_cast<f64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Now() - m_Start).count());
    }

    [[nodiscard]] f64 ElapsedMilliseconds() const {
        return ElapsedNanoseconds() / 1.0e6;
    }

    [[nodiscard]] f64 ElapsedSeconds() const {
        return ElapsedNanoseconds() / 1.0e9;
    }

  private:
    [[nodiscard]] Clock::time_point Now() const {
        return m_IsRunning ? Clock::now() : m_Stop;
    }

    Clock::time_point m_Start;
    Clock::time_point m_Stop;
    bool              m_IsRunning = false;
};

} // namespace MDAA
