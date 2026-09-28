#pragma once

#include "MDAA/Core/Timer.h"
#include "MDAA/Core/Types.h"

#include <algorithm>
#include <cstdio>
#include <format>
#include <print>
#include <string>
#include <string_view>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace MDAA {

// A single line progress bar for stdout, meant for the long stages of a lab.
// On a terminal the bar repaints in place, redirected to a file or a pipe it
// stays silent and prints one line per finished stage instead. Constructing
// the bar is what turns it on, there is no separate switch. Report is not
// thread safe, exactly one thread may paint while others report to it.
class ProgressBar final {
  public:
    static constexpr i32 DefaultWidth = 32;
    static constexpr i32 DefaultIntervalMilliseconds = 80;

    // The label is copied here, a temporary std::string may be passed safely.
    ProgressBar(
        std::string_view label,
        u64              total,
        i32              width = DefaultWidth,
        i32              intervalMilliseconds = DefaultIntervalMilliseconds) : m_Label(label),
                                                                  m_Total(total == 0 ? 1 : total),
                                                                  m_Width(width > 0 ? width : DefaultWidth),
                                                                  m_IntervalMilliseconds(
                                                                      intervalMilliseconds >= 0 ? intervalMilliseconds : DefaultIntervalMilliseconds),
                                                                  m_Interactive(IsInteractive()) {
        m_Timer.Start();
    }

    // done is clamped to total, reporting total completes the bar.
    void Report(u64 done) {
        if (m_Finished) {
            return;
        }
        if (m_Interactive) {
            Draw(done);
        }
        if (done >= m_Total) {
            Complete();
        }
    }

    // Completes the bar, does nothing when it is already finished.
    void Finish() {
        Report(m_Total);
    }

  private:
    static bool IsInteractive() {
#ifdef _WIN32
        return ::_isatty(::_fileno(stdout)) != 0;
#else
        return ::isatty(::fileno(stdout)) != 0;
#endif
    }

    // Repaints in place, throttled unless the bar is about to complete.
    void Draw(u64 done) {
        const f64 seconds = m_Timer.ElapsedSeconds();
        if (done < m_Total && seconds - m_LastPaint < (m_IntervalMilliseconds / 1000.0)) {
            return;
        }
        m_LastPaint = seconds;

        const f64 fraction =
            static_cast<f64>(std::min(done, m_Total)) / static_cast<f64>(m_Total);
        const auto filled = static_cast<i32>(fraction * static_cast<f64>(m_Width));

        std::string bar;
        bar.reserve(static_cast<usize>(m_Width));
        for (i32 i = 0; i < m_Width; i++) {
            bar += (i < filled) ? '#' : '.';
        }

        const f64 eta = (done == 0) ? 0.0 : (seconds / fraction) - seconds;
        std::print(
            "\r  {:<34} |{}| {:>5.1f}%  {:>6.2f}s  eta {:>6.2f}s",
            m_Label,
            bar,
            fraction * 100.0,
            seconds,
            eta);
        (void)std::fflush(stdout);
    }

    // Ends the line on a terminal, otherwise leaves a plain line in the log.
    void Complete() {
        m_Finished = true;
        if (m_Interactive) {
            std::println();
            return;
        }
        std::println("  {:<34} done in {:.2f}s", m_Label, m_Timer.ElapsedSeconds());
    }

    Timer       m_Timer;
    std::string m_Label;
    u64         m_Total;
    i32         m_Width;
    i32         m_IntervalMilliseconds;
    f64         m_LastPaint = -1.0e9;
    bool        m_Interactive = false;
    bool        m_Finished = false;
};

} // namespace MDAA
