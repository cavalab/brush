#ifndef BRUSH_UTIL_PROGRESS_H
#define BRUSH_UTIL_PROGRESS_H

#include <algorithm>
#include <cmath>
#include <deque>
#include <numeric>
#include <optional>
#include <string>
#include <fmt/format.h>

namespace Brush::Util {

inline std::string format_duration(double seconds)
{
    const auto total = static_cast<unsigned long long>(std::round(std::max(0.0, seconds)));
    return fmt::format("{:02}:{:02}:{:02}", total / 3600, (total / 60) % 60, total % 60);
}

class RecentGenerationTimes {
public:
    void record(double seconds)
    {
        if (durations.size() == window_size)
            durations.pop_front();
        durations.push_back(seconds);
    }

    // Time limits are checked between generations, so a generation started
    // before the deadline can finish after it.
    std::optional<double> remaining(size_t generations,
                                    std::optional<double> time_limit_remaining = std::nullopt) const
    {
        if (generations == 0 || (time_limit_remaining && *time_limit_remaining <= 0))
            return 0.0;
        if (durations.empty())
            return std::nullopt;

        const double average = std::accumulate(durations.begin(), durations.end(), 0.0)
                               / durations.size();
        double estimate = average * generations;
        if (time_limit_remaining && average > 0)
            estimate = std::min(estimate, average * std::ceil(*time_limit_remaining / average));
        return estimate;
    }

private:
    static constexpr size_t window_size = 10;
    std::deque<double> durations;
};

} // namespace Brush::Util

#endif
