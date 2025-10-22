#pragma once

#include "stopwatch.h"
#include "../external/fmt/format.h"

#include <sstream>

namespace Ember {


    struct ProgressBar {
        Stopwatch<std::chrono::milliseconds> start;

        ProgressBar() {
            start.start();
        }

        std::string report(const u64 progress, const u64 total, const u64 barWidth) {
            std::ostringstream out;

            out << fmt::format("{:>4.0f}% ", static_cast<float>(progress * 100) / total);

            const u64 pos = barWidth * progress / total;
            out << "\u2595";
            for (u64 i = 0; i < barWidth - 1; ++i) {
                if (i < pos) out << "\u2588";
                else out << " ";
            }
            out << "\u258F";

            const u64 elapsed = std::max<u64>(start.elapsed(), 1);
            const u64 msRemaining = (total - progress) * elapsed / std::max<u64>(progress, 1);

            out << fmt::format(" {}/{} at {:.2f} per sec with {} remaining", progress, total, static_cast<float>(progress) / elapsed * 1000, formatTime(msRemaining));

            return out.str();
        }
    };
}