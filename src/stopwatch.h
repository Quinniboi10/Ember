#pragma once

#include <chrono>

#include "types.h"

namespace Ember {
    inline std::string formatTime(const u64 timeInMS) {
        long long seconds = timeInMS / 1000;
        const long long hours = seconds / 3600;
        seconds %= 3600;
        const long long minutes = seconds / 60;
        seconds %= 60;

        std::string result;

        if (hours > 0)
            result += std::to_string(hours) + "h ";
        if (minutes > 0 || hours > 0)
            result += std::to_string(minutes) + "m ";
        if (seconds > 0 || minutes > 0 || hours > 0)
            result += std::to_string(seconds) + "s";
        if (result == "")
            return std::to_string(timeInMS) + "ms";
        return result;
    }

    template<typename Precision>
    class Stopwatch {
        std::chrono::high_resolution_clock::time_point startTime;
        std::chrono::high_resolution_clock::time_point pauseTime;

        bool paused;

        u64 pausedTime;

    public:
        Stopwatch() { start(); }

        void start() {
            startTime  = std::chrono::high_resolution_clock::now();
            pausedTime = 0;
            paused     = false;
        }

        void reset() { start(); }

        u64 elapsed() {
            u64 pausedTime = this->pausedTime;
            if (paused)
                pausedTime += std::chrono::duration_cast<Precision>(std::chrono::high_resolution_clock::now() - pauseTime).count();
            return std::chrono::duration_cast<Precision>(std::chrono::high_resolution_clock::now() - startTime).count() - pausedTime;
        }

        void pause() {
            paused    = true;
            pauseTime = std::chrono::high_resolution_clock::now();
        }
        void resume() {
            paused = false;
            pausedTime += std::chrono::duration_cast<Precision>(std::chrono::high_resolution_clock::now() - pauseTime).count();
        }
    };
}