#pragma once

#include "types.h"

#include <sstream>

namespace Ember {
    // Formats a number with commas
    inline std::string formatNum(const i64 v) {
        auto s = std::to_string(v);

        int n = s.length() - 3;
        if (v < 0)
            n--;
        while (n > 0) {
            s.insert(n, ",");
            n -= 3;
        }

        return s;
    }

    inline std::vector<std::string> split(const std::string& str, const char delim) {
        std::vector<std::string> result;

        std::istringstream stream(str);

        for (std::string token{}; std::getline(stream, token, delim);) {
            if (token.empty())
                continue;

            result.push_back(token);
        }

        return result;
    }
}