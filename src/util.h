#pragma once

#include "types.h"

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
}