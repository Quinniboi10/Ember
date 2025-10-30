#pragma once

#include <iostream>
#include <cstdint>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#endif

#ifndef NDEBUG
#include <boost/stacktrace.hpp>
#endif

#undef assert

namespace Ember {
    #define exitWithMsg(msg, code) { \
        std::cout << "**ERROR**  " << msg << std::endl; \
        std::exit(code); \
    }

    #ifndef NDEBUG
        #define assert(x) \
            if (!(x)) [[unlikely]] { \
                std::cout << std::endl << std::endl << boost::stacktrace::stacktrace() << std::endl << "Assertion failed: " << #x << ", file " << __FILE__ << ", line " << __LINE__ << std::endl; \
                std::terminate(); \
            }
    #else
        #define assert(x) ;
    #endif

    using u64 = uint64_t;
    using u32 = uint32_t;
    using u16 = uint16_t;
    using u8  = uint8_t;

    using i64 = int64_t;
    using i32 = int32_t;
    using i16 = int16_t;
    using i8  = int8_t;

    using usize = size_t;

    namespace internal {
        namespace cursor {
            [[maybe_unused]] inline void clearAll(std::ostream& out = std::cout) { out << "\033[2J\033[H"; }
            [[maybe_unused]] inline void clear(std::ostream& out = std::cout) { out << "\033[2K\r"; }
            [[maybe_unused]] inline void clearDown(std::ostream& out = std::cout) { out << "\x1b[J"; }
            [[maybe_unused]] inline void home(std::ostream& out = std::cout) { out << "\033[H"; }
            [[maybe_unused]] inline void up(std::ostream& out = std::cout) { out << "\033[A"; }
            [[maybe_unused]] inline void down(std::ostream& out = std::cout) { out << "\033[B"; }
            [[maybe_unused]] inline void begin(std::ostream& out = std::cout) { out << "\033[1G"; }
            [[maybe_unused]] inline void goTo(const usize x, const usize y, std::ostream& out = std::cout) { out << "\033[" << y << ";" << x << "H"; }

            [[maybe_unused]] inline void hide(std::ostream& out = std::cout) { out << "\033[?25l"; }
            [[maybe_unused]] inline void show(std::ostream& out = std::cout) { out << "\033[?25h"; }
        }

        struct UnicodeTerminalInitializer {
            UnicodeTerminalInitializer() {
                #ifdef _WIN32
                SetConsoleOutputCP(CP_UTF8);
                #endif
            }
        };

        static inline UnicodeTerminalInitializer unicodeTerminalInitializer;
    }
}