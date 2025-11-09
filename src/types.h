#pragma once

#include <iostream>
#include <cstdint>
#include <cstring>
#include <vector>

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

    template <typename T>
    struct UnifiedVector {
        static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable for UnifiedVector");

        usize size = 0;
        T* data = nullptr;

        UnifiedVector() = default;

        explicit UnifiedVector(const usize size) : size(size) {
            data = static_cast<T*>(aligned_alloc(64, sizeof(T) * size));
            std::memset(data, 0, sizeof(T) * size);
        }

        UnifiedVector(const std::vector<T>& other) {
            resize(other.size());
            std::memcpy(data, other.data(), sizeof(T) * size);
        }

        UnifiedVector(const UnifiedVector& other) : size(other.size) {
            if (size > 0) {
                data = static_cast<T*>(aligned_alloc(64, sizeof(T) * size));
                std::memcpy(data, other.data, sizeof(T) * size);
            }
            else
                data = nullptr;
        }

        UnifiedVector(UnifiedVector&& other) noexcept : size(other.size), data(other.data) {
            other.size = 0;
            other.data = nullptr;
        }

        ~UnifiedVector() { if (data) free(data); }

        void resize(const usize newSize) {
            if (newSize != size) {
                if (data)
                    free(data);
                data = static_cast<T*>(aligned_alloc(64, sizeof(T) * newSize));
                size = newSize;

                std::memset(data, 0, sizeof(T) * size);
            }
        }

        T* begin() noexcept { return data; }
        const T* begin() const noexcept { return data; }
        T* end() noexcept { return data + size; }
        const T* end() const noexcept { return data + size; }

        T& operator[](const usize i) { return data[i]; }
        const T& operator[](const usize i) const { return data[i]; }

        UnifiedVector& operator=(const UnifiedVector& other) noexcept {
            if (this != &other) {
                resize(other.size);
                std::memcpy(data, other.data, sizeof(T) * size);
            }
            return *this;
        }
        UnifiedVector& operator=(UnifiedVector&& other) noexcept {
            if (this != &other) {
                if (data)
                    free(data);
                size = other.size;
                data = other.data;
                other.size = 0;
                other.data = nullptr;
            }
            return *this;
        }
    };
}