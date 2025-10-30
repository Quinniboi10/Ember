#pragma once

#include "types.h"

#include "../external/fmt/format.h"

#include <vector>
#include <array>

namespace Ember {
    namespace internal {
        template <typename T>
        concept UsizeLike = std::is_same_v<std::decay_t<T>, usize>;
    }

    template<usize dimensionality>
    struct Tensor {
        std::array<usize, dimensionality> dimensions;
        std::vector<float> data;

        Tensor() = default;

        template<internal::UsizeLike... Args>
        explicit Tensor(const Args... args) {
            static_assert(sizeof...(Args) == dimensionality, "Tensor must have the same size and dimensionality");
            dimensions = { args... };
            data.resize((args * ...));
        }

        explicit Tensor(const std::array<usize, dimensionality>& dimensions) : dimensions(dimensions) {
            u64 size = 1;

            for (const usize d : dimensions)
                size *= d;
            data.resize(size);
        }

        Tensor(const std::vector<float>& input) requires (dimensionality == 1) {
            dimensions[0] = input.size();
            data = input;
        }

        template<internal::UsizeLike... Args>
        void resize(Args... args) {
            static_assert(sizeof...(Args) == dimensionality, "Resized tensor must have the same dimensionality");
            dimensions = { args... };
            data.resize((args * ...));
        }

        void resize(const std::array<usize, dimensionality>& newDims) {
            u64 size = 1;
            dimensions = newDims;

            for (const usize d : dimensions)
                size *= d;
            data.resize(size);
        }

        float* ptr() { return data.data(); }
        const float* ptr() const { return data.data(); }

        usize size() const { return data.size(); }
        auto begin() { return data.begin(); }
        auto begin() const { return data.begin(); }
        auto end() { return data.end(); }
        auto end() const { return data.end(); }

        void fill(const float value) {
            for (float& f : data)
                f = value;
        }

        // Get the dimensionality
        auto dims() { return dimensions; }
        const auto& dims() const { return dimensions; }
        usize dim(const usize idx) const { return dimensions[idx]; }

        template<typename... Args>
        float& operator[](Args... args) {
            static_assert(sizeof...(Args) == dimensionality, "Access must match tensor dimensionality");
            std::array<usize, dimensionality> indices{ static_cast<usize>(args)... };
            usize idx = 0;
            usize stride = 1;

            for (int i = dimensionality - 1; i >= 0; i--) {
                idx += indices[i] * stride;
                stride *= dimensions[i];
            }

            assert(idx < data.size());

            return data[idx];
        }

        template<typename... Args>
        const float& operator[](Args... args) const {
            static_assert(sizeof...(Args) == dimensionality, "Access must match tensor dimensionality");
            std::array<usize, dimensionality> indices{ static_cast<usize>(args)... };
            usize idx = 0;
            usize stride = 1;

            for (int i = dimensionality - 1; i >= 0; i--) {
                idx += indices[i] * stride;
                stride *= dimensions[i];
            }

            assert(idx < data.size());

            return data[idx];
        }
    };
}