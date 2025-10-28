#pragma once

#include "types.h"

#include "../external/fmt/format.h"

#include <vector>

namespace Ember {
    // Tensor recursive case
    // Tensor currently has a rather foolish implementation since it doesn't flatten the memory
    template<usize dimensionality>
    struct Tensor {
        static_assert(dimensionality > 1, "dimensionality must be >= 1");

        std::vector<Tensor<dimensionality - 1>> data;

        Tensor() = default;

        // Resize with a variadic pack of sizes, e.g. tensor.resize(3, 4, 5);
        template<typename... Args>
        void resize(const usize firstDim, Args... restDims) {
            static_assert(sizeof...(Args) == dimensionality - 1,
                "Number of arguments to resize() must match tensor dimensionality");

            data.resize(firstDim);
            for (auto& subTensor : data)
                subTensor.resize(restDims...);
        }

        void fill(const float value) {
            for (auto& subTensor : data)
                subTensor.fill(value);
        }

        usize size() const { return data.size(); }

        auto begin() { return data.begin(); }
        auto end() { return data.end(); }
        auto begin() const { return data.begin(); }
        auto end() const { return data.end(); }

        Tensor<dimensionality - 1>& operator[](const usize idx) { return data[idx]; }
        const Tensor<dimensionality - 1>& operator[](const usize idx) const { return data[idx]; }

        Tensor& operator=(const Tensor& other) {
            data = other.data;
            return *this;
        }
    };

    // Tensor base case
    template<>
    struct Tensor<1> {
        std::vector<float> data;

        Tensor() = default;
        Tensor(const std::vector<float> &data) : data(data) {}
        explicit Tensor(const usize size, const float def = 0.0f) : data(size, def) {}

        void resize(const usize size) {
            data.resize(size);
        }

        void fill(const float value) {
            std::fill(data.begin(), data.end(), value);
        }

        usize size() const { return data.size(); }

        auto begin() { return data.begin(); }
        auto end() { return data.end(); }
        auto begin() const { return data.begin(); }
        auto end() const { return data.end(); }

        auto ptr() { return data.data(); }
        auto ptr() const { return data.data(); }

        float& operator[](const usize idx) { return data[idx]; }
        const float& operator[](const usize idx) const { return data[idx]; }

        Tensor& operator=(const Tensor& other) = default;

        friend std::ostream& operator<<(std::ostream& os, const Tensor<1>& t) {
            os << "[";
            for (usize i = 0; i < t.size(); i++) {
                if (i < t.size() - 1)
                    os << fmt::format("{}, ", t.data[i]);
                else
                    os << fmt::format("{}]", t.data[i]);
            }
            return os;
        }
    };

    struct BlasMatrix {
        usize rows{};
        usize cols{};
        std::vector<float> data;

        BlasMatrix() = default;
        BlasMatrix(const usize rows, const usize cols) : rows(rows), cols(cols), data(rows * cols) {}

        void resize(const usize rows, const usize cols) {
            this->rows = rows;
            this->cols = cols;
            data.resize(rows * cols);
        }

        float* ptr() { return data.data(); }
        const float* ptr() const { return data.data(); }

        void fill(const float value) {
            for (float& f : data)
                f = value;
        }

        // (i, j) access (row i, column j)
        float& operator()(const usize i, const usize j)       { return data[i * cols + j]; }
        const float& operator()(const usize i, const usize j) const { return data[i * cols + j]; }
    };
}