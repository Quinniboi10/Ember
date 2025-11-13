#pragma once

#include "types.h"

#include <cblas.h>
#include <vector>
#include <array>

namespace Ember {
    enum Device { CPU, GPU };
    namespace internal {
        void sgemmDispatch(
            Device device,
            bool transA, bool transB,
            int M, int N, int K,
            float alpha,
            const float* A, int lda,
            const float* B, int ldb,
            float beta,
            float* C, int ldc);

        extern Device device;

        void memcpy(Device device, void* dest, const void* src, usize size);

        void* malloc(Device device, usize size);
        void free(Device device, void* ptr);

        struct SharedVector {
            Device loc = CPU;
            float* data = nullptr;

            usize size = 0;

            SharedVector() = default;
            // Returns a GPU tensor if available
            explicit SharedVector(const usize size);
            // Always returns a CPU tensor
            SharedVector(const std::vector<float>& other);

            SharedVector(const SharedVector& other);
            SharedVector(SharedVector&& other) noexcept : loc(other.loc), data(other.data), size(other.size) {
                other.size = 0;
                other.data = nullptr;
            }

            ~SharedVector() {
                if (data)
                    free(loc, data);
            }

            void resize(const usize newSize);

            float* begin() noexcept { return data; }
            const float* begin() const noexcept { return data; }
            float* end() noexcept { return data + size; }
            const float* end() const noexcept { return data + size; }

            float& operator[](const usize i) {
                assert(i < size);
                return data[i];
            }
            const float& operator[](const usize i) const {
                assert(i < size);
                return data[i];
            }

            SharedVector& operator=(const SharedVector& other) noexcept;
            SharedVector& operator=(SharedVector&& other) noexcept {
                if (this != &other) {
                    if (data)
                        free(loc, data);
                    size = other.size;
                    data = other.data;
                    other.size = 0;
                    other.data = nullptr;
                }
                return *this;
            }

            void moveTo(Device target);
        };

        template <typename T>
        concept UsizeLike = std::is_same_v<std::decay_t<T>, usize>;
    }

    class Tensor {
        internal::SharedVector underlying;

    public:
        usize dimensionality;

        std::vector<usize> dimensions;
        std::vector<usize> strides;

        Tensor() = default;

        template<internal::UsizeLike... Args>
        explicit Tensor(const Args... args) {
            dimensionality = sizeof...(Args);
            dimensions = { args... };
            underlying.resize((args * ...));
            calculateStrides();
        }

        explicit Tensor(const std::vector<usize>& dimensions) : dimensions(dimensions) {
            dimensionality = dimensions.size();

            u64 size = 1;
            for (const usize d : dimensions)
                size *= d;
            underlying.resize(size);
            calculateStrides();
        }

        Tensor(const std::vector<float>& input)  {
            dimensionality = 1;
            dimensions.resize(1);
            dimensions[0] = input.size();
            underlying = input;
            calculateStrides();
        }

        template<internal::UsizeLike... Args>
        void resize(Args... args) {
            dimensionality = sizeof...(Args);
            dimensions = { args... };
            underlying.resize((args * ...));
            calculateStrides();
        }

        void resize(const std::vector<usize>& newDims) {
            dimensionality = newDims.size();
            dimensions = newDims;

            u64 size = 1;
            for (const usize d : dimensions)
                size *= d;
            underlying.resize(size);
            calculateStrides();
        }

        void setDimension(const usize dimIdx, const usize newSize) {
            assert(dimIdx < dimensionality);
            dimensions[dimIdx] = newSize;

            u64 size = 1;
            for (const usize d : dimensions)
                size *= d;
            underlying.resize(size);
            calculateStrides();
        }

        // Add a leading 1 to the dimensions
        void unsqueeze() {
            std::vector<usize> newSizes(1 + dimensions.size());
            newSizes[0] = 1;
            std::memcpy(newSizes.data() + 1, dimensions.data(), dimensions.size() * sizeof(usize));

            resize(newSizes);
        }

        float* ptr() { return underlying.data; }
        const float* ptr() const { return underlying.data; }

        usize size() const { return underlying.size; }
        auto begin() { return underlying.begin(); }
        auto begin() const { return underlying.begin(); }
        auto end() { return underlying.end(); }
        auto end() const { return underlying.end(); }

        void fill(const float value) {
            std::fill(underlying.begin(), underlying.end(), value);
        }

        void calculateStrides() {
            strides.resize(dimensionality);
            if (dimensionality == 0) return;

            strides[dimensionality - 1] = 1;

            for (int i = dimensionality - 2; i >= 0; i--)
                strides[i] = strides[i + 1] * dimensions[i + 1];
        }

        // Get the dimensionality
        auto& dims() { return dimensions; }
        const auto& dims() const { return dimensions; }
        usize dim(const usize idx) const { return dimensions[idx]; }

        // Leave the data but change the dimensions
        // assumes the size doesn't change
        void reshape(const std::vector<usize>& newDims) {
            dimensionality = newDims.size();
            dimensions = newDims;

            #ifndef NDEBUG
                u64 size = 1;
                for (const usize d : dimensions)
                    size *= d;
                assert(underlying.size == size);
            #endif

            calculateStrides();
        }

        internal::SharedVector& data() {
            assert(underlying.loc == CPU);
            return underlying;
        }

        const internal::SharedVector& data() const {
            assert(underlying.loc == CPU);
            return underlying;
        }

        void to(const Device device) { underlying.moveTo(device); }

        Device getDevice() const { return underlying.loc; }

        float& operator()(const usize i) {
            assert(dimensionality == 1);
            return underlying[i];
        }

        const float& operator()(const usize i) const {
            assert(dimensionality == 1);
            return underlying[i];
        }

        float& operator()(const usize i, const usize j) {
            assert(dimensionality == 2);
            return underlying[i * strides[0] + j];
        }

        const float& operator()(const usize i, const usize j) const {
            assert(dimensionality == 2);
            return underlying[i * strides[0] + j];
        }

        template<typename... Args>
        float& operator()(Args... args) {
            assert(sizeof...(Args) == dimensionality);
            usize idx = 0;
            usize strideIdx = 0;
            ((idx += static_cast<usize>(args) * strides[strideIdx++]), ...);
            return underlying[idx];
        }

        template<typename... Args>
        const float& operator()(Args... args) const {
            assert(sizeof...(Args) == dimensionality);
            usize idx = 0;
            usize strideIdx = 0;
            ((idx += static_cast<usize>(args) * strides[strideIdx++]), ...);
            return underlying[idx];
        }

        // Matrix operations

        // Compute a * b then add to the current tensor
        void madd(const Tensor& a, const Tensor& b) { madd(a, b, false, false); }
        // Compute a * b then add to the current tensor
        void madd(const Tensor& a, const Tensor& b, const bool transposeA, const bool transposeB);

        // Compute a * b then add to the current tensor
        void madd(const bool transA, const bool transB, const int M, const int N, const int K,
         const float alpha, const float* A, const int lda, const float* B, const int ldb, const float beta) {
            internal::sgemmDispatch(
                underlying.loc,
                transA, transB,
                M, N, K,
                alpha,
                A, lda,
                B, ldb,
                beta,
                this->ptr(), static_cast<int>(this->dim(1))
            );
        }

        void axpy(const float scalar, const Tensor& other);
    };
}