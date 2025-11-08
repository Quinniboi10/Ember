#pragma once

#include "types.h"

#include "../external/fmt/format.h"

#include <cblas.h>
#include <vector>
#include <array>

namespace Ember {
    #define sgemm cblas_sgemm;

    namespace internal {
        template <typename T>
        concept UsizeLike = std::is_same_v<std::decay_t<T>, usize>;
    }

    struct Tensor {
        usize dimensionality;

        std::vector<usize> dimensions;
        std::vector<float> data;
        std::vector<usize> strides;

        Tensor() = default;

        template<internal::UsizeLike... Args>
        explicit Tensor(const Args... args) {
            dimensionality = sizeof...(Args);
            dimensions = { args... };
            data.resize((args * ...));
            calculateStrides();
        }

        explicit Tensor(const std::vector<usize>& dimensions) : dimensions(dimensions) {
            dimensionality = dimensions.size();

            u64 size = 1;
            for (const usize d : dimensions)
                size *= d;
            data.resize(size);
            calculateStrides();
        }

        Tensor(const std::vector<float>& input)  {
            dimensionality = 1;
            dimensions.resize(1);
            dimensions[0] = input.size();
            data = input;
            calculateStrides();
        }

        template<internal::UsizeLike... Args>
        void resize(Args... args) {
            dimensionality = sizeof...(Args);
            dimensions = { args... };
            data.resize((args * ...));
            calculateStrides();
        }

        void resize(const std::vector<usize>& newDims) {
            dimensionality = newDims.size();
            dimensions = newDims;

            u64 size = 1;
            for (const usize d : dimensions)
                size *= d;
            data.resize(size);
            calculateStrides();
        }

        void setDimension(const usize dimIdx, const usize newSize) {
            assert(dimIdx < dimensionality);
            dimensions[dimIdx] = newSize;

            u64 size = 1;
            for (const usize d : dimensions)
                size *= d;
            data.resize(size);
            calculateStrides();
        }

        // Add a leading 1 to the dimensions
        void unsqueeze() {
            std::vector<usize> newSizes(1 + dimensions.size());
            newSizes[0] = 1;
            std::memcpy(newSizes.data() + 1, dimensions.data(), dimensions.size() * sizeof(usize));

            resize(newSizes);
        }

        float* ptr() { return data.data(); }
        const float* ptr() const { return data.data(); }

        usize size() const { return data.size(); }
        auto begin() { return data.begin(); }
        auto begin() const { return data.begin(); }
        auto end() { return data.end(); }
        auto end() const { return data.end(); }

        void fill(const float value) {
            std::fill(data.begin(), data.end(), value);
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
                assert(data.size() == size);
            #endif

            calculateStrides();
        }

        float& operator[](const usize i) {
            assert(dimensionality == 1);
            return data[i];
        }

        const float& operator[](const usize i) const {
            assert(dimensionality == 1);
            return data[i];
        }

        float& operator[](const usize i, const usize j) {
            assert(dimensionality == 2);
            return data[i * strides[0] + j];
        }

        const float& operator[](const usize i, const usize j) const {
            assert(dimensionality == 2);
            return data[i * strides[0] + j];
        }

        template<typename... Args>
        float& operator[](Args... args) {
            assert(sizeof...(Args) == dimensionality);
            usize idx = 0;
            usize strideIdx = 0;
            ((idx += static_cast<usize>(args) * strides[strideIdx++]), ...);
            return data[idx];
        }

        template<typename... Args>
        const float& operator[](Args... args) const {
            assert(sizeof...(Args) == dimensionality);
            usize idx = 0;
            usize strideIdx = 0;
            ((idx += static_cast<usize>(args) * strides[strideIdx++]), ...);
            return data[idx];
        }


        // Matrix operations

        // Compute a * b then add to the current tensor
        void madd(const Tensor& a, const Tensor& b) { madd(a, b, false, false); }
        // Compute a * b then add to the current tensor
        void madd(const Tensor& a, const Tensor& b, const bool transposeA, const bool transposeB) {
            assert(dimensionality == 2);
            assert(a.dimensionality == 2);
            assert(b.dimensionality == 2);

            // Logical dimensions for op(A) and op(B)
            const usize aRows = transposeA ? a.dim(1) : a.dim(0);
            const usize aCols = transposeA ? a.dim(0) : a.dim(1);
            const usize bRows = transposeB ? b.dim(1) : b.dim(0);
            const usize bCols = transposeB ? b.dim(0) : b.dim(1);

            // Ensure dimensions are compatible for C = op(A) * op(B) + C
            assert(aCols == bRows);
            assert(this->dim(0) == aRows);
            assert(this->dim(1) == bCols);

            // Matrix multiplication parameters
            const int M = static_cast<int>(this->dim(0)); // rows of C / op(A)
            const int N = static_cast<int>(this->dim(1)); // cols of C / op(B)
            const int K = static_cast<int>(aCols);            // inner dimension

            const auto transA = transposeA ? CblasTrans : CblasNoTrans;
            const auto transB = transposeB ? CblasTrans : CblasNoTrans;

            // Leading dimensions assuming row major
            const int lda = static_cast<int>(a.dim(1));
            const int ldb = static_cast<int>(b.dim(1));
            const int ldc = static_cast<int>(this->dim(1));

            // Perform C = op(A) * op(B) + C
            cblas_sgemm(
                CblasRowMajor,
                transA, transB,
                M, N, K,
                1.0f,
                a.ptr(), lda,
                b.ptr(), ldb,
                1.0f,
                this->ptr(), ldc
            );
        }
        // Compute a * b then add to the current tensor
        void madd(const CBLAS_TRANSPOSE transA, const CBLAS_TRANSPOSE transB, const blasint M, const blasint N, const blasint K,
         const float alpha, const float* A, const blasint lda, const float* B, const blasint ldb, const float beta) {
            cblas_sgemm(
                CblasRowMajor, transA, transB,
                M, N, K,
                alpha,
                A, lda,
                B, ldb,
                beta,
                this->ptr(), static_cast<int>(this->dim(1))
            );
        }
    };
}