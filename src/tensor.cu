#include "tensor.h"

#ifdef EMBER_CUDA
#include <cublas_v2.h>
#endif

#ifdef EMBER_CUDA
struct HandleManager {
    cublasHandle_t handle;

    HandleManager() {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "Failed to create cuBLAS handle\n");
            std::abort();
        }

        // Enable tensor cores
        cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    }

    ~HandleManager() {
        cublasDestroy(handle);
    }
};

auto& getCublasHandle() {
    static HandleManager hm;
    return hm.handle;
}
#endif

namespace Ember {
    namespace internal {
        void memcpy([[maybe_unused]] Device device, void* dest, const void* src, usize size) {
#ifdef EMBER_CUDA
            if (device == GPU)
                cudaMemcpy(dest, src, size, cudaMemcpyDeviceToDevice);
            else
                std::memcpy(dest, src, size);
#else
            assert(device == CPU);
            std::memcpy(dest, src, size);
#endif
        }

        void memset([[maybe_unused]] Device device, void* dest, const char value, usize size) {
#ifdef EMBER_CUDA
            if (device == GPU)
                cudaMemset(dest, value, size);
            else
                std::memset(dest, value, size);
#else
            assert(device == CPU);
            std::memset(dest, value, size);
#endif
        }


        void* malloc([[maybe_unused]] Device device, usize size) {
#ifdef EMBER_CUDA
            if (device == GPU) {
                void* ptr;
                cudaError_t err = cudaMalloc(&ptr, size);

                if (err != cudaSuccess)
                    exitWithMsg(std::string("Failed to allocate memory on GPU. Error ") + cudaGetErrorName(err) + " (" + cudaGetErrorString(err) + ")", 2);

                return ptr;
            }

            auto ptr = std::malloc(size);

            if (!ptr)
                exitWithMsg("Failed to allocate RAM", 2);
            return ptr;
#else
            assert(device == CPU);
            return std::malloc(size);
#endif
        }

        void free([[maybe_unused]] Device device, void* ptr) {
#ifdef EMBER_CUDA
            if (device == GPU)
                cudaFree(ptr);
            else
                std::free(ptr);
#else
            assert(device == CPU);
            std::free(ptr);
#endif
        }

        void sgemmDispatch(
                    [[maybe_unused]] Device device,
                    bool transA, bool transB,
                    int M, int N, int K,
                    float alpha,
                    const float* A, int lda,
                    const float* B, int ldb,
                    float beta,
                    float* C, int ldc) {
#ifdef EMBER_CUDA
            if (device == GPU) {
                // cuBLAS uses column major ordering
                cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
                cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

                // Prefetch
                int deviceId;
                cudaError_t err = cudaGetDevice(&deviceId);
                if (err != cudaSuccess) {
                    fprintf(stderr, "Failed to get device ID: %s\n", cudaGetErrorString(err));
                    return;
                }

                cublasSgemm(
                    getCublasHandle(),
                    opB, opA,
                    N, M, K,
                    &alpha,
                    B, ldb,
                    A, lda,
                    &beta,
                    C, ldc
                );
            }
            else
                // CPU OpenBLAS version
                    cblas_sgemm(
                        CblasRowMajor,
                        transA ? CblasTrans : CblasNoTrans,
                        transB ? CblasTrans : CblasNoTrans,
                        M, N, K,
                        alpha,
                        A, lda,
                        B, ldb,
                        beta,
                        C, ldc
                    );
#else
            // CPU OpenBLAS version
            cblas_sgemm(
                CblasRowMajor,
                transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans,
                M, N, K,
                alpha,
                A, lda,
                B, ldb,
                beta,
                C, ldc
            );
#endif
        }


        // Shared Vector code
        SharedVector::SharedVector(const usize size) : size(size) {
            data = static_cast<float*>(malloc(loc, sizeof(float) * size));
            memset(loc, data, 0, sizeof(float) * size);
        }

        SharedVector::SharedVector(const std::vector<float>& other) {
            loc = CPU;
            resize(other.size());
            memcpy(CPU, data, other.data(), sizeof(float) * size);
        }

        SharedVector::SharedVector(const SharedVector& other) : loc(other.loc), size(other.size) {
            if (size > 0) {
                data = static_cast<float*>(malloc(loc, sizeof(float) * size));
                memcpy(loc, data, other.data, sizeof(float) * size);
            }
        }

        void SharedVector::resize(const usize newSize) {
            if (newSize != size) {
                if (data)
                    free(loc, data);
                data = static_cast<float*>(malloc(loc, sizeof(float) * newSize));
                size = newSize;

                memset(loc, data, 0, sizeof(float) * size);
            }
        }

        SharedVector& SharedVector::operator=(const SharedVector& other) noexcept {
            assert(loc == other.loc);
            if (this != &other) {
                resize(other.size);
                memcpy(loc, data, other.data, sizeof(float) * size);
            }
            return *this;
        }

        void SharedVector::moveTo([[maybe_unused]] const Device target) {
            assert(data);

            if (target == loc)
                return;

#ifdef EMBER_CUDA
            const usize bytes = sizeof(float) * size;
            float* newPtr = static_cast<float*>(internal::malloc(target, bytes));

            if (target == CPU)
                cudaMemcpy(newPtr, data, size, cudaMemcpyDeviceToHost);
            else if (target == GPU)
                cudaMemcpy(newPtr, data, size, cudaMemcpyHostToDevice);

            if (data)
                free(loc, data);
            data = newPtr;
            loc = target;
#else
            exitWithMsg("Can't move a tensor across devices without CUDA support", 2)
        #endif
        }
    }


    void Tensor::madd(const Tensor& a, const Tensor& b, const bool transposeA, const bool transposeB) {
#ifdef EMBER_CUDA
        if (underlying.loc == GPU) {
            assert(a.underlying.loc == GPU && b.underlying.loc == GPU);
        }
        else {
            assert(a.underlying.loc == CPU && b.underlying.loc == CPU);
        }
#else
        assert(a.underlying.loc == CPU && b.underlying.loc == CPU && underlying.loc == CPU);
#endif

        assert(dimensionality == 2);
        assert(a.dimensionality == 2);
        assert(b.dimensionality == 2);

        // Logical dimensions for op(A) and op(B)
        const usize aCols = transposeA ? a.dim(0) : a.dim(1);

#ifndef NDEBUG
        const usize aRows = transposeA ? a.dim(1) : a.dim(0);
        const usize bRows = transposeB ? b.dim(1) : b.dim(0);
        const usize bCols = transposeB ? b.dim(0) : b.dim(1);
#endif

        // Ensure dimensions are compatible for C = op(A) * op(B) + C
        assert(aCols == bRows);
        assert(this->dim(0) == aRows);
        assert(this->dim(1) == bCols);

        // Matrix multiplication parameters
        const int M = static_cast<int>(this->dim(0)); // rows of C / op(A)
        const int N = static_cast<int>(this->dim(1)); // cols of C / op(B)
        const int K = static_cast<int>(aCols);            // inner dimension

        // Leading dimensions assuming row major
        const int lda = static_cast<int>(a.dim(1));
        const int ldb = static_cast<int>(b.dim(1));
        const int ldc = static_cast<int>(this->dim(1));

        // Perform C = op(A) * op(B) + C
        internal::sgemmDispatch(
            underlying.loc,
            transposeA, transposeB,
            M, N, K,
            1.0f,
            a.ptr(), lda,
            b.ptr(), ldb,
            1.0f,
            this->ptr(), ldc
        );
    }

    void Tensor::axpy(const float scalar, const Tensor& other) {
#ifdef EMBER_CUDA
        if (underlying.loc == GPU) {
            cublasSaxpy(
                getCublasHandle(),
                underlying.size,
                &scalar,
                other.ptr(), 1,
                ptr(), 1
            );
        }
        else
            cblas_saxpy(
                underlying.size,
                scalar,
                other.ptr(), 1,
                ptr(), 1
            );
#else
        cblas_saxpy(
            underlying.size,
            scalar,
            other.ptr(), 1,
            ptr(), 1
        );
#endif
    }
}