#include "activation.h"

#include <algorithm>
#include <cmath>

#ifdef EMBER_CUDA
#include <cuda_runtime.h>
#endif

namespace Ember {
#ifdef EMBER_CUDA
    __global__ void reluKernel(const float* __restrict__ input, float* __restrict__ output, usize size) {
        const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < size)
            output[idx] = std::fmax(0.0f, input[idx]);
    }

    __global__ void reluDerivKernel(const float* __restrict__ prev, const float* __restrict__ grad, float* __restrict__ output, usize size) {
        const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < size)
            output[idx] = grad[idx] * (prev[idx] > 0 ? 1 : 0);
    }
#endif

    namespace internal::activations {
        float ReLU(const float x) {
            return std::max(x, 0.0f);
        }

        namespace derivatives {
            float ReLU(const float x) {
                return x > 0 ? 1 : 0;
            }
        }
    }

    namespace activations {
        void ReLU::forward(const Layer& previous) {
            const auto size = previous.values.size();
#ifdef EMBER_CUDA
            if (previous.values.getDevice() == GPU) {
                reluKernel<<<(size + 255) / 256, 256>>>(previous.values.ptr(), values.ptr(), size);
                cudaDeviceSynchronize();
            }
            else {
                for (usize prev = 0; prev < size; prev++)
                    values.data()[prev] = internal::activations::ReLU(previous.values.data()[prev]);
            }
#else
            for (usize prev = 0; prev < size; prev++)
                values.data()[prev] = internal::activations::ReLU(previous.values.data()[prev]);
#endif
        }
        Tensor ReLU::backward(const Layer& previous, const Tensor& gradOutput) const {
            Tensor result(gradOutput.dims());

            const auto size = gradOutput.size();
#ifdef EMBER_CUDA
            if (previous.values.getDevice() == GPU) {
                reluDerivKernel<<<(size + 255) / 256, 256>>>(previous.values.ptr(), gradOutput.ptr(), result.ptr(), size);
                cudaDeviceSynchronize();
            }
            else {
                for (usize prev = 0; prev < size; prev++)
                    result.data()[prev] = gradOutput.data()[prev] * internal::activations::derivatives::ReLU(previous.values.data()[prev]);
            }
#else
            for (usize prev = 0; prev < size; prev++)
                result.data()[prev] = gradOutput.data()[prev] * internal::activations::derivatives::ReLU(previous.values.data()[prev]);
#endif
            return result;
        }


        void Softmax::forward(const Layer& previous) {
            const usize batchSize = previous.values.dim(0);
            const usize numClasses = previous.values.dim(1);

            for (usize sample = 0; sample < batchSize; sample++) {
                float maxIn = previous.values(sample, 0);
                for (usize i = 1; i < numClasses; i++)
                    maxIn = std::max(maxIn, previous.values(sample, i));

                float sum = 0.0f;
                for (usize i = 0; i < numClasses; i++) {
                    values(sample, i) = std::exp(previous.values(sample, i) - maxIn);
                    sum += values(sample, i);
                }

                if (sum == 0.0f) {
                    const float uniform = 1.0f / numClasses;
                    for (usize i = 0; i < numClasses; i++)
                        values(sample, i) = uniform;
                }
                else {
                    const float sumScalar = 1.0f / sum;
                    for (usize i = 0; i < numClasses; i++)
                        values(sample, i) *= sumScalar;
                }
            }
        }
        Tensor Softmax::backward([[maybe_unused]] const Layer& previous, const Tensor& gradOutput) const {
            const usize batchSize = gradOutput.dim(0);
            const usize numClasses = gradOutput.dim(1);

            Tensor result(batchSize, numClasses);

            for (usize sample = 0; sample < batchSize; sample++) {
                float dot = 0.0f;
                for (usize i = 0; i < numClasses; i++)
                    dot += values(sample, i) * gradOutput(sample, i);

                for (usize i = 0; i < numClasses; i++)
                    result(sample, i) = values(sample, i) * (gradOutput(sample, i) - dot);
            }

            return result;
        }
    }
}
