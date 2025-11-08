#include "activation.h"

#include <algorithm>

namespace Ember {
    namespace internal::activations {
        float ReLU(const float x) {
            return std::max(x, 0.0f);
        }
        float CReLU(const float x) {
            return std::clamp(x, 0.0f, 1.0f);
        }

        namespace derivatives {
            float ReLU(const float x) {
                return x > 0 ? 1 : 0;
            }
            float CReLU(const float x) {
                return x > 0  && x < 1 ? 1 : 0;
            }
        }
    }

    namespace activations {
        void ReLU::forward(const Layer& previous) {
            for (usize prev = 0; prev < previous.values.size(); prev++)
                values.data[prev] = internal::activations::ReLU(previous.values.data[prev]);
        }
        Tensor ReLU::backward(const Layer& previous, const Tensor& gradOutput) const {
            Tensor result(gradOutput.dims());
            for (usize prev = 0; prev < gradOutput.size(); prev++)
                result.data[prev] = gradOutput.data[prev] * internal::activations::derivatives::ReLU(previous.values.data[prev]);

            return result;
        }


        void CReLU::forward(const Layer& previous) {
            for (usize prev = 0; prev < previous.values.size(); prev++)
                values.data[prev] = internal::activations::CReLU(previous.values.data[prev]);
        }
        Tensor CReLU::backward(const Layer& previous, const Tensor& gradOutput) const {
            Tensor result(gradOutput.dims());
            for (usize prev = 0; prev < gradOutput.size(); prev++)
                result.data[prev] = gradOutput.data[prev] * internal::activations::derivatives::CReLU(previous.values.data[prev]);

            return result;
        }


        void Softmax::forward(const Layer& previous) {
            const usize batchSize = previous.values.dim(0);
            const usize numClasses = previous.values.dim(1);

            for (usize sample = 0; sample < batchSize; sample++) {
                float maxIn = previous.values[sample, 0];
                for (usize i = 1; i < numClasses; i++)
                    maxIn = std::max(maxIn, previous.values[sample, i]);

                float sum = 0.0f;
                for (usize i = 0; i < numClasses; i++) {
                    values[sample, i] = std::exp(previous.values[sample, i] - maxIn);
                    sum += values[sample, i];
                }

                if (sum == 0.0f) {
                    const float uniform = 1.0f / numClasses;
                    for (usize i = 0; i < numClasses; i++)
                        values[sample, i] = uniform;
                }
                else {
                    const float sumScalar = 1.0f / sum;
                    for (usize i = 0; i < numClasses; i++)
                        values[sample, i] *= sumScalar;
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
                    dot += values[sample, i] * gradOutput[sample, i];

                for (usize i = 0; i < numClasses; i++)
                    result[sample, i] = values[sample, i] * (gradOutput[sample, i] - dot);
            }

            return result;
        }
    }
}
