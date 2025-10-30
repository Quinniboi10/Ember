#include "activation.h"

namespace Ember {
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
            for (usize prev = 0; prev < previous.size; prev++)
                values[prev] = internal::activations::ReLU(previous.values[prev]);
        }
        Tensor<1> ReLU::backward(const Layer& previous, const Tensor<1>& gradOutput) const {
            Tensor<1> result(gradOutput.size());
            for (usize prev = 0; prev < gradOutput.size(); prev++)
                result[prev] = gradOutput[prev] * internal::activations::derivatives::ReLU(previous.values[prev]);

            return result;
        }


        void Softmax::forward(const Layer& previous) {
            values.resize(previous.size);
            float maxIn = previous.values[0];
            for (usize i = 1; i < previous.size; i++)
                maxIn = std::max(maxIn, previous.values[i]);

            float sum = 0.0f;
            for (usize i = 0; i < previous.size; i++) {
                values[i] = std::exp(previous.values[i] - maxIn);
                sum += values[i];
            }

            if (sum == 0.0f)
                for (auto& v : values) v = 1.0f / previous.size;
            else
                for (auto& v : values) v /= sum;
        }
        Tensor<1> Softmax::backward([[maybe_unused]] const Layer& previous, const Tensor<1>& gradOutput) const {
            const usize n = gradOutput.size();
            Tensor<1> result(n);

            // Compute dot product of gradOutput and softmax output (values)
            float dot = 0.0f;
            for (usize i = 0; i < n; ++i)
                dot += values[i] * gradOutput[i];

            // Compute gradient for each element
            for (usize i = 0; i < n; ++i)
                result[i] = values[i] * (gradOutput[i] - dot);

            return result;
        }
    }
}