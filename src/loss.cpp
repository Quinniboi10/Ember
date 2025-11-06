#include "loss.h"

namespace Ember::loss {
    float MeanSquaredError::forward(const Tensor& output, const Tensor& target) {
        assert(output.size() == target.size());

        float loss = 0;
        for (usize i = 0; i < output.size(); i++)
                loss += std::pow(output.data[i] - target.data[i], 2);
        return loss / output.dim(1);
    }

    Tensor MeanSquaredError::backward(const Tensor& output, const Tensor& target) {
        Tensor gradient;
        gradient.resize(output.dims());

        const float scalar = 2.0f / output.dim(1);

        for (usize i = 0; i < output.size(); i++)
            gradient[i] = (output.data[i] - target.data[i]) * scalar;
        return gradient;
    }


    float SigmoidMSE::forward(const Tensor& output, const Tensor& target) {
        assert(output.size() == target.size());

        float loss = 0;
        for (usize i = 0; i < output.size(); i++) {
            const float imprecision = std::abs(output.data[i] - target.data[i]);
            const float sigmoid = 1 / (1 + std::exp(-imprecision));
            loss += std::pow(sigmoid, 2);
        }
        return loss / output.dim(1);
    }

    Tensor SigmoidMSE::backward(const Tensor& output, const Tensor& target) {
        assert(output.size() == target.size());

        Tensor gradient;
        gradient.resize(output.dims());

        const float scalar = 2.0f / output.dim(1);

        for (usize i = 0; i < output.size(); i++) {
            const float diff = output.data[i] - target.data[i];
            const float sign = (diff >= 0.0f) ? 1.0f : -1.0f;
            const float sigmoid = 1.0f / (1.0f + std::exp(-std::abs(diff)));
            const float grad = scalar * sigmoid * sigmoid * (1.0f - sigmoid) * sign * 2.0f;

            gradient.data[i] = grad;
        }

        return gradient;
    }


    float CrossEntropyLoss::forward(const Tensor& output, const Tensor& target) {
        assert(output.size() == target.size());

        float loss = 0.0;
        for (usize i = 0; i < output.size(); i++) {
            assert(target.data[i] >= 0);
            float prob = std::max(output.data[i], 1e-10f);
            prob = std::min(prob, 1.0f);
            loss -= target.data[i] * std::log(prob);
        }
        return loss / output.dim(0);
    }

    Tensor CrossEntropyLoss::backward(const Tensor& output, const Tensor& target) {
        assert(output.size() == target.size());

        Tensor gradient;
        gradient.resize(output.dims());

        const float scalar = 1.0f / output.dim(0);
        for (usize i = 0; i < output.size(); i++) {
            const float prob = std::max(output.data[i], 1e-10f);
            gradient.data[i] = -target.data[i] / prob * scalar;
        }

        return gradient;
    }
}