#include "loss.h"

namespace Ember::loss {
    float MeanSquaredError::forward(const Tensor& output, const Tensor& target) {
        assert(output.size() == target.size());

        float loss = 0;
        for (usize i = 0; i < output.size(); i++)
                loss += std::pow(output.data[i] - target.data[i], 2);
        return loss / output.size();
    }

    Tensor MeanSquaredError::backward(const Tensor& output, const Tensor& target) {
        Tensor gradient;
        gradient.resize(output.dims());

        const float scalar = 2.0f / output.size();

        for (usize i = 0; i < output.size(); i++)
            gradient[i] = (output.data[i] - target.data[i]) * scalar;
        return gradient;
    }


    float SigmoidMSE::forward(const Tensor& output, const Tensor& target) {
        assert(output.size() == target.size());

        float loss = 0;
        for (usize i = 0; i < output.size(); i++) {
            const float imprecision = std::abs(output.data[i] - target.data[i]);
            const float sigmoid = k / (1 + std::exp(a + b * imprecision));
            loss += std::pow(sigmoid, 2);
        }
        return loss / output.size();
    }

    Tensor SigmoidMSE::backward(const Tensor& output, const Tensor& target) {
        assert(output.size() == target.size());

        Tensor gradient;
        gradient.resize(output.dims());

        const float scalar = 2.0f / output.size();

        for (usize i = 0; i < output.size(); i++) {
            const float diff = output.data[i] - target.data[i];
            const float sign = (diff >= 0.0f) ? 1.0f : -1.0f;

            const float expTerm = std::exp(a + b * std::abs(diff));
            const float denominator = 1.0f + expTerm;
            const float sigmoid = k / denominator;

            const float grad = -scalar * b * sigmoid * sigmoid * (expTerm / denominator) * sign;

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
        return loss / output.size();
    }

    Tensor CrossEntropyLoss::backward(const Tensor& output, const Tensor& target) {
        assert(output.size() == target.size());

        Tensor gradient;
        gradient.resize(output.dims());

        const float scalar = 1.0f / output.size();
        for (usize i = 0; i < output.size(); i++) {
            const float prob = std::max(output.data[i], 1e-10f);
            gradient.data[i] = -target.data[i] / prob * scalar;
        }

        return gradient;
    }
}