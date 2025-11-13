#include "loss.h"

namespace Ember::loss {
    float MeanSquaredError::forward(const Tensor& output, const Tensor& target) {
        assert(output.size() == target.size());

        float loss = 0;
        for (usize i = 0; i < output.size(); i++)
                loss += std::pow(output.data()[i] - target.data()[i], 2);
        return loss / output.size();
    }

    Tensor MeanSquaredError::backward(const Tensor& output, const Tensor& target) {
        Tensor gradient;
        gradient.resize(output.dims());

        const float scalar = 2.0f / output.size();

        for (usize i = 0; i < output.size(); i++)
            gradient(i) = (output.data()[i] - target.data()[i]) * scalar;
        return gradient;
    }


    float SigmoidMSE::forward(const Tensor& output, const Tensor& target) {
        assert(output.size() == target.size());

        float loss = 0;
        for (usize i = 0; i < output.size(); i++) {
            const float imprecision = std::abs(sigmoid(output.data()[i]) - sigmoid(target.data()[i]));
            loss += std::pow(imprecision, 2);
        }
        return loss / output.size() - offset;
    }

    Tensor SigmoidMSE::backward(const Tensor& output, const Tensor& target) {
        assert(output.size() == target.size());

        Tensor gradient;
        gradient.resize(output.dims());

        const float scalar = 2.0f / output.size();

        for (usize i = 0; i < output.size(); i++) {
            const float expOutput = std::exp(a + b * output.data()[i]);
            const float expTarget = std::exp(a + b * target.data()[i]);

            const float fOutput = k / (1.0f + expOutput);
            const float fTarget = k / (1.0f + expTarget);

            const float fprimeOut = -k * b * expOutput / ((1.0f + expOutput) * (1.0f + expOutput));

            gradient.data()[i] = scalar * (fOutput - fTarget) * fprimeOut;
        }

        return gradient;
    }


    float CrossEntropyLoss::forward(const Tensor& output, const Tensor& target) {
        assert(output.size() == target.size());

        float loss = 0.0;
        for (usize i = 0; i < output.size(); i++) {
            assert(target.data()[i] >= 0);
            float prob = std::max(output.data()[i], 1e-10f);
            prob = std::min(prob, 1.0f);
            loss -= target.data()[i] * std::log(prob);
        }
        return loss / output.size();
    }

    Tensor CrossEntropyLoss::backward(const Tensor& output, const Tensor& target) {
        assert(output.size() == target.size());

        Tensor gradient;
        gradient.resize(output.dims());

        const float scalar = 1.0f / output.size();
        for (usize i = 0; i < output.size(); i++) {
            const float prob = std::max(output.data()[i], 1e-10f);
            gradient.data()[i] = -target.data()[i] / prob * scalar;
        }

        return gradient;
    }
}