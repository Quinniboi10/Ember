#include "loss.h"

namespace Ember::loss {
    float MeanSquaredError::forward(const Tensor<1>& output, const std::vector<float>& target) {
        assert(output.size() == target.size());

        float loss = 0;
        for (usize i = 0; i < output.size(); i++)
            loss += std::pow(output[i] - target[i], 2);
        return loss / output.size();
    }

    Tensor<1> MeanSquaredError::backward(const Tensor<1>& output, const std::vector<float>& target) {
        Tensor<1> gradient;
        gradient.resize(output.size());

        const float scalar = 2.0f / output.size();

        for (usize i = 0; i < output.size(); i++)
            gradient[i] = (output[i] - target[i]) * scalar;
        return gradient;
    }
}