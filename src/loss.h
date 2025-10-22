#pragma once
#include <vector>

#include "layer.h"

namespace Ember {
    namespace internal {
        struct LossFunction {
            virtual float forward(const Tensor<1>& output, const std::vector<float>& target) = 0;
            virtual Tensor<1> backward(const Tensor<1>& output, const std::vector<float>& target) = 0;

            virtual ~LossFunction() = default;
        };
    }

    namespace loss {
        struct MeanSquaredError : internal::LossFunction {
            float forward(const Tensor<1>& output, const std::vector<float> &target) override;
            Tensor<1> backward(const Tensor<1>& output, const std::vector<float> &target) override;
        };
    }
}