#pragma once
#include <vector>

#include "layer.h"

namespace Ember {
    namespace internal {
        struct LossFunction {
            virtual float forward(const Tensor& output, const std::vector<float>& target) = 0;
            virtual Tensor backward(const Tensor& output, const std::vector<float>& target) = 0;

            virtual ~LossFunction() = default;
        };
    }

    namespace loss {
        struct MeanSquaredError : internal::LossFunction {
            float forward(const Tensor& output, const std::vector<float> &target) override;
            Tensor backward(const Tensor& output, const std::vector<float> &target) override;
        };

        struct CrossEntropyLoss : internal::LossFunction {
            float forward(const Tensor& output, const std::vector<float> &target) override;
            Tensor backward(const Tensor& output, const std::vector<float> &target) override;
        };
    }
}