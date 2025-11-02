#pragma once
#include <vector>

#include "layer.h"

namespace Ember {
    namespace internal {
        struct LossFunction {
            virtual float forward(const Tensor& output, const Tensor& target) = 0;
            virtual Tensor backward(const Tensor& output, const Tensor& target) = 0;

            virtual ~LossFunction() = default;
        };
    }

    namespace loss {
        struct MeanSquaredError : internal::LossFunction {
            float forward(const Tensor& output, const Tensor& target) override;
            Tensor backward(const Tensor& output, const Tensor& target) override;
        };

        struct CrossEntropyLoss : internal::LossFunction {
            float forward(const Tensor& output, const Tensor& target) override;
            Tensor backward(const Tensor& output, const Tensor& target) override;
        };
    }
}