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

        // Apply sigmoid to the input and target before
        // calculating the loss
        // Modeled by the below function, paste into Desmos to see it
        // f\left(x\right)=\left(\frac{k}{1+e^{\left(a+bx\right)}}\right)
        struct SigmoidMSE : internal::LossFunction {
            float a = 2.3;
            float b = -0.17;
            float k = 3;

            SigmoidMSE() = default;
            SigmoidMSE(const float a, const float b, const float k) : a(a), b(b), k(k) {}
            explicit SigmoidMSE(const float horizontalStretch) { b /= horizontalStretch; }

            float forward(const Tensor& output, const Tensor& target) override;
            Tensor backward(const Tensor& output, const Tensor& target) override;
        };

        struct CrossEntropyLoss : internal::LossFunction {
            float forward(const Tensor& output, const Tensor& target) override;
            Tensor backward(const Tensor& output, const Tensor& target) override;
        };
    }
}