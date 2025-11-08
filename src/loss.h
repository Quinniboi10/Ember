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
        // calculating the loss using MSE
        // Modeled by the below function, paste into Desmos to see it
        // f\left(x\right)=\left(\frac{k}{1+e^{\left(a+bx\right)}}\right)
        struct SigmoidMSE : internal::LossFunction {
            float a = 1;
            float b = -0.25;
            float k = 1;

            float offset;

            SigmoidMSE(const float a, const float b, const float k) : a(a), b(b), k(k) {
                offset = -std::pow(sigmoid(0), 2);
            }
            explicit SigmoidMSE(const float horizontalStretch) {
                b /= horizontalStretch;
                offset = -std::pow(sigmoid(0), 2);
            }

            float sigmoid(const float x) const { return k / (1 + std::exp(a + b * x)); }

            float forward(const Tensor& output, const Tensor& target) override;
            Tensor backward(const Tensor& output, const Tensor& target) override;
        };

        struct CrossEntropyLoss : internal::LossFunction {
            float forward(const Tensor& output, const Tensor& target) override;
            Tensor backward(const Tensor& output, const Tensor& target) override;
        };
    }
}