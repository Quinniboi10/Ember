#pragma once

#include "layer.h"

namespace Ember {
    namespace internal::activations {
        float ReLU(float x);
    }

    namespace activations {
        struct ReLU : internal::ActivationLayer {
            void forward(const Layer& previous) override;

            Tensor<1> backward(const Layer& previous, const Tensor<1>& gradOutput) const override;

            std::string str() const override {
                return fmt::format("ReLU - applied to {} features", size);
            }
        };

        struct Softmax : internal::ActivationLayer {
            void forward(const Layer& previous) override;

            Tensor<1> backward(const Layer& previous, const Tensor<1>& gradOutput) const override;

            std::string str() const override {
                return fmt::format("Softmax - applied to {} features", size);
            }
        };
    }
}