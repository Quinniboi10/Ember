#pragma once

#include "layer.h"

namespace Ember {
    namespace internal::activations {
        float ReLU(float x);
    }

    namespace activations {
        struct ReLU : internal::ActivationLayer {
            void forward(const Layer& previous) override;

            Tensor backward(const Layer& previous, const Tensor& gradOutput) const override;

            std::unique_ptr<Layer> clone() override {
                return std::make_unique<ReLU>(*this);
            }

            std::string str() const override {
                return fmt::format("ReLU - applied to {} features", outputSize);
            }
        };

        struct Softmax : internal::ActivationLayer {
            void forward(const Layer& previous) override;

            Tensor backward(const Layer& previous, const Tensor& gradOutput) const override;

            std::unique_ptr<Layer> clone() override {
                return std::make_unique<Softmax>(*this);
            }

            std::string str() const override {
                return fmt::format("Softmax - applied to {} features", outputSize);
            }
        };
    }
}