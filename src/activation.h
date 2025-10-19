#pragma once

#include "layer.h"

namespace Ember {
    namespace internal::activations {
        float ReLU(float x);
    }

    namespace activations {
        struct ReLU : internal::ActivationLayer {
            void forward(const Layer& previous) override;

            std::string str() const override {
                return fmt::format("ReLU - applied to {} features", size);
            }
        };
    }
}