#pragma once

#include <vector>

#include "network.h"

namespace Ember {
    namespace internal {
        struct Optimizer {
            Network& net;

            std::vector<Tensor<1>> weightGradients;
            std::vector<Tensor<1>> biasGradients;

            explicit Optimizer(Network& net);

            Optimizer(const Optimizer& other) : net(other.net), weightGradients(other.weightGradients), biasGradients(other.biasGradients) {}

            void zeroGrad();

            void clipGrad(const float maxNorm);

            virtual void step(float lr) = 0;
            virtual std::unique_ptr<Optimizer> clone() const = 0;

            virtual ~Optimizer() = default;
        };
    }

    namespace optimizers {
        struct SGD : internal::Optimizer {
            std::vector<Tensor<1>> weightVelocities;
            std::vector<Tensor<1>> biasVelocities;

            float momentum;

            SGD(Network& net, const float momentum = 0.9f);

            SGD(const SGD& other);

            void step(const float lr) override;

            std::unique_ptr<Optimizer> clone() const override;
        };
    }
}