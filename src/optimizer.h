#pragma once

#include <vector>

#include "network.h"

namespace Ember {
    namespace internal {
        struct Optimizer {
            Network& net;

            std::vector<BlasMatrix> weightGradients;
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
            std::vector<BlasMatrix> weightVelocities;
            std::vector<Tensor<1>> biasVelocities;

            float momentum;

            SGD(Network& net, const float momentum = 0.9f);
            SGD(const SGD& other) = default;

            void step(const float lr) override;

            std::unique_ptr<Optimizer> clone() const override;
        };

        struct Adam : internal::Optimizer {
            float beta1;
            float beta2;
            float epsilon;
            float decay;
            usize iteration = 0;

            std::vector<BlasMatrix> weightVelocities;
            std::vector<Tensor<1>> biasVelocities;
            std::vector<BlasMatrix> weightMomentum;
            std::vector<Tensor<1>> biasMomentum;

            explicit Adam(Network& net, const float beta1 = 0.9f, const float beta2 = 0.999f, const float epsilon = 1e-08, const float decay = 0.01f);
            Adam(const Adam& other) = default;

            void step(const float lr) override;

            std::unique_ptr<Optimizer> clone() const override;
        };
    }
}