#pragma once

#include "activation.h"
#include "dataloader.h"
#include "optimizer.h"
#include "loss.h"

namespace Ember {
    namespace internal {
        struct Gradient {
            Tensor<1> weightGrad;
            Tensor<1> biasGrad;

            Gradient() = default;
            Gradient(const Tensor<1>& weightGrad, const Tensor<1>& biasGrad) : weightGrad(weightGrad), biasGrad(biasGrad) {}
        };
    }

    struct Learner {
        Network& net;
        internal::DataLoader& dataLoader;
        internal::Optimizer& optimizer;
        std::unique_ptr<internal::LossFunction> lossFunc;

        template<typename LossFunction>
        Learner(Network& net, internal::DataLoader& dataLoader, internal::Optimizer& optimizer, const LossFunction&& lossFunc) : net(net), dataLoader(dataLoader), optimizer(optimizer) {
            this->lossFunc = std::make_unique<std::decay_t<LossFunction>>(lossFunc);
        }

        // Returns a vector of gradients
        // RETURNS VALUES ORDERED FROM LAST TO FIRST LAYER
        std::vector<internal::Gradient> backward(const std::vector<float>& target) const;

        // Apply a gradient to the optimizer
        void applyGradients(const usize batchSize, const std::vector<Tensor<1>>& weightGradAccum, const std::vector<Tensor<1>>& biasGradAccum);

        // Main trainer functionality is through this function
        // Trains a neural network
        void learn(const float lr, const usize epochs, usize threads = 0);
    };
}