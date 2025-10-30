#pragma once

#include "dataloader.h"
#include "activation.h"
#include "optimizer.h"
#include "callback.h"
#include "loss.h"

namespace Ember {
    namespace internal {
        struct Gradient {
            Tensor<2> weightGrad;
            Tensor<1> biasGrad;

            Gradient() = default;
            Gradient(const Tensor<2>& weightGrad, const Tensor<1>& biasGrad) : weightGrad(weightGrad), biasGrad(biasGrad) {}
        };
    }

    template <typename T>
    concept CallbackLike = std::derived_from<std::decay_t<T>, internal::Callback>;

    struct Learner {
        Network& net;
        internal::DataLoader& dataLoader;
        internal::Optimizer& optimizer;
        std::unique_ptr<internal::LossFunction> lossFunc;

        std::vector<std::unique_ptr<internal::Callback>> callbacks;

        // Info for callbacks to use/change based on the last state of the learner
        float lr{};

        // Updated every epoch
        float testLoss{};
        float testAccuracy{};

        // Updated every batch
        u64 currentBatch{};
        float trainLoss{};
        usize epoch{};

        template<typename LossFunction>
        Learner(Network& net, internal::DataLoader& dataLoader, internal::Optimizer& optimizer, const LossFunction&& lossFunc) : net(net), dataLoader(dataLoader), optimizer(optimizer) {
            this->lossFunc = std::make_unique<std::decay_t<LossFunction>>(lossFunc);
        }

        // Add the given list of callbacks to the learner
        template <CallbackLike... Args>
        void addCallbacks(Args&&... args) {
            (callbacks.emplace_back(std::make_unique<std::decay_t<Args>>(std::forward<Args>(args))), ...);
        }

        // Returns a vector of gradients
        // RETURNS VALUES ORDERED FROM LAST TO FIRST LAYER
        std::vector<internal::Gradient> backward(const Network& net, const std::vector<float>& target) const;

        // Apply a gradient to the optimizer
        void applyGradients(const usize batchSize, const std::vector<Tensor<2>>& weightGradAccum, const std::vector<Tensor<1>>& biasGradAccum);

        // Main trainer functionality is through this function
        // Trains a neural network
        void learn(const float lr, const usize epochs, usize threads);
    };
}