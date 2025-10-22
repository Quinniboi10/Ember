#include "optimizer.h"

namespace Ember {
    namespace internal {
        Optimizer::Optimizer(Network& net) : net(net) {
            weightGradients.resize(net.layers.size());
            biasGradients.resize(net.layers.size());
            for (usize i = 1; i < net.layers.size(); i++) {
                const std::unique_ptr<Layer>& l = net.layers[i];
                const auto* layer = dynamic_cast<ComputeLayer*>(l.get());
                if (!layer)
                    continue;

                weightGradients[i].resize(layer->weights.size());
                biasGradients[i].resize(layer->biases.size());
            }
        }

        void Optimizer::zeroGrad() {
            for (Tensor<1>& grad : weightGradients)
                grad.fill(0);

            for (Tensor<1>& grad : biasGradients)
                grad.fill(0);
        }

        void Optimizer::clipGrad(const float maxNorm) {
            // Compute total norm of all gradients (weights and biases) across all layers
            double totalNormSq = 0.0;
            // Weights gradients
            for (const auto& layerGradients : weightGradients)
                for (const float wg : layerGradients)
                    totalNormSq += wg * wg;

            // Bias gradients
            for (const auto& layerGradients : biasGradients)
                for (const float bg : layerGradients)
                    totalNormSq += bg * bg;

            const float totalNorm = std::sqrt(totalNormSq);

            // Scale all gradients if needed
            if (totalNorm > maxNorm && totalNorm > 0.0f) {
                const float scale = maxNorm / totalNorm;

                // Weights gradients
                for (auto& layerGradients : weightGradients)
                    for (float& wg : layerGradients)
                        wg *= scale;

                // Bias gradients
                for (auto& layerGradients : biasGradients)
                    for (float& bg : layerGradients)
                        bg *= scale;
            }
        }
    }

    namespace optimizers {
        SGD::SGD(Network& net, const float momentum) : Optimizer(net), momentum(momentum) {
            weightVelocities.resize(net.layers.size());
            biasVelocities.resize(net.layers.size());

            for (usize i = 1; i < net.layers.size(); i++) {
                const std::unique_ptr<internal::Layer>& l = net.layers[i];
                const auto* layer = dynamic_cast<internal::ComputeLayer*>(l.get());
                if (!layer)
                    continue;

                weightVelocities[i].resize(layer->weights.size());
                biasVelocities[i].resize(layer->biases.size());
            }
        }

        SGD::SGD(const SGD& other)
                : Optimizer(other),
                    weightVelocities(other.weightVelocities),
                    biasVelocities(other.biasVelocities),
                    momentum(other.momentum) {}

        void SGD::step(const float lr) {
            for (usize lIdx = 1; lIdx < net.layers.size(); lIdx++) {
                std::unique_ptr<internal::Layer>& l = net.layers[lIdx];
                auto* layer = dynamic_cast<internal::ComputeLayer*>(l.get());
                if (!layer)
                    continue;

                assert(weightVelocities[lIdx].size() == layer->weights.size());
                assert(biasVelocities[lIdx].size() == layer->biases.size());
                assert(weightGradients[lIdx].size() == layer->weights.size());
                assert(biasGradients[lIdx].size() == layer->biases.size());

                // Update weights with momentum
                for (usize i = 0; i < layer->weights.size(); i++) {
                    weightVelocities[lIdx][i] = momentum * weightVelocities[lIdx][i] - lr * weightGradients[lIdx][i];
                    layer->weights[i] += weightVelocities[lIdx][i];
                }

                // Update biases with momentum
                for (usize i = 0; i < layer->biases.size(); i++) {
                    biasVelocities[lIdx][i] = momentum * biasVelocities[lIdx][i] - lr * biasGradients[lIdx][i];
                    layer->biases[i] += biasVelocities[lIdx][i];
                }
            }
        }

        std::unique_ptr<internal::Optimizer> SGD::clone() const {
            return std::make_unique<SGD>(*this);
        }
    }
}