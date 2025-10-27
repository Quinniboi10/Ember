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

                weightGradients[i].resize(layer->weights.rows, layer->weights.cols);
                biasGradients[i].resize(layer->biases.size());
            }
        }

        void Optimizer::zeroGrad() {
            for (BlasMatrix& grad : weightGradients)
                grad.fill(0);

            for (Tensor<1>& grad : biasGradients)
                grad.fill(0);
        }

        void Optimizer::clipGrad(const float maxNorm) {
            // Compute total norm of all gradients (weights and biases) across all layers
            double totalNormSq = 0.0;
            // Weights gradients
            for (const auto& layerGradients : weightGradients)
                for (const float wg : layerGradients.data)
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
                    for (float& wg : layerGradients.data)
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

                weightVelocities[i].resize(layer->weights.rows, layer->weights.cols);
                biasVelocities[i].resize(layer->biases.size());
            }
        }

        void SGD::step(const float lr) {
            for (usize lIdx = 1; lIdx < net.layers.size(); lIdx++) {
                std::unique_ptr<internal::Layer>& l = net.layers[lIdx];
                auto* layer = dynamic_cast<internal::ComputeLayer*>(l.get());
                if (!layer)
                    continue;

                assert(weightVelocities[lIdx].data.size() == layer->weights.data.size());
                assert(biasVelocities[lIdx].size() == layer->biases.size());
                assert(weightGradients[lIdx].data.size() == layer->weights.data.size());
                assert(biasGradients[lIdx].size() == layer->biases.size());

                // Update weights with momentum
                for (usize i = 0; i < layer->weights.data.size(); i++) {
                    weightVelocities[lIdx].data[i] = momentum * weightVelocities[lIdx].data[i] - lr * weightGradients[lIdx].data[i];
                    layer->weights.data[i] += weightVelocities[lIdx].data[i];
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

        Adam::Adam(Network& net, const float beta1, const float beta2, const float epsilon, const float decay) : Optimizer(net) {
            this->beta1 = beta1;
            this->beta2 = beta2;
            this->epsilon = epsilon;
            this->decay = decay;
            weightVelocities.resize(net.layers.size());
            biasVelocities.resize(net.layers.size());
            weightMomentum.resize(net.layers.size());
            biasMomentum.resize(net.layers.size());

            for (usize i = 1; i < net.layers.size(); i++) {
                const std::unique_ptr<internal::Layer>& l = net.layers[i];
                const auto* layer = dynamic_cast<internal::ComputeLayer*>(l.get());
                if (!layer)
                    continue;

                weightVelocities[i].resize(layer->weights.rows, layer->weights.cols);
                biasVelocities[i].resize(layer->biases.size());
                weightMomentum[i].resize(layer->weights.rows, layer->weights.cols);
                biasMomentum[i].resize(layer->biases.size());
            }
        }

        void Adam::step(const float lr) {
            iteration++;
            const float biasCorr1 = 1.0f - std::pow(beta1, iteration);
            const float biasCorr2 = 1.0f - std::pow(beta2, iteration);

            for (usize lIdx = 1; lIdx < net.layers.size(); lIdx++) {
                std::unique_ptr<internal::Layer>& l = net.layers[lIdx];
                auto* layer = dynamic_cast<internal::ComputeLayer*>(l.get());
                if (!layer)
                    continue;

                assert(weightVelocities[lIdx].data.size() == layer->weights.data.size());
                assert(biasVelocities[lIdx].size() == layer->biases.size());
                assert(weightGradients[lIdx].data.size() == layer->weights.data.size());
                assert(biasGradients[lIdx].size() == layer->biases.size());

                // Update weights
                for (usize i = 0; i < layer->weights.data.size(); i++) {
                    layer->weights.data[i] *= 1.0f - lr * decay;

                    weightMomentum[lIdx].data[i] = beta1 * weightMomentum[lIdx].data[i] + (1.0f - beta1) * weightGradients[lIdx].data[i];
                    weightVelocities[lIdx].data[i] = beta2 * weightVelocities[lIdx].data[i] + (1.0f - beta2) * weightGradients[lIdx].data[i] * weightGradients[lIdx].data[i];

                    // Bias correction
                    const float mHat = weightMomentum[lIdx].data[i] / biasCorr1;
                    const float vHat = weightVelocities[lIdx].data[i] / biasCorr2;

                    layer->weights.data[i] -= lr * mHat / (std::sqrt(vHat) + epsilon);
                }

                // Update biases
                for (usize i = 0; i < layer->biases.size(); i++) {
                    layer->biases[i] *= (1.0f - lr * decay);

                    biasMomentum[lIdx][i] = beta1 * biasMomentum[lIdx][i] + (1.0f - beta1) * biasGradients[lIdx][i];
                    biasVelocities[lIdx][i] = beta2 * biasVelocities[lIdx][i] + (1.0f - beta2) * biasGradients[lIdx][i] * biasGradients[lIdx][i];

                    // Bias correction
                    const float mHat = biasMomentum[lIdx][i] / biasCorr1;
                    const float vHat = biasVelocities[lIdx][i] / biasCorr2;

                    layer->biases[i] -= lr * mHat / (std::sqrt(vHat) + epsilon);
                }
            }
        }

        std::unique_ptr<internal::Optimizer> Adam::clone() const {
            return std::make_unique<Adam>(*this);
        }
    }
}