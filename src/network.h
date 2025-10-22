#pragma once

#include "layer.h"

#include <random>
#include <memory>

namespace Ember {
    enum class NetworkMode {
        EVAL,
        TRAIN
    };

    struct Network {
        std::vector<std::unique_ptr<internal::Layer>> layers;

        template <typename... Args>
        void _init(const bool useXavierInit, Args&&... args) {
            (layers.emplace_back(std::make_unique<std::decay_t<Args>>(std::forward<Args>(args))), ...);

            std::random_device rd;
            std::mt19937 gen(rd());

            for (usize l = 1; l < layers.size(); l++) {
                // Try to set the size of an activation layer
                if (auto* activationLayer = dynamic_cast<internal::ActivationLayer*>(layers[l].get()); activationLayer != nullptr) {
                    activationLayer->setSize(layers[l - 1]->size);
                    continue;
                }
                auto* layer = dynamic_cast<internal::ComputeLayer*>(layers[l].get());
                // Only give random values to compute layers
                // Dynamic cast will return nullptr if it's an activation layer
                if (layer == nullptr)
                    continue;

                const usize fanIn = layers[l - 1]->size;
                const usize fanOut = layer->size;

                if (useXavierInit) {
                    const float limit = std::sqrt(6.0f / (fanIn + fanOut));
                    std::uniform_real_distribution<float> dist(-limit, limit);
                    for (auto& w : layer->weights)
                        w = dist(gen);
                }
                else {
                    const float stddev = std::sqrt(2.0f / fanIn);
                    std::normal_distribution<float> dist(0.0f, stddev);
                    for (auto& w : layer->weights)
                        w = dist(gen);
                }

                std::fill(layer->biases.begin(), layer->biases.end(), 0.0f);
            }
        }

        template <typename... Args>
        explicit Network(Args&&... args) {
            _init(true, std::forward<Args>(args)...);
        }

        void setMode(const NetworkMode mode);

        void forward(const Tensor<1>& input);
        const Tensor<1>& output() const;

        friend std::ostream& operator<<(std::ostream& os, const Network& net);
    };
}