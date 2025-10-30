#pragma once

#include "layer.h"

#include <random>
#include <memory>

namespace Ember {
    enum class NetworkMode {
        EVAL,
        TRAIN
    };

    template <typename T>
    concept LayerLike = std::derived_from<std::decay_t<T>, internal::Layer>;

    struct Network {
        std::vector<std::unique_ptr<internal::Layer>> layers;

        template <LayerLike... Args>
        void _init(const bool useXavierInit, Args&&... args) {
            (layers.emplace_back(std::make_unique<std::decay_t<Args>>(std::forward<Args>(args))), ...);

            std::random_device rd;
            std::mt19937 gen(rd());

            for (usize l = 1; l < layers.size(); l++) {
                // Try to set the size of an activation layer
                if (auto* activationLayer = dynamic_cast<internal::ActivationLayer*>(layers[l].get())) {
                    activationLayer->setSize(layers[l - 1]->size);
                    continue;
                }
                auto* layer = dynamic_cast<internal::ComputeLayer*>(layers[l].get());
                // Only give random values to compute layers
                // Dynamic cast will return nullptr if it's an activation layer
                if (layer == nullptr)
                    continue;

                layer->init(layers[l - 1]->size);

                const usize fanIn = layers[l - 1]->size;
                const usize fanOut = layer->size;

                if (useXavierInit) {
                    const float limit = std::sqrt(6.0f / (fanIn + fanOut));
                    std::uniform_real_distribution<float> dist(-limit, limit);
                    for (auto& w : layer->weights.data)
                        w = dist(gen);
                }
                else {
                    const float stddev = std::sqrt(2.0f / fanIn);
                    std::normal_distribution<float> dist(0.0f, stddev);
                    for (auto& w : layer->weights.data)
                        w = dist(gen);
                }

                std::fill(layer->biases.begin(), layer->biases.end(), 0.0f);
            }
        }

        Network(const Network& other) {
            for (const auto& layer : other.layers) {
                layers.emplace_back(layer->clone());
            }
        }

        template <LayerLike... Args>
        explicit Network(Args&&... args) {
            _init(true, std::forward<Args>(args)...);
        }

        void forward(const Tensor& input, const usize threads);
        const Tensor& output() const;

        Network& operator=(const Network& other) {
            if (this != &other) {
                layers.clear();
                layers.reserve(other.layers.size());
                for (const auto& l : other.layers) {
                    layers.push_back(l->clone());
                }
            }
            return *this;
        }

        friend std::ostream& operator<<(std::ostream& os, const Network& net);
    };
}