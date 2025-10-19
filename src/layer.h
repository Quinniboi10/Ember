#pragma once

#include "tensor.h"
#include "../external/fmt/format.h"

namespace Ember {
    namespace internal {
        struct Layer {
            Tensor<1> values;

            usize size;

            Layer() = default;

            explicit Layer(const usize size) {
                setSize(size);
            }

            void setSize(const usize size) {
                values.resize(size);
                this->size = size;
            }

            virtual void forward(const Layer& previous) = 0;
            virtual Tensor<1>& getOutputs() { return values; };
            virtual const Tensor<1>& getOutputs() const { return values; };

            virtual std::string str() const = 0;
            virtual ~Layer() = default;
        };

        struct ComputeLayer : Layer {
            Tensor<1> weights; // Indexed [previous][current], flattened to prev * size + curr
            Tensor<1> biases;

            ComputeLayer() = delete;

            ComputeLayer(const usize previousSize, const usize size) : Layer(size) {
                this->weights.resize(previousSize * size);
                this->biases.resize(size);
            }
        };

        struct ActivationLayer : Layer {};
    }

    namespace layers {
        struct Input : internal::Layer {
            explicit Input(const usize size) : Layer(size) {}

            void forward(const Layer& previous) override {};

            std::string str() const override {
                return fmt::format("Input - {} features", size);
            }
        };

        struct Linear : internal::ComputeLayer {
            // Construct a hidden layer
            Linear(const usize previousSize, const usize size) : ComputeLayer(previousSize, size) {}

            // Forward pass
            // Fill values in the current layer
            void forward(const Layer& previous) override {
                const usize inputSize  = previous.size;
                const usize outputSize = size;

                // Move biases into the target vector
                values = biases;

                // This instruction tells the compiler to run across all threads
                #pragma omp parallel for schedule(auto)
                for (usize prev = 0; prev < inputSize; prev++) {
                    for (usize curr = 0; curr < outputSize; curr++)
                        values[curr] += previous.getOutputs()[prev] * weights[prev * size + curr];
                }
            }

            std::string str() const override {
                return fmt::format("Linear - {} input features and {} output features", weights.size() / size, size);
            }
        };
    }
}