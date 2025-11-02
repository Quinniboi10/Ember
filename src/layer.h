#pragma once

#include "tensor.h"

#include <utility>
#include <cblas.h>
#include <string>
#include <thread>

namespace Ember {
    namespace internal {
        struct Layer {
            Tensor values; // Dimensionality of 2

            usize size{};

            Layer() = default;

            explicit Layer(const usize size) {
                this->size = size;
                values.resize(static_cast<usize>(1), size);
            }

            void setSize(const usize newSize) {
                this->size = newSize;
                values.resize(static_cast<usize>(1), size);
            }

            virtual void forward(const Layer& previous) = 0;

            virtual std::unique_ptr<Layer> clone() = 0;

            virtual std::string str() const = 0;
            virtual u64 numParams() const = 0;
            virtual ~Layer() = default;
        };

        struct ComputeLayer : Layer {
            Tensor weights; // previousSize rows and size cols, dimensionality of 2
            Tensor biases; // Dimensionality of 1

            ComputeLayer() = delete;

            explicit ComputeLayer(const usize size) : Layer(size) {
                this->biases.resize(size);
            }

            void init(const usize previousSize) {
                this->weights.resize(size, previousSize);
            }

            virtual std::tuple<Tensor, Tensor, Tensor> backward(const Layer& previous, const Tensor& gradOutput) const = 0;
        };

        struct ActivationLayer : Layer {
            virtual Tensor backward(const Layer& previous, const Tensor& gradOutput) const = 0;

            u64 numParams() const override { return 0; }
        };
    }

    namespace layers {
        struct Input : internal::Layer {
            explicit Input(const usize size) : Layer(size) {}

            void forward([[maybe_unused]] const Layer& previous) override {}

            std::unique_ptr<Layer> clone() override {
                return std::make_unique<Input>(*this);
            }

            std::string str() const override {
                return fmt::format("Input - {} features", size);
            }

            u64 numParams() const override { return 0; }
        };

        struct Linear : internal::ComputeLayer {
            // Construct a hidden layer
            explicit Linear(const usize size) : ComputeLayer(size) {}

            // Forward pass
            // Fill values in the current layer
            void forward(const Layer& previous) override {
                const usize inputSize  = previous.size;
                const usize outputSize = size;

                // Copy biases to output first
                for (usize i = 0; i < values.dim(0); i++) {
                    std::memcpy(&values[i, 0], biases.ptr(), outputSize * sizeof(float));

                    // Perform y = W^T * x + y (in-place)
                    // dimensions:
                    //   W: outputSize x inputSize
                    //   x: inputSize
                    //   y: outputSize
                    cblas_sgemv(
                        CblasRowMajor,           // Memory layout
                        CblasNoTrans,            // Don't transpose W to keep outputSize x inputSize
                        outputSize,              // rows of W
                        inputSize,               // cols of W
                        1.0f,                    // alpha
                        weights.ptr(),           // W data
                        inputSize,               // lda (leading dimension, number of cols)
                        &previous.values[i, 0],  // x vector
                        1,                       // incx
                        1.0f,                    // beta (since y already holds biases)
                        &values[i, 0],           // y vector (output)
                        1                        // incy
                    );
                }
            }

            // Returns gradInput, weightGrad, biasGrad
            std::tuple<Tensor, Tensor, Tensor> backward(const Layer& previous, const Tensor& gradOutput) const override {
                const usize inputSize  = previous.size;
                const usize outputSize = size;

                Tensor gradInput(values.dim(0), inputSize);
                Tensor weightGrad(weights.dims());
                Tensor biasGrad(biases.dims());

                gradInput.fill(0);
                weightGrad.fill(0);
                biasGrad.fill(0);

                // Compute gradients
                for (usize i = 0; i < values.dim(0); i++) {
                    for (usize curr = 0; curr < outputSize; curr++) {
                        biasGrad[curr] += gradOutput[i, curr];
                        for (usize prev = 0; prev < inputSize; prev++) {
                            gradInput[i, prev] += weights[curr, prev] * gradOutput[i, curr];
                            weightGrad[curr, prev] += previous.values[i, prev] * gradOutput[i, curr];
                        }
                    }
                }

                return { gradInput, weightGrad, biasGrad };
            }

            std::unique_ptr<Layer> clone() override {
                return std::make_unique<Linear>(*this);
            }

            std::string str() const override {
                return fmt::format("Linear - {} input features and {} output features", weights.dim(1), size);
            }
            u64 numParams() const override { return weights.size() + biases.size(); }
        };
    }
}