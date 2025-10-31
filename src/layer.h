#pragma once

#include "tensor.h"

#include <utility>
#include <cblas.h>
#include <string>
#include <thread>

namespace Ember {
    namespace internal {
        // The foundation for all layers
        // The values tensor should store
        // the output of any given layer
        // regardless of any internal compute
        struct Layer {
            Tensor values;

            usize outputSize;

            Layer() = default;

            explicit Layer(const usize size) {
                setSize(size);
            }

            void setSize(const usize size) {
                outputSize = size;
                values.resize(size);
            }

            virtual void forward(const Layer& previous) = 0;

            virtual std::unique_ptr<Layer> clone() = 0;

            virtual std::string str() const = 0;
            virtual u64 numParams() const = 0;
            virtual ~Layer() = default;
        };

        struct ComputeLayer : Layer {
            Tensor weights; // previousSize rows and size cols
            Tensor biases;

            ComputeLayer() = delete;

            explicit ComputeLayer(const usize size) : Layer(size) {
                this->biases.resize(size);
            }

            virtual void init(const usize previousSize) = 0;

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
                return fmt::format("Input - {} features", outputSize);
            }

            u64 numParams() const override { return 0; }
        };

        struct Linear : internal::ComputeLayer {
            // Construct a hidden layer
            explicit Linear(const usize size) : ComputeLayer(size) {}

            // Forward pass
            // Fill values in the current layer
            void forward(const Layer& previous) override {
                const usize inputSize  = previous.values.size();
                const usize outputSize = values.size();

                // Copy biases to output first
                std::memcpy(values.ptr(), biases.ptr(), outputSize * sizeof(float));

                // Perform y = W^T * x + y (in-place)
                // dimensions:
                //   W: outputSize x inputSize
                //   x: inputSize
                //   y: outputSize
                cblas_sgemv(
                    CblasRowMajor,         // Memory layout
                    CblasNoTrans,          // Don't transpose W to keep outputSize x inputSize
                    outputSize,            // rows of W
                    inputSize,             // cols of W
                    1.0f,                  // alpha
                    weights.ptr(),         // W data
                    inputSize,             // lda (leading dimension, number of cols)
                    previous.values.ptr(), // x vector
                    1,                     // incx
                    1.0f,                  // beta (since y already holds biases)
                    values.ptr(),          // y vector (output)
                    1                      // incy
                );
            }

            void init(const usize previousSize) override {
                this->weights.resize(outputSize, previousSize);
            }

            // Returns gradInput, weightGrad, biasGrad
            std::tuple<Tensor, Tensor, Tensor> backward(const Layer& previous, const Tensor& gradOutput) const override {
                const usize inputSize  = previous.outputSize;

                Tensor gradInput(inputSize);
                Tensor weightGrad(weights.dims());
                Tensor biasGrad(outputSize);

                // Compute gradients
                for (usize curr = 0; curr < outputSize; curr++) {
                    biasGrad[curr] = gradOutput[curr];
                    for (usize prev = 0; prev < inputSize; prev++) {
                        gradInput[prev] += weights[curr, prev] * gradOutput[curr];
                        weightGrad[curr, prev] += previous.values[prev] * gradOutput[curr];
                    }
                }

                return { gradInput, weightGrad, biasGrad };
            }

            std::unique_ptr<Layer> clone() override {
                return std::make_unique<Linear>(*this);
            }

            std::string str() const override {
                return fmt::format("Linear - {} input features and {} output features", weights.dim(1), outputSize);
            }
            u64 numParams() const override { return weights.size() + biases.size(); }
        };
    }
}