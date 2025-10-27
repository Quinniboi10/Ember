#pragma once

#include "tensor.h"

#include <utility>
#include <cblas.h>
#include <string>
#include <thread>

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
                this->size = size;
                values.resize(size);
            }

            virtual void forward(const Layer& previous) = 0;

            virtual std::unique_ptr<Layer> clone() = 0;

            virtual std::string str() const = 0;
            virtual ~Layer() = default;
        };

        struct ComputeLayer : Layer {
            BlasMatrix weights; // previousSize rows and size cols
            Tensor<1> biases;

            ComputeLayer() = delete;

            explicit ComputeLayer(const usize size) : Layer(size) {
                this->biases.resize(size);
            }

            void init(const usize previousSize) {
                this->weights.resize(size, previousSize);
            }

            virtual std::tuple<Tensor<1>, BlasMatrix, Tensor<1>> backward(const Layer& previous, const Tensor<1>& gradOutput) const = 0;
        };

        struct ActivationLayer : Layer {
            virtual Tensor<1> backward(const Layer& previous, const Tensor<1>& gradOutput) const = 0;
        };
    }

    namespace layers {
        struct Input : internal::Layer {
            explicit Input(const usize size) : Layer(size) {}

            void forward(const Layer& previous) override {}

            std::unique_ptr<Layer> clone() override {
                return std::make_unique<Input>(*this);
            }

            std::string str() const override {
                return fmt::format("Input - {} features", size);
            }
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
                    weights.data.data(),   // W data
                    inputSize,             // lda (leading dimension, number of cols)
                    previous.values.ptr(), // x vector
                    1,                     // incx
                    1.0f,                  // beta (since y already holds biases)
                    values.ptr(),          // y vector (output)
                    1                      // incy
                );
            }

            std::tuple<Tensor<1>, BlasMatrix, Tensor<1>> backward(const Layer& previous, const Tensor<1>& gradOutput) const override {
                const usize inputSize  = previous.size;
                const usize outputSize = size;

                Tensor<1> gradInput(inputSize, 0.0f);
                BlasMatrix weightGrad(weights.rows, weights.cols);
                Tensor<1> biasGrad(size, 0.0f);

                // Compute gradients
                for (usize curr = 0; curr < outputSize; curr++) {
                    biasGrad[curr] = gradOutput[curr];
                    for (usize prev = 0; prev < inputSize; prev++) {
                        const usize wIndex = prev * outputSize + curr;
                        gradInput[prev] += weights.data[wIndex] * gradOutput[curr];
                        weightGrad.data[wIndex] += previous.values[prev] * gradOutput[curr];
                    }
                }

                return { gradInput, weightGrad, biasGrad };
            }

            std::unique_ptr<Layer> clone() override {
                return std::make_unique<Linear>(*this);
            }

            std::string str() const override {
                return fmt::format("Linear - {} input features and {} output features", weights.cols, size);
            }
        };
    }
}