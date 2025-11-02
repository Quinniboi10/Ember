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
                const usize batch = values.dim(0);
                const usize inputSize = previous.size;
                const usize outputSize = size;

                for (usize i = 0; i < batch; ++i)
                    std::memcpy(&values[i, 0], biases.ptr(), outputSize * sizeof(float));

                // Batched matmul across all
                cblas_sgemm(
                    CblasRowMajor,
                    CblasNoTrans,  // previous.values: batch x inputSize
                    CblasTrans,    // weights: inputSize x outputSize
                    static_cast<int>(batch),
                    static_cast<int>(outputSize),
                    static_cast<int>(inputSize),
                    1.0f,
                    previous.values.ptr(),
                    static_cast<int>(inputSize),
                    weights.ptr(),
                    static_cast<int>(inputSize),
                    1.0f,
                    values.ptr(),
                    static_cast<int>(outputSize)
                );
            }

            // Returns gradInput, weightGrad, biasGrad
            std::tuple<Tensor, Tensor, Tensor> backward(const Layer& previous, const Tensor& gradOutput) const override {
                const usize batch = values.dim(0);
                const usize inputSize = previous.size;
                const usize outputSize = size;

                Tensor gradInput(batch, inputSize);
                Tensor weightGrad(outputSize, inputSize);
                Tensor biasGrad(outputSize);

                gradInput.fill(0);
                weightGrad.fill(0);
                biasGrad.fill(0);

                // gradInput = (batch x outputSize) * (outputSize x inputSize)
                cblas_sgemm(
                    CblasRowMajor,
                    CblasNoTrans,   // A = gradOutput
                    CblasNoTrans,   // B = weights
                    static_cast<int>(batch),
                    static_cast<int>(inputSize),
                    static_cast<int>(outputSize),
                    1.0f,
                    gradOutput.ptr(),
                    static_cast<int>(outputSize),
                    weights.ptr(),
                    static_cast<int>(inputSize),
                    0.0f,
                    gradInput.ptr(),
                    static_cast<int>(inputSize)
                );

                // weightGrad = (outputSize x batch) * (batch x inputSize)
                cblas_sgemm(
                    CblasRowMajor,
                    CblasTrans,     // A = gradOutput
                    CblasNoTrans,   // B = previous.values
                    static_cast<int>(outputSize),
                    static_cast<int>(inputSize),
                    static_cast<int>(batch),
                    1.0f,
                    gradOutput.ptr(),
                    static_cast<int>(outputSize),
                    previous.values.ptr(),
                    static_cast<int>(inputSize),
                    0.0f,
                    weightGrad.ptr(),
                    static_cast<int>(inputSize)
                );

                // Sum over batch of gradOutput
                for (usize i = 0; i < batch; ++i)
                    for (usize j = 0; j < outputSize; ++j)
                        biasGrad[j] += gradOutput[i, j];

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