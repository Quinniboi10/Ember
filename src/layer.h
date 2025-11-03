#pragma once

#include "tensor.h"

#include <utility>
#include <cblas.h>
#include <string>
#include <thread>

namespace Ember {
    namespace internal {
        struct Layer {
            Tensor values; // Dimensionality >= 2

            Layer() = default;
            explicit Layer(const usize size) {
                values.resize(static_cast<usize>(1), size);
            }

            // Assumes the previous layer already has
            // the (batch, ...) dimensions
            virtual void init(const Tensor& previous) {
                values.resize(previous.dims());
            }

            virtual void setBatchSize(const usize batchSize) {
                values.setDimension(0, batchSize);
            }

            virtual void forward(const Layer& previous) = 0;

            virtual std::unique_ptr<Layer> clone() = 0;

            virtual std::string str() const = 0;
            std::string dims() const {
                std::string s{};

                for (usize i = 1; i < values.dimensionality; i++) {
                    s += fmt::format("{}", values.dim(i));
                    if (i < values.dimensionality - 1)
                        s += "x";
                }

                return s;
            }
            virtual u64 numParams() const = 0;
            virtual ~Layer() = default;
        };

        struct ComputeLayer : Layer {
            Tensor weights;  // previousSize rows and size cols, dimensionality of 2
            Tensor biases;   // Dimensionality of 1

            ComputeLayer() = delete;

            explicit ComputeLayer(const usize size) : Layer(size) {
                this->biases.resize(size);
            }

            void init(const Tensor& previous) override {
                this->weights.resize(values.size(), previous.size());
            }

            virtual std::tuple<Tensor, Tensor, Tensor> backward(const Layer& previous, const Tensor& gradOutput) const = 0;
        };

        struct NonComputeLayer : Layer {
            virtual Tensor backward(const Layer& previous, const Tensor& gradOutput) const = 0;

            u64 numParams() const override { return 0; }
        };
    }

    namespace layers {
        struct Input : internal::Layer {
            template <typename... Args>
            explicit Input(const Args... args) {
                values.resize(std::vector<usize>{ 1, static_cast<usize>(args)... });
            }

            void forward([[maybe_unused]] const Layer& previous) override {}

            std::unique_ptr<Layer> clone() override {
                return std::make_unique<Input>(*this);
            }

            std::string str() const override {
                return fmt::format("Input - {}", dims());
            }

            u64 numParams() const override { return 0; }
        };

        struct Flatten : internal::NonComputeLayer {
            std::vector<usize> originalDimensions;

            void init(const Tensor& previous) override {
                originalDimensions = previous.dims();
                values.resize(static_cast<usize>(1), previous.size());
            }

            void setBatchSize(const usize batchSize) override {
                values.setDimension(0, batchSize);
                originalDimensions[0] = batchSize;
            }

            void forward(const Layer& previous) override { values.data = previous.values.data; }
            Tensor backward([[maybe_unused]] const Layer& previous, const Tensor& gradOutput) const override {
                Tensor reshapedGrad = gradOutput;
                reshapedGrad.reshape(originalDimensions);
                return reshapedGrad;
            }

            std::unique_ptr<Layer> clone() override {
                return std::make_unique<Flatten>(*this);
            }

            std::string str() const override {
                return fmt::format("Flatten - {}", dims());
            }
        };

        struct Linear : internal::ComputeLayer {
            // Construct a hidden layer
            explicit Linear(const usize size) : ComputeLayer(size) {}

            // Forward pass
            // Fill values in the current layer
            void forward(const Layer& previous) override {
                const usize batchSize = values.dim(0);
                const usize inputSize = previous.values.size() / batchSize;
                const usize outputSize = values.size() / batchSize;

                for (usize i = 0; i < batchSize; i++)
                    std::memcpy(&values[i, 0], biases.ptr(), outputSize * sizeof(float));

                // Batched matmul across all
                cblas_sgemm(
                    CblasRowMajor,
                    CblasNoTrans,  // previous.values: batch x inputSize
                    CblasTrans,    // weights: inputSize x outputSize
                    static_cast<int>(batchSize),
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
                const usize batchSize = values.dim(0);
                const usize inputSize = previous.values.size() / batchSize;
                const usize outputSize = values.size() / batchSize;

                Tensor gradInput(batchSize, inputSize);
                Tensor weightGrad(outputSize, inputSize);
                Tensor biasGrad(outputSize);

                // gradInput = (batch x outputSize) * (outputSize x inputSize)
                cblas_sgemm(
                    CblasRowMajor,
                    CblasNoTrans,   // A = gradOutput
                    CblasNoTrans,   // B = weights
                    static_cast<int>(batchSize),
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
                    static_cast<int>(batchSize),
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
                for (usize i = 0; i < batchSize; i++)
                    for (usize j = 0; j < outputSize; j++)
                        biasGrad[j] += gradOutput[i, j];

                return { gradInput, weightGrad, biasGrad };
            }

            std::unique_ptr<Layer> clone() override {
                return std::make_unique<Linear>(*this);
            }

            std::string str() const override {
                return fmt::format("Linear - {} input features and {} output features", weights.dim(1), values.dim(1));
            }
            u64 numParams() const override { return weights.size() + biases.size(); }
        };
    }
}