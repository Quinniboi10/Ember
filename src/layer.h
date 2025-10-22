#pragma once

#include "tensor.h"

#include <utility>
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
                values.resize(size);
                this->size = size;
            }

            virtual void forward(const Layer& previous) = 0;

            virtual std::string str() const = 0;
            virtual ~Layer() = default;
        };

        struct ComputeLayer : Layer {
            Tensor<1> weights; // Indexed [previous][current], flattened to prev * size + curr
            Tensor<1> biases;

            usize threadCount;

            ComputeLayer() = delete;

            ComputeLayer(const usize previousSize, const usize size) : Layer(size) {
                threadCount = std::max<usize>(1, std::thread::hardware_concurrency());
                threadCount = std::min<usize>(threadCount, size / 2);

                this->weights.resize(previousSize * size);
                this->biases.resize(size);
            }

            void setThreadCount(const usize threadCount) {
                this->threadCount = std::max<usize>(1, threadCount);
                this->threadCount = std::min<usize>(threadCount, size / 2);
            }

            virtual std::tuple<Tensor<1>, Tensor<1>, Tensor<1>> backward(const Layer& previous, const Tensor<1>& gradOutput) const = 0;
        };

        struct ActivationLayer : Layer {
            virtual Tensor<1> backward(const Layer& previous, const Tensor<1>& gradOutput) const = 0;
        };
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

                std::vector<std::thread> threads;

                threadCount = 1;

                const auto worker = [&](const usize threadId) {
                    // Divide the range across threads
                    const usize start = (outputSize * threadId) / threadCount;
                    const usize end   = std::min((outputSize * (threadId + 1)) / threadCount, outputSize);

                    for (usize curr = start; curr < end; curr++) {
                        float sum = biases[curr];
                        for (usize prev = 0; prev < inputSize; prev++)
                            sum += previous.values[prev] * weights[prev * size + curr];
                        values[curr] = sum;
                    }
                };

                // Launch worker threads
                for (usize t = 1; t < threadCount; t++)
                    threads.emplace_back(worker, t);

                // Run thread 0 on the main thread
                worker(0);

                // Join all threads
                for (std::thread& t : threads)
                    if (t.joinable())
                        t.join();
            }

            std::tuple<Tensor<1>, Tensor<1>, Tensor<1>> backward(const Layer& previous, const Tensor<1>& gradOutput) const override {
                const usize inputSize  = previous.size;
                const usize outputSize = size;

                Tensor<1> gradInput(inputSize, 0.0f);
                Tensor<1> weightGrad(weights.size(), 0.0f);
                Tensor<1> biasGrad(size, 0.0f);

                // Compute gradients
                for (usize curr = 0; curr < outputSize; curr++) {
                    biasGrad[curr] = gradOutput[curr];
                    for (usize prev = 0; prev < inputSize; prev++) {
                        const usize wIndex = prev * outputSize + curr;
                        gradInput[prev] += weights[wIndex] * gradOutput[curr];
                        weightGrad[wIndex] += previous.values[prev] * gradOutput[curr];
                    }
                }

                return { gradInput, weightGrad, biasGrad };
            }

            std::string str() const override {
                return fmt::format("Linear - {} input features and {} output features", weights.size() / size, size);
            }
        };
    }
}