#include "learner.h"
#include "progbar.h"

#include <algorithm>


namespace Ember {
    std::vector<internal::Gradient> Learner::backward(const std::vector<float> &target) const {
        std::vector<internal::Gradient> gradients(net.layers.size());

        Tensor<1> error = lossFunc->backward(net.output(), target);

        for (usize idx = net.layers.size() - 1; idx > 0; idx--) {
            auto* layer = net.layers[idx].get();

            if (const auto* actLayer = dynamic_cast<internal::ActivationLayer*>(layer)) {
                error = actLayer->backward(*net.layers[idx - 1], error);
            }
            else if (const auto* compLayer = dynamic_cast<internal::ComputeLayer*>(layer)) {
                auto [gradInput, weightGrad, biasGrad] = compLayer->backward(*net.layers[idx - 1], error);
                gradients[idx] = internal::Gradient(weightGrad, biasGrad);
                error = gradInput;
            }
        }

        return gradients;
    }

    void Learner::applyGradients(const usize batchSize, const std::vector<Tensor<1>>& weightGradAccum, const std::vector<Tensor<1>>& biasGradAccum) {
        const float batchScalar = 1.0f / batchSize;
        // Apply gradients to weights and biases
        for (usize l = net.layers.size() - 1; l > 0; l--) {
            if (const auto& currLayer = dynamic_cast<internal::ComputeLayer*>(net.layers[l].get())) {
                assert(optimizer.weightGradients[l].size() == currLayer->weights.size());
                assert(optimizer.biasGradients[l].size() == currLayer->biases.size());

                for (usize i = 0; i < optimizer.weightGradients[l].size(); i++)
                    optimizer.weightGradients[l][i] += weightGradAccum[l][i] * batchScalar;
                for (usize i = 0; i < currLayer->size; i++)
                    optimizer.biasGradients[l][i] += biasGradAccum[l][i] * batchScalar;
            }
        }
    }

    void Learner::learn(const float lr, const usize epochs, usize threads) {
        if (threads == 0)
            threads = std::thread::hardware_concurrency();
        if (threads == 0) {
            std::cerr << "Failed to detect number of threads. Defaulting to 1" << std::endl;
            threads = 1;
        }

        const u64 batchSize = dataLoader.batchSize;
        const u64 batchesPerEpoch = dataLoader.numSamples / batchSize;

        fmt::println("Training for {} batches with {} batches per epoch", batchesPerEpoch * epochs, batchesPerEpoch);

        std::cout << "Epoch    Train loss    Test loss    Test accuracy\n\n" << std::endl;

        // Returns { test loss, test accuracy }
        const auto getTestLossAcc = [&]() {
            float loss = 0;
            usize numCorrect = 0;
            dataLoader.loadTestSet();
            const usize testSize = dataLoader.batchData().size();
            while (dataLoader.hasNext()) {
                internal::DataPoint data = dataLoader.next();
                net.forward(data.input);
                loss += lossFunc->forward(net.layers.back()->values, data.target);
                usize guess = 0;
                usize goal = 0;
                for (usize i = 0; i < data.target.size(); i++) {
                    if (net.layers.back()->values[i] > net.layers.back()->values[guess])
                        guess = i;
                    if (data.target[i] > data.target[goal])
                        goal = i;
                }
                numCorrect += (guess == goal);
            }
            return std::pair<float, float>{ loss / (testSize ? testSize : 1), numCorrect / static_cast<float>(testSize ? testSize : 1) };
        };

        // Initialize accumulators
        std::vector<Tensor<1>> weightGradAccum(net.layers.size());
        std::vector<Tensor<1>> biasGradAccum(net.layers.size());

        for (usize i = 1; i < net.layers.size(); i++) {
            if (const auto* compLayer = dynamic_cast<internal::ComputeLayer*>(net.layers[i].get())) {
                weightGradAccum[i].resize(compLayer->weights.size());
                biasGradAccum[i].resize(compLayer->biases.size());
            }
        }

        // Preload first batch
        dataLoader.asyncPreloadBatch();

        // Main loop
        for (usize epoch = 0; epoch < epochs; epoch++) {
            double trainLoss = 0;

            ProgressBar progressBar{};

            for (u64 batchIdx = 0; batchIdx < batchesPerEpoch; batchIdx++) {
                // Reset accumulators per mini-batch
                for (auto& t : weightGradAccum)
                    t.fill(0);
                for (auto& t : biasGradAccum)
                    t.fill(0);

                dataLoader.waitForBatch();
                dataLoader.swapBuffers();

                // Instantly start loading next batch
                dataLoader.asyncPreloadBatch();

                for (u64 sample = 0; sample < batchSize; sample++) {
                    const internal::DataPoint& data = dataLoader.next();

                    net.forward(data.input);

                    // Accumulate training loss
                    trainLoss += lossFunc->forward(net.layers.back()->values, data.target);

                    const auto gradients = backward(data.target);

                    // Accumulate gradients
                    for (usize l = 1; l < net.layers.size(); l++) {
                        const auto& prevLayer = net.layers[l - 1];
                        if (const auto* compLayer = dynamic_cast<internal::ComputeLayer*>(net.layers[l].get())) {
                            for (usize i = 0; i < compLayer->size; i++) {
                                for (usize j = 0; j < prevLayer->size; j++) {
                                    const usize idx = j * compLayer->size + i;
                                    assert(l < weightGradAccum.size());
                                    assert(idx < weightGradAccum[l].size());
                                    assert(idx < gradients[l].weightGrad.size());

                                    weightGradAccum[l][idx] += gradients[l].weightGrad[idx];
                                }

                                assert(l < biasGradAccum.size());
                                assert(i < biasGradAccum[l].size());
                                assert(i < gradients[l].biasGrad.size());

                                biasGradAccum[l][i] += gradients[l].biasGrad[i];
                            }
                        }
                    }
                }

                applyGradients(batchSize, weightGradAccum, biasGradAccum);
                optimizer.clipGrad(1);
                optimizer.step(lr);
                optimizer.zeroGrad();

                internal::cursor::up();
                internal::cursor::up();
                internal::cursor::begin();
                fmt::println("{:>5L}{:>14.5f}{:>13}{:>17}", epoch, trainLoss / batchIdx / batchSize, "Pending", "Pending");
                std::cout << progressBar.report(batchIdx, batchesPerEpoch, 63) << "      " << std::endl;
            }
            const auto [testLoss, testAccuracy] = getTestLossAcc();

            internal::cursor::up();
            internal::cursor::clear();
            internal::cursor::up();
            fmt::println("{:>5L}{:>14.5f}{:>13.5f}{:>17.2f}%\n\n", epoch, trainLoss / batchesPerEpoch / batchSize, testLoss, testAccuracy * 100);
        }
    }
}