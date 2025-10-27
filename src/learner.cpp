#include "learner.h"
#include "progbar.h"
#include "omp.h"

#include <algorithm>

namespace Ember {
    std::vector<internal::Gradient> Learner::backward(const Network& net, const std::vector<float> &target) const {
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

    void Learner::learn(const float initialLr, const usize epochs, usize threads) {
        // Initialize the learner's callback shared states
        lr = initialLr;
        testLoss = std::numeric_limits<float>::infinity();
        testAccuracy = std::numeric_limits<float>::infinity();

        currentBatch = 0;
        trainLoss = std::numeric_limits<float>::infinity();
        epoch = 0;

        // Initialization of stuff
        std::pair<float, float> test{};

        // Accumulators
        std::vector<std::vector<Tensor<1>>> threadWeightGradAccum(threads);
        std::vector<std::vector<Tensor<1>>> threadBiasGradAccum(threads);

        std::vector<Tensor<1>> weightGradAccum(net.layers.size());
        std::vector<Tensor<1>> biasGradAccum(net.layers.size());

        const u64 batchSize = dataLoader.batchSize;
        const u64 batchesPerEpoch = dataLoader.numSamples / batchSize;

        std::vector<Network> networks;

        double trainLoss{};

        ProgressBar progressBar{};

        // Returns { test loss, test accuracy }
        const auto getTestLossAcc = [&]() {
            float loss = 0;
            usize numCorrect = 0;
            dataLoader.loadTestSet();
            const usize testSize = dataLoader.testSetSize();
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

        Stopwatch<std::chrono::milliseconds> stopwatch;

        for (const auto& c : callbacks)
            c->setLearner(this);

        try {
            for (const auto& c : callbacks)
                c->run(internal::BEFORE_FIT);
        }
        catch (const internal::CallbackException& e) {
            if (const auto* error = dynamic_cast<const internal::CancelFitException*>(&e))
                goto afterFit;
        }

        // Get number of threads to use
        if (threads == 0)
            threads = std::thread::hardware_concurrency();
        if (threads == 0) {
            std::cerr << "Failed to detect number of threads" << std::endl;
            threads = 1;
        }
        fmt::println("Using {} threads", threads);

        fmt::println("Training for {} batches with {} batches per epoch", batchesPerEpoch * epochs, batchesPerEpoch);

        std::cout << "Epoch    Train loss    Test loss    Test accuracy        Time\n\n" << std::endl;

        for (auto& accum : threadWeightGradAccum)
            accum.resize(net.layers.size());
        for (auto& accum : threadBiasGradAccum)
            accum.resize(net.layers.size());

        for (usize i = 1; i < net.layers.size(); i++) {
            if (const auto* compLayer = dynamic_cast<internal::ComputeLayer*>(net.layers[i].get())) {
                weightGradAccum[i].resize(compLayer->weights.size());
                biasGradAccum[i].resize(compLayer->biases.size());

                for (auto& accum : threadWeightGradAccum)
                    accum[i].resize(compLayer->weights.size());
                for (auto& accum : threadBiasGradAccum)
                    accum[i].resize(compLayer->biases.size());
            }
        }

        networks.reserve(threads);
        for (usize t = 0; t < threads; t++)
            networks.push_back(net);

        // Preload first batch
        dataLoader.asyncPreloadBatch();

        stopwatch.reset();

        // Set the network to only use 1 thread on the forward pass
        net.setMode(NetworkMode::TRAIN);

        // Main loop
        for (epoch = 0; epoch < epochs; epoch++) {
            try {
                for (const auto& c : callbacks)
                    c->run(internal::BEFORE_EPOCH);
            }
            catch (const internal::CallbackException& e) {
                if (const auto& error = dynamic_cast<const internal::CancelEpochException*>(&e))
                    goto afterEpoch;
                if (const auto* error = dynamic_cast<const internal::CancelFitException*>(&e))
                    goto afterFit;
            }

            trainLoss = 0;

            progressBar = ProgressBar();

            for (currentBatch = 0; currentBatch < batchesPerEpoch; currentBatch++) {
                try {
                    for (const auto& c : callbacks)
                        c->run(internal::BEFORE_BATCH);
                }
                catch (const internal::CallbackException& e) {
                    if (const auto& error = dynamic_cast<const internal::CancelBatchException*>(&e))
                        goto afterBatch;
                    if (const auto& error = dynamic_cast<const internal::CancelEpochException*>(&e))
                        goto afterEpoch;
                    if (const auto* error = dynamic_cast<const internal::CancelFitException*>(&e))
                        goto afterFit;
                }

                // Reset accumulators per mini-batch
                for (auto& t : weightGradAccum)
                    t.fill(0);
                for (auto& t : biasGradAccum)
                    t.fill(0);
                for (auto& accum : threadWeightGradAccum)
                    for (auto& t : accum)
                        t.fill(0);
                for (auto& accum : threadBiasGradAccum)
                    for (auto& t : accum)
                        t.fill(0);

                for (auto& n : networks)
                    n = net;

                dataLoader.waitForBatch();
                dataLoader.swapBuffers();

                // Instantly start loading next batch
                dataLoader.asyncPreloadBatch();

                #pragma omp parallel for num_threads(threads) reduction(+:trainLoss)
                for (u64 sample = 0; sample < batchSize; sample++) {
                    const usize tID = omp_get_thread_num();

                    Network& thisNet = networks[tID];

                    const internal::DataPoint& data = dataLoader.batchData(sample);

                    thisNet.forward(data.input);

                    // Accumulate training loss
                    trainLoss += lossFunc->forward(thisNet.output(), data.target);

                    const auto gradients = backward(thisNet, data.target);

                    // Accumulate gradients
                    for (usize l = 1; l < thisNet.layers.size(); l++) {
                        const auto& prevLayer = thisNet.layers[l - 1];
                        if (const auto* compLayer = dynamic_cast<internal::ComputeLayer*>(thisNet.layers[l].get())) {
                            for (usize i = 0; i < compLayer->size; i++) {
                                for (usize j = 0; j < prevLayer->size; j++) {
                                    const usize idx = j * compLayer->size + i;
                                    assert(l < weightGradAccum.size());
                                    assert(idx < weightGradAccum[l].size());
                                    assert(idx < gradients[l].weightGrad.size());

                                    threadWeightGradAccum[tID][l][idx] += gradients[l].weightGrad[idx];
                                }

                                assert(l < biasGradAccum.size());
                                assert(i < biasGradAccum[l].size());
                                assert(i < gradients[l].biasGrad.size());

                                threadBiasGradAccum[tID][l][i] += gradients[l].biasGrad[i];
                            }
                        }
                    }
                }

                // Reduce across threads
                for (usize t = 0; t < threads; t++) {
                    for (usize l = 1; l < net.layers.size(); l++) {
                        const auto& prevLayer = net.layers[l - 1];
                        if (const auto* compLayer = dynamic_cast<internal::ComputeLayer*>(net.layers[l].get())) {
                            for (usize i = 0; i < compLayer->size; i++) {
                                for (usize j = 0; j < prevLayer->size; j++) {
                                    const usize idx = j * compLayer->size + i;

                                    weightGradAccum[l][idx] += threadWeightGradAccum[t][l][idx];
                                }

                                biasGradAccum[l][i] += threadBiasGradAccum[t][l][i];
                            }
                        }
                    }
                }

                applyGradients(batchSize, weightGradAccum, biasGradAccum);
                optimizer.clipGrad(1);
                optimizer.step(lr);
                optimizer.zeroGrad();

                this->trainLoss = trainLoss / currentBatch / batchSize;

                internal::cursor::up();
                internal::cursor::up();
                internal::cursor::begin();
                fmt::println("{:>5L}{:>14.5f}{:>13}{:>17}{:>12}", epoch, trainLoss / currentBatch / batchSize, "Pending", "Pending", formatTime(stopwatch.elapsed()));
                std::cout << progressBar.report(currentBatch, batchesPerEpoch, 63) << "      " << std::endl;

                afterBatch:
                for (const auto& c : callbacks)
                    c->run(internal::AFTER_BATCH);
            }
            test = getTestLossAcc();

            testLoss = test.first;
            testAccuracy = test.second;

            internal::cursor::up();
            internal::cursor::clear();
            internal::cursor::up();
            internal::cursor::clear();

            afterEpoch:
            try {
                for (const auto& c : callbacks)
                    c->run(internal::AFTER_EPOCH);
            }
            catch (const internal::CallbackException& e) {
                if (const auto* error = dynamic_cast<const internal::CancelFitException*>(&e))
                    goto afterFit;
            }

            fmt::println("{:>5L}{:>14.5f}{:>13.5f}{:>16.2f}%{:>12}\n\n", epoch, trainLoss / batchesPerEpoch / batchSize, testLoss, testAccuracy * 100, formatTime(stopwatch.elapsed()));
        }

        afterFit:
        for (const auto& c : callbacks)
            c->run(internal::AFTER_FIT);

        net.setMode(NetworkMode::EVAL);
    }
}