#include "learner.h"
#include "progbar.h"

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

    void Learner::applyGradients(const usize batchSize, const std::vector<BlasMatrix>& weightGradAccum, const std::vector<Tensor<1>>& biasGradAccum) {
        const float batchScalar = 1.0f / batchSize;
        // Apply gradients to the optimizer
        for (usize l = net.layers.size() - 1; l > 0; l--) {
            if (const auto* currLayer = dynamic_cast<internal::ComputeLayer*>(net.layers[l].get())) {
                assert(optimizer.weightGradients[l].data.size() == currLayer->weights.data.size());
                assert(optimizer.biasGradients[l].size() == currLayer->biases.size());

                // Weights
                cblas_saxpy(optimizer.weightGradients[l].data.size(), batchScalar,
                            weightGradAccum[l].ptr(), 1,
                            optimizer.weightGradients[l].ptr(), 1);

                // Biases
                cblas_saxpy(optimizer.biasGradients[l].size(), batchScalar,
                            biasGradAccum[l].ptr(), 1,
                            optimizer.biasGradients[l].ptr(), 1);
            }
        }
    }

    void Learner::learn(const float initialLr, const usize epochs, const usize threads) {
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
        std::vector<BlasMatrix> weightGradAccum(net.layers.size());
        std::vector<Tensor<1>> biasGradAccum(net.layers.size());

        const u64 batchSize = dataLoader.batchSize;
        const u64 batchesPerEpoch = dataLoader.numSamples / batchSize;

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
                net.forward(data.input, threads);
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

        // Store the compute layers so RTTI isn't done on-the-fly
        std::vector<internal::ComputeLayer*> computeLayers;
        std::vector<usize> computeLayerIndexes;

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

        fmt::println("Training for {} batches with {} batches per epoch", batchesPerEpoch * epochs, batchesPerEpoch);

        std::cout << "Epoch    Train loss    Test loss    Test accuracy        Time\n\n" << std::endl;

        for (usize i = 1; i < net.layers.size(); i++) {
            if (auto* compLayer = dynamic_cast<internal::ComputeLayer*>(net.layers[i].get())) {
                weightGradAccum[i].resize(compLayer->weights.rows, compLayer->weights.cols);
                biasGradAccum[i].resize(compLayer->biases.size());

                computeLayers.push_back(compLayer);
                computeLayerIndexes.push_back(i);
            }
        }

        // Preload first batch
        dataLoader.asyncPreloadBatch();

        stopwatch.reset();

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

                dataLoader.waitForBatch();
                dataLoader.swapBuffers();

                // Instantly start loading next batch
                dataLoader.asyncPreloadBatch();

                for (u64 sample = 0; sample < batchSize; sample++) {
                    const internal::DataPoint& data = dataLoader.batchData(sample);

                    net.forward(data.input, threads);

                    // Accumulate training loss
                    trainLoss += lossFunc->forward(net.output(), data.target);

                    const auto gradients = backward(net, data.target);

                    // Accumulate gradients
                    for (usize i = 0; i < computeLayers.size(); i++) {
                        const usize l = computeLayerIndexes[i];
                        const auto* layer = computeLayers[i];
                        cblas_saxpy(layer->weights.data.size(), 1.0f,
                                    gradients[l].weightGrad.ptr(), 1,
                                    weightGradAccum[l].ptr(), 1);

                        cblas_saxpy(layer->biases.size(), 1.0f,
                                    gradients[l].biasGrad.ptr(), 1,
                                    biasGradAccum[l].ptr(), 1);
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
    }
}