#include "learner.h"
#include "progbar.h"
#include "util.h"

#include <algorithm>

namespace Ember {
    void Learner::backward(const Network& net, const Tensor& target) const {
        Tensor error = lossFunc->backward(net.output(), target);

        const float batchScalar = 1.0f / net.layers[0]->values.dim(0);
        for (usize idx = net.layers.size() - 1; idx > 0; idx--) {
            auto* layer = net.layers[idx].get();

            if (const auto* actLayer = dynamic_cast<internal::ActivationLayer*>(layer)) {
                error = actLayer->backward(*net.layers[idx - 1], error);
            }
            else if (const auto* compLayer = dynamic_cast<internal::ComputeLayer*>(layer)) {
                auto [gradInput, weightGrad, biasGrad] = compLayer->backward(*net.layers[idx - 1], error);

                // Weights
                cblas_saxpy(optimizer.weightGradients[idx].size(), batchScalar,
                            weightGrad.ptr(), 1,
                            optimizer.weightGradients[idx].ptr(), 1);

                // Biases
                cblas_saxpy(optimizer.biasGradients[idx].size(), batchScalar,
                            biasGrad.ptr(), 1,
                            optimizer.biasGradients[idx].ptr(), 1);

                error = gradInput;
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

        const u64 batchSize = dataLoader.batchSize;
        const u64 batchesPerEpoch = dataLoader.numSamples / batchSize;

        ProgressBar progressBar{};

        // Returns { test loss, test accuracy }
        const auto getTestLossAcc = [&]() {
            usize numCorrect = 0;
            dataLoader.loadTestSet();
            const internal::DataPoint& data = dataLoader.batchData();
            const usize testSize = data.input.dim(0);

            net.forward(data.input, threads);

            const float loss = lossFunc->forward(net.output(), data.target);

            for (usize i = 0; i < data.target.dim(0); i++) {
                usize guess = 0;
                usize goal = 0;
                for (usize j = 0; j < data.target.dim(1); j++) {
                    if (net.output()[i, j] > net.output()[i, guess])
                        guess = j;
                    if (data.target[i, j] > data.target[goal])
                        goal = j;
                }
                numCorrect += (guess == goal);
            }

            return std::pair<float, float>{ loss / std::max<usize>(testSize, 1), numCorrect / static_cast<float>(testSize ? testSize : 1) };
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

        fmt::println("Training for {} batches with {} batches per epoch", formatNum(batchesPerEpoch * epochs), formatNum(batchesPerEpoch));

        std::cout << "Epoch    Train loss    Test loss    Test accuracy        Time\n\n" << std::endl;

        for (usize i = 1; i < net.layers.size(); i++) {
            if (auto* compLayer = dynamic_cast<internal::ComputeLayer*>(net.layers[i].get())) {
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

                dataLoader.waitForBatch();
                dataLoader.swapBuffers();

                // Instantly start loading next batch
                dataLoader.asyncPreloadBatch();

                net.forward(dataLoader.batchData().input, threads);
                trainLoss += lossFunc->forward(net.output(), dataLoader.batchData().target);

                backward(net, dataLoader.batchData().target);

                optimizer.clipGrad(1);
                optimizer.step(lr);
                optimizer.zeroGrad();

                internal::cursor::up();
                internal::cursor::up();
                internal::cursor::begin();
                fmt::println("{:>5L}{:>14.5f}{:>13}{:>17}{:>12}", epoch, trainLoss, "Pending", "Pending", formatTime(stopwatch.elapsed()));
                std::cout << progressBar.report(currentBatch + 1, batchesPerEpoch, 63) << "      " << std::endl;

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
