#pragma once

#include "types.h"
#include "tensor.h"

#include <vector>
#include <future>
#include <random>
#include <array>


namespace Ember {
    namespace internal {
        struct DataPoint {
            Tensor input;
            Tensor target;

            DataPoint() = default;
        };

        struct DataLoader {
            u64 threads;
            u64 batchSize;
            float trainSplit;

            u64 numSamples;

            usize currBatch;
            std::future<void> dataFuture;
            std::array<DataPoint, 2> data;

            DataLoader(const u64 batchSize, const float trainSplit, const u64 threads) {
                this->threads = threads;
                this->batchSize = batchSize;
                this->trainSplit = trainSplit;

                this->numSamples = 0;

                this->currBatch = 0;
            }

            // Loads batch into other buffer
            virtual void loadBatch(const usize batchIdx) = 0;
            virtual void loadTestSet() = 0;

            // Attempts to load data asynchronously if threads > 0
            void asyncPreloadBatch() {
                dataFuture = std::async(threads > 0 ? std::launch::async : std::launch::deferred, [this]() { loadBatch(currBatch ^ 1); });
            }

            void waitForBatch() {
                if (dataFuture.valid())
                    dataFuture.get();
            }

            const DataPoint& batchData() const {
                return data[currBatch];
            }

            void swapBuffers() {
                currBatch ^= 1;
            }

            virtual ~DataLoader() = default;
        };
    }

    namespace dataloaders {
        struct ImageDataLoader : internal::DataLoader {
            std::string dataDir;
            std::vector<std::string> types;
            std::vector<u64> samplesPerType;
            std::vector<std::vector<std::string>> allImages;

            std::vector<usize> trainSamplesPerType;
            usize numTrainSamples;
            usize numTestSamples;

            usize width;
            usize height;

            ImageDataLoader(const std::string& dataDir, const u64 batchSize, const float trainSplit, const u64 threads = 0, const usize width = 0, const usize height = 0);

            void loadBatch(const usize batchIdx) override;
            void loadTestSet() override;
        };
    }
}
