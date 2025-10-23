#pragma once

#include "types.h"

#include <vector>
#include <future>
#include <random>
#include <array>

namespace Ember {
    namespace internal {
        struct DataPoint {
            std::vector<float> input;
            std::vector<float> target;

            DataPoint(const std::vector<float>& input, const std::vector<float>& target) : input(input), target(target) {}
        };

        struct DataLoader {
            u64 threads;
            u64 batchSize;
            float trainSplit;

            u64 numSamples;

            usize currBatch;
            std::future<void> dataFuture;
            std::array<std::vector<DataPoint>, 2> data;

            DataLoader(const u64 batchSize, const float trainSplit, const u64 threads) {
                this->threads = threads;
                this->batchSize = batchSize;
                this->trainSplit = trainSplit;

                this->numSamples = 0;

                this->currBatch = 0;

                data[0].reserve(batchSize);
                data[1].reserve(batchSize);
            }

            // Loads batch into other buffer
            virtual void loadBatch(const usize batchIdx) = 0;
            virtual void loadTestSet() = 0;

            bool hasNext() const {
                return data[currBatch].size() > 0;
            }

            DataPoint next() {
                assert(hasNext());
                const DataPoint dataPoint = data[currBatch].back();
                data[currBatch].pop_back();
                return dataPoint;
            }

            // Attempts to load data asynchronously if threads > 0
            void asyncPreloadBatch() {
                dataFuture = std::async(threads > 0 ? std::launch::async : std::launch::deferred, [this]() { loadBatch(currBatch ^ 1); });
            }

            void waitForBatch() {
                if (dataFuture.valid())
                    dataFuture.get();
            }

            const DataPoint& batchData(const usize idx) const {
                return data[currBatch][idx];
            }

            usize testSetSize() const {
                return data[currBatch].size();
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
            std::mt19937 rng{ std::random_device{}() };

            usize width;
            usize height;

            ImageDataLoader(const std::string& dataDir, const u64 batchSize, const float trainSplit, const u64 threads = 0, const usize width = 0, const usize height = 0);

            void loadBatch(const usize batchIdx) override;
            void loadTestSet() override;
        };
    }
}
