#pragma once

#include "types.h"
#include "tensor.h"
#include "./chess/board.h"

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

            u64 numSamples = 0;

            usize currBatch = 0;

            std::future<void> dataFuture;
            std::array<DataPoint, 2> data;

            DataLoader(const u64 batchSize, const u64 threads) {
                this->threads = threads;
                this->batchSize = batchSize;
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

            virtual void swapBuffers() {
                currBatch ^= 1;
            }

            // Returns the number of "correct" outputs
            // from the network
            virtual u64 countCorrect(const Tensor& output, const Tensor& target) = 0;

            virtual ~DataLoader() = default;
        };
    }

    namespace dataloaders {
        struct ImageDataLoader : internal::DataLoader {
            std::string dataDir;
            std::vector<std::string> types;
            std::vector<u64> samplesPerType;
            std::vector<std::vector<std::string>> allImages;

            std::vector<u64> trainSamplesPerType;
            u64 numTrainSamples;
            u64 numTestSamples;

            float trainSplit;

            usize width;
            usize height;

            ImageDataLoader(const std::string& dataDir, const u64 batchSize, const u64 threads, const float trainSplit, const usize width = 0, const usize height = 0);

            void loadBatch(const usize batchIdx) override;
            void loadTestSet() override;

            u64 countCorrect(const Tensor& output, const Tensor& target) override;
        };

        // Defined in ./chess/*
        namespace chess {
            struct BulletTextDataLoader : internal::DataLoader {
                std::string filePath;

                u64 batchNumber = 0;
                usize evalScale = 0;

                BulletTextDataLoader(const std::string& filePath, const u64 batchSize, const usize evalScale, const u64 threads = 0);

                void loadBatch(const usize batchIdx) override;
                void loadTestSet() override;

                u64 countCorrect(const Tensor& output, const Tensor& target) override;


                void swapBuffers() override {
                    batchNumber++;
                    currBatch ^= 1;
                }
            };
        }
    }
}
