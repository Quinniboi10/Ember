#include "dataloader.h"

#include "../external/fmt/format.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../external/stb_image.h"

#include <filesystem>
#include <omp.h>

std::vector<float> loadGreyscaleImage(const std::string& path, const Ember::usize w, const Ember::usize h) {
    int width, height, channels;
    unsigned char* data = stbi_load(path.data(), &width, &height, &channels, 1);
    if (!data)
        throw std::runtime_error("Failed to load image: " + path);

    std::vector<float> vec(width * height);

    if ((w == static_cast<Ember::usize>(width) || w == 0) && (h == static_cast<Ember::usize>(height) || h == 0)) {
        for (Ember::usize i = 0; i < width * height; i++)
            vec[i] = data[i] / 255.0f;
    }
    else {
        // Simple nearest-neighbor resize
        for (Ember::usize y = 0; y < h; ++y) {
            for (Ember::usize x = 0; x < w; ++x) {
                const int sourceX = x * width / w;
                const int sourceY = y * height / h;
                const int sourceIdx = sourceY * width + sourceX;
                const int destIdx = y * w + x;
                vec[destIdx] = data[sourceIdx] / 255.0f;
            }
        }
    }

    stbi_image_free(data);
    return vec;
}

namespace Ember::dataloaders {
    ImageDataLoader::ImageDataLoader(const std::string& dataDir, const u64 batchSize, const u64 threads, const float trainSplit, const usize width, const usize height)
        : DataLoader(batchSize, threads), dataDir(dataDir), trainSplit(trainSplit), width(width), height(height) {
        fmt::println("Attempting to open data dir '{}'", dataDir);
        if (!std::filesystem::exists(dataDir) || !std::filesystem::is_directory(dataDir))
            exitWithMsg("Data directory does not exist or is not a directory: " + dataDir, 1);

        for (const auto &entry: std::filesystem::directory_iterator(this->dataDir)) {
            if (entry.is_directory())
                types.push_back(entry.path().string());
        }

        fmt::println("Found {} types", types.size());

        samplesPerType.resize(types.size());

        allImages.resize(types.size());
        numSamples = 0;
        for (usize typeIdx = 0; typeIdx < types.size(); typeIdx++) {
            for (const auto &entry: std::filesystem::directory_iterator(types[typeIdx])) {
                if (entry.is_regular_file()) {
                    allImages[typeIdx].push_back(entry.path().string());
                    numSamples++;
                    samplesPerType[typeIdx]++;
                }
            }
        }

        this->numTrainSamples = 0;
        this->numTestSamples = 0;

        trainSamplesPerType.resize(types.size());
        for (usize typeIdx = 0; typeIdx < types.size(); typeIdx++) {
            trainSamplesPerType[typeIdx] = samplesPerType[typeIdx] * trainSplit;
            numTrainSamples += trainSamplesPerType[typeIdx];
            numTestSamples += samplesPerType[typeIdx] - trainSamplesPerType[typeIdx];
        }

        fmt::println("Using train to test ratio of {:.2f} with {} train samples and {} test samples", trainSplit / (1 - trainSplit), numTrainSamples, numTestSamples);
    }

    void ImageDataLoader::loadBatch(const usize batchIdx) {
        data[batchIdx].input.resize(batchSize, width * height);
        data[batchIdx].target.resize(batchSize, types.size());

        data[batchIdx].target.fill(0);
        data[batchIdx].input.fill(0);

        if (types.empty())
            exitWithMsg(fmt::format("No types found in '{}'", dataDir), 1);

        std::vector<std::vector<internal::DataPoint>> localData(threads);

        #pragma omp parallel for num_threads(std::max<usize>(threads, 1))
        for (usize i = 0; i < batchSize; i++) {
            std::mt19937 rng{ std::random_device{}() + omp_get_thread_num()};

            // Randomly pick a type
            std::uniform_int_distribution<usize> typeDist(0, types.size() - 1);
            const usize typeIdx = typeDist(rng);

            // Randomly pick an image
            std::uniform_int_distribution<usize> imgDist(0, trainSamplesPerType[typeIdx] - 1);
            const usize imgIdx = imgDist(rng);

            std::vector<float> input = loadGreyscaleImage(allImages[typeIdx][imgIdx], width, height);
            std::vector<float> target(types.size(), 0);
            target[typeIdx] = 1;

            std::memcpy(&data[batchIdx].input[i, 0], input.data(), sizeof(float) * input.size());
            std::memcpy(&data[batchIdx].target[i, 0], target.data(), sizeof(float) * target.size());
        }
    }

    void ImageDataLoader::loadTestSet() {
        data[currBatch].input.resize(numTestSamples, width * height);
        data[currBatch].target.resize(numTestSamples, types.size());

        data[currBatch].input.fill(0.0f);
        data[currBatch].target.fill(0.0f);

        if (types.empty())
            exitWithMsg(fmt::format("No types found in '{}'", dataDir), 1);

        u64 idx = 0;
        for (usize typeIdx = 0; typeIdx < types.size(); typeIdx++) {
            for (usize imgIdx = trainSamplesPerType[typeIdx]; imgIdx < allImages[typeIdx].size(); imgIdx++) {
                assert(idx < numTestSamples);

                std::vector<float> input = loadGreyscaleImage(allImages[typeIdx][imgIdx], width, height);
                std::vector<float> target(types.size(), 0.0f);
                target[typeIdx] = 1.0f;

                std::memcpy(&data[currBatch].input[idx, 0], input.data(), sizeof(float) * input.size());
                std::memcpy(&data[currBatch].target[idx, 0], target.data(), sizeof(float) * target.size());

                idx++;
            }
        }
    }

    bool ImageDataLoader::countCorrect(const Tensor& output, const Tensor& target) {
        u64 numCorrect = 0;

        for (usize i = 0; i < target.dim(0); i++) {
            usize guess = 0;
            usize goal = 0;
            for (usize j = 0; j < target.dim(1); j++) {
                if (output[i, j] > output[i, guess])
                    guess = j;
                if (target[i, j] > target[i, goal])
                    goal = j;
            }
            numCorrect += (guess == goal);
        }

        return numCorrect;
    }
}