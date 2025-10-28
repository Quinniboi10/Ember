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
    ImageDataLoader::ImageDataLoader(const std::string& dataDir, const u64 batchSize, const float trainSplit, const u64 threads, const usize width, const usize height)
        : DataLoader(batchSize, trainSplit, threads) {
        this->width = width;
        this->height = height;

        fmt::println("Attempting to open data dir '{}'", dataDir);
        if (!std::filesystem::exists(dataDir) || !std::filesystem::is_directory(dataDir))
            exitWithMsg("Data directory does not exist or is not a directory: " + dataDir, 1);

        this->dataDir = dataDir;

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

        fmt::println("Using train to test ratio of {:.2f} with approximately {:.0f} train samples and {:.0f} test samples", trainSplit / (1 - trainSplit), numSamples * trainSplit, numSamples * (1 - trainSplit));
    }

    void ImageDataLoader::loadBatch(const usize batchIdx) {
        data[batchIdx].clear();
        data[batchIdx].reserve(batchSize);

        if (types.empty())
            exitWithMsg(fmt::format("No types found in '{}'", dataDir), 1);

        std::vector<std::vector<internal::DataPoint>> localData(threads);

        #pragma omp parallel for num_threads(threads)
        for (usize i = 0; i < batchSize; i++) {
            std::mt19937 rng{ std::random_device{}() + omp_get_thread_num()};

            auto& threadData = localData[omp_get_thread_num()];
            threadData.reserve(batchSize / threads + 1);

            // Randomly pick a type
            std::uniform_int_distribution<usize> typeDist(0, types.size() - 1);
            const usize typeIdx = typeDist(rng);

            // Randomly pick an image
            std::uniform_int_distribution<usize> imgDist(0, samplesPerType[typeIdx] * trainSplit - 1);
            const usize imgIdx = imgDist(rng);

            std::vector<float> input = loadGreyscaleImage(allImages[typeIdx][imgIdx], width, height);
            std::vector<float> target(types.size(), 0);
            target[typeIdx] = 1;

            threadData.emplace_back(std::move(input), std::move(target));
        }

        for (auto& threadData : localData)
            data[batchIdx].insert(data[batchIdx].end(),
                                  std::make_move_iterator(threadData.begin()),
                                  std::make_move_iterator(threadData.end()));
    }

    void ImageDataLoader::loadTestSet() {
        data[currBatch].clear();

        if (types.empty())
            exitWithMsg(fmt::format("No types found in '{}'", dataDir), 1);

        for (usize typeIdx = 0; typeIdx < types.size(); typeIdx++) {
            u64 currIdx = 0;
            for (const auto& entry : std::filesystem::directory_iterator(types[typeIdx])) {
                if (entry.is_regular_file()) {
                    if (currIdx < samplesPerType[typeIdx] * trainSplit - 1) {
                        currIdx++;
                        continue;
                    }

                    std::vector<float> input = loadGreyscaleImage(entry.path().string(), width, height);
                    std::vector<float> target(types.size());
                    target[typeIdx] = 1;

                    data[currBatch].emplace_back(std::move(input), std::move(target));
                }
            }
        }
    }
}
