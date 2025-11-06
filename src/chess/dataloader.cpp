#include "../dataloader.h"

#include "board.h"

#include "../../external/fmt/format.h"

#include <filesystem>
#include <algorithm>
#include <fstream>
#include <omp.h>

#include "../util.h"

namespace Ember::dataloaders::chess {
    BulletTextDataLoader::BulletTextDataLoader(const std::string& filePath, const u64 batchSize, const u64 threads) : DataLoader(batchSize, threads), filePath(filePath) {
        fmt::println("Attempting to open file '{}'", filePath);
        if (!std::filesystem::exists(filePath) || std::filesystem::is_directory(filePath))
            exitWithMsg("Data file does not exist or is a directory: " + filePath, 1);


        std::string l;
        std::ifstream file(filePath);
        while (std::getline(file, l))
            numSamples++;

        fmt::println("Found {} positions", formatNum(numSamples));
    }

    void BulletTextDataLoader::loadBatch(const usize batchIdx) {
        std::ifstream file(filePath);

        data[batchIdx].input.resize(batchSize, static_cast<usize>(2 * 6 * 64));
        data[batchIdx].target.resize(batchSize, static_cast<usize>(1));

        data[batchIdx].target.fill(0);
        data[batchIdx].input.fill(0);

        std::vector<std::vector<internal::DataPoint>> localData(threads);

        std::string l;
        std::vector<std::string> lines;
        usize linesRead;


        linesRead = 0;
        while (std::getline(file, l) && linesRead < batchNumber * batchSize)
            linesRead++;

        // Load lines into a buffer
        linesRead = 0;
        while (linesRead < batchSize) {
            if (!std::getline(file, l)) {
                batchNumber = 0;
                file = std::ifstream(filePath);
            }
            // Discard null chars because windows uses encoding that
            // places a \0 after every character
            std::erase_if(l, [](const char c) { return c == '\0'; });

            if (std::ranges::all_of(l.begin(), l.end(), [](const char c) { return std::isspace(c); }))
                continue;

            lines.emplace_back(l);
            linesRead++;
        }

        assert(lines.size() == batchSize);

        #pragma omp parallel for num_threads(std::max<usize>(threads, 1))
        for (usize i = 0; i < batchSize; i++) {
            std::string& line = lines[i];

            // Strip UTF-16 BOM if present
            if (line.size() >= 2 && static_cast<unsigned char>(line[0]) == 0xFF && static_cast<unsigned char>(line[1]) == 0xFE) {
                line = line.substr(2);
            }

            const auto tokens = split(line, '|');

            if (tokens.size() != 3)
                exitWithMsg(fmt::format("Expected 3 tokens, got {}. Failed to parse line: {}", tokens.size(), line), 1);
            assert(tokens.size() == 3);

            const std::string& fen = tokens[0];
            const float eval = std::stof(tokens[1]);
            // Token 2 is discarded b/c it's the WDL which is not
            // used yet

            Ember::chess::Board board{};
            board.loadFromFEN(fen);

            std::vector<float> input = board.asInputLayer();
            std::vector<float> target(1, 0);

            std::memcpy(&data[batchIdx].input[i, 0], input.data(), sizeof(float) * input.size());
            data[batchIdx].target[i, 0] = eval;
        }
    }

    void BulletTextDataLoader::loadTestSet() {
        loadBatch(currBatch);
    }

    bool BulletTextDataLoader::countCorrect(const Tensor& output, const Tensor& target) {
        u64 numCorrect = 0;

        for (usize i = 0; i < target.dim(0); i++)
            numCorrect += (static_cast<i64>(output[i, 0]) == static_cast<i64>(target[i, 0]));

        return numCorrect;
    }
}