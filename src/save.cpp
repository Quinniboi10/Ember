#include "save.h"


namespace Ember {
    void saveParams(const std::string& path, const Network& net) {
        std::ofstream file(path, std::ios::binary);

        const auto write = [&](const auto& val) {
            file.write(reinterpret_cast<const char*>(&val), sizeof(val));
        };

        for (const auto& l : net.layers) {
            const auto* layer = dynamic_cast<internal::ComputeLayer*>(l.get());
            if (!layer)
                continue;

            for (const float weight : layer->weights.data())
                write(weight);

            for (const float bias : layer->biases)
                write(bias);
        }
    }

    void loadParams(const std::string& path, Network& net) {
        std::ifstream file(path, std::ios::binary);

        const auto read = [&file](auto& value) {
            file.read(reinterpret_cast<char*>(&value), sizeof(value));
        };

        for (auto& l : net.layers) {
            auto* layer = dynamic_cast<internal::ComputeLayer*>(l.get());
            if (!layer)
                continue;

            for (float& weight : layer->weights.data())
                read(weight);

            for (float& bias : layer->biases)
                read(bias);
        }
    }
}