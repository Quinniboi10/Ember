#include "network.h"
#include "util.h"

namespace Ember {
    void Network::forward(const Tensor& input, const usize threads) {
        openblas_set_num_threads(threads);
        layers[0]->values = input;

        for (usize i = 1; i < layers.size(); i++)
            layers[i]->forward(*layers[i - 1]);
    }

    const Tensor& Network::output() const {
        return layers.back()->values;
    }

    std::ostream& operator<<(std::ostream& os, const Network& net) {
        u64 params = 0;
        os << fmt::format("Neural network consisting of {} layers\n", net.layers.size());
        for (usize i = 0; i < net.layers.size(); i++) {
            os << fmt::format("    {}: {}\n", i, net.layers[i]->str());
            params += net.layers[i]->numParams();
        }
        os << fmt::format("Network consists of {} learnable parameters", formatNum(params));
        return os;
    }
};