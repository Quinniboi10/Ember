#include "network.h"

namespace Ember {
    void Network::forward(const Tensor<1>& input, const usize threads) {
        openblas_set_num_threads(threads);
        layers[0]->values = input;

        for (usize i = 1; i < layers.size(); i++)
            layers[i]->forward(*layers[i - 1]);
    }

    const Tensor<1>& Network::output() const {
        return layers.back()->values;
    }

    std::ostream& operator<<(std::ostream& os, const Network& net) {
        os << fmt::format("Neural network consisting of {} layers\n", net.layers.size());
        for (usize i = 0; i < net.layers.size(); i++)
            os << fmt::format("    {}: {}\n", i, net.layers[i]->str());
        return os;
    }
};