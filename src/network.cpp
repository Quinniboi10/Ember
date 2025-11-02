#include "network.h"

#include "dataloader.h"
#include "util.h"

namespace Ember {
    void Network::forward(const Tensor& input, const usize threads) {
        assert(input.dimensionality == 2);
        assert(input.dim(1) == layers[0]->size);
        openblas_set_num_threads(threads);

        for (auto& l : layers)
            l->values.resize(input.dims()[0], l->size);

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
        os << fmt::format("Network contains a total of {} learnable parameters", formatNum(params));
        return os;
    }
};