#include "activation.h"

namespace Ember {
    namespace internal::activations {
        float ReLU(const float x) {
            return std::max(x, 0.0f);
        }
    }

    namespace activations {
        void ReLU::forward(const Layer& previous) {
            for (usize prev = 0; prev < previous.size; prev++)
                values[prev] = internal::activations::ReLU(previous.getOutputs()[prev]);
        }
    }
}