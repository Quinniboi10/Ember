#include "learner.h"

int main() {
    Ember::Network net(
        Ember::layers::Input(2 * 6 * 64),
        Ember::layers::Linear(4096),
        Ember::activations::ReLU(),
        Ember::layers::Linear(1)
     );

    net.to(Ember::GPU);

    constexpr Ember::usize evalScale = 400;

    Ember::dataloaders::chess::BulletTextDataLoader dataloader("../datasets/preludeData.txt", 1024 * 16, evalScale, 1);
    Ember::optimizers::Adam optimizer(net);

    Ember::Learner learner(net, dataloader, optimizer, Ember::loss::SigmoidMSE(evalScale));

    std::cout << net << std::endl;

    learner.addCallbacks(
        Ember::callbacks::DropLROnPlateau(3, 0.3, Ember::Metric::TRAIN_LOSS),
        Ember::callbacks::StopWhenNoProgress(5, Ember::Metric::TRAIN_LOSS),
        Ember::callbacks::AutosaveBest("../net.bin")
    );

    learner.learn(0.005, 40, 1);
}