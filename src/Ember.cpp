#include "learner.h"
#include "maxpool.h"
#include "convolution.h"

int main() {
    Ember::Network net(
        Ember::layers::Input(2 * 6 * 64),
        Ember::layers::Linear(1024),
        Ember::activations::ReLU(),
        Ember::layers::Linear(1)
     );

    Ember::dataloaders::chess::BulletTextDataLoader dataloader("../datasets/preludeData.txt", 64, 6);
    Ember::optimizers::Adam optimizer(net);

    Ember::Learner learner(net, dataloader, optimizer, Ember::loss::SigmoidMSE());

    std::cout << net << std::endl;

    learner.addCallbacks(
        Ember::callbacks::DropLROnPlateau(3, 0.3),
        Ember::callbacks::StopWhenNoProgress(5),
        Ember::callbacks::AutosaveBest("../net.bin")
    );

    learner.learn(0.001, 20, 1);
}
