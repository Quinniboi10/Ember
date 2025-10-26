#include "learner.h"
#include "save.h"

int main() {
    Ember::Network net(
        Ember::layers::Input(28 * 28),
        Ember::layers::Linear(28 * 28, 64),
        Ember::activations::ReLU(),
        Ember::layers::Linear(64, 10),
        Ember::activations::Softmax()
     );

    Ember::dataloaders::ImageDataLoader dataloader("../datasets/MNIST/", 128, 0.9, 6, 28, 28);
    Ember::optimizers::Adam optimizer(net);

    Ember::Learner learner(net, dataloader, optimizer, Ember::loss::CrossEntropyLoss());

    net.setMode(Ember::NetworkMode::TRAIN);

    std::cout << net << std::endl;

    learner.addCallbacks(
        Ember::callbacks::DropLROnPlateau(1, 0.3),
        Ember::callbacks::StopWhenNoProgress(3)
    );

    learner.learn(0.01, 20, 1);

    Ember::saveParams("../net.bin", net);
}