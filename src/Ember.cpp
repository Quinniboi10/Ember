#include "learner.h"

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

    Ember::Learner learner(net, dataloader, optimizer, Ember::loss::MeanSquaredError());

    net.setMode(Ember::NetworkMode::TRAIN);

    std::cout << net << std::endl;

    learner.learn(0.05, 2, 1);
}