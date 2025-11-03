#include "learner.h"
#include "maxpool.h"
#include "convolution.h"

int main() {
    Ember::Network net(
        Ember::layers::Input(28, 28),
        Ember::layers::Convolution(8, 3),
        Ember::activations::ReLU(),
        Ember::layers::MaxPool(),
        Ember::layers::Flatten(),
        Ember::layers::Linear(128),
        Ember::activations::ReLU(),
        Ember::layers::Linear(64),
        Ember::activations::ReLU(),
        Ember::layers::Linear(10),
        Ember::activations::Softmax()
     );

    Ember::dataloaders::ImageDataLoader dataloader("../datasets/MNIST/", 32, 0.9, 6, 28, 28);
    Ember::optimizers::Adam optimizer(net);

    Ember::Learner learner(net, dataloader, optimizer, Ember::loss::CrossEntropyLoss());

    std::cout << net << std::endl;

    learner.addCallbacks(
        Ember::callbacks::DropLROnPlateau(3, 0.3),
        Ember::callbacks::StopWhenNoProgress(5),
        Ember::callbacks::AutosaveBest("../net.bin")
    );

    learner.learn(0.001, 20, 1);
}
