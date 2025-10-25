#include "callback.h"
#include "learner.h"

namespace Ember::callbacks {
    void DropLROnPlateau::run(const internal::LearnerLoopState state) {
        assert(learner);
        if (state != internal::AFTER_EPOCH)
            return;

        if (learner->trainLoss < best) {
            best = learner->trainLoss;
            sinceLast = 0;
        }
        else if (sinceLast >= patience) {
            learner->lr *= factor;
            sinceLast = 0;

            fmt::println("Dropping LR to {}", learner->lr);
        }
        else
            sinceLast++;
    }

    void StopWhenNoProgress::run(const internal::LearnerLoopState state) {
        assert(learner);
        if (state != internal::AFTER_EPOCH)
            return;

        if (learner->trainLoss < best) {
            best = learner->trainLoss;
            sinceLast = 0;
        }
        else if (sinceLast >= patience)
            throw internal::CancelFitException();
        else
            sinceLast++;
    }
}