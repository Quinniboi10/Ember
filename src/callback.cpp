#include "callback.h"

#include "learner.h"
#include "save.h"

// Metrics are lower-is-better so some must be inverted
float getMetric(const Ember::Metric metric, const Ember::Learner* learner) {
    switch (metric) {
        case Ember::Metric::TRAIN_LOSS:
            return learner->trainLoss;
        case Ember::Metric::TEST_LOSS:
            return learner->testLoss;
        case Ember::Metric::TEST_ACCURACY:
            return 1.0f - learner->testAccuracy;
    }

    return 0;
}

namespace Ember::callbacks {
    void DropLROnPlateau::run(const internal::LearnerLoopState state) {
        assert(learner);
        if (state != internal::AFTER_EPOCH)
            return;

        const float current = getMetric(metric, learner);

        if (current < best) {
            best = current;
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

        const float current = getMetric(metric, learner);

        if (current < best) {
            best = current;
            sinceLast = 0;
        }
        else if (sinceLast >= patience)
            throw internal::CancelFitException();
        else
            sinceLast++;
    }


    void AutosaveBest::run(const internal::LearnerLoopState state) {
        assert(learner);
        if (state != internal::AFTER_EPOCH)
            return;

        const float current = getMetric(metric, learner);

        if (current < best) {
            best = current;
            saveParams(path, learner->net);
        }
    }
}