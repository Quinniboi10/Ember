#pragma once

#include "types.h"

#include <limits>
#include <utility>

namespace Ember {
    struct Learner;

    namespace internal {
        enum LearnerLoopState {
            BEFORE_FIT,
            BEFORE_EPOCH,
            BEFORE_BATCH,
            AFTER_BATCH,
            AFTER_EPOCH,
            AFTER_FIT
        };

        struct CallbackException : std::exception {};

        struct CancelBatchException : CallbackException {};
        struct CancelEpochException : CallbackException {};
        struct CancelFitException : CallbackException {};

        struct Callback {
            Learner* learner;

            Callback() { learner = nullptr; }

            void setLearner(Learner* learner) { this->learner = learner; }

            virtual void run(const LearnerLoopState state) = 0;

            virtual ~Callback() = default;
        };
    }

    enum class Metric {
        TRAIN_LOSS,
        TEST_LOSS,
        TEST_ACCURACY
    };
    
    namespace callbacks {
        struct DropLROnPlateau : internal::Callback {
            Metric metric;
            u64 patience;
            float factor;

            u64 sinceLast;
            float best;

            DropLROnPlateau(const u64 patience, const float factor, const Metric metric = Metric::TEST_ACCURACY) : metric(metric), patience(patience), factor(factor) {
                sinceLast = 0;
                best = std::numeric_limits<float>::infinity();
            }

            void run(const internal::LearnerLoopState state) override;
        };

        struct StopWhenNoProgress : internal::Callback {
            Metric metric;
            u64 patience;

            u64 sinceLast;
            float best;

            explicit StopWhenNoProgress(const u64 patience, const Metric metric = Metric::TEST_ACCURACY) : metric(metric), patience(patience) {
                sinceLast = 0;
                best = std::numeric_limits<float>::infinity();
            }

            void run(const internal::LearnerLoopState state) override;
        };

        struct AutosaveBest : internal::Callback {
            Metric metric;
            std::string path;
            float best;

            explicit AutosaveBest(std::string  path, const Metric metric = Metric::TEST_ACCURACY) : metric(metric), path(std::move(path)) {
                best = std::numeric_limits<float>::infinity();
            }

            void run(const internal::LearnerLoopState state) override;
        };
    }
}
