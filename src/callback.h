#pragma once

#include "types.h"

#include <limits>

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
    namespace callbacks {
        struct DropLROnPlateau : internal::Callback {
            u64 patience;
            float factor;

            u64 sinceLast;
            float best;

            DropLROnPlateau(const u64 patience, const float factor) : patience(patience), factor(factor) {
                sinceLast = 0;
                best = std::numeric_limits<float>::infinity();
            }

            void run(const internal::LearnerLoopState state) override;
        };

        struct StopWhenNoProgress : internal::Callback {
            u64 patience;

            u64 sinceLast;
            float best;

            explicit StopWhenNoProgress(const u64 patience) : patience(patience) {
                sinceLast = 0;
                best = std::numeric_limits<float>::infinity();
            }

            void run(const internal::LearnerLoopState state) override;
        };
    }
}
