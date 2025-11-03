#include "layer.h"
#include <limits>

namespace Ember::layers {

    struct MaxPool : internal::NonComputeLayer {
        usize x, y;
        usize stride;
        usize outX, outY;

        usize numChannels;

        std::vector<usize> maxIndex;

        explicit MaxPool(const usize stride = 2) : stride(stride) {
            x = y = 0;
            outX = outY = 0;
            numChannels = 0;
        }

        void setBatchSize(const usize batchSize) override {
            values.setDimension(0, batchSize);
            maxIndex.resize(batchSize * outX * outY * numChannels);
        }

        void init(const Tensor& previous) override {
            // Batch size, x, y, z
            assert(previous.dimensionality == 4);

            x = previous.dim(1);
            y = previous.dim(2);
            numChannels = previous.dim(3);

            assert(x % stride == 0);
            assert(y % stride == 0);

            outX = x / stride;
            outY = y / stride;

            values.resize(static_cast<usize>(1), outX, outY, numChannels);
        }

        void forward(const Layer& previous) override {
            const usize batchSize = previous.values.dim(0);

            for (usize b = 0; b < batchSize; b++) {
                for (usize c = 0; c < numChannels; c++) {
                    for (usize oy = 0; oy < outY; oy++) {
                        for (usize ox = 0; ox < outX; ox++) {
                            float bestVal = -std::numeric_limits<float>::infinity();
                            usize bestIdxX = 0;
                            usize bestIdxY = 0;

                            for (usize ky = 0; ky < stride; ky++) {
                                for (usize kx = 0; kx < stride; kx++) {
                                    const usize ix = ox * stride + kx;
                                    const usize iy = oy * stride + ky;

                                    const float v = previous.values[b, ix, iy, c];
                                    if (v > bestVal) {
                                        bestVal = v;
                                        bestIdxX = ix;
                                        bestIdxY = iy;
                                    }
                                }
                            }

                            values[b, ox, oy, c] = bestVal;
                            const usize flatOut = b * outX * outY * numChannels
                                                + c * outX * outY
                                                + oy * outX
                                                + ox;

                            const usize flatMax = b * x * y * numChannels
                                                + c * x * y
                                                + bestIdxY * x
                                                + bestIdxX;

                            maxIndex[flatOut] = flatMax;
                        }
                    }
                }
            }
        }

        Tensor backward(const Layer& previous, const Tensor& gradOutput) const override {
            const usize batchSize = gradOutput.dim(0);

            Tensor gradInput(previous.values.dims());

            const usize inputXY        = x * y;
            const usize inputXYZ       = x * y * numChannels;
            const usize outputXY       = outX * outY;
            const usize outputXYZ      = outX * outY * numChannels;

            for (usize b = 0; b < batchSize; b++) {
                for (usize c = 0; c < numChannels; c++) {
                    for (usize oy = 0; oy < outY; oy++) {
                        for (usize ox = 0; ox < outX; ox++) {

                            // Flat output index
                            const usize flatOut = b * outputXYZ
                                                + c * outputXY
                                                + oy * outX
                                                + ox;

                            // Flat input index where max occurred
                            const usize flatIn = maxIndex[flatOut];

                            // Convert flat input index to the indexes
                            const usize bc  = flatIn / inputXYZ;
                            const usize rem = flatIn % inputXYZ;

                            const usize cc  = rem / inputXY;
                            const usize rem2 = rem % inputXY;

                            const usize iy = rem2 / x;
                            const usize ix = rem2 % x;

                            gradInput[bc, ix, iy, cc] += gradOutput[b, ox, oy, c];
                        }
                    }
                }
            }

            return gradInput;
        }

        std::unique_ptr<Layer> clone() override {
            return std::make_unique<MaxPool>(*this);
        }

        std::string str() const override {
            return fmt::format("MaxPool {}x{}x{} to {}x{}x{}", x, y, numChannels, outX, outY, numChannels);
        }

        u64 numParams() const override { return 0; }
    };
}