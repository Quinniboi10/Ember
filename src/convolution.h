#pragma once

#include "types.h"
#include "layer.h"

namespace Ember::layers {
    struct Convolution : internal::ComputeLayer {
        // Weights are stored by kernel
        // Assuming the kernel is N x M
        // and K is the number of kernels
        // there will be N x M x K weights
        // Indexed weights[kernel number][weight (flattened)]

        usize outX; // Output x
        usize outY; // Output y

        usize numKernels; // Number of kernels, output Z
        usize kernelSize; // Size of the kernel

        usize x;
        usize y;

        usize stride;

        usize rows;
        usize cols;
        usize inputChannels;

        std::vector<float> patchMatrix;

        mutable std::vector<float> colGrad;
        mutable std::vector<float> localPatch;

        Convolution(const usize numKernels, const usize kernelSize, const usize stride = 1) : ComputeLayer(0), numKernels(numKernels), kernelSize(kernelSize), stride(stride) {
            outX = outY = 0;
            x = y = 0;
            rows = cols = 0;

            biases.resize(numKernels);
        }

        // Given the values of the previous
        // layer we can interpret what
        // input this layer will receive
        void init(const Tensor& previous) override {
            // Batch size, x, y, channels
            assert(previous.dimensionality == 4);

            x = previous.dim(1);
            y = previous.dim(2);
            inputChannels = previous.dim(3);

            outX = (x - kernelSize) / stride + 1;
            outY = (y - kernelSize) / stride + 1;

            values.resize(static_cast<usize>(1), outX, outY, numKernels);

            rows = outX * outY;
            cols = kernelSize * kernelSize * inputChannels;

            weights.resize(numKernels, cols);

            patchMatrix.resize(rows * cols);

            colGrad.resize(rows * cols);
            localPatch.resize(rows * cols);
        }

        // Forward pass
        void forward(const Layer& previous) override {
            const usize batchSize = values.dim(0);

            // Copy biases to output
            for (usize i = 0; i < batchSize; i++) {
                for (usize ox = 0; ox < outX; ox++) {
                    for (usize oy = 0; oy < outY; oy++) {
                        for (usize j = 0; j < numKernels; j++) {
                            values[i, ox, oy, j] = biases[j];
                        }
                    }
                }
            }

            for (usize i = 0; i < batchSize; i++) {
                // Convert the image into columns
                // The goal of this is to allow the use
                // of OpenBLAS for the matrix math
                usize row = 0;
                for (usize ox = 0; ox < outX; ox++) {
                    for (usize oy = 0; oy < outY; oy++) {
                        float* rowPtr = &patchMatrix[row * cols];
                        usize idx = 0;
                        for (usize ch = 0; ch < inputChannels; ch++) {
                            for (usize ky = 0; ky < kernelSize; ky++) {
                                for (usize kx = 0; kx < kernelSize; kx++) {
                                    const usize ix = ox * stride + kx;
                                    const usize iy = oy * stride + ky;
                                    rowPtr[idx++] = previous.values[i, ix, iy, ch];
                                }
                            }
                        }
                        row++;
                    }
                }

                // Run the matrix math
                cblas_sgemm(
                    CblasRowMajor, CblasNoTrans, CblasTrans,
                    rows, numKernels, cols,
                    1.0f,
                    patchMatrix.data(), cols,
                    weights.ptr(), cols,
                    1.0f,
                    &values[i, 0, 0, 0], numKernels
                );
            }
        }

    // Returns gradInput, weightGrad, biasGrad
    std::tuple<Tensor, Tensor, Tensor> backward(const Layer& previous, const Tensor& gradOutput) const override {
        const usize batchSize = values.dim(0);
        const usize outputSize = rows * numKernels;

        Tensor gradInput(previous.values.dims());

        Tensor weightGrad(weights.dims());

        Tensor biasGrad(numKernels);

        for (usize i = 0; i < batchSize; i++) {
            for (usize k = 0; k < numKernels; k++) {
                float sum = 0.0f;
                for (usize ox = 0; ox < outX; ox++)
                    for (usize oy = 0; oy < outY; oy++)
                        sum += gradOutput[i, ox, oy, k];
                biasGrad[k] += sum;
            }
        }

        for (usize i = 0; i < batchSize; i++) {
            usize row = 0;
            for (usize ox = 0; ox < outX; ox++) {
                for (usize oy = 0; oy < outY; oy++) {
                    float* rowPtr = &localPatch[row * cols];
                    usize idx = 0;

                    for (usize ch = 0; ch < inputChannels; ch++) {
                        for (usize ky = 0; ky < kernelSize; ky++) {
                            for (usize kx = 0; kx < kernelSize; kx++) {
                                const usize ix = ox * stride + kx;
                                const usize iy = oy * stride + ky;
                                rowPtr[idx++] = previous.values[i, ix, iy, ch];
                            }
                        }
                    }
                    row++;
                }
            }

            const float* goPtr = gradOutput.ptr() + i * outputSize;

            // gradOutput: (rows x numKernels)
            // localPatch: (rows x cols)
            // weightGrad: (numKernels x cols)
            weightGrad.madd(
                CblasTrans, CblasNoTrans,
                numKernels, cols, rows,
                1.0f,
                goPtr, numKernels,
                localPatch.data(), cols,
                (i == 0 ? 0.0f : 1.0f)
            );

            sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rows, cols, numKernels,
                1.0f,
                goPtr, numKernels,
                weights.ptr(), cols,
                0.0f,
                colGrad.data(), cols
            );

            // Reconstruct the grad input
            for (usize ox = 0; ox < outX; ox++) {
                for (usize oy = 0; oy < outY; oy++) {
                    const usize row = ox * outY + oy;
                    const float* rowPtr = &colGrad[row * cols];
                    usize idx = 0;

                    for (usize ch = 0; ch < inputChannels; ch++) {
                        for (usize ky = 0; ky < kernelSize; ky++) {
                            for (usize kx = 0; kx < kernelSize; kx++) {
                                const usize ix = ox * stride + kx;
                                const usize iy = oy * stride + ky;
                                gradInput[i, ix, iy, ch] += rowPtr[idx++];
                            }
                        }
                    }
                }
            }
        }

        return { gradInput, weightGrad, biasGrad };
    }

        std::unique_ptr<Layer> clone() override {
            return std::make_unique<Convolution>(*this);
        }

        std::string str() const override {
            return fmt::format("Convolution - {} {}x{} kernels and {} input channels to {}x{}x{} output features", numKernels, kernelSize, kernelSize, inputChannels, outX, outY, numKernels);
        }
        u64 numParams() const override { return weights.size() + biases.size(); }
    };
}