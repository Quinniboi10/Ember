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
        std::vector<float> patchMatrix;

        mutable std::vector<float> colGrad;

        explicit Convolution(const usize x, const usize y, const usize numKernels, const usize kernelSize, const usize stride = 1) : ComputeLayer(0), outX((x - kernelSize) / stride + 1), outY((y - kernelSize) / stride + 1), numKernels(numKernels), kernelSize(kernelSize), x(x), y(y), stride(stride) {
            outputSize = outX * outY * numKernels;
            values.resize(outputSize);
            weights.resize(numKernels, kernelSize * kernelSize);
            biases.resize(outputSize);

            cols = kernelSize * kernelSize;
            rows = outX * outY;
            patchMatrix.resize(rows * cols);

            colGrad.resize(rows * cols);
        }

        // Forward pass
        void forward(const Layer& previous) override {
            // Copy biases to output
            std::memcpy(values.ptr(), biases.ptr(), outputSize * sizeof(float));

            // Convert the image into columns
            // The goal of this is to allow the use
            // of OpenBLAS for the matrix math
            usize row = 0;
            for (usize oy = 0; oy < outY; oy++) {
                for (usize ox = 0; ox < outX; ox++) {
                    float* rowPtr = &patchMatrix[row * cols];
                    usize idx = 0;

                    for (usize ky = 0; ky < kernelSize; ky++) {
                        for (usize kx = 0; kx < kernelSize; kx++) {
                            const usize ix = ox * stride + kx;
                            const usize iy = oy * stride + ky;
                            rowPtr[idx++] = previous.values[iy * x + ix];
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
                values.ptr(), numKernels
            );
        }

        void init(const usize previousSize) override {}

        // Returns gradInput, weightGrad, biasGrad
        std::tuple<Tensor, Tensor, Tensor> backward(const Layer& previous, const Tensor& gradOutput) const override {
            const usize inputSize = previous.outputSize;

            Tensor gradInput(inputSize);
            Tensor weightGrad(weights.dims());
            Tensor biasGrad(outputSize);

            for (usize i = 0; i < outputSize; i++)
                biasGrad[i] = gradOutput[i];

            // gradOutput: (rows x numKernels)
            // patchMatrix: (rows x cols)
            // weightGrad: (numKernels x cols)
            cblas_sgemm(
                CblasRowMajor, CblasTrans, CblasNoTrans,
                numKernels, cols, rows,
                1.0f,
                gradOutput.ptr(), numKernels,
                patchMatrix.data(), cols,
                0.0f,
                weightGrad.ptr(), cols
            );

            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rows, cols, numKernels,
                1.0f,
                gradOutput.ptr(), numKernels,
                weights.ptr(), cols,
                0.0f,
                colGrad.data(), cols
            );

            // Reconstruct the grad input
            for (usize oy = 0; oy < outY; oy++) {
                for (usize ox = 0; ox < outX; ox++) {

                    const usize row = oy * outX + ox;
                    const float* rowPtr = &colGrad[row * cols];
                    usize idx = 0;

                    for (usize ky = 0; ky < kernelSize; ky++) {
                        for (usize kx = 0; kx < kernelSize; kx++) {
                            const usize ix = ox * stride + kx;
                            const usize iy = oy * stride + ky;
                            gradInput[iy * x + ix] += rowPtr[idx++];
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
            return fmt::format("Convolution - {} {}x{} kernels and {} output features", numKernels, kernelSize, kernelSize, outputSize);
        }
        u64 numParams() const override { return weights.size() + biases.size(); }
    };
}