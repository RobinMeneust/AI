//
// Created by robin on 14/01/2024.
//

#include <iostream>
#include <cmath>
#include <random>
#include "../include/Conv2DLayer.h"
#include "../include/MaxPoolingLayer.h"

Conv2DLayer::Conv2DLayer(int nbKernels, const std::vector<int>& kernelDimSizes, int stride, int padding, ActivationFunction *activationFunction) : Layer({}, {}, activationFunction), nbKernels(nbKernels), kernelDimSizes(kernelDimSizes), stride(stride), padding(padding) {
    if(nbKernels<=0 || padding<0 || stride<=0 || kernelDimSizes.size() != 2 || kernelDimSizes[0] <= 0 || kernelDimSizes[1] <= 0) {
        std::cerr << "ERROR: Invalid kernel, input padding or stride (we must have: padding>=0 and stride>0)" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void Conv2DLayer::changeInputShape(const std::vector<int> &newInputShape) {
    //TODO: Change the name of this function (changeInputShape is inaccurate)
    std::vector<int> newOutputShape;
    if(newInputShape.size() == 2) {
        newOutputShape.push_back(nbKernels);
    } else if(newInputShape.size() == 3) {
        newOutputShape.push_back(newInputShape[0]*nbKernels);
    } else {
        std::cerr << "ERROR: Invalid input shape" << std::endl;
        exit(EXIT_FAILURE);
    }
    // width output = (number of iterations with i from 1 to (width input with padding) with a step of (stride) (i+=stride)) - (number of kernels that don't fit in the input matrix)
    //              = (width input with padding - width kernel + 1) / stride

    int dim1 = (newInputShape[0]+padding-kernelDimSizes[0]+1)/stride;
    int dim2 = (newInputShape[1]+padding-kernelDimSizes[1]+1)/stride;

    newOutputShape.push_back(dim1);
    newOutputShape.push_back(dim2);

    changeShapes(newInputShape, newOutputShape);

    if(dim1<=0 || dim2<=0) {
        std::cerr << "ERROR: Invalid layer parameters" << std::endl;
        exit(EXIT_FAILURE);
    }

    for(int i=0; i<kernels.size(); i++) {
        delete kernels[0];
    }
    kernels.clear();

    // Create kernels
    for(int i=0; i<newOutputShape[0]; i++) {
        kernels.push_back(createKernel());
    }
}

Conv2DLayer::~Conv2DLayer() {
    for(int i=0; i<kernels.size(); i++) {
        delete kernels[0];
    }
    kernels.clear();
}

Tensor * Conv2DLayer::createKernel() {
    Tensor* newKernel = new Tensor(kernelDimSizes);
    float* kernelData = newKernel->getData();

    // Uniform Xavier Initialization
    float upperBound = (float) sqrt(6.0/(double)(getInputSize()+getOutputSize()));
    float lowerBound = -upperBound;

    // random generator
    int seed = 5;
    std::default_random_engine gen(seed);
    std::uniform_real_distribution<float> distribution(lowerBound,upperBound);

    for(int i=0; i<newKernel->getSize(); i++) {
        kernelData[i] = distribution(gen);
    }

    return newKernel;
}

Tensor *Conv2DLayer::getOutput(const Tensor &input) {
    Tensor* preActivationValues = getPreActivationValues(input);
    Tensor* output = getActivationValues(*preActivationValues);
    delete preActivationValues;
    return output;
}

void Conv2DLayer::adjustParams(float learningRate, Tensor *currentCostDerivatives, Tensor *prevLayerOutput) {
    //TODO
    float* currentCostDerivativesData = currentCostDerivatives->getData();
    float* prevLayerOutputData = prevLayerOutput->getData();

    int batchSize = currentCostDerivatives->getDimSize(0);
    int currentLayerOutputDim1 = currentCostDerivatives->getSize()/ batchSize;
    int prevLayerOutputDim1 = prevLayerOutput->getSize() / batchSize;

    int nb2DFramesOutput = getOutputSize(0);

    int outputWidth = getOutputDim() == 3 ? getOutputSize(1) : getOutputSize(0);
    int outputHeight = getOutputDim() == 3 ? getOutputSize(2) : getOutputSize(1);

    int inputWidth = getInputDim() == 3 ? getInputSize(1) : getInputSize(0);
    int inputHeight = getInputDim() == 3 ? getInputSize(2) : getInputSize(1);

    for (int k = 0; k < nb2DFramesOutput; k++) {
        int correspondingInput2DFrame = k / nbKernels;
        float *kernelData = kernels[k]->getData();
        // For each 2D outputs
        for (int yOut = 0; yOut <= outputHeight; yOut++) {
            for (int xOut = 0; xOut <= outputWidth; xOut++) {
                int p1 = k * yOut * outputWidth + xOut;
                int p2 = 0;

                int xStart = xOut * stride;
                int yStart = yOut * stride;

                for(int i=0; i<kernelDimSizes[1]; i++) {
                    for(int j=0; j<kernelDimSizes[0]; j++) {
                        if(xStart + j < inputWidth && yStart + i < inputHeight) {
                            int p1Copy = p1;
                            double delta = 0.02 * kernelData[p2]; // F_i = F_x,y
                            int p3 = correspondingInput2DFrame * (yStart + i) * kernelDimSizes[0] + xStart + j;
                            for (int b = 0; b < batchSize; b++) {
                                delta += currentCostDerivativesData[p1Copy] * prevLayerOutputData[p3]; // delta = dC/da_k * da_k/dz_k * dz_k/dw_i,j
                                p1Copy += currentLayerOutputDim1;
                                p3 += prevLayerOutputDim1;
                            }
                            delta /= (double) batchSize;
                            kernelData[p2] = kernelData[p2] - learningRate * delta;
                        }
                        p2++;
                    }
                }
            }
        }
    }
}

Tensor *Conv2DLayer::getPreActivationDerivatives(const Tensor &input) {
    if(input.getNDim() != getInputDim()+1) {
        std::cerr << "ERROR: Invalid input (check the dimensions)" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::vector<int> outputShapeWithBatch = {input.getDimSize(0)};

    // Here the output tensor is like a tensor of the same dimensions of the output of this layer where each of its component is a tensor with the same dimension of the input.
    // So that we have dz/da for each "z" and "a" of the output of the layer and input of the layer
    // The shape is: (batch size, nb 2D outputs, height of 2D outputs, width, nb 2D inputs, height of 2D inputs, width)

    for(int i=0; i<outputShape.size(); i++) {
        outputShapeWithBatch.push_back(outputShape[i]);
    }
    for(int i=0; i<inputShape.size(); i++) {
        outputShapeWithBatch.push_back(inputShape[i]);
    }

    Tensor* derivatives = new Tensor(outputShapeWithBatch);
    float* derivativesData = derivatives->getData();

    int nb2DFramesInput = 1;
    if(getInputDim() == 3) {
        nb2DFramesInput = getInputSize(0);
    }
    int nb2DFramesOutput = getOutputSize(0);

    int inputWidth = input.getNDim() == 3 ? input.getDimSize(1) : input.getDimSize(0);
    int inputHeight = input.getNDim() == 3 ? input.getDimSize(2) : input.getDimSize(1);

    inputWidth += padding;
    inputHeight += padding;

    int outputWidth = getOutputDim() == 3 ? getOutputSize(1) : getOutputSize(0);
    int outputHeight = getOutputDim() == 3 ? getOutputSize(2) : getOutputSize(1);

    for(int b=0; b<outputShapeWithBatch[0]; b++) {
        for(int k=0; k<nb2DFramesOutput; k++) {
            float *kernelData = kernels[k]->getData();
            // For each 2D outputs
            for (int yOut = 0; yOut <= outputHeight; yOut++) {
                for (int xOut = 0; xOut <= outputWidth; xOut++) {
                    // For each output element, compute the derivatives in respect to the previous layer elements
                    for (int nIn = 0; nIn < nb2DFramesInput * outputShapeWithBatch[0]; nIn++) {
                        // For each 2D inputs
                        int p = b * k * yOut * xOut * nIn * inputHeight * inputWidth; // TODO: it's not optimized

                        int xStart = xOut * stride;
                        int yStart = yOut * stride;
                        int xEnd = xStart + kernelDimSizes[0];
                        int yEnd = yStart + kernelDimSizes[1];

                        for (int yIn = 0; yIn <= inputHeight; yIn++) {
                            int p2 = (yIn-yOut) * inputWidth;
                            for (int xIn = 0; xIn <= inputWidth; xIn++) {
                                if (xIn >= xStart && xIn <= xEnd && yIn >= yStart && yIn <= yEnd) {
                                    derivativesData[p] = kernelData[p2 + xIn - xOut];
                                } else {
                                    derivativesData[p] = 0;
                                }
                                p++;
                            }
                        }
                    }
                }
            }
        }
    }

    return derivatives;
}

Tensor *Conv2DLayer::getPreActivationValues(const Tensor &input) {
    if(input.getNDim() != getInputDim()+1) {
        std::cerr << "ERROR: Invalid input (check the dimensions)" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::vector<int> outputShapeWithBatch = {input.getDimSize(0)};
    for(int i=0; i<outputShape.size(); i++) {
        outputShapeWithBatch.push_back(outputShape[i]);
    }

    Tensor* output = new Tensor(outputShapeWithBatch);
    float* outputData = output->getData();

    Tensor* inputWithPadding;
    if(padding != 0) {
        inputWithPadding = MaxPoolingLayer::addPaddingToBatchData(input, padding); //TODO This function should be moved to a separate file
    } else {
        inputWithPadding = (Tensor *) & input;
    }

    float* inputData = inputWithPadding->getData();

    int nb2DInputs = 1;
    if(getInputDim() == 4) {
        nb2DInputs = getInputSize(1);
    }

    int inputWidth = input.getNDim() == 3 ? inputWithPadding->getDimSize(1) : inputWithPadding->getDimSize(0);
    int inputHeight = input.getNDim() == 3 ? inputWithPadding->getDimSize(2) : inputWithPadding->getDimSize(1);

    int xUpperBound = inputWidth-kernelDimSizes[0];
    xUpperBound = (xUpperBound/stride) * stride;
    int yUpperBound = inputHeight-kernelDimSizes[1];
    yUpperBound = (yUpperBound/stride) * stride;

    int remainderInput = inputWidth - xUpperBound - stride + (stride-1)*inputWidth;


    int p = 0;

    for(int b=0; b<outputShapeWithBatch[0]; b++) {
        for (int k = 0; k < nbKernels; k++) {
            float* kernelData = kernels[k]->getData();
            int pInput = b*nb2DInputs;
            for (int n = 0; n < nb2DInputs; n++) {
                // For each 2D inputs compute Max-pooling
                for (int y = 0; y <= yUpperBound; y += stride) {
                    for (int x = 0; x <= xUpperBound; x += stride) {
                        // Apply the kernel to the following elements
                        outputData[p] = 0;

                        int k3 = 0;
                        for (int k1 = 0; k1 < kernelDimSizes[1]; k1++) {
                            int p2 = pInput + k1 * inputWidth;
                            for (int k2 = 0; k2 < kernelDimSizes[0]; k2++) {
                                outputData[p] += inputData[p2] * kernelData[k3];
                                k3++;
                                p2++;
                            }
                        }
                        p++;
                        pInput += stride;
                    }
                    pInput += remainderInput;
                }
            }
        }
    }

    if(padding != 0) {
        delete inputWithPadding;
    }

    return output;
}

std::string Conv2DLayer::toString() {
    return "Conv2D layer";
}