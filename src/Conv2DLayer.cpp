//
// Created by robin on 14/01/2024.
//

#include <iostream>
#include <cmath>
#include <random>
#include "../include/Conv2DLayer.h"

Conv2DLayer::Conv2DLayer(int nbKernels, const std::vector<int>& kernelDimSizes, int stride, int padding, ActivationFunction *activationFunction) : Layer({}, {}, activationFunction), nbKernels(nbKernels), kernelDimSizes(kernelDimSizes), stride(stride), padding(padding) {
    if(nbKernels<=0 || padding<0 || stride<=0 || kernelDimSizes.size() != 2 || kernelDimSizes[0] <= 0 || kernelDimSizes[1] <= 0) {
        std::cerr << "ERROR: Invalid kernel, input padding or stride (we must have: padding>=0 and stride>0)" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void Conv2DLayer::changeInputShape(const std::vector<int> &newInputShape) {
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
        kernels[i] = createKernel();
    }
}

Conv2DLayer::~Conv2DLayer() {
    for(int i=0; i<kernels.size(); i++) {
        delete kernels[0];
    }
    kernels.clear();
}

Tensor * Conv2DLayer::createKernel() {
    Tensor* newKernel = new Tensor(2, kernelDimSizes);
    float* kernelData = newKernel->getData();

    // Uniform Xavier Initialization
    float upperBound = (float) sqrt(6.0/(double)(getInputSize()+getOutputSize()));
    float lowerBound = -upperBound;

    // random generator
    int seed = 5;
    std::default_random_engine gen(seed);
    std::uniform_real_distribution<float> distribution(lowerBound,upperBound);

    for(int i=0; i<newKernel->size(); i++) {
        kernelData[i] = distribution(gen);
    }

    return newKernel;
}

Tensor *Conv2DLayer::getOutput(const Tensor &input) {
    return nullptr;//TODO
}

void Conv2DLayer::adjustParams(float learningRate, Tensor *currentCostDerivatives, Tensor *prevLayerOutput) {
//TODO
}

Tensor *Conv2DLayer::getPreActivationDerivatives(int currentLayerOutputIndex, int prevLayerOutputIndex) {
    return nullptr; //TODO
}

Tensor *Conv2DLayer::getPreActivationDerivatives() {
    return nullptr; //TODO
}

Tensor *Conv2DLayer::getPreActivationValues(const Tensor &input) {
    return nullptr; //TODO
}

std::string Conv2DLayer::toString() {
    return "Conv2D layer";
}