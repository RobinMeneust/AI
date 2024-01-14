//
// Created by robin on 14/01/2024.
//

#include <iostream>
#include "../include/MaxPoolingLayer.h"
#include "../include/Identity.h"

MaxPoolingLayer::MaxPoolingLayer(const std::vector<int> &kernelDimSizes, int stride, int padding) : Layer({}, {}, new Identity()), kernelDimSizes(kernelDimSizes), stride(stride), padding(padding) {
    if(padding<0 || stride<=0 || kernelDimSizes.size() != 2 || kernelDimSizes[0] <= 0 || kernelDimSizes[1] <= 0) {
        std::cerr << "ERROR: Invalid kernel, padding or stride (we must have: padding>=0 and stride>0)" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void MaxPoolingLayer::changeInputShape(const std::vector<int> &newInputShape) {
    std::vector<int> newOutputShape;
    int i = 0;
    if(newInputShape.size() == 3) {
        newOutputShape.push_back(newInputShape[0]);
        i++;
    } else if(newInputShape.size() != 2) {
        std::cerr << "ERROR: Invalid input shape" << std::endl;
        exit(EXIT_FAILURE);
    }
    // width output = (number of iterations with i from 1 to (width input with padding) with a step of (stride) (i+=stride)) - (number of kernels that don't fit in the input matrix)
    //              = (width input with padding - width kernel + 1) / stride

    int dim1 = (newInputShape[i]+padding-kernelDimSizes[0]+1)/stride;
    i++;
    int dim2 = (newInputShape[i]+padding-kernelDimSizes[1]+1)/stride;

    newOutputShape.push_back(dim1);
    newOutputShape.push_back(dim2);

    changeShapes(newInputShape, newOutputShape);

    if(dim1<=0 || dim2<=0) {
        std::cerr << "ERROR: Invalid layer parameters" << std::endl;
        exit(EXIT_FAILURE);
    }
}

Tensor *MaxPoolingLayer::getOutput(const Tensor &input) {
    return nullptr; //TODO
}

void MaxPoolingLayer::adjustParams(float learningRate, Tensor *currentCostDerivatives, Tensor *prevLayerOutput) {
    //TODO
}

Tensor *MaxPoolingLayer::getPreActivationDerivatives(int currentLayerOutputIndex, int prevLayerOutputIndex) {
    return nullptr; //TODO
}

Tensor *MaxPoolingLayer::getPreActivationDerivatives() {
    return nullptr; //TODO
}

Tensor *MaxPoolingLayer::getPreActivationValues(const Tensor &input) {
    return nullptr; //TODO
}

std::string MaxPoolingLayer::toString() {
    return "Max-pooling layer";
}

