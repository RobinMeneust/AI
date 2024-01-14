#include "../include/FlattenLayer.h"
#include "../include/Identity.h"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <bits/stdc++.h>

FlattenLayer::FlattenLayer(const std::vector<int> &inputShape) : Layer(inputShape, {}, new Identity()) {
    setOutputShape({getInputSize()});
}

FlattenLayer::FlattenLayer() : Layer({}, {}, new Identity()) {}

Tensor* FlattenLayer::getPreActivationValues(const Tensor &input) {
    return new Tensor(2, {input.getDimSize(0),outputShape[0]}, input.getData());
}


Tensor* FlattenLayer::getOutput(const Tensor &input) {
    return getPreActivationValues(input);
}

void FlattenLayer::adjustParams(float learningRate, Tensor* currentCostDerivatives, Tensor* prevLayerOutput) {}

Tensor* FlattenLayer::getPreActivationDerivatives(int currentLayerOutputIndex, int prevLayerOutputIndex) {
    Tensor* output = new Tensor(1, {1});
    output->set({0},currentLayerOutputIndex == prevLayerOutputIndex ? 1 : 0);
    return output;
}

Tensor *FlattenLayer::getPreActivationDerivatives() {
    Tensor* output = new Tensor(2, {getOutputSize(),getInputSize()});
    float* outputData = output->getData();

    int k = 0;
    for(int i=0; i<getOutputSize(); i++) {
        for(int j=0; j<getInputSize(); j++) {
            outputData[k] = i==j;
            k++;
        }
    }
    return output;
}

std::string FlattenLayer::toString() {
    std::string s = "Flatten layer";
    return s;
}


void FlattenLayer::changeInputShape(const std::vector<int> &newInputShape) {
    int size = 1;
    for(auto &s: newInputShape) {
        size *= s;
    }

    changeShapes(newInputShape, {size});

    if(getInputSize()==0 || getInputSize() != getOutputSize() || getOutputDim() != 1) {
        std::cerr << "ERROR: Invalid layer parameters" << std::endl;
        exit(EXIT_FAILURE);
    }
}
