/**
 * @file DenseLayer.cpp
 * @author Robin MENEUST
 * @brief Functions used to manipulate neuron layers
 * @date 2022-12-14
 */

#include "../include/Layer.h"

Layer::Layer(const std::vector<int> &inputShape, const std::vector<int> &outputShape, ActivationFunction* activationFunction) : inputShape(inputShape), outputShape(outputShape), activationFunction(activationFunction) {}

Layer::Layer(const Layer &copy) : inputShape(copy.inputShape), outputShape(copy.outputShape), activationFunction(copy.activationFunction)  {}

Layer::~Layer() {
    inputShape.clear();
    outputShape.clear();
}

int Layer::getDimInput() {
    return inputShape.size();
}

int Layer::getOutputDim() {
    return outputShape.size();
}

int Layer::getInputSize(int dim) {
    if(dim<getOutputDim())
        return inputShape[dim];
    else
        return -1;
}

int Layer::getOutputSize(int dim) {
    if(dim<getOutputDim())
        return outputShape[dim];
    else
        return -1;
}

/**
 * Get the derivatives (in a tensor) of this layer activation function evaluated at the given input
 * @param input Tensor where are evaluated the derivatives
 * @return Derivatives tensor
 */
Tensor* Layer::getActivationDerivatives(const Tensor& input) {
    return activationFunction->getDerivatives(input, input.getDimSize(0));
}

Tensor* Layer::getActivationValues(const Tensor &input) {
    return activationFunction->getValues(input, input.getDimSize(0));
}
