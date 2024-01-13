/**
 * @file LeakyRelu.cpp
 * @author Robin MENEUST
 * @brief Methods of the class LeakyRelu used by a neuron layer. This class defines both the LeakyRelu function and its derivatives
 * @date 2024-01-08
 */

#include "../include/LeakyRelu.h"
#include <iostream>

/**
 * For all component xi of the input vector, calculate LeakyReLU(xi) and return a vector that contains the result for each xi
 * @param input Input vector
 * @return Vector of the output of the function for each component of the input vector
 */
Tensor * LeakyRelu::getValues(const Tensor &input, int batchSize) {
    Tensor* output = new Tensor(input.getNDim(), input.getDimSizes());
    float* outputData = output->getData();
    float* inputData = input.getData();

    for(int i=0; i<output->size(); i++) {
        outputData[i] = inputData[i] <= 0 ? 0.01f * inputData[i] : inputData[i];
    }
    return output;
}

/**
 * For all component xi of the input vector, calculate the derivative dLeakyReLU(xi)/dxi and return a vector that contains the result for each xi
 * @param input Input vector
 * @return Vector of the derivative of the function for each component of the input vector
 */
Tensor* LeakyRelu::getDerivatives(const Tensor &input, int batchSize) {
    Tensor* output = new Tensor(input.getNDim(), input.getDimSizes());
    float* outputData = output->getData();
    float* inputData = input.getData();

    for(int i=0; i<output->size() ; i++) {
        outputData[i] = inputData[i] <= 0 ? 0.01f : 1;
    }
    return output;
}