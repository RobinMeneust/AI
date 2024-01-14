/**
 * @file Sigmoid.cpp
 * @author Robin MENEUST
 * @brief Methods of the class Sigmoid used by a layer of an AI model. This class defines both the Sigmoid function and its derivatives
 * @date 2023-12-15
 */

#include "../include/Sigmoid.h"
#include <cmath>

/**
 * For all component xi of the input tensor, calculate Sigmoid(xi) and return a tensor that contains the result for each xi
 * @param input Input tensor whose rank is greater than or equal to 1
 * @param batchSize Size of the batch. It's not used for this function, but it is required by the parent class.
 * @return Tensor of the output of the function for each component of the input tensor
 */

Tensor * Sigmoid::getValues(const Tensor &input, int batchSize) {
    Tensor* output = new Tensor(input.getNDim(), input.getDimSizes());
    float* outputData = output->getData();
    float* inputData = input.getData();

    for(int i=0; i<input.size(); i++) {
        outputData[i] = 1.0f / (1 + exp(-inputData[i]));
    }
    return output;
}

/**
 * For all component xi of the input tensor, calculate the derivative dSigmoid(xi)/dxi and return a tensor that contains the result for each xi.
 * @param input Input tensor whose rank is greater than or equal to 1
 * @param batchSize Size of the batch. It's not used for this function, but it is required by the parent class.
 * @return Tensor of the derivatives of the function for each component of the input tensor
 */

Tensor* Sigmoid::getDerivatives(const Tensor &input, int batchSize) {
    Tensor* output = getValues(input, batchSize);
    float* outputData = output->getData();

    for(int i=0; i<input.size(); i++) {
        outputData[i] = outputData[i]*(1-outputData[1]);
    }
    return output;
}