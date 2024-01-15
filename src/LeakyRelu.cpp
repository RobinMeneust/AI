/**
 * @file LeakyRelu.cpp
 * @author Robin MENEUST
 * @brief Methods of the class LeakyRelu used by a layer of an AI model. This class defines both the LeakyRelu function and its derivatives
 * @date 2024-01-08
 */

#include "../include/LeakyRelu.h"

/**
 * For all component xi of the input tensor, calculate LeakyReLU(xi) and return a tensor that contains the result for each xi
 * @param input Input tensor whose rank is greater than or equal to 1
 * @param batchSize Size of the batch. It's not used for this function, but it is required by the parent class.
 * @return Tensor of the output of the function for each component of the input tensor
 */
Tensor * LeakyRelu::getValues(const Tensor &input, int batchSize) {
    Tensor* output = new Tensor(input.getDimSizes());
    float* outputData = output->getData();
    float* inputData = input.getData();

    for(int i=0; i<output->getSize(); i++) {
        outputData[i] = inputData[i] <= 0 ? 0.01f * inputData[i] : inputData[i];
    }
    return output;
}

/**
 * For all component xi of the input tensor, calculate the derivative dLeakyReLU(xi)/dxi and return a tensor that contains the result for each xi
 * @param input Input tensor whose rank is greater than or equal to 1
 * @param batchSize Size of the batch. It's not used for this function, but it is required by the parent class.
 * @return Tensor of the derivatives of the function for each component of the input tensor
 */
Tensor* LeakyRelu::getDerivatives(const Tensor &input, int batchSize) {
    Tensor* output = new Tensor(input.getDimSizes());
    float* outputData = output->getData();
    float* inputData = input.getData();

    for(int i=0; i<output->getSize() ; i++) {
        outputData[i] = inputData[i] <= 0 ? 0.01f : 1;
    }
    return output;
}