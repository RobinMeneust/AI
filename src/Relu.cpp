/**
 * @file Relu.cpp
 * @author Robin MENEUST
 * @brief Methods of the class Relu used by a layer of an AI model. This class defines both the Relu function and its derivatives
 * @date 2024-01-02
 */

#include "../include/Relu.h"

/**
 * For all component xi of the input tensor, calculate Relu(xi) and return a tensor that contains the result for each xi
 * @param input Input tensor whose rank is greater than or equal to 1
 * @param batchSize Size of the batch. It's not used for this function, but it is required by the parent class.
 * @return Tensor of the output of the function for each component of the input tensor
 */

Tensor * Relu::getValues(const Tensor &input, int batchSize) {
    Tensor* output = new Tensor(input.getNDim(), input.getDimSizes());
    float* outputData = output->getData();
    float* inputData = input.getData();

    for(int i=0; i<output->size(); i++) {
        outputData[i] = inputData[i] <= 0 ? 0 : inputData[i];
    }
    return output;
}

/**
 * For all component xi of the input tensor, calculate the derivative dRelu(xi)/dxi and return a tensor that contains the result for each xi
 * @param input Input tensor whose rank is greater than or equal to 1
 * @param batchSize Size of the batch. It's not used for this function, but it is required by the parent class.
 * @return Tensor of the derivatives of the function for each component of the input tensor
 */

Tensor* Relu::getDerivatives(const Tensor &input, int batchSize) {
    Tensor* output = new Tensor(input.getNDim(), input.getDimSizes());
    float* outputData = output->getData();
    float* inputData = input.getData();

    for(int i=0; i<output->size() ; i++) {
        outputData[i] = inputData[i] <= 0 ? 0 : 1;
    }
    return output;
}