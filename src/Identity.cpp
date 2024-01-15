/**
 * @file Identity.cpp
 * @author Robin MENEUST
 * @brief Methods of the class Identity used by a layer of an AI model. This class defines both the Sigmoid function and its derivatives
 * @date 2024-01-10
 */

#include "../include/Identity.h"

/**
 * For all component xi of the input tensor, calculate Identity(xi) and return a tensor that contains the result for each xi
 * @param input Input tensor whose rank is greater than or equal to 1
 * @param batchSize Size of the batch. It's not used for this function, but it is required by the parent class.
 * @return Tensor of the output of the function for each component of the input tensor
 */
Tensor * Identity::getValues(const Tensor &input, int batchSize) {
    return new Tensor(input);
}

/**
 * For all component xi of the input tensor, calculate the derivative dIdentity(xi)/dxi and return a tensor that contains the result for each xi.
 * @param input Input tensor whose rank is greater than or equal to 1
 * @param batchSize Size of the batch. It's not used for this function, but it is required by the parent class.
 * @return Tensor of the derivatives of the function for each component of the input tensor
 */

Tensor* Identity::getDerivatives(const Tensor &input, int batchSize) {
    Tensor* output = new Tensor(input.getNDim(), input.getDimSizes());
    float* outputData = output->getData();

    for(int i=0; i<input.size(); i++) {
        outputData[i] = 1;
    }

    return output;
}