/**
 * @file Identity.cpp
 * @author Robin MENEUST
 * @brief Methods of the class Sigmoid used by a neuron layer. This class defines both the Sigmoid function and its derivatives
 * @date 2024-01-10
 */

#include "../include/Identity.h"
#include <cmath>
#include <iostream>

/**
 * For all component xi of the input vector, calculate Identity(xi) and return a vector that contains the result for each xi
 * @param input Input vector
 * @return Vector of the output of the function for each component of the input vector
 */
Tensor* Identity::getValues(const Tensor &input) {
    return new Tensor(input);
}

/**
 * For all component xi of the input vector, calculate the derivative dIdentity(xi)/dxi and return a vector that contains the result for each xi.
 * @param input Input vector
 * @return Vector of the derivative of the function for each component of the input vector
 */

Tensor* Identity::getDerivatives(const Tensor &input) {
    float* output = new float[input.getDimSizes()[0]];
    for(int i=0; i<input.getDimSizes()[0]; i++) {
        output[i] = 1;
    }
    Tensor* outputTensor = new Tensor(input.getNDim(), input.getDimSizes(), output);
    delete[] output;
    return outputTensor;
}