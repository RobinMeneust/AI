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
Tensor* LeakyRelu::getValues(const Tensor &input) {
    Tensor* output = new Tensor(input.getNDim(), input.getDimSizes());
    for(int i=0; i<output->getDimSizes()[0]; i++) {
        float inputValue = input.get({i});
        output->set({i}, inputValue <= 0 ? 0.01f * inputValue : inputValue);
    }
    return output;
}

/**
 * For all component xi of the input vector, calculate the derivative dLeakyReLU(xi)/dxi and return a vector that contains the result for each xi
 * @param input Input vector
 * @return Vector of the derivative of the function for each component of the input vector
 */
Tensor* LeakyRelu::getDerivatives(const Tensor &input) {
    Tensor* output = new Tensor(input.getNDim(), input.getDimSizes());
    for(int i=0; i<output->getDimSizes()[0]; i++) {
        output->set({i}, input.get({i}) <= 0 ? 0.01 : 1);
    }
    return output;
}