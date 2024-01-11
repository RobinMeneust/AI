/**
 * @file Relu.cpp
 * @author Robin MENEUST
 * @brief Methods of the class Relu used by a neuron layer. This class defines both the Relu function and its derivatives
 * @date 2024-01-02
 */

#include "../include/Relu.h"
#include <iostream>

/**
 * For all component xi of the input vector, calculate Relu(xi) and return a vector that contains the result for each xi
 * @param input Input vector
 * @return Vector of the output of the function for each component of the input vector
 */

Tensor* Relu::getValues(const Tensor &input) {
    Tensor* output = new Tensor(input.getNDim(), input.getDimSizes());
    for(int i=0; i<input.getDimSizes()[0]; i++) {
        float inputValue = input.get({i});
        output->set({i}, inputValue <= 0 ? 0 : inputValue);
    }
    return output;
}

/**
 * For all component xi of the input vector, calculate the derivative dRelu(xi)/dxi and return a vector that contains the result for each xi
 * @param input Input vector
 * @param size Size of the input vector
 * @return Vector of the derivative of the function for each component of the input vector
 */

Tensor* Relu::getDerivatives(const Tensor &input) {
    Tensor* output = new Tensor(input.getNDim(), input.getDimSizes());
    for(int i=0; i<input.getDimSizes()[0]; i++) {
        output->set({i}, input.get({i}) <= 0 ? 0 : 1);
    }
    return output;
}