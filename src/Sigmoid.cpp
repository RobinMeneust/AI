/**
 * @file Sigmoid.cpp
 * @author Robin MENEUST
 * @brief Methods of the class Sigmoid used by a neuron layer. This class defines both the Sigmoid function and its derivatives
 * @date 2023-12-15
 */

#include "../include/Sigmoid.h"
#include <cmath>
#include <iostream>

/**
 * For all component xi of the input vector, calculate Sigmoid(xi) and return a vector that contains the result for each xi
 * @param input Input vector
 * @return Vector of the output of the function for each component of the input vector
 */

Tensor* Sigmoid::getValues(const Tensor &input) {
    Tensor* output = new Tensor(input.getNDim(), input.getDimSizes());
    for(int i=0; i<input.getDimSizes()[0]; i++) {
        output->set({i}, 1.0f/(1+exp(-input.get({i}))));
    }
    return output;
}

/**
 * For all component xi of the input vector, calculate the derivative dSigmoid(xi)/dxi and return a vector that contains the result for each xi.
 * @param input Input vector
 * @return Vector of the derivative of the function for each component of the input vector
 */

Tensor* Sigmoid::getDerivatives(const Tensor &input) {
    Tensor* output = getValues(input);
    for(int i=0; i<input.getDimSizes()[0]; i++) {
        float temp = output->get({i});
        output->set({i}, temp * (1-temp));
    }
    return output;
}