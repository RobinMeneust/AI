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
 * @param size Size of the input vector
 * @return Vector of the output of the function for each component of the input vector
 */

float* Sigmoid::getValues(float* input, int size) {
    float* output = new float[size];
    for(int i=0; i<size; i++) {
        output[i] = 1.0f/(1+exp(-input[i]));
    }
    return output;
}

/**
 * For all component xi of the input vector, calculate the derivative dSigmoid(xi)/dxi and return a vector that contains the result for each xi.
 * @param input Input vector
 * @param size Size of the input vector
 * @return Vector of the derivative of the function for each component of the input vector
 */

float* Sigmoid::getDerivatives(float* input, int size) {
    float* output = getValues(input, size);
    for(int i=0; i<size; i++) {
        output[i] = output[i] * (1 - output[i]);
    }
    return output;
}