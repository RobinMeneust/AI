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
 * @param size Size of the input vector
 * @return Vector of the output of the function for each component of the input vector
 */
float* LeakyRelu::getValues(float* input, int size) {
    float* output = new float[size];
    for(int i=0; i<size; i++) {
        output[i] = input[i] <= 0 ? 0.01f*input[i] : input[i];
    }
    return output;
}

/**
 * For all component xi of the input vector, calculate the derivative dLeakyReLU(xi)/dxi and return a vector that contains the result for each xi
 * @param input Input vector
 * @param size Size of the input vector
 * @return Vector of the derivative of the function for each component of the input vector
 */
float* LeakyRelu::getDerivatives(float* input, int size) {
    float* output = getValues(input, size);
    for(int i=0; i<size; i++) {
        output[i] = input[i] <= 0 ? 0.01 : 1;
    }
    return output;
}