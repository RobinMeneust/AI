//
// Created by robin on 15/12/2023.
//

#include "../include/Softmax.h"
#include <cmath>
#include <iostream>

float Softmax::getValue(float* input, inputIndex, int size) {
    float result = exp(input[inputIndex]);
    float denominator = 0.0f;

    for(int j=0; j<size; j++) {
        denominator += exp(input[j]);
    }
    result /= denominator;
    return result;
}

/**
 * Calculate (d softmax(x_i)) / (d x_k)
 * Softmax takes the input (x_1, x_2, ..., x_n) and a x_i component of this vector
 * Here x_i and x_k are both components of the input vector
 * @return
 */
float Softmax::getDerivative(float* input, int i, int k, int size) {
    float sXi = getValue(input, i, size);
    if(i == k) {
        return sXi * (1-sXi);
    }
    return sXi * getValue(input, k, size);
}