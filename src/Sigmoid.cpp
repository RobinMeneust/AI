//
// Created by robin on 15/12/2023.
//

#include "../include/Sigmoid.h"
#include <cmath>
#include <iostream>


float* Sigmoid::getValues(float* input, int size) {
    float* output = new float[size];
    for(int i=0; i<size; i++) {
        output[i] = 1.0f/(1+exp(-input[i]));
    }
    return output;
}

float** Sigmoid::getDerivatives(float* input, int size) {
    float** output = new float*[1];
    output[0] = getValues(input, size);
    for(int i=0; i<size; i++) {
        output[0][i] = output[0][i] * (1 - output[0][i]);
    }
    return output;
}

bool Sigmoid::isActivationFunctionMultiDim() {
    return false;
}