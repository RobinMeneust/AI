//
// Created by robin on 15/12/2023.
//

#include "../include/LeakyRelu.h"
#include <cmath>
#include <iostream>

float* LeakyRelu::getValues(float* input, int size) {
    float* output = new float[size];
    for(int i=0; i<size; i++) {
        output[i] = input[i] <= 0 ? 0.01f*input[i] : input[i];
    }
    return output;
}

float* LeakyRelu::getDerivatives(float* input, int size) {
    float* output = getValues(input, size);
    for(int i=0; i<size; i++) {
        output[i] = input[i] <= 0 ? 0.01 : 1;
    }
    return output;
}