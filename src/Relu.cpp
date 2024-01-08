//
// Created by robin on 15/12/2023.
//

#include "../include/Relu.h"
#include <cmath>
#include <iostream>

float* Relu::getValues(float* input, int size) {
    float* output = new float[size];
    for(int i=0; i<size; i++) {
        output[i] = input[i] <= 0 ? 0 : input[i];
    }
    return output;
}

float** Relu::getDerivatives(float* input, int size) {
    float** output = new float*[1];
    output[0] = getValues(input, size);
    for(int i=0; i<size; i++) {
        output[0][i] = input[i] <= 0 ? 0 : 1;
    }
    return output;
}