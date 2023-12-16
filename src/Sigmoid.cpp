//
// Created by robin on 15/12/2023.
//

#include "../include/Sigmoid.h"
#include <cmath>
#include <iostream>

float Sigmoid::getValue(float* input, inputIndex, int size) {
    return 1/(1+exp(-input[inputIndex]));
}

float Sigmoid::getDerivative(float* input, int i, int k, int size) {
    if(i != k) {
        return 0.0f;
    }
    float sigmoidXi = getValue(input, i, size);
    return sigmoidXi * (1 - sigmoidXi);
}