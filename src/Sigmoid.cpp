//
// Created by robin on 15/12/2023.
//

#include "../include/Sigmoid.h"
#include <cmath>
#include <iostream>

float Sigmoid::getValue(float* input, int inputIndex, int size) {
    return 1/(1+exp(-input[inputIndex]));
}

float Sigmoid::getDerivative(float* input, int i, int k, int size) {
    if(i != k) {
        std::cerr << "Error: i must be equals to k" << std::endl;
        exit(EXIT_FAILURE);
    }
    float sigmoidXi = getValue(input, i, size);
    return sigmoidXi * (1 - sigmoidXi);
}

bool Sigmoid::isInputMultidimensional() {
    return false;
}