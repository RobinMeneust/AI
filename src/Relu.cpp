//
// Created by robin on 15/12/2023.
//

#include "../include/Relu.h"
#include <cmath>
#include <iostream>

float Relu::getValue(float* input, int inputIndex, int size) {
    std::cout << " get val " << input << " " << inputIndex << " " << size << std::endl;
    return input[inputIndex] <= 0 ? 0 : input[inputIndex];
}

float Relu::getDerivative(float* input, int i, int k, int size) {
    std::cout << " get deriv " << input << " " << i << " " << k << " " << size << std::endl;
    if(i != k) {
        std::cerr << "Error: i must be equals to k" << std::endl;
        exit(EXIT_FAILURE);
    }
    return input[i] <= 0 ? 0 : 1;
}

bool Relu::isInputMultidimensional() {
    return false;
}