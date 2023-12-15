//
// Created by robin on 15/12/2023.
//

#include "../include/Sigmoid.h"
#include <cmath>
#include <iostream>

float* Sigmoid::getValue(float* input, int size) {
    float* result = new float[size];
    for(int i=0; i<size; i++) {
        result[i] = 1/(1+exp(-input[i]));
    }
    return result;
}