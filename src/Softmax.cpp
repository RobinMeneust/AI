//
// Created by robin on 15/12/2023.
//

#include "../include/Sigmoid.h"
#include <cmath>
#include <iostream>

float* Softmax::getValue(float* input, int size) {
    float* result = new float[size];
    for(int i=0; i<size; i++) {
        result[i] = exp(input(i));
        float denominator = 0.0f;
        for(int j=0; j<size; j++) {
            denominator += expr(input[j]);
        }
        result[i] /= denominator;
    }
    return result;
}