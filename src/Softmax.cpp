//
// Created by robin on 15/12/2023.
//

#include "../include/Softmax.h"
#include <cmath>
#include <iostream>

float Softmax::getMax(float* input, int size) {
    float max = input[0];
    for(int i=1; i<size; i++) {
        if(max < input[i]) {
            max = input[i];
        }
    }
    return max;
}

float* Softmax::getValues(float* input, int size) {
    float max = getMax(input, size); // used to avoid overflow
    if(max<0) {
        max = 0.0f;
    }

    float* output = new float[size];
    float* expTemp = new float[size];
    float sumExp = 0.0f;
    for(int i=0; i<size; i++) {
        expTemp[i] = exp(input[i]-max);
        sumExp += expTemp[i];
    }
    for(int i=0; i<size; i++) {
        output[i] = expTemp[i] / sumExp;
    }
    delete[] expTemp;
    return output;
}

float** Softmax::getDerivatives(float* input, int size) {
    float** output = new float*[size];
    float* temp = getValues(input, size);
    for(int i=0; i<size; i++) {
        output[i] = new float[size];
        for(int j=0; j<size; j++) {
            if(i==j) {
                output[i][j] = temp[i] * (1 - temp[i]);
            } else {
                output[i][j] = -temp[i]* temp[j];
            }
        }
    }
    delete temp;
    return output;
}


bool Softmax::isActivationFunctionMultiDim() {
    return true;
}