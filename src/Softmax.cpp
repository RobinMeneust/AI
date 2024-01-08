//
// Created by robin on 15/12/2023.
//

#include "../include/Softmax.h"
#include <cmath>
#include <iostream>

float getAbsMax(float* input, int size) {
    float max = input[0];
    for(int i=1; i<size; i++) {
        float absVal = input[i] < 0 ? -input[i] : input[i];
        if(max < absVal) {
            max = absVal;
        }
    }
    return max;
}

float* Softmax::getValues(float* input, int size) {
    // Used to avoid overflow. The output doesn't change because e^(a*x) / (sum e^(a*x)) = (e^a * e^x) / (e^a * sum e^x) = e^x / (sum e^x)
    // We take 40 because exp(40) < 10^18 < 10^38 = float max value. So, if we have less than 10^20 neurons for the layer then we won't get an overflow
    // Here the exponent is between -40 and 40 since the "max" is the absolute maximum (max(abs(min),abs(max)))

    float max = getAbsMax(input, size);
    float factor = 40.0f/max;

    float* output = new float[size];
    float* expTemp = new float[size];
    float sumExp = 0.0f;
    for(int i=0; i<size; i++) {
        expTemp[i] = exp(input[i]*factor);
        sumExp += expTemp[i];
    }
    for(int i=0; i<size; i++) {
        output[i] = expTemp[i] / sumExp;
    }
    delete[] expTemp;
    return output;
}

float* Softmax::getDerivatives(float* input, int size) {
    float* output = getValues(input, size);
    for(int i=0; i<size; i++) {
        output[i] = output[i] * (1 - output[i]);
    }
    return output;
}