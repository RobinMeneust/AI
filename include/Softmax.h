//
// Created by robin on 15/12/2023.
//
#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "ActivationFunction.h"

class Softmax : public ActivationFunction {
public:
    float getValue(float* input, inputIndex, int size);
    float getDerivative(float* input, int i, int k, int size);
};

#endif