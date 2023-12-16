//
// Created by robin on 15/12/2023.
//
#ifndef SIGMOID_H
#define SIGMOID_H

#include "ActivationFunction.h"

class Sigmoid : public ActivationFunction {
public:
    float getValue(float* input, inputIndex, int size);
    float getDerivative(float* input, int i, int k, int size);
};

#endif