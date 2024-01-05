//
// Created by robin on 15/12/2023.
//
#ifndef SIGMOID_H
#define SIGMOID_H

#include "ActivationFunction.h"

class Sigmoid : public ActivationFunction {
public:
    float* getValues(float* input, int size);
    float* getDerivatives(float* input, int size);
};

#endif