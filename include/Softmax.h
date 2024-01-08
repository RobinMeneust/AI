//
// Created by robin on 15/12/2023.
//
#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "ActivationFunction.h"

class Softmax : public ActivationFunction {
public:
    float *getValues(float *input, int size);
    float *getDerivatives(float *input, int size);
};
#endif