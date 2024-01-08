#ifndef LEAKY_RELU_H
#define LEAKY_RELU_H

#include "ActivationFunction.h"

class LeakyRelu : public ActivationFunction {
public:
    float* getValues(float* input, int size);
    float* getDerivatives(float* input, int size);
};

#endif