#ifndef RELU_H
#define RELU_H

#include "ActivationFunction.h"

class Relu : public ActivationFunction {
public:
    float getValue(float* input, int inputIndex, int size);
    float getDerivative(float* input, int i, int k, int size);
    bool isInputMultidimensional();
};

#endif