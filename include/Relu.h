#ifndef RELU_H
#define RELU_H

#include "ActivationFunction.h"

class Relu : public ActivationFunction {
public:
    float* getValues(float* input, int size);
    float** getDerivatives(float* input, int size);
};

#endif