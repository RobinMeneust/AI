/**
 * @file LeakyRelu.h
 * @author Robin MENEUST
 * @brief Functions prototypes and class definitions of LeakyRelu.cpp
 * @date 2024-01-08
 */

#ifndef LEAKY_RELU_H
#define LEAKY_RELU_H

#include "ActivationFunction.h"

/**
 * @class LeakyRelu
 * @brief LeakyRelu function used by an neuron layer. This class defines both the LeakyRelu function and its derivatives
 */

class LeakyRelu : public ActivationFunction {
public:
    float* getValues(float* input, int size);
    float* getDerivatives(float* input, int size);
};

#endif