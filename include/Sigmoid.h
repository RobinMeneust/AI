/**
 * @file Sigmoid.h
 * @author Robin MENEUST
 * @brief Functions prototypes and class definitions of Sigmoid.cpp
 * @date 2023-12-15
 */

#ifndef SIGMOID_H
#define SIGMOID_H

#include "ActivationFunction.h"

/**
 * @class Sigmoid
 * @brief Sigmoid function used by an neuron layer. This class defines both the Sigmoid function and its derivatives
 */

class Sigmoid : public ActivationFunction {
public:
    float* getValues(float* input, int size);
    float* getDerivatives(float* input, int size);
};

#endif