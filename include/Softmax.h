/**
 * @file Softmax.h
 * @author Robin MENEUST
 * @brief Functions prototypes and class definitions of Softmax.cpp
 * @date 2023-12-15
 */

#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "ActivationFunction.h"

/**
 * @class Softmax
 * @brief Softmax function used by an neuron layer. This class defines both the Softmax function and its derivatives
 */

class Softmax : public ActivationFunction {
public:
    float *getValues(float *input, int size);
    float *getDerivatives(float *input, int size);
private:
    float getAbsMax(float* input, int size);
};
#endif