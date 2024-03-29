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
 * @brief Softmax function used by a layer of an AI model. This class defines both the Softmax function and its derivatives
 */

class Softmax : public ActivationFunction {
public:
    Tensor *getValues(const Tensor &input, int batchSize);
    Tensor* getDerivatives(const Tensor &input, int batchSize);
private:
    float getAbsMax(float* input, int size);
};
#endif