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
 * @brief Sigmoid function used by a layer of an AI model. This class defines both the Sigmoid function and its derivatives
 */

class Sigmoid : public ActivationFunction {
public:
    Tensor *getValues(const Tensor &input, int batchSize);
    Tensor* getDerivatives(const Tensor &input, int batchSize);
};

#endif