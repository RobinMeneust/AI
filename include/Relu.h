/**
 * @file Relu.h
 * @author Robin MENEUST
 * @brief Functions prototypes and class definitions of Relu.cpp
 * @date 2024-01-02
 */

#ifndef RELU_H
#define RELU_H

#include "ActivationFunction.h"

/**
 * @class Relu
 * @brief Relu function used by a layer of an AI model. This class defines both the Relu function and its derivatives
 */

class Relu : public ActivationFunction {
public:
    Tensor *getValues(const Tensor &input, int batchSize);
    Tensor* getDerivatives(const Tensor &input, int batchSize);
};

#endif