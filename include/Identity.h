/**
 * @file Identity.h
 * @author Robin MENEUST
 * @brief Functions prototypes and class definitions of Identity.cpp
 * @date 2024-01-10
 */

#ifndef IDENTITY_H
#define IDENTITY_H

#include "ActivationFunction.h"

/**
 * @class Identity
 * @brief Identity function used by an neuron layer. This class defines both the Identity function and its derivatives
 */

class Identity : public ActivationFunction {
public:
    Tensor *getValues(const Tensor &input, int batchSize);
    Tensor* getDerivatives(const Tensor &input, int batchSize);
};

#endif