/**
 * @file ActivationFunction.h
 * @author Robin MENEUST
 * @brief Functions prototypes and class definitions of ActivationFunction.cpp
 * @date 2023-12-15
 */


#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include "Tensor.h"

/**
 * @class ActivationFunction
 * @brief Interface for functions used by a layer of an AI model. This class defines both the function and its derivatives
 */

class ActivationFunction {
public:
    /**
     * Let f be the activation function. If input is a tensor (whose dimension is accepted by the function) then for all component xi of the input tensor, calculate f(xi) and return a tensor, whose size is the same as the input, that contains the result for each xi.
     * @param input Input tensor
     * @param batchSize Size of the batch. It's not used for this function, but it is required by the parent class.
     * @return Tensor of the outputs of the function for each component of the input tensor
     */
    virtual Tensor *getValues(const Tensor &input, int batchSize) = 0;

    /**
     * Let f be the activation function. If input is a tensor (whose dimension is accepted by the function) then for all component xi of the input tensor, calculate df(xi)/dxi and return a tensor, whose size is the same as the input, that contains the result for each xi.
     * @param input Input tensor
     * @param batchSize Size of the batch. It's not used for this function, but it is required by the parent class.
     * @return Tensor of the derivative of the function for each component of the input tensor
     */
    virtual Tensor* getDerivatives(const Tensor &input, int batchSize) = 0;
};
#endif