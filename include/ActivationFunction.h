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
 * @brief Interface for functions used by an neuron layer. This class defines both the function and its derivatives
 */

class ActivationFunction {
public:
    /**
     * Let f be the activation function. If input is a vector for example: for all component xi of the input vector, calculate f(xi) and return a vector that contains the result for each xi. Some functions only accept specific tensor dimensions.
     * @param Tensor Input tensor
     * @return Tensor of the outputs of the function for each component of the input tensor
     */
    virtual Tensor* getValues(const Tensor &input) = 0;

    /**
     * Let f be the activation function. If input is a vector for example: for all component xi of the input vector, calculate the derivative df(xi)/dxi and return a vector that contains the result for each xi. Some functions only accept specific tensor dimensions.
     * @param input Input tensor
     * @param size Size of the input vector
     * @return Vector of the derivative of the function for each component of the input vector
     */
    virtual Tensor* getDerivatives(const Tensor &input) = 0;
};
#endif